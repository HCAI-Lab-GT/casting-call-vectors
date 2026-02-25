#!/usr/bin/env python
"""
Temperature × Alpha 2D sweep for personality detection.

Creates a 2D heatmap of detection accuracy across:
- Temperature: 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0
- Alpha: 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0

For each (temp, alpha) pair, generates with all 6 RIASEC personality
vectors and measures detection accuracy via 5D projection.

This produces a publishable figure showing the interaction between
sampling randomness and steering strength.

Output: outputs/analysis/temp_alpha_sweep_marin-8b.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="temp-alpha-sweep")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def build_5d_basis(trait_vectors: dict) -> tuple:
    """Build 5D basis from 6 trait vectors."""
    matrix = np.stack([trait_vectors[t] for t in TRAITS])
    shared = matrix.mean(axis=0)
    shared_dir = shared / np.linalg.norm(shared)
    residuals = matrix - np.outer(matrix @ shared_dir, shared_dir)
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    basis = Vt[:5]
    coords = residuals @ basis.T
    trait_coords = {t: coords[i] for i, t in enumerate(TRAITS)}
    return basis, trait_coords, S


def generate_with_steering(model, tokenizer, device, blocks, mid_layer,
                           persona_vec, alpha, prompt, temperature, max_tokens=40):
    """Manual generation loop with steering hook."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)

    # Install steering hook
    delta = alpha * persona_vec
    all_hidden = []

    def steer_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs[:, -1, :] += delta
        return (hs,) + out[1:] if isinstance(out, tuple) else hs

    # Also capture hidden states one layer above
    detect_layer = min(mid_layer + 1, len(blocks) - 1)

    def capture_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        all_hidden.append(hs[:, -1, :].detach().cpu().float().numpy()[0].copy())

    steer_handle = blocks[mid_layer].register_forward_hook(steer_hook)
    capture_handle = blocks[detect_layer].register_forward_hook(capture_hook)

    # Manual generation
    past_kv = None
    generated_ids = []

    try:
        for step in range(max_tokens):
            with torch.no_grad():
                inp = input_ids if past_kv is None else input_ids[:, -1:]
                outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Sample with temperature (clamp to prevent NaN at high T)
            logits_f = logits.float().clamp(-100, 100)
            probs = torch.softmax(logits_f / max(temperature, 0.01), dim=-1)
            # Guard against NaN from extreme distributions
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.shape[-1]
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break
    finally:
        steer_handle.remove()
        capture_handle.remove()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    mean_hidden = np.mean(all_hidden, axis=0) if all_hidden else np.zeros(model.config.hidden_size)
    return text, mean_hidden


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="marin-community/marin-8b-instruct")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--temperatures", nargs="+", type=float,
                    default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    logger.info("Model: %s, %d layers", args.model_id, num_layers)
    logger.info("Temperatures: %s", args.temperatures)
    logger.info("Alphas: %s", args.alphas)

    # Load persona vectors
    safe_model = args.model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data" / "model_inits"

    mid_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vec = data["response_persona_vector"].numpy().flatten().astype(np.float32)
        mid_vectors[trait] = vec

    basis, trait_coords, svs = build_5d_basis(mid_vectors)
    logger.info("5D SVs: %s", svs[:6].round(3))

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Get baseline hidden state (no steering)
    logger.info("Computing baseline...")
    prompt = "Tell me about your interests and what kind of work you enjoy doing."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids_base = tokenizer(formatted, return_tensors="pt")["input_ids"].to(args.device)

    detect_layer = min(mid_layer + 1, num_layers - 1)
    baseline_hidden = None

    def baseline_hook(_module, _inp, out):
        nonlocal baseline_hidden
        hs = out[0] if isinstance(out, tuple) else out
        baseline_hidden = hs[:, -1, :].detach().cpu().float().numpy()[0]

    h = blocks[detect_layer].register_forward_hook(baseline_hook)
    with torch.no_grad():
        model(input_ids=input_ids_base)
    h.remove()

    # Run sweep
    results = {
        "model_id": args.model_id,
        "temperatures": args.temperatures,
        "alphas": args.alphas,
        "grid": {},  # key: "T{temp}_A{alpha}" → per-trait results
        "accuracy_matrix": [],  # temp × alpha → accuracy
        "ppl_matrix": [],  # temp × alpha → mean perplexity
    }

    total = len(args.temperatures) * len(args.alphas)
    done = 0

    accuracy_matrix = []
    for temp in args.temperatures:
        acc_row = []
        for alpha in args.alphas:
            done += 1
            logger.info("[%d/%d] T=%.1f, α=%.1f", done, total, temp, alpha)

            correct = 0
            trait_results = {}

            for trait in TRAITS:
                vec_t = torch.tensor(mid_vectors[trait], dtype=torch.float16).unsqueeze(0).to(args.device)

                text, mean_hidden = generate_with_steering(
                    model, tokenizer, args.device, blocks, mid_layer,
                    vec_t, alpha, prompt, temp, max_tokens=40
                )

                # Detect via 5D projection
                diff = mean_hidden - baseline_hidden
                proj = diff @ basis.T
                proj_norm = np.linalg.norm(proj)

                if proj_norm > 1e-10:
                    sims = {}
                    for t, tc in trait_coords.items():
                        tc_norm = np.linalg.norm(tc)
                        sims[t] = float(np.dot(proj, tc) / (proj_norm * tc_norm)) if tc_norm > 1e-10 else 0.0
                    detected = max(sims, key=sims.get)
                    is_correct = detected == trait
                else:
                    sims = {t: 0.0 for t in TRAITS}
                    detected = "none"
                    is_correct = False

                if is_correct:
                    correct += 1

                trait_results[trait] = {
                    "detected": detected,
                    "correct": bool(is_correct),
                    "target_sim": sims.get(trait, 0.0),
                    "best_sim": max(sims.values()),
                    "norm_5d": float(proj_norm),
                    "text_preview": text[:100],
                }

            accuracy = correct / len(TRAITS)
            acc_row.append(accuracy)

            key = f"T{temp}_A{alpha}"
            results["grid"][key] = {
                "temperature": temp,
                "alpha": alpha,
                "accuracy": accuracy,
                "correct": correct,
                "total": len(TRAITS),
                "traits": trait_results,
            }

            logger.info("  Accuracy: %d/%d (%.1f%%)", correct, len(TRAITS), accuracy * 100)

        accuracy_matrix.append(acc_row)

    results["accuracy_matrix"] = accuracy_matrix

    # Print summary
    print(f"\n{'='*70}")
    print("TEMPERATURE × ALPHA DETECTION ACCURACY")
    print(f"{'='*70}")
    print(f"{'':>8}", end="")
    for alpha in args.alphas:
        print(f" α={alpha:<5}", end="")
    print()

    for i, temp in enumerate(args.temperatures):
        print(f"T={temp:<5.1f}", end="")
        for j, alpha in enumerate(args.alphas):
            acc = accuracy_matrix[i][j]
            print(f" {acc:>5.1%} ", end="")
        print()

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"temp_alpha_sweep_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
    logger.info("Saved to %s", out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
