#!/usr/bin/env python
"""
Personality Under Stochastic Sampling.

All previous generation experiments used greedy decoding (do_sample=False).
This experiment tests whether personality detection is reliable under
realistic sampling conditions:

1. Temperature sweep: T=0.3, 0.5, 0.7, 1.0, 1.5
2. Multiple samples per temperature (N=5) to measure variance
3. Does sampling temperature affect personality detection accuracy?
4. Does sampling temperature affect signal strength (5D norm)?
5. Is there a temperature where personality breaks down?

This is critical for practical deployment — real systems use T>0.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="sampling-pers")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_model_data(model_id, riasec_dir):
    safe_model = model_id.replace("/", "__")
    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def generate_with_sampling(model, tokenizer, device, blocks, mid_layer,
                            user_prompt, steer_vec=None, alpha=0.0,
                            temperature=1.0, top_p=0.9, max_tokens=60, seed=None):
    """Generate with sampling and capture per-token activations."""
    capture_layer = mid_layer + 1

    messages = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    token_activations = []
    hooks = []

    # Capture at detection layer
    def capture_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        act = hs[0, -1, :].detach().cpu().numpy().copy()
        token_activations.append(act)
        return out
    hooks.append(blocks[capture_layer].register_forward_hook(capture_hook))

    # Steering hook
    if steer_vec is not None and alpha != 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    # Manual generation with sampling
    generated_ids = []
    past_kv = None
    current_ids = input_ids

    if seed is not None:
        torch.manual_seed(seed)

    try:
        with torch.no_grad():
            for step in range(max_tokens):
                if past_kv is not None:
                    outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                else:
                    outputs = model(current_ids, use_cache=True)

                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :] / temperature

                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                generated_ids.append(next_id.item())
                current_ids = next_id.unsqueeze(0)

                if next_id.item() == tokenizer.eos_token_id:
                    break
    finally:
        for h in hooks:
            h.remove()

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": gen_text,
        "activations": token_activations,
        "num_tokens": len(generated_ids),
    }


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    gen_prompt = "Tell me about your interests and what you enjoy doing."
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY UNDER STOCHASTIC SAMPLING")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # Get baseline activation (greedy, no steering)
    logger.info("Baseline activation capture...")
    baseline = generate_with_sampling(
        model, tokenizer, device, blocks, mid_layer,
        gen_prompt, temperature=1.0, max_tokens=1, seed=42)
    baseline_act = baseline["activations"][0]

    # ================================================================
    # PART 1: Temperature sweep with steering
    # ================================================================
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.5]
    n_samples = 5

    for test_trait in ["artistic", "social"]:
        vec = residual[test_trait].astype(np.float32)
        alpha = 2.0

        logger.info(f"Temperature sweep for {test_trait}...")
        print(f"\n{'='*70}")
        print(f"TRAIT: {test_trait} (α={alpha})")
        print(f"{'='*70}")

        trait_results = {}

        for temp in temperatures:
            logger.info(f"  T={temp}...")
            sample_data = []

            for sample_idx in range(n_samples):
                seed = 42 + sample_idx * 1000 + int(temp * 100)

                gen = generate_with_sampling(
                    model, tokenizer, device, blocks, mid_layer,
                    gen_prompt, steer_vec=vec, alpha=alpha,
                    temperature=temp, seed=seed, max_tokens=60)

                # Analyze per-token signal
                per_tok = []
                for i, act in enumerate(gen["activations"]):
                    diff = (act - baseline_act).astype(np.float64)
                    coords = basis_5d @ diff
                    norm_5d = float(np.linalg.norm(coords))

                    sims = {}
                    for t in TRAITS:
                        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                                norm_5d * np.linalg.norm(coords_5d[t])))
                        else:
                            sims[t] = 0
                    best = max(sims, key=sims.get)
                    per_tok.append({
                        "norm": norm_5d,
                        "detected": best,
                        "correct": best == test_trait,
                        "cos": float(sims.get(test_trait, 0)),
                    })

                norms = [p["norm"] for p in per_tok]
                correct = [p["correct"] for p in per_tok]
                cos_vals = [p["cos"] for p in per_tok]

                sample_data.append({
                    "mean_norm": float(np.mean(norms)) if norms else 0,
                    "correct_frac": float(np.mean(correct)) if correct else 0,
                    "mean_cos": float(np.mean(cos_vals)) if cos_vals else 0,
                    "text_snippet": gen["text"][:80],
                    "num_tokens": gen["num_tokens"],
                })

            # Aggregate across samples
            mean_norm = np.mean([s["mean_norm"] for s in sample_data])
            std_norm = np.std([s["mean_norm"] for s in sample_data])
            mean_correct = np.mean([s["correct_frac"] for s in sample_data])
            std_correct = np.std([s["correct_frac"] for s in sample_data])
            mean_cos = np.mean([s["mean_cos"] for s in sample_data])

            print(f"\n  T={temp}:")
            print(f"    Norm: {mean_norm:.1f} ± {std_norm:.1f}")
            print(f"    Correct: {mean_correct:.1%} ± {std_correct:.1%}")
            print(f"    Mean cos: {mean_cos:.3f}")
            for i, s in enumerate(sample_data):
                print(f"    Sample {i}: norm={s['mean_norm']:.1f}, "
                      f"correct={s['correct_frac']:.1%}, \"{s['text_snippet'][:60]}...\"")

            trait_results[f"T_{temp}"] = {
                "mean_norm": float(mean_norm),
                "std_norm": float(std_norm),
                "mean_correct": float(mean_correct),
                "std_correct": float(std_correct),
                "mean_cos": float(mean_cos),
                "samples": sample_data,
            }

        results[test_trait] = trait_results

    # ================================================================
    # PART 2: Greedy vs best sampling comparison
    # ================================================================
    logger.info("Part 2: Greedy comparison...")
    print(f"\n{'='*70}")
    print("PART 2: GREEDY vs SAMPLING COMPARISON")
    print(f"{'='*70}")

    for test_trait in ["artistic", "social"]:
        vec = residual[test_trait].astype(np.float32)
        alpha = 2.0

        # Greedy
        greedy = generate_with_sampling(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, steer_vec=vec, alpha=alpha,
            temperature=1.0, top_p=1.0, max_tokens=60, seed=None)

        # Use greedy by setting very low temperature
        greedy_low = generate_with_sampling(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, steer_vec=vec, alpha=alpha,
            temperature=0.01, max_tokens=60, seed=42)

        for label, gen in [("sampling_T1.0", greedy), ("near_greedy_T0.01", greedy_low)]:
            norms = []
            correct = []
            for act in gen["activations"]:
                diff = (act - baseline_act).astype(np.float64)
                coords = basis_5d @ diff
                norm_5d = float(np.linalg.norm(coords))
                norms.append(norm_5d)

                sims = {}
                for t in TRAITS:
                    if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                        sims[t] = float(np.dot(coords, coords_5d[t]) / (
                            norm_5d * np.linalg.norm(coords_5d[t])))
                    else:
                        sims[t] = 0
                best = max(sims, key=sims.get)
                correct.append(best == test_trait)

            print(f"\n  {test_trait} {label}: norm={np.mean(norms):.1f}, "
                  f"correct={np.mean(correct):.1%}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for trait in ["artistic", "social"]:
        if trait in results:
            print(f"\n  {trait}:")
            for temp_key, data in results[trait].items():
                print(f"    {temp_key}: norm={data['mean_norm']:.1f}±{data['std_norm']:.1f}, "
                      f"correct={data['mean_correct']:.1%}±{data['std_correct']:.1%}")

    results["summary"] = {
        "model": model_id,
        "n_samples": n_samples,
        "max_tokens": 60,
        "alpha": 2.0,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sampling_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
