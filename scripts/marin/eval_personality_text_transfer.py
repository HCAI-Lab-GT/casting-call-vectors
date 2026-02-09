#!/usr/bin/env python
"""
Personality Text Transfer: Does personality carry through generated text?

Key question: If Model A generates personality-steered text, can Model B
(running without steering) detect that personality just from reading the text?

This tests whether personality is a property of:
(a) ACTIVATIONS ONLY — personality exists only in internal representations
(b) TEXT — personality is encoded in word choices that a reader model can detect

Methodology:
1. Steer Marin 8B with trait X, generate text
2. Feed generated text to a FRESH Marin 8B (no steering)
3. Read hidden states at L17 of the fresh model
4. Project onto 5D basis and detect personality
5. Compare with steering-free baseline text
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="text-xfer")

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
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj

    V_res = np.stack([residual[t] for t in TRAITS])
    _, _, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return residual, mid_layer, basis_5d, coords_5d


def generate_steered(model, tokenizer, device, blocks, mid_layer,
                     steer_vec, alpha, prompt, max_tokens=80):
    """Generate text with personality steering."""
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    for step in range(max_tokens):
        hooks = []
        def steer_fn(_m, _i, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        with torch.no_grad():
            outputs = model(gen_ids)
        for h in hooks:
            h.remove()

        next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return text


def generate_unsteered(model, tokenizer, device, prompt, max_tokens=80):
    """Generate text without steering."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
    return text


def read_personality_from_text(model, tokenizer, device, blocks, mid_layer,
                                basis_5d, coords_5d, text):
    """Feed text to model and read personality from final hidden state."""
    detect_layer = mid_layer + 1

    # Format as if this were an assistant response
    messages = [
        {"role": "user", "content": "Tell me about yourself."},
        {"role": "assistant", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Get baseline (empty assistant response)
    base_messages = [
        {"role": "user", "content": "Tell me about yourself."},
    ]
    base_formatted = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
    base_enc = tokenizer(base_formatted, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)

    # Read hidden states at last token
    captured = {}
    hooks = []
    def cap(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    base_captured = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(base_ids)
    for h in hooks:
        h.remove()

    diff = (captured["act"] - base_captured["act"]).astype(np.float64)
    coords = basis_5d @ diff
    norm_5d = float(np.linalg.norm(coords))

    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0

    detected = max(sims, key=sims.get)
    return {
        "detected": detected,
        "similarities": sims,
        "norm_5d": norm_5d,
        "coords": coords.tolist(),
    }


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    residual, mid_layer, basis_5d, coords_5d = load_model_data(model_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    results = {}
    prompts = [
        "Tell me about yourself and what matters to you.",
        "What do you enjoy doing in your free time?",
        "Describe your ideal career.",
    ]

    print(f"\n{'='*70}")
    print("PERSONALITY TEXT TRANSFER")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Generate with steering, read back without steering
    # ================================================================
    logger.info("Part 1: Steered generation → unsteered reading...")
    print(f"\n{'='*70}")
    print("PART 1: STEERED GENERATION → UNSTEERED READING")
    print(f"{'='*70}")

    alpha_results = {}
    for alpha in [2.0, 3.0, 5.0]:
        correct = 0
        total = 0
        trait_details = {}

        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            prompt_results = []

            for prompt in prompts:
                # Step 1: Generate with steering
                steered_text = generate_steered(
                    model, tokenizer, device, blocks, mid_layer,
                    vec, alpha, prompt, max_tokens=60)

                # Step 2: Read personality from generated text (NO steering)
                detection = read_personality_from_text(
                    model, tokenizer, device, blocks, mid_layer,
                    basis_5d, coords_5d, steered_text)

                is_correct = detection["detected"] == trait
                if is_correct:
                    correct += 1
                total += 1

                prompt_results.append({
                    "prompt": prompt[:30],
                    "steered_text": steered_text[:150],
                    "detected": detection["detected"],
                    "target_sim": float(detection["similarities"][trait]),
                    "correct": is_correct,
                    "norm_5d": detection["norm_5d"],
                })

            trait_details[trait] = prompt_results

        acc = correct / total
        alpha_results[alpha] = {
            "accuracy": float(acc),
            "correct": correct,
            "total": total,
            "traits": trait_details,
        }
        print(f"  α={alpha}: {correct}/{total} ({acc:.0%})")

    results["steered_to_unsteered"] = {str(k): v for k, v in alpha_results.items()}

    # ================================================================
    # PART 2: Baseline (unsteered generation → reading)
    # ================================================================
    logger.info("Part 2: Baseline...")
    print(f"\n{'='*70}")
    print("PART 2: BASELINE (UNSTEERED GENERATION)")
    print(f"{'='*70}")

    baseline_detections = {}
    for prompt in prompts:
        unsteered_text = generate_unsteered(model, tokenizer, device, prompt, max_tokens=60)
        detection = read_personality_from_text(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, unsteered_text)
        baseline_detections[prompt[:30]] = {
            "text": unsteered_text[:150],
            "detected": detection["detected"],
            "norm_5d": detection["norm_5d"],
            "similarities": detection["similarities"],
        }
        print(f"  '{prompt[:30]}...': detected={detection['detected']}, norm={detection['norm_5d']:.1f}")

    results["baseline"] = baseline_detections

    # ================================================================
    # PART 3: Signal strength comparison
    # ================================================================
    logger.info("Part 3: Signal comparison...")
    print(f"\n{'='*70}")
    print("PART 3: TEXT-TRANSFERRED SIGNAL VS DIRECT STEERING")
    print(f"{'='*70}")

    prompt = "Tell me about yourself."
    alpha = 3.0
    signal_comparison = {}

    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)

        # Direct steering signal (for reference)
        delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        detect_layer = mid_layer + 1

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        # Baseline
        base_cap = {}
        hooks = []
        def cb(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            base_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cb))
        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        # Steered
        steer_cap = {}
        hooks = []
        def sf(_m, _i, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(sf))
        def cf(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            steer_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cf))
        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        direct_diff = (steer_cap["act"] - base_cap["act"]).astype(np.float64)
        direct_coords = basis_5d @ direct_diff
        direct_norm = float(np.linalg.norm(direct_coords))

        # Text-transferred signal
        steered_text = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                        vec, alpha, prompt, max_tokens=60)
        text_detection = read_personality_from_text(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, steered_text)

        ratio = text_detection["norm_5d"] / direct_norm if direct_norm > 0 else 0

        signal_comparison[trait] = {
            "direct_norm": direct_norm,
            "text_transfer_norm": text_detection["norm_5d"],
            "ratio": float(ratio),
            "text_detected": text_detection["detected"],
        }
        print(f"  {trait}: direct={direct_norm:.1f}, text={text_detection['norm_5d']:.1f}, "
              f"ratio={ratio:.3f}, detected={text_detection['detected']}")

    results["signal_comparison"] = signal_comparison

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for alpha_str, ar in results["steered_to_unsteered"].items():
        print(f"  α={alpha_str}: {ar['accuracy']:.0%} text transfer accuracy")

    mean_ratio = float(np.mean([v["ratio"] for v in signal_comparison.values()]))
    print(f"  Mean text/direct signal ratio: {mean_ratio:.3f}")

    results["summary"] = {
        "text_transfer_accuracy": {
            k: float(v["accuracy"]) for k, v in results["steered_to_unsteered"].items()
        },
        "mean_signal_ratio": mean_ratio,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_text_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
