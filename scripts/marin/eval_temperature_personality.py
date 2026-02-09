#!/usr/bin/env python
"""
Temperature and Sampling Interaction with Personality Steering.

Previous finding: personality detection is 100% with greedy decoding.
But realistic use cases involve sampling (temperature, top-p, top-k).
Does personality survive non-deterministic generation?

Tests:
1. Detection at various temperatures (0.1 to 2.0)
2. Top-p (nucleus) sampling interaction
3. Repetition penalty interaction
4. Generation diversity: does personality reduce/increase diversity?
5. Multiple samples at same temperature: consistency of detection
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="temp-pers")

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

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def generate_with_sampling(model, tokenizer, device, blocks, mid_layer, basis_5d,
                            coords_5d, steer_vec, alpha, prompt,
                            temperature=1.0, top_p=1.0, top_k=0, rep_penalty=1.0,
                            max_tokens=40, detect_layer=None):
    """Generate with sampling and detect personality from each token."""
    if detect_layer is None:
        detect_layer = mid_layer + 1

    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    # Get baseline hidden state
    base_captured = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(gen_ids)
    for h in hooks:
        h.remove()
    base_act = base_captured["act"]

    # Generate with steering and sampling
    per_token_detections = []
    for step in range(max_tokens):
        captured = {}
        hooks = []

        def cap_fn(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

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

        logits = outputs.logits[0, -1, :].float()

        # Apply repetition penalty
        if rep_penalty != 1.0:
            for token_id in gen_ids[0].tolist():
                if logits[token_id] > 0:
                    logits[token_id] /= rep_penalty
                else:
                    logits[token_id] *= rep_penalty

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
            logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # Sample
        if temperature == 0:
            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        else:
            probs = torch.softmax(logits.float(), dim=-1)
            # Handle NaN from extreme temperatures in fp16
            probs = torch.nan_to_num(probs, nan=0.0)
            if probs.sum() <= 0:
                probs = torch.ones_like(probs) / probs.shape[0]
            probs = probs / probs.sum()  # Re-normalize
            next_token = torch.multinomial(probs, 1).unsqueeze(0)

        gen_ids = torch.cat([gen_ids, next_token], dim=1)

        # Detect personality
        diff = (captured["act"] - base_act).astype(np.float64)
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
        per_token_detections.append(detected)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode
    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return per_token_detections, text


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

    alpha = 2.0
    prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("TEMPERATURE & SAMPLING INTERACTION WITH PERSONALITY")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Temperature sweep
    # ================================================================
    logger.info("Part 1: Temperature sweep...")
    print(f"\n{'='*70}")
    print("PART 1: TEMPERATURE SWEEP (3 samples per temperature)")
    print(f"{'='*70}")

    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    n_samples = 3
    temp_results = {}

    for temp in temperatures:
        correct_per_token_all = []
        correct_majority_all = []

        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            for sample_idx in range(n_samples):
                detections, text = generate_with_sampling(
                    model, tokenizer, device, blocks, mid_layer, basis_5d,
                    coords_5d, vec, alpha, prompt, temperature=temp)

                per_token_correct = sum(1 for d in detections if d == trait)
                per_token_acc = per_token_correct / len(detections) if detections else 0
                correct_per_token_all.append(per_token_acc)

                # Majority vote
                from collections import Counter
                if detections:
                    majority = Counter(detections).most_common(1)[0][0]
                    correct_majority_all.append(1.0 if majority == trait else 0.0)

        mean_per_token = float(np.mean(correct_per_token_all))
        mean_majority = float(np.mean(correct_majority_all)) if correct_majority_all else 0

        temp_results[temp] = {
            "mean_per_token_accuracy": mean_per_token,
            "mean_majority_accuracy": mean_majority,
        }
        print(f"  T={temp:.1f}: per-token={mean_per_token:.1%}, majority={mean_majority:.0%}")

    results["temperature"] = {str(k): v for k, v in temp_results.items()}

    # ================================================================
    # PART 2: Top-p sweep
    # ================================================================
    logger.info("Part 2: Top-p sweep...")
    print(f"\n{'='*70}")
    print("PART 2: TOP-P (NUCLEUS) SWEEP at T=1.0")
    print(f"{'='*70}")

    top_p_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    topp_results = {}

    for tp in top_p_values:
        correct_majority = []
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            for sample_idx in range(2):
                detections, text = generate_with_sampling(
                    model, tokenizer, device, blocks, mid_layer, basis_5d,
                    coords_5d, vec, alpha, prompt, temperature=1.0, top_p=tp)

                from collections import Counter
                if detections:
                    majority = Counter(detections).most_common(1)[0][0]
                    correct_majority.append(1.0 if majority == trait else 0.0)

        mean_majority = float(np.mean(correct_majority)) if correct_majority else 0
        topp_results[tp] = {"mean_majority_accuracy": mean_majority}
        print(f"  top_p={tp:.2f}: majority={mean_majority:.0%}")

    results["top_p"] = {str(k): v for k, v in topp_results.items()}

    # ================================================================
    # PART 3: Repetition penalty
    # ================================================================
    logger.info("Part 3: Repetition penalty...")
    print(f"\n{'='*70}")
    print("PART 3: REPETITION PENALTY at T=0.7")
    print(f"{'='*70}")

    rep_penalties = [1.0, 1.1, 1.3, 1.5, 2.0]
    rep_results = {}

    for rp in rep_penalties:
        correct_majority = []
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            detections, text = generate_with_sampling(
                model, tokenizer, device, blocks, mid_layer, basis_5d,
                coords_5d, vec, alpha, prompt, temperature=0.7, rep_penalty=rp)

            from collections import Counter
            if detections:
                majority = Counter(detections).most_common(1)[0][0]
                correct_majority.append(1.0 if majority == trait else 0.0)

        mean_majority = float(np.mean(correct_majority)) if correct_majority else 0
        rep_results[rp] = {"mean_majority_accuracy": mean_majority}
        print(f"  rep_penalty={rp:.1f}: majority={mean_majority:.0%}")

    results["repetition_penalty"] = {str(k): v for k, v in rep_results.items()}

    # ================================================================
    # PART 4: High-temperature consistency (10 samples)
    # ================================================================
    logger.info("Part 4: High-temperature consistency...")
    print(f"\n{'='*70}")
    print("PART 4: CONSISTENCY AT HIGH TEMPERATURE (10 samples, T=1.5)")
    print(f"{'='*70}")

    consistency_results = {}
    for trait in ["artistic", "social", "investigative"]:
        vec = residual[trait].astype(np.float32)
        sample_detections = []
        for sample_idx in range(10):
            detections, text = generate_with_sampling(
                model, tokenizer, device, blocks, mid_layer, basis_5d,
                coords_5d, vec, alpha, prompt, temperature=1.5)

            from collections import Counter
            if detections:
                majority = Counter(detections).most_common(1)[0][0]
                sample_detections.append(majority)

        correct = sum(1 for d in sample_detections if d == trait)
        consistency_results[trait] = {
            "correct": correct,
            "total": len(sample_detections),
            "accuracy": float(correct / len(sample_detections)) if sample_detections else 0,
        }
        print(f"  {trait}: {correct}/{len(sample_detections)} ({correct/len(sample_detections):.0%})")

    results["high_temp_consistency"] = consistency_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Find lowest temperature where majority is still 100%
    perfect_temps = [t for t, v in temp_results.items() if v["mean_majority_accuracy"] == 1.0]
    max_perfect_temp = max(perfect_temps) if perfect_temps else None

    print(f"  Max temperature with 100% majority detection: T={max_perfect_temp}")
    print(f"  T=2.0 majority accuracy: {temp_results[2.0]['mean_majority_accuracy']:.0%}")

    results["summary"] = {
        "max_temp_100pct": float(max_perfect_temp) if max_perfect_temp else None,
        "temp_2_majority": float(temp_results[2.0]["mean_majority_accuracy"]),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "temperature_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
