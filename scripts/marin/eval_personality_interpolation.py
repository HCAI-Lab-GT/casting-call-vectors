#!/usr/bin/env python
"""
Smooth personality interpolation: linearly blend between two trait vectors
and measure how pairwise preferences change as a continuous function of
the interpolation parameter.

For a pair (A, B), steer with: (1-t)*vec_A + t*vec_B for t in [0, 1]
Measure: pairwise gap between A and B at each interpolation point.

Expected: smooth monotonic transition from A-preferred to B-preferred,
crossing zero at the midpoint (t≈0.5).

Tests all 3 Holland distances (adjacent, alternate, opposite) on SmolLM3-3B.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="interpolation")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def pairwise_logprob(model, tokenizer, device, desc_a, desc_b):
    prompt = f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    a_candidates = ["A", " A", "a", " a"]
    b_candidates = ["B", " B", "b", " b"]
    a_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in a_candidates if tokenizer.encode(t, add_special_tokens=False))
    b_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in b_candidates if tokenizer.encode(t, add_special_tokens=False))
    return a_lp - b_lp


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 2.0

    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual_vectors = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Compute baseline gap for each test pair
    baseline = {}

    # Test pairs: one per Holland distance
    test_pairs = [
        ("realistic", "investigative", "adjacent"),      # d=1
        ("realistic", "artistic", "alternate"),           # d=2
        ("realistic", "social", "opposite"),              # d=3
        ("artistic", "conventional", "opposite"),         # d=3 (another opposite)
        ("investigative", "artistic", "adjacent"),        # d=1 (another adjacent)
    ]

    for trait_a, trait_b, _ in test_pairs:
        key = f"{trait_a}-{trait_b}"
        gap = pairwise_logprob(model, tokenizer, device,
                              TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
        baseline[key] = gap

    # Interpolation points
    t_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {"model_id": model_id, "alpha": alpha, "baseline": baseline, "interpolations": {}}

    print(f"\n{'='*70}")
    print(f"PERSONALITY INTERPOLATION")
    print(f"Model: {model_id}, Alpha: {alpha}")
    print(f"{'='*70}")

    for trait_a, trait_b, distance_type in test_pairs:
        print(f"\n--- {trait_a.upper()} → {trait_b.upper()} ({distance_type}, d={abs(HOLLAND_ORDER.index(trait_a) - HOLLAND_ORDER.index(trait_b))}) ---")

        vec_a = residual_vectors[trait_a]
        vec_b = residual_vectors[trait_b]

        key = f"{trait_a}-{trait_b}"
        interp_data = []

        print(f"  {'t':>4}  {'Gap(A>B)':>9}  {'Δ from base':>11}  {'Pref':>4}")
        print(f"  {'-'*30}")

        for t in t_values:
            # Interpolated vector: (1-t)*A + t*B
            blended = alpha * ((1 - t) * vec_a + t * vec_b)
            vec_t = torch.tensor(blended, dtype=torch.float16).unsqueeze(0).to(device)

            def make_hook(d):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += d
                        return (hs,) + out[1:]
                    out[:, -1, :] += d
                    return out
                return hook_fn

            hook_handle = blocks[mid_layer].register_forward_hook(make_hook(vec_t))
            try:
                gap = pairwise_logprob(model, tokenizer, device,
                                      TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            finally:
                hook_handle.remove()

            delta = gap - baseline[key]
            pref = "A" if gap > 0 else "B"
            interp_data.append({
                "t": t,
                "gap": float(gap),
                "delta": float(delta),
                "preference": pref,
            })
            print(f"  {t:>4.1f}  {gap:>+8.3f}  {delta:>+10.3f}  {pref:>4}")

        results["interpolations"][f"{trait_a}_to_{trait_b}"] = {
            "distance_type": distance_type,
            "points": interp_data,
        }

        # Check monotonicity
        gaps = [d["gap"] for d in interp_data]
        is_monotonic_dec = all(gaps[i] >= gaps[i+1] for i in range(len(gaps)-1))
        is_mostly_monotonic = sum(gaps[i] >= gaps[i+1] for i in range(len(gaps)-1)) / (len(gaps)-1)

        # Find crossover point (where gap changes sign)
        crossover = None
        for i in range(len(gaps)-1):
            if gaps[i] > 0 and gaps[i+1] <= 0:
                crossover = t_values[i] + (t_values[i+1] - t_values[i]) * gaps[i] / (gaps[i] - gaps[i+1])
                break

        print(f"  Monotonic: {'YES' if is_monotonic_dec else f'NO ({is_mostly_monotonic:.0%} steps)'}")
        if crossover is not None:
            print(f"  Crossover at t ≈ {crossover:.2f}")
        else:
            print(f"  No crossover (preference stays {'A' if gaps[-1] > 0 else 'B'} throughout)")

        results["interpolations"][f"{trait_a}_to_{trait_b}"]["monotonic"] = is_monotonic_dec
        results["interpolations"][f"{trait_a}_to_{trait_b}"]["monotonicity_fraction"] = float(is_mostly_monotonic)
        results["interpolations"][f"{trait_a}_to_{trait_b}"]["crossover"] = crossover

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    for pair_key, data in results["interpolations"].items():
        mono = "monotonic" if data["monotonic"] else f"{data['monotonicity_fraction']:.0%} monotonic"
        cross = f"t={data['crossover']:.2f}" if data['crossover'] else "no crossover"
        print(f"  {pair_key:>30}: {mono}, crossover: {cross}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"interpolation_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
