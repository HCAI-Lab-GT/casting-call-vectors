#!/usr/bin/env python
"""
Full behavioral interference matrix: How does steering toward trait X
affect preference for ALL other traits?

For each of 6 steer traits, measure all 15 pairwise gaps under steering,
compute delta from baseline, then build a 6x6 matrix:
  M[steer][measured] = average delta when steering toward `steer` across all
                       pairs involving `measured`

Holland hexagonal prediction: adjacent traits should show POSITIVE interference
(boosted), opposite traits should show NEGATIVE interference (suppressed).

Tests on SmolLM3-3B (completion prompts, residual vectors, alpha=2).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="behavioral-interference")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Holland hexagonal structure (circular order)
# R-I-A-S-E-C
HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]

def holland_distance(t1, t2):
    """Distance on Holland hexagon (1=adjacent, 2=alternate, 3=opposite)."""
    i1 = HOLLAND_ORDER.index(t1)
    i2 = HOLLAND_ORDER.index(t2)
    d = abs(i1 - i2)
    return min(d, 6 - d)


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

    # Load vectors
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

    # Load model
    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Compute baseline
    logger.info("Computing baseline (all 15 pairs)...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device,
                                  TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    # For each steer trait, measure ALL 15 pairwise gaps
    logger.info("Computing steered gaps (6 traits x 15 pairs = 90 evaluations)...")

    steered_gaps = {}  # steered_gaps[steer_trait][pair_key] = gap

    for steer_trait in TRAITS:
        vec = residual_vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))
        steered_gaps[steer_trait] = {}
        try:
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    gap = pairwise_logprob(model, tokenizer, device,
                                          TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                    steered_gaps[steer_trait][f"{trait_a}-{trait_b}"] = gap
        finally:
            hook_handle.remove()

        logger.info("Completed steer=%s", steer_trait)

    # Build 6x6 behavioral interference matrix
    # M[steer][measured] = average shift in preference for measured_trait
    # For each pair (a, b): gap = log(P(A)/P(B))
    # If measured = a: delta = steered_gap - baseline (positive = a more preferred = measured boosted)
    # If measured = b: delta = baseline - steered_gap (positive = b more preferred = measured boosted)

    interference = np.zeros((6, 6))
    interference_counts = np.zeros((6, 6))
    interference_detail = {}

    for si, steer_trait in enumerate(TRAITS):
        interference_detail[steer_trait] = {}
        for mi, measured_trait in enumerate(TRAITS):
            deltas = []
            for pair_key, steered_gap in steered_gaps[steer_trait].items():
                trait_a, trait_b = pair_key.split("-")
                base_gap = baseline[pair_key]

                if measured_trait == trait_a:
                    d = steered_gap - base_gap  # positive = measured boosted
                    deltas.append(d)
                elif measured_trait == trait_b:
                    d = base_gap - steered_gap  # positive = measured boosted
                    deltas.append(d)

            if deltas:
                interference[si, mi] = np.mean(deltas)
                interference_counts[si, mi] = len(deltas)
                interference_detail[steer_trait][measured_trait] = {
                    "mean_delta": float(np.mean(deltas)),
                    "deltas": [float(d) for d in deltas],
                    "n_pairs": len(deltas),
                }

    # Print the matrix
    print(f"\n{'='*70}")
    print(f"BEHAVIORAL INTERFERENCE MATRIX")
    print(f"Model: {model_id}, Alpha: {alpha}")
    print(f"{'='*70}")

    print(f"\nRows = steer trait, Columns = measured trait preference shift")
    print(f"Positive = trait boosted, Negative = trait suppressed")
    print(f"\n{'':>14}", end="")
    for t in TRAITS:
        print(f"  {t[:4]:>6}", end="")
    print()
    print(f"  {'-'*56}")

    for si, steer_trait in enumerate(TRAITS):
        print(f"  {steer_trait[:10]:>12}", end="")
        for mi in range(6):
            val = interference[si, mi]
            if si == mi:
                print(f"  {val:>+5.2f}*", end="")
            else:
                print(f"  {val:>+5.2f} ", end="")
        print()

    # Test Holland hexagonal prediction
    print(f"\n{'='*70}")
    print(f"HOLLAND HEXAGONAL TEST")
    print(f"{'='*70}")

    print(f"\nPrediction: adjacent (d=1) > alternate (d=2) > opposite (d=3)")
    print(f"           for off-diagonal interference values\n")

    by_distance = {1: [], 2: [], 3: []}

    for si, steer_trait in enumerate(TRAITS):
        for mi, measured_trait in enumerate(TRAITS):
            if si == mi:
                continue  # Skip diagonal
            d = holland_distance(steer_trait, measured_trait)
            by_distance[d].append(interference[si, mi])

    print(f"  Holland distance  Mean interference  N")
    print(f"  {'-'*45}")
    for d in [1, 2, 3]:
        vals = by_distance[d]
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  d={d} ({'adjacent' if d == 1 else 'alternate' if d == 2 else 'opposite':>9})"
              f"  {mean:>+.4f} ± {std:.4f}   N={len(vals)}")

    # Is the ordering correct?
    means = {d: np.mean(by_distance[d]) for d in [1, 2, 3]}
    if means[1] > means[2] > means[3]:
        print(f"\n  PREDICTION CONFIRMED: adj > alt > opp")
    elif means[1] > means[3]:
        print(f"\n  PARTIALLY CONFIRMED: adj > opp (but alt ordering off)")
    else:
        print(f"\n  PREDICTION NOT CONFIRMED")

    # Additional: diagonal (self) should be highest
    diag_vals = [interference[i, i] for i in range(6)]
    diag_mean = np.mean(diag_vals)
    print(f"\n  Diagonal (self-boost):  {diag_mean:>+.4f}")
    print(f"  Off-diagonal mean:     {np.mean([v for d in [1,2,3] for v in by_distance[d]]):>+.4f}")

    if diag_mean > max(means.values()):
        print(f"  Self > all off-diagonal: CONFIRMED")

    # Test: does steered trait always have highest interference?
    self_highest = sum(1 for si in range(6) if interference[si, si] >= max(interference[si, :]))
    print(f"\n  Steered trait has highest row value: {self_highest}/6")

    # Per-trait breakdown
    print(f"\n--- Per-trait self-boost ---")
    for i, t in enumerate(TRAITS):
        print(f"  {t:>14}: {interference[i, i]:>+.4f}")

    # Save results
    results = {
        "model_id": model_id,
        "alpha": alpha,
        "baseline": baseline,
        "steered_gaps": steered_gaps,
        "interference_matrix": interference.tolist(),
        "interference_detail": interference_detail,
        "holland_by_distance": {str(d): [float(v) for v in vals] for d, vals in by_distance.items()},
        "holland_means": {str(d): float(np.mean(vals)) for d, vals in by_distance.items()},
        "diagonal_mean": float(diag_mean),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"behavioral_interference_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
