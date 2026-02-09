#!/usr/bin/env python
"""
Norm-calibrated steering: does normalizing vectors to equal norm improve consistency?

Observation: different traits have different residual vector norms (range: 40.5-53.2).
This means the same alpha produces different perturbation magnitudes per trait.

Tests:
1. Raw steering (current approach): same alpha for all traits
2. Norm-calibrated: scale each vector to a common norm, then apply alpha
3. Does calibration improve consistency of pairwise discrimination across traits?
4. Does it reduce the per-trait PPL variance?

Also tests the relationship between vector norm and:
- Pairwise discrimination delta
- PPL cost
- Extrapolation linearity
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="norm-cal")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the old oak tree.",
    "Machine learning algorithms have transformed how we process large datasets.",
    "The restaurant served an excellent pasta with fresh tomatoes and basil.",
    "Democracy requires active participation from citizens in the political process.",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_residual_vectors(model_id, riasec_dir):
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
    return residual, mid_layer


def pairwise_logprob_chat(model, tokenizer, device, desc_a, desc_b):
    messages = [
        {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
        {"role": "user", "content": f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    return log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()


def eval_steering(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline):
    """Evaluate steering for a single trait vector."""
    vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
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
    try:
        trait_deltas = {t: 0.0 for t in TRAITS}
        trait_counts = {t: 0 for t in TRAITS}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                           TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                base_gap = baseline[f"{trait_a}-{trait_b}"]
                shift = gap - base_gap
                trait_deltas[trait_a] += shift
                trait_counts[trait_a] += 1
                trait_deltas[trait_b] -= shift
                trait_counts[trait_b] += 1
        for t in TRAITS:
            if trait_counts[t] > 0:
                trait_deltas[t] /= trait_counts[t]
    finally:
        hook_handle.remove()
    return trait_deltas


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)

    # Compute norms
    norms = {t: np.linalg.norm(residual[t]) for t in TRAITS}
    mean_norm = np.mean(list(norms.values()))
    print(f"\n--- Residual Vector Norms ---")
    for t in TRAITS:
        print(f"  {t:>15}: {norms[t]:.2f}")
    print(f"  {'Mean':>15}: {mean_norm:.2f}")
    print(f"  {'Std':>15}: {np.std(list(norms.values())):.2f}")
    print(f"  {'Max/Min ratio':>15}: {max(norms.values())/min(norms.values()):.2f}")

    # Create norm-calibrated vectors
    calibrated = {}
    for t in TRAITS:
        calibrated[t] = (residual[t] / norms[t] * mean_norm).astype(np.float32)

    # Load model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"RAW vs NORM-CALIBRATED STEERING")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    results = {}

    for method_name, vectors in [("raw", residual), ("calibrated", calibrated)]:
        logger.info(f"Testing {method_name} steering...")
        method_results = {}

        for steer_trait in TRAITS:
            vec = vectors[steer_trait].astype(np.float32)
            profile = eval_steering(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline)

            target_delta = profile[steer_trait]
            sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
            top_trait = sorted_prof[0][0]

            method_results[steer_trait] = {
                "target_delta": float(target_delta),
                "top_trait": top_trait,
                "correct": top_trait == steer_trait,
                "profile": {t: float(d) for t, d in profile.items()},
                "vec_norm": float(np.linalg.norm(vec)),
            }

        # Summary for this method
        deltas = [method_results[t]["target_delta"] for t in TRAITS]
        correct = sum(1 for t in TRAITS if method_results[t]["correct"])
        results[method_name] = method_results

        print(f"\n  {method_name.upper()}:")
        print(f"    Correct top trait: {correct}/6")
        print(f"    Target deltas: mean={np.mean(deltas):+.3f}, std={np.std(deltas):.3f}, "
              f"min={min(deltas):+.3f}, max={max(deltas):+.3f}")
        for t in TRAITS:
            r = method_results[t]
            mark = "✓" if r["correct"] else "✗"
            print(f"    {t:>15}: delta={r['target_delta']:+.3f}, top={r['top_trait']} {mark}, "
                  f"norm={r['vec_norm']:.1f}")

    # Comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")

    raw_deltas = [results["raw"][t]["target_delta"] for t in TRAITS]
    cal_deltas = [results["calibrated"][t]["target_delta"] for t in TRAITS]

    raw_correct = sum(1 for t in TRAITS if results["raw"][t]["correct"])
    cal_correct = sum(1 for t in TRAITS if results["calibrated"][t]["correct"])

    print(f"\n  {'Metric':>25}  {'Raw':>10}  {'Calibrated':>10}")
    print(f"  {'Correct top trait':>25}  {raw_correct:>10}/6  {cal_correct:>10}/6")
    print(f"  {'Mean target delta':>25}  {np.mean(raw_deltas):>+10.3f}  {np.mean(cal_deltas):>+10.3f}")
    print(f"  {'Std target delta':>25}  {np.std(raw_deltas):>10.3f}  {np.std(cal_deltas):>10.3f}")
    print(f"  {'Min target delta':>25}  {min(raw_deltas):>+10.3f}  {min(cal_deltas):>+10.3f}")
    print(f"  {'Max target delta':>25}  {max(raw_deltas):>+10.3f}  {max(cal_deltas):>+10.3f}")
    print(f"  {'CV (std/mean)':>25}  {np.std(raw_deltas)/np.mean(raw_deltas):>10.3f}  {np.std(cal_deltas)/np.mean(cal_deltas):>10.3f}")

    # Correlation between norm and delta
    from scipy.stats import pearsonr, spearmanr
    raw_norms = [norms[t] for t in TRAITS]
    r_nd, p_nd = pearsonr(raw_norms, raw_deltas)
    rho_nd, p_rho_nd = spearmanr(raw_norms, raw_deltas)
    print(f"\n  Norm → Raw Delta: r={r_nd:.3f} (p={p_nd:.3f}), ρ={rho_nd:.3f} (p={p_rho_nd:.3f})")

    if np.std(cal_deltas) < np.std(raw_deltas):
        print(f"\n  CONCLUSION: Calibration REDUCES delta variance ({np.std(raw_deltas):.3f} → {np.std(cal_deltas):.3f})")
    else:
        print(f"\n  CONCLUSION: Calibration does NOT reduce delta variance ({np.std(raw_deltas):.3f} → {np.std(cal_deltas):.3f})")

    results["analysis"] = {
        "raw_norms": {t: float(norms[t]) for t in TRAITS},
        "mean_norm": float(mean_norm),
        "norm_delta_pearson": {"r": float(r_nd), "p": float(p_nd)},
        "norm_delta_spearman": {"rho": float(rho_nd), "p": float(p_rho_nd)},
        "raw_delta_std": float(np.std(raw_deltas)),
        "calibrated_delta_std": float(np.std(cal_deltas)),
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "norm_calibrated_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
