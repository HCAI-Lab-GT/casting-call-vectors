#!/usr/bin/env python
"""
Weighted personality composition: precise control via linear mixing in 5D.

Can we create arbitrary personality blends with predictable behavior?
E.g., "70% artistic + 30% investigative" should produce a profile
where artistic dominates at ~70% of its pure delta and investigative
contributes ~30%.

Tests:
1. Binary blends at various ratios: 100/0, 75/25, 50/50, 25/75, 0/100
2. Predicted vs observed profiles: is the mapping linear?
3. Triple blends: does linearity hold for 3-way combinations?
4. Holland-respecting blends: adjacent traits should compose smoothly
5. Holland-violating blends: opposite traits should cancel

If linearity holds, this establishes the 5D personality space as a
true CONTROL PANEL with continuous, predictable knobs.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="weighted-comp")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_residual_and_basis(model_id, riasec_dir):
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

    return residual, coords_5d, basis_5d, mid_layer


def reconstruct_from_5d(coords_5d, basis_5d):
    return (basis_5d.T @ coords_5d).astype(np.float32)


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


def measure_profile(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline):
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
    alpha = 1.0  # Use alpha=1 to stay in linear regime
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    logger.info("Loading vectors...")
    residual, coords_5d, basis_5d, mid_layer = load_residual_and_basis(target_id, riasec_dir)

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

    # First: measure pure trait profiles for prediction
    logger.info("Measuring pure trait profiles...")
    pure_profiles = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        profile = measure_profile(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline)
        pure_profiles[trait] = profile
        print(f"  Pure {trait:>15}: " + " ".join(f"{t[:4]}={d:+.3f}" for t, d in
              sorted(profile.items(), key=lambda x: -x[1])))

    print(f"\n{'='*70}")
    print(f"WEIGHTED PERSONALITY COMPOSITION")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    results = {"pure_profiles": {t: {k: float(v) for k, v in pure_profiles[t].items()} for t in TRAITS}}

    # ================================================================
    # PART 1: Binary blends with varying ratios
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 1: BINARY BLENDS")
    print(f"{'='*70}")

    blend_pairs = [
        ("artistic", "investigative"),    # Holland adjacent
        ("artistic", "conventional"),     # Holland opposite
        ("social", "enterprising"),       # Holland adjacent
        ("realistic", "investigative"),   # Holland adjacent
    ]

    ratios = [1.0, 0.75, 0.5, 0.25, 0.0]  # Weight for trait A

    binary_results = {}

    for trait_a, trait_b in blend_pairs:
        pair_key = f"{trait_a}-{trait_b}"
        pair_results = {}

        print(f"\n  Pair: {trait_a} + {trait_b}")

        for w_a in ratios:
            w_b = 1.0 - w_a
            label = f"{w_a:.0%}A+{w_b:.0%}B"

            # Create blended 5D coordinates
            blend_coords = w_a * coords_5d[trait_a] + w_b * coords_5d[trait_b]
            blend_vec = reconstruct_from_5d(blend_coords, basis_5d)

            # Predicted profile (linear combination of pure profiles)
            predicted = {t: w_a * pure_profiles[trait_a][t] + w_b * pure_profiles[trait_b][t]
                        for t in TRAITS}

            # Measured profile
            logger.info(f"Testing {pair_key} at {label}...")
            observed = measure_profile(
                model, tokenizer, device, blocks, mid_layer, blend_vec, alpha, baseline)

            # Compare
            pred_vals = [predicted[t] for t in TRAITS]
            obs_vals = [observed[t] for t in TRAITS]
            r_pred, p_pred = pearsonr(pred_vals, obs_vals)

            sorted_obs = sorted(observed.items(), key=lambda x: -x[1])
            top = sorted_obs[0][0]

            # Expected top: whichever has higher weight should be top (or close)
            expected_top = trait_a if w_a > w_b else (trait_b if w_b > w_a else "either")

            print(f"    {label}: top={top} (expected={expected_top}), "
                  f"pred~obs r={r_pred:.3f}, "
                  f"A={observed[trait_a]:+.3f}({predicted[trait_a]:+.3f}), "
                  f"B={observed[trait_b]:+.3f}({predicted[trait_b]:+.3f})")

            pair_results[label] = {
                "weight_a": w_a,
                "weight_b": w_b,
                "predicted": {t: float(predicted[t]) for t in TRAITS},
                "observed": {t: float(observed[t]) for t in TRAITS},
                "pred_obs_r": float(r_pred),
                "top_trait": top,
                "expected_top": expected_top,
            }

        binary_results[pair_key] = pair_results

    results["binary_blends"] = binary_results

    # ================================================================
    # PART 2: Linearity test — is prediction accurate?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 2: LINEARITY ANALYSIS")
    print(f"{'='*70}")

    all_pred = []
    all_obs = []

    for pair_key, pair_results in binary_results.items():
        pair_preds = []
        pair_obs = []
        for label, data in pair_results.items():
            for t in TRAITS:
                pair_preds.append(data["predicted"][t])
                pair_obs.append(data["observed"][t])
                all_pred.append(data["predicted"][t])
                all_obs.append(data["observed"][t])

        r_pair, _ = pearsonr(pair_preds, pair_obs)
        print(f"  {pair_key:>30}: pred~obs r = {r_pair:.3f}")

    r_all, p_all = pearsonr(all_pred, all_obs)
    print(f"\n  Overall predicted vs observed: r = {r_all:.4f} (p = {p_all:.2e})")

    # RMSE
    rmse = np.sqrt(np.mean([(p - o)**2 for p, o in zip(all_pred, all_obs)]))
    print(f"  RMSE: {rmse:.4f}")

    results["linearity"] = {
        "overall_r": float(r_all),
        "overall_p": float(p_all),
        "rmse": float(rmse),
    }

    # ================================================================
    # PART 3: Triple blend
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 3: TRIPLE BLEND")
    print(f"{'='*70}")

    triple_combos = [
        ({"artistic": 0.5, "investigative": 0.3, "social": 0.2}, "creative-scientist-helper"),
        ({"enterprising": 0.4, "conventional": 0.3, "realistic": 0.3}, "business-practical"),
        ({"artistic": 0.33, "social": 0.33, "enterprising": 0.34}, "equal-third-ASE"),
    ]

    triple_results = {}

    for weights, name in triple_combos:
        logger.info(f"Testing triple blend: {name}...")

        # Create blended coordinates
        blend_coords = sum(w * coords_5d[t] for t, w in weights.items())
        blend_vec = reconstruct_from_5d(blend_coords, basis_5d)

        # Predicted profile
        predicted = {t: sum(w * pure_profiles[src][t] for src, w in weights.items())
                    for t in TRAITS}

        # Measured
        observed = measure_profile(
            model, tokenizer, device, blocks, mid_layer, blend_vec, alpha, baseline)

        pred_vals = [predicted[t] for t in TRAITS]
        obs_vals = [observed[t] for t in TRAITS]
        r_pred, _ = pearsonr(pred_vals, obs_vals)

        sorted_obs = sorted(observed.items(), key=lambda x: -x[1])
        top = sorted_obs[0][0]

        # Expected top
        expected_tops = sorted(weights.items(), key=lambda x: -x[1])
        expected_top = expected_tops[0][0]

        print(f"\n  {name}:")
        print(f"    Weights: {weights}")
        print(f"    Top: {top} (expected: {expected_top})")
        print(f"    pred~obs r: {r_pred:.3f}")
        print(f"    Predicted: " + " ".join(f"{t[:4]}={predicted[t]:+.3f}" for t in TRAITS))
        print(f"    Observed:  " + " ".join(f"{t[:4]}={observed[t]:+.3f}" for t in TRAITS))

        triple_results[name] = {
            "weights": weights,
            "predicted": {t: float(predicted[t]) for t in TRAITS},
            "observed": {t: float(observed[t]) for t in TRAITS},
            "pred_obs_r": float(r_pred),
            "top_trait": top,
            "expected_top": expected_top,
        }

    results["triple_blends"] = triple_results

    # ================================================================
    # PART 4: Continuous sweep — vary the ratio smoothly and track
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 4: CONTINUOUS RATIO SWEEP (artistic → investigative)")
    print(f"{'='*70}")

    sweep_ratios = np.arange(0, 1.01, 0.2)
    sweep_results = []

    for w_art in sweep_ratios:
        w_inv = 1.0 - w_art
        blend_coords = w_art * coords_5d["artistic"] + w_inv * coords_5d["investigative"]
        blend_vec = reconstruct_from_5d(blend_coords, basis_5d)

        logger.info(f"Sweep: artistic={w_art:.1f}...")
        observed = measure_profile(
            model, tokenizer, device, blocks, mid_layer, blend_vec, alpha, baseline)

        sweep_results.append({
            "w_artistic": float(w_art),
            "w_investigative": float(w_inv),
            "artistic_delta": float(observed["artistic"]),
            "investigative_delta": float(observed["investigative"]),
            "profile": {t: float(observed[t]) for t in TRAITS},
        })

        print(f"  w_art={w_art:.1f}: art={observed['artistic']:+.3f}, "
              f"inv={observed['investigative']:+.3f}, "
              f"top={sorted(observed.items(), key=lambda x: -x[1])[0][0]}")

    # Check monotonicity
    art_deltas = [s["artistic_delta"] for s in sweep_results]
    inv_deltas = [s["investigative_delta"] for s in sweep_results]

    art_monotonic = all(art_deltas[i] >= art_deltas[i+1] for i in range(len(art_deltas)-1))
    inv_monotonic = all(inv_deltas[i] <= inv_deltas[i+1] for i in range(len(inv_deltas)-1))

    r_art_w, _ = pearsonr([s["w_artistic"] for s in sweep_results], art_deltas)
    r_inv_w, _ = pearsonr([s["w_investigative"] for s in sweep_results], inv_deltas)

    print(f"\n  Artistic monotonic with weight: {art_monotonic}")
    print(f"  Investigative monotonic with weight: {inv_monotonic}")
    print(f"  Artistic delta~weight: r = {r_art_w:.3f}")
    print(f"  Investigative delta~weight: r = {r_inv_w:.3f}")

    results["sweep"] = {
        "data": sweep_results,
        "art_monotonic": art_monotonic,
        "inv_monotonic": inv_monotonic,
        "art_delta_weight_r": float(r_art_w),
        "inv_delta_weight_r": float(r_inv_w),
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Overall linearity: r = {r_all:.4f} (RMSE = {rmse:.4f})")

    if r_all > 0.9:
        print(f"  CONCLUSION: 5D personality space is HIGHLY LINEAR")
        print(f"  Arbitrary weighted blends produce predictable profiles (r > 0.9)")
    elif r_all > 0.7:
        print(f"  CONCLUSION: 5D personality space is approximately linear (r > 0.7)")
    else:
        print(f"  CONCLUSION: Non-linear interactions dominate (r < 0.7)")

    # Count correct top predictions
    correct_top = 0
    total_top = 0
    for pair_key, pair_results in binary_results.items():
        for label, data in pair_results.items():
            if data["expected_top"] != "either":
                total_top += 1
                if data["top_trait"] == data["expected_top"]:
                    correct_top += 1

    print(f"  Correct top-trait predictions: {correct_top}/{total_top}")

    results["summary"] = {
        "overall_linearity_r": float(r_all),
        "rmse": float(rmse),
        "correct_top_predictions": correct_top,
        "total_predictions": total_top,
        "sweep_art_monotonic": art_monotonic,
        "sweep_inv_monotonic": inv_monotonic,
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weighted_composition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
