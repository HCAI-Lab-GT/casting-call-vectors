#!/usr/bin/env python
"""
Blind alpha estimation: estimate the applied steering strength from activations alone.

The neutralization experiment (finding #59) showed we can detect WHICH trait
was applied with 100% accuracy. But can we also estimate HOW MUCH (alpha)?

This is crucial for a practical personality firewall:
- If we can estimate alpha, we can apply exactly the right correction
- Without alpha estimation, we might over- or under-correct

TEST:
1. Apply each trait at alpha = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
2. Capture activations at L17
3. Project onto 5D basis → detected coordinates
4. Estimate alpha from the magnitude of detected coordinates
5. Build a calibration curve: detected_norm vs true_alpha
6. Test: can we predict alpha on held-out data?

Also tests:
- Is the relationship linear? (expected: yes, from finding #36 alpha detection r=0.999)
- Can we build a SINGLE calibration curve across all traits?
- What's the minimum detectable alpha?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr, linregress
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="blind-alpha")

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


def load_all_data(model_id, riasec_dir):
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


def capture_activations(model, tokenizer, device, blocks, layer_idx,
                        prompt, steer_vec=None, alpha=0.0, mid_layer=None):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}

    def make_capture_hook(lidx):
        def hook_fn(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[lidx] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        return hook_fn

    cap_hook = blocks[layer_idx].register_forward_hook(make_capture_hook(layer_idx))

    steer_hook = None
    if steer_vec is not None and alpha > 0 and mid_layer is not None:
        vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta_vec
                return (hs,) + out[1:]
            out[:, -1, :] += delta_vec
            return out

        steer_hook = blocks[mid_layer].register_forward_hook(steer_fn)

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        cap_hook.remove()
        if steer_hook:
            steer_hook.remove()

    return captured[layer_idx]


def measure_profile(model, tokenizer, device, blocks, mid_layer,
                    vec, alpha, baseline, correction_vec=None, correction_alpha=0.0):
    hooks = []

    if vec is not None and alpha > 0:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def make_steer_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hooks.append(blocks[mid_layer].register_forward_hook(make_steer_hook(delta_vec)))

    if correction_vec is not None and correction_alpha != 0:
        corr_t = torch.tensor(correction_vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_corr = correction_alpha * corr_t

        def make_corr_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hooks.append(blocks[mid_layer].register_forward_hook(make_corr_hook(delta_corr)))

    try:
        trait_logprobs = {}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                messages = [
                    {"role": "user", "content": f"Which describes you better? Answer with just A or B.\n"
                                                 f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                                                 f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(formatted, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                a_ids = tokenizer.encode("A", add_special_tokens=False)
                b_ids = tokenizer.encode("B", add_special_tokens=False)
                gap = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()
                trait_logprobs[f"{trait_a}-{trait_b}"] = gap
    finally:
        for h in hooks:
            h.remove()

    trait_deltas = {t: 0.0 for t in TRAITS}
    trait_counts = {t: 0 for t in TRAITS}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            key = f"{trait_a}-{trait_b}"
            shift = trait_logprobs[key] - baseline[key]
            trait_deltas[trait_a] += shift
            trait_counts[trait_a] += 1
            trait_deltas[trait_b] -= shift
            trait_counts[trait_b] += 1
    for t in TRAITS:
        if trait_counts[t] > 0:
            trait_deltas[t] /= trait_counts[t]
    return trait_deltas


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    logger.info("Loading vectors and basis...")
    residual, coords_5d, basis_5d, mid_layer = load_all_data(target_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    capture_layer = mid_layer + 1

    detect_prompts = [
        "Tell me about yourself.",
        "What do you think about teamwork?",
        "How would you describe your ideal day?",
    ]

    # Baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [
                {"role": "user", "content": f"Which describes you better? Answer with just A or B.\n"
                                             f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                                             f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"},
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
            baseline[f"{trait_a}-{trait_b}"] = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()

    logger.info("Capturing baseline activations...")
    baseline_acts = {}
    for prompt in detect_prompts:
        baseline_acts[prompt] = capture_activations(
            model, tokenizer, device, blocks, capture_layer, prompt)

    print(f"\n{'='*70}")
    print(f"BLIND ALPHA ESTIMATION FROM ACTIVATIONS")
    print(f"Target: Marin 8B, detect at L{capture_layer}")
    print(f"{'='*70}")

    test_alphas = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    results = {}

    # ================================================================
    # PART 1: Per-trait calibration curves
    # ================================================================
    all_true_alphas = []
    all_detected_norms = []
    all_projected_norms = []

    for steer_trait in TRAITS:
        vec = residual[steer_trait].astype(np.float32)
        trait_coord = coords_5d[steer_trait]
        trait_dir = trait_coord / np.linalg.norm(trait_coord)

        print(f"\n{'='*70}")
        print(f"TRAIT: {steer_trait.upper()}")
        print(f"{'='*70}")

        trait_true_alphas = []
        trait_detected_norms = []
        trait_projected_magnitudes = []
        trait_data = {}

        for alpha in test_alphas:
            logger.info(f"{steer_trait} alpha={alpha}...")
            diffs = []
            for prompt in detect_prompts:
                steered_act = capture_activations(
                    model, tokenizer, device, blocks, capture_layer,
                    prompt, vec, alpha, mid_layer)
                diff = steered_act - baseline_acts[prompt]
                diffs.append(diff)

            mean_diff = np.mean(diffs, axis=0)
            detected_coords = basis_5d @ mean_diff

            # Full 5D norm
            detected_norm = float(np.linalg.norm(detected_coords))

            # Projection onto trait direction
            projected_mag = float(np.dot(detected_coords, trait_dir))

            # Cosine similarity with trait
            if detected_norm > 0:
                trait_cos = float(np.dot(detected_coords, trait_coord) / (
                    detected_norm * np.linalg.norm(trait_coord)))
            else:
                trait_cos = 0

            trait_true_alphas.append(alpha)
            trait_detected_norms.append(detected_norm)
            trait_projected_magnitudes.append(projected_mag)
            all_true_alphas.append(alpha)
            all_detected_norms.append(detected_norm)
            all_projected_norms.append(projected_mag)

            print(f"  alpha={alpha:4.2f}: norm={detected_norm:.4f}, "
                  f"proj={projected_mag:.4f}, cos={trait_cos:.4f}")

            trait_data[f"alpha_{alpha}"] = {
                "true_alpha": alpha,
                "detected_norm": detected_norm,
                "projected_magnitude": projected_mag,
                "trait_cosine": trait_cos,
            }

        # Linear regression: alpha ~ detected_norm
        slope, intercept, r_value, _, _ = linregress(trait_true_alphas, trait_detected_norms)
        r_sq = r_value ** 2
        print(f"\n  Linear fit (norm): alpha = {slope:.4f} * norm + {intercept:.4f}, R²={r_sq:.4f}")

        slope_p, intercept_p, r_p, _, _ = linregress(trait_true_alphas, trait_projected_magnitudes)
        r_sq_p = r_p ** 2
        print(f"  Linear fit (proj): alpha = {slope_p:.4f} * proj + {intercept_p:.4f}, R²={r_sq_p:.4f}")

        # LOO prediction error
        loo_errors = []
        for i in range(len(trait_true_alphas)):
            # Train on all except i
            train_x = trait_true_alphas[:i] + trait_true_alphas[i+1:]
            train_y = trait_detected_norms[:i] + trait_detected_norms[i+1:]
            s, b, _, _, _ = linregress(train_x, train_y)
            # Predict alpha from norm
            if abs(s) > 1e-8:
                pred_alpha = (trait_detected_norms[i] - b) / s
            else:
                pred_alpha = 0
            loo_errors.append(abs(pred_alpha - trait_true_alphas[i]))

        mean_loo_error = float(np.mean(loo_errors))
        print(f"  LOO mean absolute error: {mean_loo_error:.3f} alpha units")

        results[steer_trait] = {
            "data": trait_data,
            "linear_fit_norm": {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_sq)},
            "linear_fit_proj": {"slope": float(slope_p), "intercept": float(intercept_p), "r_squared": float(r_sq_p)},
            "loo_mean_error": mean_loo_error,
        }

    # ================================================================
    # PART 2: Universal calibration (all traits together)
    # ================================================================
    print(f"\n{'='*70}")
    print("UNIVERSAL CALIBRATION (all traits)")
    print(f"{'='*70}")

    # Normalize norms by trait's coordinate magnitude
    normalized_norms = []
    for i, (alpha, norm) in enumerate(zip(all_true_alphas, all_detected_norms)):
        trait_idx = i // len(test_alphas)
        trait = TRAITS[trait_idx]
        trait_norm = np.linalg.norm(coords_5d[trait])
        if trait_norm > 0:
            normalized_norms.append(norm / trait_norm)
        else:
            normalized_norms.append(0)

    slope_u, intercept_u, r_u, _, _ = linregress(all_true_alphas, normalized_norms)
    r_sq_u = r_u ** 2
    print(f"\n  Universal fit: alpha = {slope_u:.4f} * (norm/trait_norm) + {intercept_u:.4f}, R²={r_sq_u:.4f}")

    # LOO on universal model
    loo_errors_u = []
    for i in range(len(all_true_alphas)):
        train_x = all_true_alphas[:i] + all_true_alphas[i+1:]
        train_y = normalized_norms[:i] + normalized_norms[i+1:]
        s, b, _, _, _ = linregress(train_x, train_y)
        if abs(s) > 1e-8:
            pred = (normalized_norms[i] - b) / s
        else:
            pred = 0
        loo_errors_u.append(abs(pred - all_true_alphas[i]))

    mean_loo_u = float(np.mean(loo_errors_u))
    print(f"  Universal LOO mean error: {mean_loo_u:.3f} alpha units")

    # ================================================================
    # PART 3: Blind neutralization test
    # ================================================================
    print(f"\n{'='*70}")
    print("BLIND NEUTRALIZATION (estimate alpha, apply correction)")
    print(f"{'='*70}")

    # Pick 3 test cases with different alphas
    blind_tests = [
        ("artistic", 1.5),
        ("investigative", 2.5),
        ("realistic", 3.5),  # Out of calibration range (extrapolation!)
    ]

    blind_results = {}
    for steer_trait, true_alpha in blind_tests:
        vec = residual[steer_trait].astype(np.float32)

        # Detect
        diffs = []
        for prompt in detect_prompts:
            steered_act = capture_activations(
                model, tokenizer, device, blocks, capture_layer,
                prompt, vec, true_alpha, mid_layer)
            diff = steered_act - baseline_acts[prompt]
            diffs.append(diff)

        mean_diff = np.mean(diffs, axis=0)
        detected_coords = basis_5d @ mean_diff

        # Identify trait
        detected_sims = {}
        for t in TRAITS:
            c = coords_5d[t]
            if np.linalg.norm(detected_coords) > 0 and np.linalg.norm(c) > 0:
                detected_sims[t] = float(np.dot(detected_coords, c) / (
                    np.linalg.norm(detected_coords) * np.linalg.norm(c)))
            else:
                detected_sims[t] = 0
        detected_trait = max(detected_sims, key=detected_sims.get)

        # Estimate alpha using universal calibration
        trait_norm = np.linalg.norm(coords_5d[detected_trait])
        detected_norm_normalized = float(np.linalg.norm(detected_coords)) / trait_norm if trait_norm > 0 else 0
        if abs(slope_u) > 1e-8:
            estimated_alpha = (detected_norm_normalized - intercept_u) / slope_u
        else:
            estimated_alpha = 0

        # Build correction using estimated alpha
        correction_vec = -(basis_5d.T @ detected_coords).astype(np.float32)
        # Scale correction by estimated_alpha / detected_norm to get unit correction
        # Actually, the detected_coords already contain the full magnitude,
        # so the correction should work at alpha=1.0 (like in neutralization experiment)
        # But let's test BOTH: fixed alpha=1.0 vs estimated alpha scaling

        # Method 1: Direct correction (alpha=1.0, as in neutralization)
        steered_deltas = measure_profile(
            model, tokenizer, device, blocks, mid_layer,
            vec, true_alpha, baseline)
        corrected_direct = measure_profile(
            model, tokenizer, device, blocks, mid_layer,
            vec, true_alpha, baseline,
            correction_vec=correction_vec, correction_alpha=1.0)

        # Method 2: Alpha-scaled correction
        # Reconstruct from trait direction with estimated alpha
        trait_dir_5d = coords_5d[detected_trait] / np.linalg.norm(coords_5d[detected_trait])
        correction_scaled = -(basis_5d.T @ (estimated_alpha * coords_5d[detected_trait])).astype(np.float32)
        corrected_scaled = measure_profile(
            model, tokenizer, device, blocks, mid_layer,
            vec, true_alpha, baseline,
            correction_vec=correction_scaled, correction_alpha=1.0)

        steered_target = steered_deltas[steer_trait]
        corrected_d = corrected_direct[steer_trait]
        corrected_s = corrected_scaled[steer_trait]

        neut_direct = 1.0 - (corrected_d / steered_target) if abs(steered_target) > 0.01 else 0
        neut_scaled = 1.0 - (corrected_s / steered_target) if abs(steered_target) > 0.01 else 0

        alpha_error = abs(estimated_alpha - true_alpha)

        print(f"\n  {steer_trait} (true α={true_alpha}):")
        print(f"    Detected: {detected_trait} (correct: {detected_trait == steer_trait})")
        print(f"    Estimated α: {estimated_alpha:.2f} (error: {alpha_error:.2f})")
        print(f"    Steered delta: {steered_target:+.3f}")
        print(f"    Direct correction:  {corrected_d:+.3f} ({neut_direct:.1%} neutralized)")
        print(f"    Scaled correction:  {corrected_s:+.3f} ({neut_scaled:.1%} neutralized)")

        blind_results[f"{steer_trait}_alpha{true_alpha}"] = {
            "true_trait": steer_trait,
            "true_alpha": true_alpha,
            "detected_trait": detected_trait,
            "estimated_alpha": float(estimated_alpha),
            "alpha_error": float(alpha_error),
            "steered_delta": float(steered_target),
            "direct_neutralization": float(neut_direct),
            "scaled_neutralization": float(neut_scaled),
        }

    results["blind_neutralization"] = blind_results

    # ================================================================
    # OVERALL SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    per_trait_r2 = [results[t]["linear_fit_norm"]["r_squared"] for t in TRAITS]
    per_trait_loo = [results[t]["loo_mean_error"] for t in TRAITS]
    mean_r2 = float(np.mean(per_trait_r2))
    mean_loo = float(np.mean(per_trait_loo))

    print(f"\n  Per-trait mean R²:       {mean_r2:.4f}")
    print(f"  Per-trait mean LOO err:  {mean_loo:.3f} alpha units")
    print(f"  Universal R²:           {r_sq_u:.4f}")
    print(f"  Universal LOO err:      {mean_loo_u:.3f} alpha units")

    blind_alpha_errors = [r["alpha_error"] for r in blind_results.values()]
    blind_neut = [r["direct_neutralization"] for r in blind_results.values()]
    print(f"  Blind alpha error:      {np.mean(blind_alpha_errors):.3f}")
    print(f"  Blind neutralization:   {np.mean(blind_neut):.1%}")

    if mean_r2 > 0.99:
        conclusion = "Alpha estimation is NEAR-PERFECT (R² > 0.99)"
    elif mean_r2 > 0.95:
        conclusion = "Alpha estimation is HIGHLY ACCURATE (R² > 0.95)"
    elif mean_r2 > 0.9:
        conclusion = "Alpha estimation is GOOD (R² > 0.9)"
    else:
        conclusion = f"Alpha estimation is MODERATE (R² = {mean_r2:.3f})"

    print(f"\n  CONCLUSION: {conclusion}")

    results["summary"] = {
        "per_trait_mean_r_squared": mean_r2,
        "per_trait_mean_loo_error": mean_loo,
        "universal_r_squared": float(r_sq_u),
        "universal_loo_error": mean_loo_u,
        "blind_mean_alpha_error": float(np.mean(blind_alpha_errors)),
        "blind_mean_neutralization": float(np.mean(blind_neut)),
        "conclusion": conclusion,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "blind_alpha_estimation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
