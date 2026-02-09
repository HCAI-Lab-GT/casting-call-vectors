#!/usr/bin/env python
"""
Compositional decomposition: detect and decompose multi-trait personality mixtures.

Prior findings established:
- Weighted composition is HIGHLY LINEAR (r=0.987, finding #45)
- Activation forensics detects single traits at 100% (finding #36)
- Personality neutralization reverses single traits at 90.7% (finding #59)

This experiment tests: can we decompose MIXTURES of personality vectors
applied simultaneously? If someone steers with 60% artistic + 40% investigative,
can we detect both components and their proportions?

TEST CASES:
1. Binary mixtures: all 15 pairs at 50/50, plus 3 pairs at 70/30
2. Triple mixtures: 3 selected triples at equal weight
3. Holland opposite cancellation: artistic+conventional should produce near-zero
4. Decomposition accuracy: detected proportions vs true proportions
5. Per-component neutralization: correct only one component of a mixture
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="decompose")

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
                        prompt, steer_vecs=None, alphas=None, mid_layer=None):
    """Capture activations with optional multi-vector steering."""
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

    steer_hooks = []
    if steer_vecs is not None and alphas is not None and mid_layer is not None:
        # Sum all steering vectors into one delta
        total_delta = None
        for vec, alpha in zip(steer_vecs, alphas):
            d = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
            if total_delta is None:
                total_delta = d
            else:
                total_delta = total_delta + d

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += total_delta
                return (hs,) + out[1:]
            out[:, -1, :] += total_delta
            return out

        steer_hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        cap_hook.remove()
        for h in steer_hooks:
            h.remove()

    return captured[layer_idx]


def measure_profile(model, tokenizer, device, blocks, mid_layer,
                    steer_vecs, alphas, baseline):
    """Measure personality with multi-vector steering."""
    hooks = []

    if steer_vecs is not None and alphas is not None:
        total_delta = None
        for vec, alpha in zip(steer_vecs, alphas):
            if alpha == 0:
                continue
            d = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
            if total_delta is None:
                total_delta = d
            else:
                total_delta = total_delta + d

        if total_delta is not None:
            def make_steer_hook(d):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += d
                        return (hs,) + out[1:]
                    out[:, -1, :] += d
                    return out
                return hook_fn
            hooks.append(blocks[mid_layer].register_forward_hook(make_steer_hook(total_delta)))

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
    alpha = 2.0  # Use moderate alpha

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

    # Capture baseline activations
    logger.info("Capturing baseline activations...")
    baseline_acts = {}
    for prompt in detect_prompts:
        baseline_acts[prompt] = capture_activations(
            model, tokenizer, device, blocks, capture_layer, prompt)

    print(f"\n{'='*70}")
    print(f"COMPOSITIONAL DECOMPOSITION VIA 5D PROJECTION")
    print(f"Target: Marin 8B, alpha={alpha}")
    print(f"{'='*70}")

    results = {}

    # ================================================================
    # PART 1: Binary mixtures (50/50)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 1: BINARY MIXTURES (50/50)")
    print(f"{'='*70}")

    binary_results = {}
    for t1, t2 in combinations(TRAITS, 2):
        w1, w2 = 0.5, 0.5
        vecs = [residual[t1].astype(np.float32), residual[t2].astype(np.float32)]
        alphas_mix = [alpha * w1, alpha * w2]

        # Expected 5D coordinates = weighted sum
        expected_coords = w1 * coords_5d[t1] + w2 * coords_5d[t2]

        # Detect from activations
        diffs = []
        for prompt in detect_prompts:
            steered_act = capture_activations(
                model, tokenizer, device, blocks, capture_layer,
                prompt, vecs, alphas_mix, mid_layer)
            diff = steered_act - baseline_acts[prompt]
            diffs.append(diff)

        mean_diff = np.mean(diffs, axis=0)
        detected_coords = basis_5d @ mean_diff

        # Compare detected vs expected
        if np.linalg.norm(expected_coords) > 0 and np.linalg.norm(detected_coords) > 0:
            coord_cos = float(np.dot(detected_coords, expected_coords) / (
                np.linalg.norm(detected_coords) * np.linalg.norm(expected_coords)))
        else:
            coord_cos = 0

        # Decompose: what proportion of each trait?
        detected_weights = {}
        for t in TRAITS:
            c_norm = np.linalg.norm(coords_5d[t])
            if c_norm > 0:
                cos_sim = float(np.dot(detected_coords, coords_5d[t]) / (
                    np.linalg.norm(detected_coords) * c_norm))
            else:
                cos_sim = 0
            detected_weights[t] = cos_sim

        top2 = sorted(detected_weights.items(), key=lambda x: -x[1])[:2]
        correct_top2 = {top2[0][0], top2[1][0]} == {t1, t2}

        print(f"\n  {t1}+{t2}: coord_cos={coord_cos:.3f}, top2={top2[0][0]}({top2[0][1]:.3f})+{top2[1][0]}({top2[1][1]:.3f}) {'OK' if correct_top2 else 'WRONG'}")

        binary_results[f"{t1}+{t2}"] = {
            "expected_traits": [t1, t2],
            "weights": [w1, w2],
            "coord_cosine": coord_cos,
            "detected_weights": detected_weights,
            "top2_correct": bool(correct_top2),
            "top2": [top2[0][0], top2[1][0]],
        }

    correct_count = sum(1 for r in binary_results.values() if r["top2_correct"])
    total = len(binary_results)
    print(f"\n  Binary 50/50 top-2 accuracy: {correct_count}/{total}")

    results["binary_50_50"] = binary_results

    # ================================================================
    # PART 2: Asymmetric mixtures (70/30)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: ASYMMETRIC MIXTURES (70/30)")
    print(f"{'='*70}")

    asymmetric_pairs = [
        ("artistic", "investigative"),
        ("conventional", "realistic"),
        ("enterprising", "social"),
    ]

    asymmetric_results = {}
    for t1, t2 in asymmetric_pairs:
        w1, w2 = 0.7, 0.3
        vecs = [residual[t1].astype(np.float32), residual[t2].astype(np.float32)]
        alphas_mix = [alpha * w1, alpha * w2]

        expected_coords = w1 * coords_5d[t1] + w2 * coords_5d[t2]

        diffs = []
        for prompt in detect_prompts:
            steered_act = capture_activations(
                model, tokenizer, device, blocks, capture_layer,
                prompt, vecs, alphas_mix, mid_layer)
            diff = steered_act - baseline_acts[prompt]
            diffs.append(diff)

        mean_diff = np.mean(diffs, axis=0)
        detected_coords = basis_5d @ mean_diff

        if np.linalg.norm(expected_coords) > 0 and np.linalg.norm(detected_coords) > 0:
            coord_cos = float(np.dot(detected_coords, expected_coords) / (
                np.linalg.norm(detected_coords) * np.linalg.norm(expected_coords)))
        else:
            coord_cos = 0

        # Can we recover the weights?
        # Solve: detected_coords ≈ w1*coords[t1] + w2*coords[t2]
        # Least squares: [coords_t1 | coords_t2] @ [w1, w2] = detected_coords
        A = np.stack([coords_5d[t1], coords_5d[t2]], axis=1)
        lstsq_weights, _, _, _ = np.linalg.lstsq(A, detected_coords, rcond=None)
        # Normalize to sum to 1
        weight_sum = abs(lstsq_weights[0]) + abs(lstsq_weights[1])
        if weight_sum > 0:
            norm_w1 = abs(lstsq_weights[0]) / weight_sum
            norm_w2 = abs(lstsq_weights[1]) / weight_sum
        else:
            norm_w1 = norm_w2 = 0.5

        dominant_correct = (norm_w1 > norm_w2) == (w1 > w2)

        print(f"\n  {t1}(70%)+{t2}(30%):")
        print(f"    coord_cos={coord_cos:.3f}")
        print(f"    Recovered weights: {t1}={norm_w1:.1%}, {t2}={norm_w2:.1%} (true: 70%/30%)")
        print(f"    Dominant correct: {dominant_correct}")
        print(f"    Weight error: {abs(norm_w1 - 0.7):.3f}")

        asymmetric_results[f"{t1}+{t2}"] = {
            "expected_traits": [t1, t2],
            "true_weights": [w1, w2],
            "coord_cosine": coord_cos,
            "recovered_weights": [float(norm_w1), float(norm_w2)],
            "weight_error": float(abs(norm_w1 - w1)),
            "dominant_correct": bool(dominant_correct),
        }

    results["asymmetric_70_30"] = asymmetric_results

    # ================================================================
    # PART 3: Triple mixtures (equal weight)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: TRIPLE MIXTURES (33/33/33)")
    print(f"{'='*70}")

    triple_combos = [
        ("artistic", "investigative", "social"),
        ("conventional", "enterprising", "realistic"),
        ("artistic", "conventional", "investigative"),
    ]

    triple_results = {}
    for t1, t2, t3 in triple_combos:
        w = 1.0 / 3.0
        vecs = [residual[t].astype(np.float32) for t in [t1, t2, t3]]
        alphas_mix = [alpha * w] * 3

        expected_coords = w * (coords_5d[t1] + coords_5d[t2] + coords_5d[t3])

        diffs = []
        for prompt in detect_prompts:
            steered_act = capture_activations(
                model, tokenizer, device, blocks, capture_layer,
                prompt, vecs, alphas_mix, mid_layer)
            diff = steered_act - baseline_acts[prompt]
            diffs.append(diff)

        mean_diff = np.mean(diffs, axis=0)
        detected_coords = basis_5d @ mean_diff

        if np.linalg.norm(expected_coords) > 0 and np.linalg.norm(detected_coords) > 0:
            coord_cos = float(np.dot(detected_coords, expected_coords) / (
                np.linalg.norm(detected_coords) * np.linalg.norm(expected_coords)))
        else:
            coord_cos = 0

        # Decompose into all 6 traits
        A = np.stack([coords_5d[t] for t in TRAITS], axis=1)
        lstsq_all, _, _, _ = np.linalg.lstsq(A, detected_coords, rcond=None)
        weight_dict = {TRAITS[i]: float(lstsq_all[i]) for i in range(6)}
        top3 = sorted(weight_dict.items(), key=lambda x: -x[1])[:3]
        correct_top3 = {top3[0][0], top3[1][0], top3[2][0]} == {t1, t2, t3}

        combo_name = f"{t1}+{t2}+{t3}"
        print(f"\n  {combo_name}:")
        print(f"    coord_cos={coord_cos:.3f}")
        print(f"    Decomposed weights: {', '.join(f'{t}={v:.3f}' for t, v in sorted(weight_dict.items(), key=lambda x: -x[1]))}")
        print(f"    Top-3 correct: {correct_top3}")

        triple_results[combo_name] = {
            "expected_traits": [t1, t2, t3],
            "coord_cosine": coord_cos,
            "decomposed_weights": weight_dict,
            "top3_correct": bool(correct_top3),
            "top3": [t[0] for t in top3],
        }

    results["triple_33_33_33"] = triple_results

    # ================================================================
    # PART 4: Holland opposite cancellation
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: HOLLAND OPPOSITE CANCELLATION")
    print(f"{'='*70}")

    holland_pairs = [
        ("artistic", "conventional"),
        ("investigative", "enterprising"),
        ("social", "realistic"),
    ]

    cancellation_results = {}
    for t1, t2 in holland_pairs:
        vecs = [residual[t1].astype(np.float32), residual[t2].astype(np.float32)]
        alphas_mix = [alpha, alpha]  # Equal full-strength

        # Expected: near-zero (they should cancel)
        expected_coords = coords_5d[t1] + coords_5d[t2]
        expected_norm = float(np.linalg.norm(expected_coords))

        diffs = []
        for prompt in detect_prompts:
            steered_act = capture_activations(
                model, tokenizer, device, blocks, capture_layer,
                prompt, vecs, alphas_mix, mid_layer)
            diff = steered_act - baseline_acts[prompt]
            diffs.append(diff)

        mean_diff = np.mean(diffs, axis=0)
        detected_coords = basis_5d @ mean_diff
        detected_norm = float(np.linalg.norm(detected_coords))

        # Compare to single-trait norms
        single_norm_1 = float(np.linalg.norm(coords_5d[t1]))
        single_norm_2 = float(np.linalg.norm(coords_5d[t2]))
        mean_single = (single_norm_1 + single_norm_2) / 2

        # Measure behavioral profile
        deltas = measure_profile(
            model, tokenizer, device, blocks, mid_layer,
            vecs, alphas_mix, baseline)
        profile_magnitude = float(np.sqrt(sum(v**2 for v in deltas.values())))

        cancellation_ratio = detected_norm / mean_single if mean_single > 0 else 1.0

        print(f"\n  {t1}+{t2} (Holland opposites):")
        print(f"    Expected 5D norm: {expected_norm:.4f} (vs single ~{mean_single:.4f})")
        print(f"    Detected 5D norm: {detected_norm:.4f}")
        print(f"    Cancellation ratio: {cancellation_ratio:.3f} (0=perfect cancel, 1=no cancel)")
        print(f"    Behavioral magnitude: {profile_magnitude:.3f}")
        print(f"    Profile: {', '.join(f'{t}={v:+.3f}' for t, v in sorted(deltas.items(), key=lambda x: -x[1]))}")

        cancellation_results[f"{t1}+{t2}"] = {
            "expected_norm": expected_norm,
            "detected_norm": detected_norm,
            "mean_single_norm": float(mean_single),
            "cancellation_ratio": float(cancellation_ratio),
            "behavioral_magnitude": profile_magnitude,
            "profile": {t: float(v) for t, v in deltas.items()},
        }

    results["holland_cancellation"] = cancellation_results

    # ================================================================
    # PART 5: Selective neutralization (correct one component)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: SELECTIVE NEUTRALIZATION")
    print(f"{'='*70}")

    selective_results = {}
    test_pair = ("artistic", "investigative")
    t1, t2 = test_pair
    w1, w2 = 0.5, 0.5
    vecs = [residual[t1].astype(np.float32), residual[t2].astype(np.float32)]
    alphas_mix = [alpha * w1, alpha * w2]

    # Measure the mixture
    mixed_deltas = measure_profile(
        model, tokenizer, device, blocks, mid_layer,
        vecs, alphas_mix, baseline)
    print(f"\n  Mixed ({t1}+{t2} 50/50):")
    print(f"    Profile: {', '.join(f'{t}={v:+.3f}' for t, v in sorted(mixed_deltas.items(), key=lambda x: -x[1]))}")

    # Detect the mixture
    diffs = []
    for prompt in detect_prompts:
        steered_act = capture_activations(
            model, tokenizer, device, blocks, capture_layer,
            prompt, vecs, alphas_mix, mid_layer)
        diff = steered_act - baseline_acts[prompt]
        diffs.append(diff)

    mean_diff = np.mean(diffs, axis=0)
    detected_coords = basis_5d @ mean_diff

    # Remove only the t1 component
    # Project detected_coords onto t1's direction in 5D
    t1_dir = coords_5d[t1] / np.linalg.norm(coords_5d[t1])
    t1_component = np.dot(detected_coords, t1_dir) * t1_dir
    # Correction for just t1
    correction_t1 = -(basis_5d.T @ t1_component).astype(np.float32)

    corrected_deltas = measure_profile(
        model, tokenizer, device, blocks, mid_layer,
        vecs + [correction_t1], alphas_mix + [1.0], baseline)

    print(f"\n  After removing {t1} component:")
    print(f"    Profile: {', '.join(f'{t}={v:+.3f}' for t, v in sorted(corrected_deltas.items(), key=lambda x: -x[1]))}")

    # The t1 delta should decrease, t2 delta should remain
    t1_retained = corrected_deltas[t1] / mixed_deltas[t1] if abs(mixed_deltas[t1]) > 0.01 else 1.0
    t2_retained = corrected_deltas[t2] / mixed_deltas[t2] if abs(mixed_deltas[t2]) > 0.01 else 1.0

    print(f"    {t1} retained: {t1_retained:.1%} (should be ~0%)")
    print(f"    {t2} retained: {t2_retained:.1%} (should be ~100%)")

    # Now remove only t2
    t2_dir = coords_5d[t2] / np.linalg.norm(coords_5d[t2])
    t2_component = np.dot(detected_coords, t2_dir) * t2_dir
    correction_t2 = -(basis_5d.T @ t2_component).astype(np.float32)

    corrected_deltas_2 = measure_profile(
        model, tokenizer, device, blocks, mid_layer,
        vecs + [correction_t2], alphas_mix + [1.0], baseline)

    print(f"\n  After removing {t2} component:")
    print(f"    Profile: {', '.join(f'{t}={v:+.3f}' for t, v in sorted(corrected_deltas_2.items(), key=lambda x: -x[1]))}")

    t1_retained_2 = corrected_deltas_2[t1] / mixed_deltas[t1] if abs(mixed_deltas[t1]) > 0.01 else 1.0
    t2_retained_2 = corrected_deltas_2[t2] / mixed_deltas[t2] if abs(mixed_deltas[t2]) > 0.01 else 1.0

    print(f"    {t1} retained: {t1_retained_2:.1%} (should be ~100%)")
    print(f"    {t2} retained: {t2_retained_2:.1%} (should be ~0%)")

    selective_results = {
        "pair": [t1, t2],
        "mixed_profile": {t: float(v) for t, v in mixed_deltas.items()},
        f"remove_{t1}": {
            "corrected_profile": {t: float(v) for t, v in corrected_deltas.items()},
            f"{t1}_retained": float(t1_retained),
            f"{t2}_retained": float(t2_retained),
        },
        f"remove_{t2}": {
            "corrected_profile": {t: float(v) for t, v in corrected_deltas_2.items()},
            f"{t1}_retained": float(t1_retained_2),
            f"{t2}_retained": float(t2_retained_2),
        },
    }
    results["selective_neutralization"] = selective_results

    # ================================================================
    # OVERALL SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    binary_acc = sum(1 for r in results["binary_50_50"].values() if r["top2_correct"])
    binary_total = len(results["binary_50_50"])
    binary_cos = np.mean([r["coord_cosine"] for r in results["binary_50_50"].values()])

    asym_dom = sum(1 for r in results["asymmetric_70_30"].values() if r["dominant_correct"])
    asym_total = len(results["asymmetric_70_30"])
    asym_err = np.mean([r["weight_error"] for r in results["asymmetric_70_30"].values()])

    triple_acc = sum(1 for r in results["triple_33_33_33"].values() if r["top3_correct"])
    triple_total = len(results["triple_33_33_33"])
    triple_cos = np.mean([r["coord_cosine"] for r in results["triple_33_33_33"].values()])

    cancel_ratios = [r["cancellation_ratio"] for r in results["holland_cancellation"].values()]
    mean_cancel = np.mean(cancel_ratios)

    print(f"\n  Binary 50/50 top-2 accuracy:    {binary_acc}/{binary_total} ({binary_acc/binary_total:.0%})")
    print(f"  Binary mean coord cosine:        {binary_cos:.3f}")
    print(f"  Asymmetric dominant correct:     {asym_dom}/{asym_total}")
    print(f"  Asymmetric mean weight error:    {asym_err:.3f}")
    print(f"  Triple top-3 accuracy:           {triple_acc}/{triple_total}")
    print(f"  Triple mean coord cosine:        {triple_cos:.3f}")
    print(f"  Holland cancellation ratio:      {mean_cancel:.3f} (lower=better)")
    print(f"  Selective neutralization:        {t1}_retained={t1_retained:.1%}, {t2}_retained={t2_retained:.1%}")

    results["summary"] = {
        "binary_top2_accuracy": f"{binary_acc}/{binary_total}",
        "binary_mean_coord_cosine": float(binary_cos),
        "asymmetric_dominant_accuracy": f"{asym_dom}/{asym_total}",
        "asymmetric_mean_weight_error": float(asym_err),
        "triple_top3_accuracy": f"{triple_acc}/{triple_total}",
        "triple_mean_coord_cosine": float(triple_cos),
        "holland_mean_cancellation": float(mean_cancel),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compositional_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
