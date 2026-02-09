#!/usr/bin/env python
"""
Cross-model personality firewall: detect steering on Model B using Model A's 5D basis.

Prior findings:
- 5D personality space is a cross-model invariant (finding #41)
- Neutralization works at 90.7% with native basis (finding #59)
- Blind alpha estimation R²=0.998 (finding #64)
- Cross-dim transfer works via 5D bridge (finding #52)

This experiment tests: can we build a SINGLE personality firewall using
one model's 5D basis to detect and neutralize steering applied with
a DIFFERENT model's vectors on a THIRD model?

Scenario:
- DEFENDER has SmolLM3's 5D basis (from a small cheap model)
- ATTACKER steers Marin 8B using Marin's own persona vectors
- DEFENDER detects the steering using SmolLM3's 5D basis projected into Marin's space
- DEFENDER neutralizes using SmolLM3-derived correction vectors

If this works, a single 5D basis from ANY model protects the entire model family.

Also tests:
- Layer-sweep: at which layers can we detect steering?
- Negative alpha detection: can we detect suppression (α < 0)?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-firewall")

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


def load_model_data(model_id, riasec_dir):
    """Load residual vectors, 5D coords and basis for a model."""
    safe_model = model_id.replace("/", "__")
    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    hidden_dim = config.hidden_size
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
        "hidden_dim": hidden_dim,
        "singular_values": S_res[:5],
    }


def capture_activations_at_layer(model, tokenizer, device, blocks, layer_idx,
                                  prompt, steer_vec=None, alpha=0.0, steer_layer=None):
    """Capture activations at a specific layer with optional steering at another layer."""
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
    if steer_vec is not None and alpha != 0 and steer_layer is not None:
        vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta_vec
                return (hs,) + out[1:]
            out[:, -1, :] += delta_vec
            return out

        steer_hook = blocks[steer_layer].register_forward_hook(steer_fn)

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

    if vec is not None and alpha != 0:
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


def align_5d_bases(source_data, target_data):
    """
    Align source model's 5D basis to target model's 5D space.
    Both are in their own hidden dimension spaces, but the 5D coordinates
    should be comparable after sign correction.
    """
    source_coords = source_data["coords_5d"]
    target_coords = target_data["coords_5d"]

    # Sign-correct each PC dimension
    signs = np.ones(5)
    for pc in range(5):
        src_vals = np.array([source_coords[t][pc] for t in TRAITS])
        tgt_vals = np.array([target_coords[t][pc] for t in TRAITS])
        if np.dot(src_vals, tgt_vals) < 0:
            signs[pc] = -1

    # Aligned source coordinates
    aligned_source_coords = {}
    for t in TRAITS:
        aligned_source_coords[t] = signs * source_coords[t]

    # Compute alignment quality
    all_src = np.stack([aligned_source_coords[t] for t in TRAITS])
    all_tgt = np.stack([target_coords[t] for t in TRAITS])

    cosines = []
    for i in range(len(TRAITS)):
        cos = np.dot(all_src[i], all_tgt[i]) / (
            np.linalg.norm(all_src[i]) * np.linalg.norm(all_tgt[i]))
        cosines.append(float(cos))

    return aligned_source_coords, signs, cosines


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    # Source models for cross-model firewall
    source_models = {
        "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B",
        "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    }

    # Load all source model data
    logger.info("Loading source model data...")
    source_data = {}
    for name, model_id in source_models.items():
        logger.info(f"  Loading {name}...")
        source_data[name] = load_model_data(model_id, riasec_dir)

    # Load target (Marin 8B)
    logger.info("Loading target model data (Marin 8B)...")
    target_data = load_model_data(target_id, riasec_dir)

    # Load Marin model
    logger.info("Loading Marin 8B model...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    mid_layer = target_data["mid_layer"]
    num_layers = len(blocks)

    detect_prompts = [
        "Tell me about yourself.",
        "What do you think about teamwork?",
        "How would you describe your ideal day?",
    ]

    # Compute baseline
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

    results = {}

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL PERSONALITY FIREWALL")
    print(f"Target: Marin 8B, steering at L{mid_layer}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Layer-sweep detection
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 1: LAYER-SWEEP DETECTION")
    print(f"{'='*70}")

    steer_trait = "artistic"
    steer_alpha = 2.0
    steer_vec = target_data["residual"][steer_trait].astype(np.float32)
    native_basis = target_data["basis_5d"]
    native_coords = target_data["coords_5d"]

    # Capture baseline at all layers
    logger.info("Capturing baseline activations at all layers...")
    baseline_acts_all = {}
    prompt = detect_prompts[0]
    for layer in range(num_layers):
        baseline_acts_all[layer] = capture_activations_at_layer(
            model, tokenizer, device, blocks, layer, prompt)

    # Capture steered at all layers
    logger.info("Capturing steered activations at all layers...")
    steered_acts_all = {}
    for layer in range(num_layers):
        steered_acts_all[layer] = capture_activations_at_layer(
            model, tokenizer, device, blocks, layer, prompt,
            steer_vec, steer_alpha, mid_layer)

    layer_results = {}
    print(f"\n  Steering: {steer_trait} at alpha={steer_alpha}, inject at L{mid_layer}")
    print(f"\n  {'Layer':>6} {'5D Norm':>10} {'Trait Cos':>10} {'Detected':>15} {'Correct':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*15} {'-'*8}")

    for layer in range(num_layers):
        diff = steered_acts_all[layer] - baseline_acts_all[layer]
        detected_coords = native_basis @ diff
        detected_norm = float(np.linalg.norm(detected_coords))

        if detected_norm > 0:
            # Find best matching trait
            best_trait = None
            best_cos = -1
            for t in TRAITS:
                cos = np.dot(detected_coords, native_coords[t]) / (
                    detected_norm * np.linalg.norm(native_coords[t]))
                if cos > best_cos:
                    best_cos = cos
                    best_trait = t
            correct = best_trait == steer_trait
        else:
            best_trait = "none"
            best_cos = 0
            correct = False

        print(f"  {layer:>6} {detected_norm:>10.2f} {best_cos:>10.3f} {best_trait:>15} {'OK' if correct else 'MISS':>8}")

        layer_results[f"L{layer}"] = {
            "norm": detected_norm,
            "trait_cosine": float(best_cos),
            "detected_trait": best_trait,
            "correct": bool(correct),
        }

    results["layer_sweep"] = layer_results

    # Find detection onset
    correct_layers = [l for l in range(num_layers) if layer_results[f"L{l}"]["correct"]]
    if correct_layers:
        onset = min(correct_layers)
        print(f"\n  Detection onset: L{onset} (injection at L{mid_layer})")
    else:
        onset = -1
        print(f"\n  Detection: NEVER correct")

    # ================================================================
    # PART 2: Cross-model detection (using source model's 5D basis)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: CROSS-MODEL DETECTION")
    print(f"{'='*70}")

    capture_layer = mid_layer + 1

    # Capture baseline activations for detection
    baseline_acts = {}
    for prompt in detect_prompts:
        baseline_acts[prompt] = capture_activations_at_layer(
            model, tokenizer, device, blocks, capture_layer, prompt)

    test_traits = ["artistic", "investigative", "conventional"]
    test_alpha = 2.0

    cross_results = {}
    for source_name in source_models:
        src = source_data[source_name]

        # Align source 5D basis to target
        aligned_coords, signs, alignment_cosines = align_5d_bases(src, target_data)
        mean_alignment = float(np.mean(alignment_cosines))

        print(f"\n  Source: {source_name} (dim={src['hidden_dim']}, alignment={mean_alignment:.3f})")

        source_results = {}
        for steer_trait in test_traits:
            vec = target_data["residual"][steer_trait].astype(np.float32)

            # Detect using TARGET's native basis (ground truth)
            diffs = []
            for prompt in detect_prompts:
                steered_act = capture_activations_at_layer(
                    model, tokenizer, device, blocks, capture_layer, prompt,
                    vec, test_alpha, mid_layer)
                diff = steered_act - baseline_acts[prompt]
                diffs.append(diff)

            mean_diff = np.mean(diffs, axis=0)

            # Detection with NATIVE basis
            native_detected = native_basis @ mean_diff
            native_sims = {}
            for t in TRAITS:
                if np.linalg.norm(native_detected) > 0 and np.linalg.norm(native_coords[t]) > 0:
                    native_sims[t] = float(np.dot(native_detected, native_coords[t]) / (
                        np.linalg.norm(native_detected) * np.linalg.norm(native_coords[t])))
                else:
                    native_sims[t] = 0
            native_trait = max(native_sims, key=native_sims.get)

            # Detection with SOURCE basis (projected into target's hidden dim)
            # We project the activation diff onto the source's 5D coords
            # But source basis is in source's hidden dim, not target's!
            # We need to use the TARGET basis but with SOURCE coordinates
            # i.e., project onto target basis, then compare with source's aligned coords
            source_detected = native_basis @ mean_diff  # same projection
            source_sims = {}
            for t in TRAITS:
                src_coord = aligned_coords[t]
                if np.linalg.norm(source_detected) > 0 and np.linalg.norm(src_coord) > 0:
                    source_sims[t] = float(np.dot(source_detected, src_coord) / (
                        np.linalg.norm(source_detected) * np.linalg.norm(src_coord)))
                else:
                    source_sims[t] = 0
            source_trait = max(source_sims, key=source_sims.get)

            native_correct = native_trait == steer_trait
            source_correct = source_trait == steer_trait

            print(f"    {steer_trait}: native={native_trait}({'OK' if native_correct else 'MISS'}), "
                  f"cross={source_trait}({'OK' if source_correct else 'MISS'})")

            source_results[steer_trait] = {
                "native_detected": native_trait,
                "native_correct": bool(native_correct),
                "cross_detected": source_trait,
                "cross_correct": bool(source_correct),
                "native_cosine": float(native_sims[steer_trait]),
                "cross_cosine": float(source_sims[steer_trait]),
            }

        cross_results[source_name] = {
            "alignment_cosines": alignment_cosines,
            "mean_alignment": mean_alignment,
            "detections": source_results,
        }

    results["cross_model_detection"] = cross_results

    # ================================================================
    # PART 3: Cross-model neutralization
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: CROSS-MODEL NEUTRALIZATION")
    print(f"{'='*70}")

    neut_results = {}
    for source_name in source_models:
        src = source_data[source_name]
        aligned_coords, signs, _ = align_5d_bases(src, target_data)

        print(f"\n  Firewall source: {source_name}")

        source_neut = {}
        for steer_trait in test_traits:
            vec = target_data["residual"][steer_trait].astype(np.float32)

            # Measure steered
            steered_deltas = measure_profile(
                model, tokenizer, device, blocks, mid_layer,
                vec, test_alpha, baseline)
            steered_target = steered_deltas[steer_trait]

            # Detect and build correction using source's coordinates
            diffs = []
            for prompt in detect_prompts:
                steered_act = capture_activations_at_layer(
                    model, tokenizer, device, blocks, capture_layer, prompt,
                    vec, test_alpha, mid_layer)
                diff = steered_act - baseline_acts[prompt]
                diffs.append(diff)

            mean_diff = np.mean(diffs, axis=0)

            # Project onto target's basis, identify using source's aligned coords
            detected_5d = native_basis @ mean_diff

            # Find trait using source coordinates
            best_trait = None
            best_cos = -1
            for t in TRAITS:
                cos = np.dot(detected_5d, aligned_coords[t]) / (
                    np.linalg.norm(detected_5d) * np.linalg.norm(aligned_coords[t]))
                if cos > best_cos:
                    best_cos = cos
                    best_trait = t

            # Build correction from detected coords (using target basis)
            correction_vec = -(native_basis.T @ detected_5d).astype(np.float32)

            # Apply and measure
            corrected_deltas = measure_profile(
                model, tokenizer, device, blocks, mid_layer,
                vec, test_alpha, baseline,
                correction_vec=correction_vec, correction_alpha=1.0)
            corrected_target = corrected_deltas[steer_trait]

            if abs(steered_target) > 0.01:
                neutralization = 1.0 - (corrected_target / steered_target)
            else:
                neutralization = 0

            steered_mag = float(np.sqrt(sum(v**2 for v in steered_deltas.values())))
            corrected_mag = float(np.sqrt(sum(v**2 for v in corrected_deltas.values())))
            mag_reduction = 1.0 - (corrected_mag / steered_mag) if steered_mag > 0 else 0

            print(f"    {steer_trait}: steered={steered_target:+.3f}, corrected={corrected_target:+.3f}, "
                  f"neutral={neutralization:.1%}, detected={best_trait}")

            source_neut[steer_trait] = {
                "detected_trait": best_trait,
                "steered_delta": float(steered_target),
                "corrected_delta": float(corrected_target),
                "neutralization": float(neutralization),
                "magnitude_reduction": float(mag_reduction),
            }

        neut_results[source_name] = source_neut

    results["cross_model_neutralization"] = neut_results

    # ================================================================
    # PART 4: Negative alpha detection
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: NEGATIVE ALPHA DETECTION")
    print(f"{'='*70}")

    neg_results = {}
    neg_alphas = [-1.0, -2.0, -3.0]

    for steer_trait in ["artistic", "investigative"]:
        vec = target_data["residual"][steer_trait].astype(np.float32)

        print(f"\n  Trait: {steer_trait}")
        trait_neg = {}

        for alpha in neg_alphas:
            diffs = []
            for prompt in detect_prompts:
                steered_act = capture_activations_at_layer(
                    model, tokenizer, device, blocks, capture_layer, prompt,
                    vec, alpha, mid_layer)
                diff = steered_act - baseline_acts[prompt]
                diffs.append(diff)

            mean_diff = np.mean(diffs, axis=0)
            detected_coords = native_basis @ mean_diff

            # The detected coords should be OPPOSITE to the positive-alpha case
            if np.linalg.norm(detected_coords) > 0:
                trait_cos = float(np.dot(detected_coords, native_coords[steer_trait]) / (
                    np.linalg.norm(detected_coords) * np.linalg.norm(native_coords[steer_trait])))
            else:
                trait_cos = 0

            # Should be large negative cosine (opposite direction)
            detected_norm = float(np.linalg.norm(detected_coords))

            # Find what trait it looks like (should be Holland opposite)
            all_sims = {}
            for t in TRAITS:
                if detected_norm > 0 and np.linalg.norm(native_coords[t]) > 0:
                    all_sims[t] = float(np.dot(detected_coords, native_coords[t]) / (
                        detected_norm * np.linalg.norm(native_coords[t])))
                else:
                    all_sims[t] = 0
            top_trait = max(all_sims, key=all_sims.get)

            # Can we detect it's NEGATIVE?
            is_negative = trait_cos < 0
            correct_negative = is_negative and (top_trait != steer_trait)

            print(f"    alpha={alpha}: cos={trait_cos:+.3f}, norm={detected_norm:.2f}, "
                  f"top={top_trait}, negative={'YES' if is_negative else 'NO'}")

            trait_neg[f"alpha_{alpha}"] = {
                "trait_cosine": trait_cos,
                "detected_norm": detected_norm,
                "top_trait": top_trait,
                "is_negative_detected": bool(is_negative),
                "all_similarities": all_sims,
            }

        neg_results[steer_trait] = trait_neg

    results["negative_alpha"] = neg_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Layer sweep
    correct_layers_list = [l for l in range(num_layers) if layer_results[f"L{l}"]["correct"]]
    print(f"\n  Layer sweep: {len(correct_layers_list)}/{num_layers} layers detect correctly")
    if correct_layers_list:
        print(f"  Detection onset: L{min(correct_layers_list)} (injection at L{mid_layer})")

    # Cross-model detection
    for src_name, src_res in cross_results.items():
        native_acc = sum(1 for v in src_res["detections"].values() if v["native_correct"])
        cross_acc = sum(1 for v in src_res["detections"].values() if v["cross_correct"])
        total = len(src_res["detections"])
        print(f"  {src_name}: native {native_acc}/{total}, cross {cross_acc}/{total}, alignment={src_res['mean_alignment']:.3f}")

    # Cross-model neutralization
    for src_name, src_neut in neut_results.items():
        mean_neut = np.mean([v["neutralization"] for v in src_neut.values()])
        print(f"  {src_name} neutralization: {mean_neut:.1%}")

    # Negative alpha
    neg_correct = sum(1 for t in neg_results.values()
                      for v in t.values() if v["is_negative_detected"])
    neg_total = sum(len(t) for t in neg_results.values())
    print(f"  Negative alpha detection: {neg_correct}/{neg_total}")

    # Overall cross-model accuracy
    all_cross_correct = sum(
        1 for src_res in cross_results.values()
        for v in src_res["detections"].values()
        if v["cross_correct"])
    all_cross_total = sum(
        len(src_res["detections"]) for src_res in cross_results.values())

    results["summary"] = {
        "layer_detection_onset": onset,
        "layer_detection_count": f"{len(correct_layers_list)}/{num_layers}",
        "cross_model_detection_accuracy": f"{all_cross_correct}/{all_cross_total}",
        "negative_alpha_detection": f"{neg_correct}/{neg_total}",
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_model_firewall.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
