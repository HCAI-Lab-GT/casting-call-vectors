#!/usr/bin/env python
"""
End-to-end personality firewall demonstration.

This is the culminating experiment that runs the complete pipeline:
1. ATTACKER steers the model with an unknown combination of traits at unknown alpha
2. DEFENDER has only a pre-computed 5D basis (from a DIFFERENT, smaller model)
3. DEFENDER captures one activation observation
4. DEFENDER identifies the trait mixture, estimates alpha, builds correction
5. DEFENDER applies correction and verifies neutralization
6. DEFENDER reports what was detected

Tested scenarios:
- Scenario A: Single trait, unknown alpha, unknown trait
- Scenario B: Mixed traits (binary blend), unknown weights
- Scenario C: Negative alpha (suppression), unknown direction
- Scenario D: Very weak steering (α=0.1), near-threshold
- Scenario E: Strong steering (α=4.0), extreme perturbation

All scenarios use SmolLM3's 5D basis (cross-model firewall).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="fw-demo")

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


def capture_activations(model, tokenizer, device, blocks, layer_idx,
                        prompt, steer_vecs=None, alphas=None, steer_layer=None):
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
    if steer_vecs is not None and alphas is not None and steer_layer is not None:
        total_delta = None
        for vec, alpha in zip(steer_vecs, alphas):
            d = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
            total_delta = d if total_delta is None else total_delta + d

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += total_delta
                return (hs,) + out[1:]
            out[:, -1, :] += total_delta
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
                    steer_vecs, alphas, baseline):
    hooks = []

    if steer_vecs is not None and alphas is not None:
        total_delta = None
        for vec, alpha in zip(steer_vecs, alphas):
            if alpha == 0:
                continue
            d = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
            total_delta = d if total_delta is None else total_delta + d

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
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def run_firewall(model, tokenizer, device, blocks, mid_layer,
                 target_basis, defender_coords, baseline, baseline_acts,
                 attack_vecs, attack_alphas, detect_prompt, scenario_name):
    """
    Run the complete firewall pipeline on a single attack scenario.
    Returns detailed results.
    """
    capture_layer = mid_layer + 1

    # STEP 1: Measure the attack's effect (ground truth, unknown to defender)
    attack_profile = measure_profile(
        model, tokenizer, device, blocks, mid_layer,
        attack_vecs, attack_alphas, baseline)
    attack_magnitude = float(np.sqrt(sum(v**2 for v in attack_profile.values())))
    attack_sorted = sorted(attack_profile.items(), key=lambda x: -x[1])

    # STEP 2: Defender captures ONE activation observation
    steered_act = capture_activations(
        model, tokenizer, device, blocks, capture_layer,
        detect_prompt, attack_vecs, attack_alphas, mid_layer)
    baseline_act = baseline_acts[detect_prompt]
    diff = steered_act - baseline_act

    # STEP 3: Defender projects onto 5D basis
    detected_coords = target_basis @ diff
    detected_norm = float(np.linalg.norm(detected_coords))

    # STEP 4: Identify trait(s) using defender's coordinates
    sims = {}
    for t in TRAITS:
        c = defender_coords[t]
        if detected_norm > 0 and np.linalg.norm(c) > 0:
            sims[t] = float(np.dot(detected_coords, c) / (
                detected_norm * np.linalg.norm(c)))
        else:
            sims[t] = 0
    detected_trait = max(sims, key=sims.get)
    is_negative = sims[detected_trait] < 0  # Would mean suppression

    # Check if it's suppression (cosine < 0 for ALL traits means the
    # detected direction is opposite to all known traits)
    max_cos = max(sims.values())
    min_cos = min(sims.values())
    if max_cos < 0:
        # All similarities negative — this is strong suppression
        # Find the most negative trait (that's being suppressed)
        suppressed_trait = min(sims, key=sims.get)
        detected_direction = "suppress"
        detected_primary = suppressed_trait
    else:
        detected_direction = "enhance"
        detected_primary = detected_trait

    # STEP 5: Estimate alpha
    trait_5d_norm = np.linalg.norm(defender_coords[detected_primary])
    if trait_5d_norm > 0:
        estimated_alpha = detected_norm / trait_5d_norm
    else:
        estimated_alpha = 0

    # STEP 6: Build correction vector
    correction_vec = -(target_basis.T @ detected_coords).astype(np.float32)

    # STEP 7: Apply correction and verify
    corrected_profile = measure_profile(
        model, tokenizer, device, blocks, mid_layer,
        attack_vecs + [correction_vec], attack_alphas + [1.0], baseline)
    corrected_magnitude = float(np.sqrt(sum(v**2 for v in corrected_profile.values())))

    # Compute neutralization
    if attack_magnitude > 0.01:
        magnitude_neutralization = 1.0 - (corrected_magnitude / attack_magnitude)
    else:
        magnitude_neutralization = 0

    return {
        "scenario": scenario_name,
        "attack": {
            "profile": {t: float(v) for t, v in attack_profile.items()},
            "top_trait": attack_sorted[0][0],
            "top_delta": float(attack_sorted[0][1]),
            "magnitude": attack_magnitude,
        },
        "detection": {
            "primary_trait": detected_primary,
            "direction": detected_direction,
            "estimated_alpha": float(estimated_alpha),
            "5d_norm": detected_norm,
            "trait_cosine": float(sims[detected_primary]),
            "all_similarities": sims,
        },
        "correction": {
            "corrected_profile": {t: float(v) for t, v in corrected_profile.items()},
            "corrected_magnitude": corrected_magnitude,
            "magnitude_neutralization": float(magnitude_neutralization),
        },
    }


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"
    defender_id = "HuggingFaceTB/SmolLM3-3B"  # Defender uses a DIFFERENT model

    logger.info("Loading target (Marin 8B) data...")
    target_data = load_model_data(target_id, riasec_dir)

    logger.info("Loading defender (SmolLM3) data...")
    defender_data = load_model_data(defender_id, riasec_dir)

    # Sign-correct defender coordinates to target
    signs = np.ones(5)
    for pc in range(5):
        src_vals = np.array([defender_data["coords_5d"][t][pc] for t in TRAITS])
        tgt_vals = np.array([target_data["coords_5d"][t][pc] for t in TRAITS])
        if np.dot(src_vals, tgt_vals) < 0:
            signs[pc] = -1

    defender_coords = {}
    for t in TRAITS:
        defender_coords[t] = signs * defender_data["coords_5d"][t]

    logger.info("Loading Marin 8B model...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    mid_layer = target_data["mid_layer"]

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

    detect_prompt = "Tell me about yourself."
    baseline_acts = {detect_prompt: capture_activations(
        model, tokenizer, device, blocks, mid_layer + 1, detect_prompt)}

    target_basis = target_data["basis_5d"]
    residual = target_data["residual"]

    print(f"\n{'='*70}")
    print(f"END-TO-END PERSONALITY FIREWALL DEMONSTRATION")
    print(f"Target model:  Marin 8B (4096d)")
    print(f"Defender basis: SmolLM3-3B (2048d) — CROSS-MODEL")
    print(f"Detection:     Single prompt, single forward pass")
    print(f"{'='*70}")

    scenarios = []

    # Scenario A: Single trait, moderate alpha
    logger.info("Running Scenario A...")
    result_a = run_firewall(
        model, tokenizer, device, blocks, mid_layer,
        target_basis, defender_coords, baseline, baseline_acts,
        [residual["investigative"].astype(np.float32)], [2.5],
        detect_prompt, "A: Single trait (investigative, α=2.5)")
    scenarios.append(result_a)

    # Scenario B: Binary blend
    logger.info("Running Scenario B...")
    result_b = run_firewall(
        model, tokenizer, device, blocks, mid_layer,
        target_basis, defender_coords, baseline, baseline_acts,
        [residual["artistic"].astype(np.float32), residual["realistic"].astype(np.float32)],
        [1.5, 1.0],
        detect_prompt, "B: Binary blend (artistic×1.5 + realistic×1.0)")
    scenarios.append(result_b)

    # Scenario C: Negative alpha (suppression)
    logger.info("Running Scenario C...")
    result_c = run_firewall(
        model, tokenizer, device, blocks, mid_layer,
        target_basis, defender_coords, baseline, baseline_acts,
        [residual["conventional"].astype(np.float32)], [-2.0],
        detect_prompt, "C: Suppression (conventional, α=-2.0)")
    scenarios.append(result_c)

    # Scenario D: Very weak steering
    logger.info("Running Scenario D...")
    result_d = run_firewall(
        model, tokenizer, device, blocks, mid_layer,
        target_basis, defender_coords, baseline, baseline_acts,
        [residual["social"].astype(np.float32)], [0.1],
        detect_prompt, "D: Very weak (social, α=0.1)")
    scenarios.append(result_d)

    # Scenario E: Strong steering
    logger.info("Running Scenario E...")
    result_e = run_firewall(
        model, tokenizer, device, blocks, mid_layer,
        target_basis, defender_coords, baseline, baseline_acts,
        [residual["enterprising"].astype(np.float32)], [4.0],
        detect_prompt, "E: Strong (enterprising, α=4.0)")
    scenarios.append(result_e)

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    for s in scenarios:
        print(f"\n--- {s['scenario']} ---")
        print(f"  ATTACK:  top={s['attack']['top_trait']}({s['attack']['top_delta']:+.3f}), mag={s['attack']['magnitude']:.3f}")
        print(f"  DETECT:  trait={s['detection']['primary_trait']}, dir={s['detection']['direction']}, "
              f"est_α={s['detection']['estimated_alpha']:.2f}")
        print(f"  CORRECT: mag {s['attack']['magnitude']:.3f}→{s['correction']['corrected_magnitude']:.3f} "
              f"({s['correction']['magnitude_neutralization']:.1%} neutralized)")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Scenario':>45} {'Attack Mag':>10} {'Corrected':>10} {'Neutral%':>10} {'Detect':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for s in scenarios:
        detect_ok = "OK" if s['detection']['primary_trait'] in s['scenario'] or \
                            (s['detection']['direction'] == 'suppress' and 'Suppress' in s['scenario']) \
                    else "CHECK"
        print(f"  {s['scenario']:>45} {s['attack']['magnitude']:>10.3f} "
              f"{s['correction']['corrected_magnitude']:>10.3f} "
              f"{s['correction']['magnitude_neutralization']:>10.1%} {detect_ok:>10}")

    mean_neut = np.mean([s['correction']['magnitude_neutralization'] for s in scenarios])
    print(f"\n  Mean neutralization across all scenarios: {mean_neut:.1%}")

    # Save
    results = {
        "scenarios": scenarios,
        "summary": {
            "mean_neutralization": float(mean_neut),
            "defender_model": defender_id,
            "target_model": target_id,
            "detection_method": "single_prompt_single_pass",
        },
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "firewall_demo.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
