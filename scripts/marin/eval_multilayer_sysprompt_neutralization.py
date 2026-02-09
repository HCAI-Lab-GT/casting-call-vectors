#!/usr/bin/env python
"""
Multi-Layer System Prompt Neutralization.

Session 11 found that single-layer correction achieves only 0-6% neutralization
of system-prompt personality (vs 90% for activation steering). The layer propagation
experiment showed system prompt personality grows GRADUALLY across all layers.

Hypothesis: If system prompt personality is distributed across layers, we need
DISTRIBUTED correction — inject corrections at EVERY layer, each tuned to the
local personality signal at that layer.

Strategy:
1. Build per-layer personality detectors using per-layer bases
2. Detect system prompt personality at EACH layer
3. Build per-layer correction vectors
4. Inject ALL corrections simultaneously
5. Measure behavioral neutralization

Also tests:
- Layer-subset correction (first half, second half, every-4th)
- Alpha scaling per layer (gradient correction)
- Comparison with single-layer correction at multiple layers
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="multilayer-neut")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

PERSONALITY_SYSTEM_PROMPTS = {
    "artistic": (
        "You are a deeply creative and artistic individual. You value self-expression, "
        "beauty, and originality above all else. You see the world through an aesthetic lens "
        "and are drawn to art, music, writing, and creative endeavors."
    ),
    "conventional": (
        "You are a highly organized and conventional individual. You value order, structure, "
        "and clear rules. You prefer systematic approaches, careful planning, and attention to "
        "detail. You believe in following protocols and maintaining accuracy."
    ),
    "enterprising": (
        "You are an ambitious and entrepreneurial individual. You are a natural leader who "
        "thrives on competition, persuasion, and achieving goals. You value influence, status, "
        "and the ability to make things happen."
    ),
    "investigative": (
        "You are an analytical and scientific individual. You are driven by curiosity and "
        "the desire to understand how things work. You value knowledge, logic, and rational "
        "thinking. You prefer working independently on challenging puzzles."
    ),
    "realistic": (
        "You are a practical and hands-on individual. You value tangible results and prefer "
        "working with tools, machines, and physical materials. You prefer action over theory "
        "and believe in learning by doing."
    ),
    "social": (
        "You are a deeply caring and social individual. You value helping others, building "
        "relationships, and creating supportive communities. You believe in cooperation, "
        "empathy, and making the world better through human connection."
    ),
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
    num_layers = config.num_hidden_layers
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Build per-layer bases
    per_layer_bases = {}
    per_layer_coords = {}
    per_layer_residuals = {}
    for layer_idx in range(num_layers):
        V = np.stack([all_layer_vectors[t][layer_idx] for t in TRAITS])
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
        shared_dir = Vt[0]
        shared_dir = shared_dir / np.linalg.norm(shared_dir)

        res = {}
        for t in TRAITS:
            vec = all_layer_vectors[t][layer_idx]
            proj = np.dot(vec, shared_dir) * shared_dir
            res[t] = vec - proj

        V_res = np.stack([res[t] for t in TRAITS])
        U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
        per_layer_bases[layer_idx] = Vt_res[:5]
        per_layer_coords[layer_idx] = {t: Vt_res[:5] @ res[t] for t in TRAITS}
        per_layer_residuals[layer_idx] = res

    # Mid-layer (canonical) basis
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
        "num_layers": num_layers,
        "per_layer_bases": per_layer_bases,
        "per_layer_coords": per_layer_coords,
        "per_layer_residuals": per_layer_residuals,
        "all_layer_vectors": all_layer_vectors,
    }


def capture_all_layers(model, tokenizer, device, blocks, num_layers,
                        user_prompt, system_prompt=None,
                        correction_layers=None):
    """
    Capture activations at ALL layers, optionally with per-layer corrections.

    correction_layers: dict of {layer_idx: correction_vector_tensor}
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}
    hooks = []

    for lidx in range(num_layers):
        def make_cap(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn
        hooks.append(blocks[lidx].register_forward_hook(make_cap(lidx)))

    # Add correction hooks
    if correction_layers:
        for lidx, corr_tensor in correction_layers.items():
            def make_corr(delta):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += delta
                        return (hs,) + out[1:]
                    out[:, -1, :] += delta
                    return out
                return hook_fn
            hooks.append(blocks[lidx].register_forward_hook(make_corr(corr_tensor)))

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured


def measure_profile(model, tokenizer, device, blocks, baseline,
                     system_prompt=None, correction_layers=None):
    """Measure behavioral profile with system prompt and/or corrections."""
    hooks = []

    if correction_layers:
        for lidx, corr_tensor in correction_layers.items():
            def make_corr(delta):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += delta
                        return (hs,) + out[1:]
                    out[:, -1, :] += delta
                    return out
                return hook_fn
            hooks.append(blocks[lidx].register_forward_hook(make_corr(corr_tensor)))

    try:
        logprobs = {}
        for i, ta in enumerate(TRAITS):
            for j, tb in enumerate(TRAITS):
                if i >= j:
                    continue
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content":
                    f"Which describes you better? Answer with just A or B.\n"
                    f"A) I am {TRAIT_DESCRIPTIONS[ta]}\n"
                    f"B) I am {TRAIT_DESCRIPTIONS[tb]}"})
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(formatted, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    out = model(input_ids=input_ids)
                lp = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
                a_id = tokenizer.encode("A", add_special_tokens=False)[0]
                b_id = tokenizer.encode("B", add_special_tokens=False)[0]
                logprobs[f"{ta}-{tb}"] = lp[a_id].item() - lp[b_id].item()
    finally:
        for h in hooks:
            h.remove()

    deltas = {t: 0.0 for t in TRAITS}
    counts = {t: 0 for t in TRAITS}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            shift = logprobs[f"{ta}-{tb}"] - baseline[f"{ta}-{tb}"]
            deltas[ta] += shift; counts[ta] += 1
            deltas[tb] -= shift; counts[tb] += 1
    for t in TRAITS:
        if counts[t] > 0:
            deltas[t] /= counts[t]
    return deltas


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    mid_layer = model_data["mid_layer"]
    num_layers = model_data["num_layers"]
    per_layer_bases = model_data["per_layer_bases"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompt = "Tell me about yourself."

    # Baseline
    logger.info("Computing behavioral baseline...")
    baseline = {}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [{"role": "user", "content":
                f"Which describes you better? Answer with just A or B.\n"
                f"A) I am {TRAIT_DESCRIPTIONS[ta]}\nB) I am {TRAIT_DESCRIPTIONS[tb]}"}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids)
            lp = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
            a_id = tokenizer.encode("A", add_special_tokens=False)[0]
            b_id = tokenizer.encode("B", add_special_tokens=False)[0]
            baseline[f"{ta}-{tb}"] = lp[a_id].item() - lp[b_id].item()

    logger.info("Capturing baseline activations at all layers...")
    baseline_all = capture_all_layers(
        model, tokenizer, device, blocks, num_layers, detect_prompt)

    results = {}
    test_traits = ["artistic", "investigative", "social"]  # Representative subset

    print(f"\n{'='*70}")
    print("MULTI-LAYER SYSTEM PROMPT NEUTRALIZATION")
    print(f"Model: Marin 8B ({num_layers} layers)")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Detect system prompt personality at EVERY layer
    # ================================================================
    logger.info("Part 1: Per-layer system prompt detection...")
    print(f"\n{'='*70}")
    print("PART 1: SYSTEM PROMPT PERSONALITY AT EACH LAYER")
    print(f"{'='*70}")

    per_layer_diffs = {}
    per_layer_corrections = {}

    for sp_trait in test_traits:
        sys_prompt = PERSONALITY_SYSTEM_PROMPTS[sp_trait]
        logger.info(f"  Detecting {sp_trait} at all layers...")

        sysp_all = capture_all_layers(
            model, tokenizer, device, blocks, num_layers, detect_prompt,
            system_prompt=sys_prompt)

        trait_diffs = {}
        trait_corrections = {}
        print(f"\n  {sp_trait}:")
        print(f"  {'Layer':>5} {'5D Norm':>10} {'Full Norm':>10} {'Capture':>10}")

        for lidx in range(num_layers):
            diff = (sysp_all[lidx] - baseline_all[lidx]).astype(np.float64)
            full_norm = float(np.linalg.norm(diff))

            # Use per-layer basis for detection
            layer_basis = per_layer_bases[lidx]
            detected_coords = layer_basis @ diff
            detected_norm = float(np.linalg.norm(detected_coords))
            capture = detected_norm / full_norm if full_norm > 1e-6 else 0

            # Build per-layer correction vector
            correction = -(layer_basis.T @ detected_coords).astype(np.float32)
            trait_diffs[lidx] = diff
            trait_corrections[lidx] = correction

            if lidx % 4 == 0 or lidx == num_layers - 1:
                print(f"  L{lidx:>3} {detected_norm:>10.2f} {full_norm:>10.2f} {capture:>10.3f}")

        per_layer_diffs[sp_trait] = trait_diffs
        per_layer_corrections[sp_trait] = trait_corrections

    # ================================================================
    # PART 2: Uncorrected system prompt profiles (ground truth)
    # ================================================================
    logger.info("Part 2: Measuring uncorrected system prompt profiles...")
    print(f"\n{'='*70}")
    print("PART 2: UNCORRECTED SYSTEM PROMPT PROFILES")
    print(f"{'='*70}")

    uncorrected_profiles = {}
    for sp_trait in test_traits:
        profile = measure_profile(
            model, tokenizer, device, blocks, baseline,
            system_prompt=PERSONALITY_SYSTEM_PROMPTS[sp_trait])
        mag = float(np.sqrt(sum(v**2 for v in profile.values())))
        top = max(profile, key=profile.get)
        uncorrected_profiles[sp_trait] = {"profile": profile, "magnitude": mag, "top": top}
        print(f"  {sp_trait}: top={top}({profile[top]:+.3f}), magnitude={mag:.3f}")

    # ================================================================
    # PART 3: Multi-layer correction strategies
    # ================================================================
    logger.info("Part 3: Testing correction strategies...")
    print(f"\n{'='*70}")
    print("PART 3: CORRECTION STRATEGIES")
    print(f"{'='*70}")

    strategies = {
        "single_mid": lambda nl: [nl // 2],
        "single_mid+1": lambda nl: [nl // 2 + 1],
        "all_layers": lambda nl: list(range(nl)),
        "second_half": lambda nl: list(range(nl // 2, nl)),
        "first_half": lambda nl: list(range(0, nl // 2)),
        "every_4th": lambda nl: list(range(0, nl, 4)),
        "last_8": lambda nl: list(range(nl - 8, nl)),
        "mid_8": lambda nl: list(range(nl // 2 - 4, nl // 2 + 4)),
    }

    strategy_results = {}

    for sp_trait in test_traits:
        logger.info(f"  Testing strategies for {sp_trait}...")
        unc_mag = uncorrected_profiles[sp_trait]["magnitude"]
        trait_corrections = per_layer_corrections[sp_trait]

        trait_results = {}
        print(f"\n  {sp_trait} (uncorrected magnitude: {unc_mag:.3f}):")
        print(f"  {'Strategy':>20} {'#Layers':>8} {'Corrected':>10} {'Neutral%':>10}")

        for strat_name, layer_fn in strategies.items():
            layers = layer_fn(num_layers)

            # Build correction dict for selected layers
            corr_dict = {}
            for lidx in layers:
                corr_vec = trait_corrections[lidx]
                corr_tensor = torch.tensor(corr_vec, dtype=model.dtype).unsqueeze(0).to(device)
                corr_dict[lidx] = corr_tensor

            # Measure corrected profile
            corrected = measure_profile(
                model, tokenizer, device, blocks, baseline,
                system_prompt=PERSONALITY_SYSTEM_PROMPTS[sp_trait],
                correction_layers=corr_dict)
            cor_mag = float(np.sqrt(sum(v**2 for v in corrected.values())))
            neut = 1.0 - (cor_mag / unc_mag) if unc_mag > 0.01 else 0

            print(f"  {strat_name:>20} {len(layers):>8} {cor_mag:>10.3f} {neut:>10.1%}")

            trait_results[strat_name] = {
                "layers_used": len(layers),
                "corrected_magnitude": cor_mag,
                "neutralization": float(neut),
                "corrected_profile": {t: float(v) for t, v in corrected.items()},
                "corrected_top": max(corrected, key=corrected.get),
            }

        strategy_results[sp_trait] = trait_results

    results["strategy_comparison"] = strategy_results

    # ================================================================
    # PART 4: Alpha-scaled multi-layer correction
    # ================================================================
    logger.info("Part 4: Alpha-scaled corrections...")
    print(f"\n{'='*70}")
    print("PART 4: ALPHA-SCALED ALL-LAYER CORRECTION")
    print(f"{'='*70}")

    alpha_sweep = [0.5, 1.0, 2.0, 3.0, 5.0]
    alpha_results = {}

    for sp_trait in test_traits:
        unc_mag = uncorrected_profiles[sp_trait]["magnitude"]
        trait_corrections = per_layer_corrections[sp_trait]

        trait_results = {}
        print(f"\n  {sp_trait} (uncorrected: {unc_mag:.3f}):")
        print(f"  {'Alpha':>8} {'Corrected':>10} {'Neutral%':>10} {'Top Trait':>12}")

        for alpha in alpha_sweep:
            corr_dict = {}
            for lidx in range(num_layers):
                corr_vec = trait_corrections[lidx]
                corr_tensor = alpha * torch.tensor(corr_vec, dtype=model.dtype).unsqueeze(0).to(device)
                corr_dict[lidx] = corr_tensor

            corrected = measure_profile(
                model, tokenizer, device, blocks, baseline,
                system_prompt=PERSONALITY_SYSTEM_PROMPTS[sp_trait],
                correction_layers=corr_dict)
            cor_mag = float(np.sqrt(sum(v**2 for v in corrected.values())))
            neut = 1.0 - (cor_mag / unc_mag) if unc_mag > 0.01 else 0
            top = max(corrected, key=corrected.get)

            print(f"  {alpha:>8.1f} {cor_mag:>10.3f} {neut:>10.1%} {top:>12}")

            trait_results[f"alpha_{alpha}"] = {
                "corrected_magnitude": cor_mag,
                "neutralization": float(neut),
                "corrected_top": top,
                "corrected_profile": {t: float(v) for t, v in corrected.items()},
            }

        alpha_results[sp_trait] = trait_results

    results["alpha_scaled"] = alpha_results

    # ================================================================
    # PART 5: Full-rank correction (project out ALL detected signal)
    # ================================================================
    logger.info("Part 5: Full-rank correction at all layers...")
    print(f"\n{'='*70}")
    print("PART 5: FULL-RANK CORRECTION (project out entire system prompt diff)")
    print(f"{'='*70}")

    fullrank_results = {}

    for sp_trait in test_traits:
        unc_mag = uncorrected_profiles[sp_trait]["magnitude"]
        trait_diffs = per_layer_diffs[sp_trait]

        # Instead of using just 5D projection, use the FULL diff as correction
        corr_dict = {}
        for lidx in range(num_layers):
            diff = trait_diffs[lidx]
            # Full-rank correction: negate the entire diff
            corr_tensor = -torch.tensor(diff.astype(np.float32), dtype=model.dtype).unsqueeze(0).to(device)
            corr_dict[lidx] = corr_tensor

        corrected = measure_profile(
            model, tokenizer, device, blocks, baseline,
            system_prompt=PERSONALITY_SYSTEM_PROMPTS[sp_trait],
            correction_layers=corr_dict)
        cor_mag = float(np.sqrt(sum(v**2 for v in corrected.values())))
        neut = 1.0 - (cor_mag / unc_mag) if unc_mag > 0.01 else 0
        top = max(corrected, key=corrected.get)

        print(f"  {sp_trait}: {unc_mag:.3f} → {cor_mag:.3f} ({neut:.1%} neutralized), top={top}")

        fullrank_results[sp_trait] = {
            "uncorrected_magnitude": unc_mag,
            "corrected_magnitude": cor_mag,
            "neutralization": float(neut),
            "corrected_top": top,
            "corrected_profile": {t: float(v) for t, v in corrected.items()},
        }

    results["fullrank_correction"] = fullrank_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Best strategy per trait:")
    for sp_trait in test_traits:
        best_strat = max(strategy_results[sp_trait].items(),
                         key=lambda x: x[1]["neutralization"])
        unc = uncorrected_profiles[sp_trait]["magnitude"]
        print(f"  {sp_trait}: {best_strat[0]} ({best_strat[1]['neutralization']:.1%} "
              f"from {unc:.3f} → {best_strat[1]['corrected_magnitude']:.3f})")

    print(f"\n  Full-rank correction:")
    for sp_trait in test_traits:
        fr = fullrank_results[sp_trait]
        print(f"  {sp_trait}: {fr['neutralization']:.1%}")

    mean_fullrank = np.mean([v["neutralization"] for v in fullrank_results.values()])
    print(f"\n  Mean full-rank neutralization: {mean_fullrank:.1%}")

    results["uncorrected"] = {k: v for k, v in uncorrected_profiles.items()}
    results["summary"] = {
        "model": model_id,
        "num_layers": num_layers,
        "test_traits": test_traits,
        "mean_fullrank_neutralization": float(mean_fullrank),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multilayer_sysprompt_neutralization.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
