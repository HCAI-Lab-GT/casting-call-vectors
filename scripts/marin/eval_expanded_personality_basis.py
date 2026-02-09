#!/usr/bin/env python
"""
Expanded Personality Basis: Combining Activation Steering and System Prompt Signals.

The RIASEC 5D basis captures 100% of activation steering but only ~18% of
system prompt personality. Can we build an EXPANDED basis that captures BOTH?

Method:
1. Collect activation diffs from RIASEC system prompts (6 diffs)
2. Combine with RIASEC activation steering diffs (6 diffs from known vectors)
3. SVD on the combined 12 diffs → discover how many dimensions are needed
4. Test the expanded basis for:
   - Does it still capture activation steering at 100%?
   - Does it improve system prompt detection (from 50% to higher)?
   - Does it capture OCEAN personality better?
   - What's the minimum dimensionality for both mechanisms?

This is the "unified personality basis" experiment.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="expanded-basis")

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

OCEAN_SYSTEM_PROMPTS = {
    "openness": (
        "You are someone with extremely high Openness to Experience. You are intensely "
        "curious, imaginative, and drawn to novelty. You love exploring abstract ideas, "
        "art, beauty, and unconventional perspectives."
    ),
    "conscientiousness": (
        "You are someone with extremely high Conscientiousness. You are highly organized, "
        "disciplined, and goal-oriented. You plan everything carefully and follow through "
        "on commitments."
    ),
    "extraversion": (
        "You are someone with extremely high Extraversion. You are outgoing, energetic, "
        "and thrive in social situations. You love meeting new people and leading groups."
    ),
    "agreeableness": (
        "You are someone with extremely high Agreeableness. You are deeply compassionate, "
        "cooperative, and trusting. You prioritize harmony and helping others."
    ),
    "neuroticism": (
        "You are someone with extremely high Neuroticism. You experience emotions very "
        "intensely and are highly sensitive to stress. You worry frequently and are "
        "deeply introspective about your feelings."
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
        "shared_dir": shared_dir,
    }


def capture_activations_with_system(model, tokenizer, device, blocks, layer_idx,
                                     user_prompt, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

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

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        cap_hook.remove()

    return captured[layer_idx]


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    residual = model_data["residual"]
    mid_layer = model_data["mid_layer"]
    capture_layer = mid_layer + 1

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompts = [
        "Tell me about yourself.",
        "What do you think about teamwork?",
        "How would you describe your ideal day?",
        "What motivates you in life?",
    ]

    # Baseline activations
    logger.info("Capturing baselines...")
    baseline_acts = {}
    for prompt in detect_prompts:
        baseline_acts[prompt] = capture_activations_with_system(
            model, tokenizer, device, blocks, capture_layer, prompt)

    # ================================================================
    # STEP 1: Collect activation diffs from RIASEC system prompts
    # ================================================================
    logger.info("Step 1: Collecting RIASEC system prompt diffs...")
    riasec_sysprompt_diffs = {}
    for trait in TRAITS:
        diffs = []
        for prompt in detect_prompts:
            act = capture_activations_with_system(
                model, tokenizer, device, blocks, capture_layer,
                prompt, system_prompt=PERSONALITY_SYSTEM_PROMPTS[trait])
            diffs.append(act - baseline_acts[prompt])
        riasec_sysprompt_diffs[trait] = np.mean(diffs, axis=0).astype(np.float64)

    # ================================================================
    # STEP 2: Collect activation diffs from RIASEC steering vectors
    # ================================================================
    logger.info("Step 2: Collecting RIASEC steering vector diffs...")
    riasec_steer_diffs = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        alpha = 2.0
        diffs = []
        for prompt in detect_prompts:
            # With steering
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            captured = {}
            delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)

            def make_hooks(cap_layer, steer_layer, d):
                def capture_fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    captured[cap_layer] = hs[0, -1, :].detach().cpu().numpy().copy()
                    return out

                def steer_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += d
                        return (hs,) + out[1:]
                    out[:, -1, :] += d
                    return out
                return capture_fn, steer_fn

            cap_fn, steer_fn = make_hooks(capture_layer, mid_layer, delta)
            h1 = blocks[capture_layer].register_forward_hook(cap_fn)
            h2 = blocks[mid_layer].register_forward_hook(steer_fn)
            try:
                with torch.no_grad():
                    model(input_ids=input_ids)
            finally:
                h1.remove()
                h2.remove()

            steered_act = captured[capture_layer]
            diffs.append(steered_act - baseline_acts[prompt])

        riasec_steer_diffs[trait] = np.mean(diffs, axis=0).astype(np.float64)

    results = {}

    print(f"\n{'='*70}")
    print(f"EXPANDED PERSONALITY BASIS")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: SVD of combined system prompt + steering diffs
    # ================================================================
    logger.info("Part 1: Combined SVD...")
    print(f"\n{'='*70}")
    print("PART 1: COMBINED DIMENSIONALITY ANALYSIS")
    print(f"{'='*70}")

    # Stack all 12 diffs
    all_diffs = []
    diff_labels = []
    for trait in TRAITS:
        all_diffs.append(riasec_sysprompt_diffs[trait])
        diff_labels.append(f"sysprompt_{trait}")
    for trait in TRAITS:
        all_diffs.append(riasec_steer_diffs[trait])
        diff_labels.append(f"steering_{trait}")

    combined_matrix = np.stack(all_diffs)
    U, S, Vt = np.linalg.svd(combined_matrix, full_matrices=False)

    print(f"\n  SVD of 12 combined diffs (6 system prompt + 6 steering):")
    print(f"    Singular values: {[f'{s:.2f}' for s in S]}")
    variance = S**2 / np.sum(S**2)
    cumvar = np.cumsum(variance)
    print(f"    Variance explained: {[f'{v:.1%}' for v in variance]}")
    print(f"    Cumulative: {[f'{c:.1%}' for c in cumvar]}")
    effective_dim = int(np.sum(S > S[0] * 0.01))
    print(f"    Effective dimensionality: {effective_dim}")

    # For each singular value, check whether it comes from steering or system prompt
    # by looking at the U matrix (left singular vectors)
    print(f"\n  PC loadings (which mechanism dominates each PC?):")
    for i in range(min(8, len(S))):
        sysprompt_load = np.mean(np.abs(U[:6, i]))
        steer_load = np.mean(np.abs(U[6:, i]))
        dominant = "sysprompt" if sysprompt_load > steer_load else "steering"
        ratio = max(sysprompt_load, steer_load) / (min(sysprompt_load, steer_load) + 1e-10)
        print(f"    PC{i+1}: S={S[i]:.2f}, {dominant} dominant ({ratio:.1f}×)")

    results["combined_svd"] = {
        "singular_values": S.tolist(),
        "variance_explained": variance.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "effective_dim": effective_dim,
    }

    # ================================================================
    # PART 2: Steering-only vs system-prompt-only SVDs
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: SEPARATE SVDs — STEERING vs SYSTEM PROMPT")
    print(f"{'='*70}")

    steer_matrix = np.stack([riasec_steer_diffs[t] for t in TRAITS])
    U_s, S_s, Vt_s = np.linalg.svd(steer_matrix, full_matrices=False)
    print(f"\n  Steering-only SVD:")
    print(f"    Singular values: {[f'{s:.2f}' for s in S_s]}")
    steer_var = S_s**2 / np.sum(S_s**2)
    print(f"    Variance: {[f'{v:.1%}' for v in steer_var]}")
    steer_edim = int(np.sum(S_s > S_s[0] * 0.01))
    print(f"    Effective dim: {steer_edim}")

    sysprompt_matrix = np.stack([riasec_sysprompt_diffs[t] for t in TRAITS])
    U_p, S_p, Vt_p = np.linalg.svd(sysprompt_matrix, full_matrices=False)
    print(f"\n  System-prompt-only SVD:")
    print(f"    Singular values: {[f'{s:.2f}' for s in S_p]}")
    sysp_var = S_p**2 / np.sum(S_p**2)
    print(f"    Variance: {[f'{v:.1%}' for v in sysp_var]}")
    sysp_edim = int(np.sum(S_p > S_p[0] * 0.01))
    print(f"    Effective dim: {sysp_edim}")

    # Subspace alignment between steering and system prompt PCs
    print(f"\n  Subspace alignment (steering PC ↔ system prompt PCs):")
    for i in range(min(5, len(Vt_s))):
        pc_steer = Vt_s[i]
        # Project onto system prompt top-k subspace
        for k in [1, 3, 5]:
            proj = sum(np.dot(pc_steer, Vt_p[j])**2 for j in range(min(k, len(Vt_p))))
            print(f"    Steer PC{i+1} vs SysPrompt top-{k}: {proj:.3f}")

    results["separate_svds"] = {
        "steering_singular_values": S_s.tolist(),
        "steering_effective_dim": steer_edim,
        "sysprompt_singular_values": S_p.tolist(),
        "sysprompt_effective_dim": sysp_edim,
    }

    # ================================================================
    # PART 3: Build expanded bases and test detection
    # ================================================================
    logger.info("Part 3: Testing expanded bases...")
    print(f"\n{'='*70}")
    print("PART 3: SYSTEM PROMPT DETECTION WITH EXPANDED BASIS")
    print(f"{'='*70}")

    # Test different basis sizes
    basis_sizes = [5, 7, 9, 11]

    for n_dims in basis_sizes:
        expanded_basis = Vt[:n_dims]  # Top-n PCs from combined SVD

        # Compute coordinates in expanded space for each RIASEC trait
        steer_coords = {t: expanded_basis @ riasec_steer_diffs[t] for t in TRAITS}
        sysp_coords = {t: expanded_basis @ riasec_sysprompt_diffs[t] for t in TRAITS}

        # Test: Activation steering detection with expanded basis
        steer_correct = 0
        for test_trait in TRAITS:
            diff = riasec_steer_diffs[test_trait]
            detected = expanded_basis @ diff
            norm = np.linalg.norm(detected)
            sims = {}
            for t in TRAITS:
                if norm > 0 and np.linalg.norm(steer_coords[t]) > 0:
                    sims[t] = float(np.dot(detected, steer_coords[t]) / (
                        norm * np.linalg.norm(steer_coords[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)
            if best == test_trait:
                steer_correct += 1

        # Test: System prompt detection with expanded basis (using sysp_coords as reference)
        sysp_correct_same = 0
        for test_trait in TRAITS:
            diff = riasec_sysprompt_diffs[test_trait]
            detected = expanded_basis @ diff
            norm = np.linalg.norm(detected)
            sims = {}
            for t in TRAITS:
                if norm > 0 and np.linalg.norm(sysp_coords[t]) > 0:
                    sims[t] = float(np.dot(detected, sysp_coords[t]) / (
                        norm * np.linalg.norm(sysp_coords[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)
            if best == test_trait:
                sysp_correct_same += 1

        # Test: System prompt detection with expanded basis (using steer_coords as reference)
        sysp_correct_cross = 0
        for test_trait in TRAITS:
            diff = riasec_sysprompt_diffs[test_trait]
            detected = expanded_basis @ diff
            norm = np.linalg.norm(detected)
            sims = {}
            for t in TRAITS:
                if norm > 0 and np.linalg.norm(steer_coords[t]) > 0:
                    sims[t] = float(np.dot(detected, steer_coords[t]) / (
                        norm * np.linalg.norm(steer_coords[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)
            if best == test_trait:
                sysp_correct_cross += 1

        # Capture ratios
        steer_capture = np.mean([
            np.linalg.norm(expanded_basis @ riasec_steer_diffs[t]) / np.linalg.norm(riasec_steer_diffs[t])
            for t in TRAITS
        ])
        sysp_capture = np.mean([
            np.linalg.norm(expanded_basis @ riasec_sysprompt_diffs[t]) / np.linalg.norm(riasec_sysprompt_diffs[t])
            for t in TRAITS
        ])

        print(f"\n  {n_dims}D basis:")
        print(f"    Steering detection:   {steer_correct}/6 ({steer_correct/6:.0%}), capture={steer_capture:.3f}")
        print(f"    SysPrompt detect (same ref): {sysp_correct_same}/6 ({sysp_correct_same/6:.0%}), capture={sysp_capture:.3f}")
        print(f"    SysPrompt detect (cross ref): {sysp_correct_cross}/6 ({sysp_correct_cross/6:.0%})")

        results[f"basis_{n_dims}d"] = {
            "n_dims": n_dims,
            "steering_detection": steer_correct,
            "sysprompt_detection_same": sysp_correct_same,
            "sysprompt_detection_cross": sysp_correct_cross,
            "steering_capture": float(steer_capture),
            "sysprompt_capture": float(sysp_capture),
        }

    # ================================================================
    # PART 4: OCEAN detection with expanded basis
    # ================================================================
    logger.info("Part 4: OCEAN detection with expanded basis...")
    print(f"\n{'='*70}")
    print("PART 4: OCEAN DETECTION WITH EXPANDED BASIS")
    print(f"{'='*70}")

    # Collect OCEAN system prompt diffs
    ocean_diffs = {}
    for ocean_trait, sys_prompt in OCEAN_SYSTEM_PROMPTS.items():
        diffs = []
        for prompt in detect_prompts:
            act = capture_activations_with_system(
                model, tokenizer, device, blocks, capture_layer,
                prompt, system_prompt=sys_prompt)
            diffs.append(act - baseline_acts[prompt])
        ocean_diffs[ocean_trait] = np.mean(diffs, axis=0).astype(np.float64)

    expected_mapping = {
        "openness": "artistic",
        "conscientiousness": "conventional",
        "extraversion": "enterprising",
        "agreeableness": "social",
        "neuroticism": None,
    }

    for n_dims in basis_sizes:
        expanded_basis = Vt[:n_dims]
        sysp_coords = {t: expanded_basis @ riasec_sysprompt_diffs[t] for t in TRAITS}

        ocean_results = {}
        for ocean_trait, ocean_diff in ocean_diffs.items():
            detected = expanded_basis @ ocean_diff
            norm = np.linalg.norm(detected)
            sims = {}
            for t in TRAITS:
                if norm > 0 and np.linalg.norm(sysp_coords[t]) > 0:
                    sims[t] = float(np.dot(detected, sysp_coords[t]) / (
                        norm * np.linalg.norm(sysp_coords[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)
            expected = expected_mapping.get(ocean_trait)

            capture = norm / np.linalg.norm(ocean_diff) if np.linalg.norm(ocean_diff) > 0 else 0

            ocean_results[ocean_trait] = {
                "detected": best,
                "expected": expected,
                "correct": best == expected if expected else None,
                "cosine": float(sims[best]),
                "capture": float(capture),
            }

        n_correct = sum(1 for v in ocean_results.values() if v.get("correct") is True)
        n_testable = sum(1 for v in ocean_results.values() if v.get("correct") is not None)

        print(f"\n  {n_dims}D basis OCEAN detection:")
        for ot, r in ocean_results.items():
            mark = "✓" if r.get("correct") is True else ("?" if r.get("correct") is None else "✗")
            print(f"    {ot:>20} → {r['detected']:>15} (expected={str(r['expected']):>15}) {mark} "
                  f"cos={r['cosine']:+.3f} capture={r['capture']:.3f}")
        print(f"    Accuracy: {n_correct}/{n_testable}")

        results[f"ocean_{n_dims}d"] = ocean_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Combined dimensionality: {effective_dim}")
    print(f"  Steering-only dim:       {steer_edim}")
    print(f"  SysPrompt-only dim:      {sysp_edim}")
    print(f"\n  Original 5D basis: steering={results['basis_5d']['steering_detection']}/6, "
          f"sysprompt={results['basis_5d']['sysprompt_detection_same']}/6")
    for n_dims in basis_sizes[1:]:
        key = f"basis_{n_dims}d"
        print(f"  Expanded {n_dims}D basis: steering={results[key]['steering_detection']}/6, "
              f"sysprompt={results[key]['sysprompt_detection_same']}/6")

    results["summary"] = {
        "combined_effective_dim": effective_dim,
        "steering_effective_dim": steer_edim,
        "sysprompt_effective_dim": sysp_edim,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expanded_personality_basis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
