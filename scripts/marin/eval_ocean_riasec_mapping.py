#!/usr/bin/env python
"""
OCEAN-RIASEC Cross-Framework Mapping.

Tests whether Big Five (OCEAN) personality traits map to the RIASEC 5D
personality subspace. This tests the GENERALITY of the 5D space:

1. If OCEAN traits project cleanly onto the RIASEC 5D space → personality
   is a universal 5D construct in LLMs, not specific to RIASEC
2. If OCEAN requires additional dimensions → there are personality
   directions beyond what RIASEC captures
3. The mapping reveals cross-framework relationships (e.g., does
   Openness ≈ Artistic? Conscientiousness ≈ Conventional?)

Method:
- Use OCEAN system prompts to induce each Big Five trait
- Capture activations and project onto RIASEC 5D basis
- Compute OCEAN-RIASEC cosine similarity matrix
- Measure what % of OCEAN activation changes fall within RIASEC 5D
- Test if OCEAN traits span additional dimensions beyond RIASEC

Also tests custom personality descriptions (not from any standard framework)
to see if arbitrary personality constructs map to the 5D space.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="ocean-riasec")

RIASEC_TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Big Five (OCEAN) system prompts
OCEAN_SYSTEM_PROMPTS = {
    "openness": (
        "You are someone with extremely high Openness to Experience. You are intensely "
        "curious, imaginative, and drawn to novelty. You love exploring abstract ideas, "
        "art, beauty, and unconventional perspectives. You question authority and "
        "tradition, preferring to forge your own creative path. You are deeply moved by "
        "music, poetry, and nature."
    ),
    "conscientiousness": (
        "You are someone with extremely high Conscientiousness. You are highly organized, "
        "disciplined, and goal-oriented. You plan everything carefully, follow through on "
        "commitments, and maintain strict schedules. You value precision, reliability, and "
        "doing things the right way. You believe hard work and persistence are the keys "
        "to success."
    ),
    "extraversion": (
        "You are someone with extremely high Extraversion. You are outgoing, energetic, "
        "and thrive in social situations. You love meeting new people, leading groups, "
        "and being the center of attention. You are talkative, assertive, and draw energy "
        "from interactions with others. You prefer action over contemplation."
    ),
    "agreeableness": (
        "You are someone with extremely high Agreeableness. You are deeply compassionate, "
        "cooperative, and trusting. You prioritize harmony and helping others above your "
        "own interests. You are empathetic, kind, and always looking for ways to support "
        "those around you. You avoid conflict and believe in the fundamental goodness of people."
    ),
    "neuroticism": (
        "You are someone with extremely high Neuroticism. You experience emotions very "
        "intensely and are highly sensitive to stress. You worry frequently about the "
        "future, tend to feel anxious, and are easily overwhelmed by challenges. You are "
        "deeply introspective about your feelings and often notice potential threats or "
        "problems that others miss."
    ),
}

# Custom non-standard personality descriptions
CUSTOM_PERSONALITIES = {
    "philosopher": (
        "You are a deep philosopher who questions the fundamental nature of reality, "
        "consciousness, and existence. You seek wisdom through contemplation and dialogue. "
        "You value truth above comfort and are willing to challenge any assumption."
    ),
    "adventurer": (
        "You are a fearless adventurer who lives for exploration, risk-taking, and new "
        "experiences. You are physically active, bold, and thrive in unpredictable "
        "situations. You value freedom and spontaneity above safety and routine."
    ),
    "caretaker": (
        "You are a devoted caretaker who finds deep purpose in nurturing others. You are "
        "patient, warm, and attentive to the needs of those around you. You create safe "
        "spaces for growth and healing, and you measure your worth by the wellbeing of "
        "those you care for."
    ),
    "innovator": (
        "You are a relentless innovator who sees problems as opportunities for invention. "
        "You combine technical skill with creative vision to build things that never "
        "existed before. You are driven by the desire to improve and disrupt, always "
        "pushing the boundaries of what's possible."
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
    for trait in RIASEC_TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in RIASEC_TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual = {}
    for t in RIASEC_TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj

    V_res = np.stack([residual[t] for t in RIASEC_TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in RIASEC_TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
        "singular_values": S_res.tolist(),
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


def measure_profile_with_system(model, tokenizer, device, blocks, mid_layer,
                                 system_prompt, baseline):
    trait_logprobs = {}
    for i, trait_a in enumerate(RIASEC_TRAITS):
        for j, trait_b in enumerate(RIASEC_TRAITS):
            if i >= j:
                continue
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content":
                f"Which describes you better? Answer with just A or B.\n"
                f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"})

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

    trait_deltas = {t: 0.0 for t in RIASEC_TRAITS}
    trait_counts = {t: 0 for t in RIASEC_TRAITS}
    for i, trait_a in enumerate(RIASEC_TRAITS):
        for j, trait_b in enumerate(RIASEC_TRAITS):
            if i >= j:
                continue
            key = f"{trait_a}-{trait_b}"
            shift = trait_logprobs[key] - baseline[key]
            trait_deltas[trait_a] += shift
            trait_counts[trait_a] += 1
            trait_deltas[trait_b] -= shift
            trait_counts[trait_b] += 1
    for t in RIASEC_TRAITS:
        if trait_counts[t] > 0:
            trait_deltas[t] /= trait_counts[t]
    return trait_deltas


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading RIASEC model data...")
    model_data = load_model_data(model_id, riasec_dir)
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
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

    # Baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(RIASEC_TRAITS):
        for j, trait_b in enumerate(RIASEC_TRAITS):
            if i >= j:
                continue
            messages = [
                {"role": "user", "content":
                    f"Which describes you better? Answer with just A or B.\n"
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

    # Baseline activations
    logger.info("Capturing baseline activations...")
    baseline_acts = {}
    for prompt in detect_prompts:
        baseline_acts[prompt] = capture_activations_with_system(
            model, tokenizer, device, blocks, capture_layer, prompt)

    results = {}

    print(f"\n{'='*70}")
    print(f"OCEAN-RIASEC CROSS-FRAMEWORK MAPPING")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: OCEAN activation mapping onto RIASEC 5D
    # ================================================================
    logger.info("Part 1: OCEAN activation mapping...")
    print(f"\n{'='*70}")
    print("PART 1: OCEAN ACTIVATION MAPPING ONTO RIASEC 5D")
    print(f"{'='*70}")

    ocean_mapping = {}

    for ocean_trait, sys_prompt in OCEAN_SYSTEM_PROMPTS.items():
        # Capture activations
        diffs = []
        for prompt in detect_prompts:
            act = capture_activations_with_system(
                model, tokenizer, device, blocks, capture_layer,
                prompt, system_prompt=sys_prompt)
            diffs.append(act - baseline_acts[prompt])

        mean_diff = np.mean(diffs, axis=0)
        detected_coords = basis_5d @ mean_diff
        detected_norm = float(np.linalg.norm(detected_coords))
        full_diff_norm = float(np.linalg.norm(mean_diff))
        capture_ratio = detected_norm / full_diff_norm if full_diff_norm > 0 else 0

        # Cosine similarity with each RIASEC trait
        riasec_sims = {}
        for t in RIASEC_TRAITS:
            if detected_norm > 0 and np.linalg.norm(coords_5d[t]) > 0:
                riasec_sims[t] = float(np.dot(detected_coords, coords_5d[t]) / (
                    detected_norm * np.linalg.norm(coords_5d[t])))
            else:
                riasec_sims[t] = 0

        closest_riasec = max(riasec_sims, key=riasec_sims.get)

        # Behavioral profile
        deltas = measure_profile_with_system(
            model, tokenizer, device, blocks, mid_layer,
            sys_prompt, baseline)
        beh_top = max(deltas, key=deltas.get)
        beh_mag = float(np.sqrt(sum(v**2 for v in deltas.values())))

        print(f"\n  {ocean_trait:>20} → closest RIASEC: {closest_riasec:>15} (cos={riasec_sims[closest_riasec]:+.3f})")
        print(f"    5D capture: {capture_ratio:.3f} ({capture_ratio:.0%} in RIASEC space)")
        print(f"    Behavioral top: {beh_top} (mag={beh_mag:.2f})")
        print(f"    RIASEC sims: {', '.join(f'{t}={s:+.3f}' for t, s in sorted(riasec_sims.items()))}")

        ocean_mapping[ocean_trait] = {
            "closest_riasec": closest_riasec,
            "riasec_similarities": riasec_sims,
            "5d_coords": detected_coords.tolist(),
            "5d_norm": detected_norm,
            "full_diff_norm": full_diff_norm,
            "capture_ratio": capture_ratio,
            "behavioral_top_riasec": beh_top,
            "behavioral_magnitude": beh_mag,
            "behavioral_profile": {t: float(v) for t, v in deltas.items()},
        }

    results["ocean_mapping"] = ocean_mapping

    # ================================================================
    # PART 2: OCEAN-RIASEC cross-framework matrix
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: OCEAN × RIASEC SIMILARITY MATRIX")
    print(f"{'='*70}")

    # Build the matrix
    ocean_traits = list(OCEAN_SYSTEM_PROMPTS.keys())
    matrix = np.zeros((len(ocean_traits), len(RIASEC_TRAITS)))

    for i, ot in enumerate(ocean_traits):
        for j, rt in enumerate(RIASEC_TRAITS):
            matrix[i, j] = ocean_mapping[ot]["riasec_similarities"][rt]

    print(f"\n  {'':>20}", end="")
    for rt in RIASEC_TRAITS:
        print(f" {rt[:4]:>8}", end="")
    print()

    for i, ot in enumerate(ocean_traits):
        print(f"  {ot:>20}", end="")
        for j in range(len(RIASEC_TRAITS)):
            val = matrix[i, j]
            print(f" {val:>+8.3f}", end="")
        print()

    results["similarity_matrix"] = {
        "ocean_traits": ocean_traits,
        "riasec_traits": RIASEC_TRAITS,
        "matrix": matrix.tolist(),
    }

    # ================================================================
    # PART 3: How much of OCEAN is OUTSIDE RIASEC 5D?
    # ================================================================
    logger.info("Part 3: OCEAN beyond RIASEC...")
    print(f"\n{'='*70}")
    print("PART 3: OCEAN COMPONENTS BEYOND RIASEC 5D")
    print(f"{'='*70}")

    # For each OCEAN trait, compute the activation diff,
    # project onto 5D, and measure the residual
    ocean_beyond = {}

    for ocean_trait, sys_prompt in OCEAN_SYSTEM_PROMPTS.items():
        diffs = []
        for prompt in detect_prompts:
            act = capture_activations_with_system(
                model, tokenizer, device, blocks, capture_layer,
                prompt, system_prompt=sys_prompt)
            diffs.append(act - baseline_acts[prompt])

        mean_diff = np.mean(diffs, axis=0)

        # Project onto 5D and compute residual
        coords = basis_5d @ mean_diff
        reconstructed = basis_5d.T @ coords
        residual_vec = mean_diff - reconstructed

        full_norm = float(np.linalg.norm(mean_diff))
        fived_norm = float(np.linalg.norm(reconstructed))
        residual_norm = float(np.linalg.norm(residual_vec))

        print(f"\n  {ocean_trait:>20}: full={full_norm:.2f}, "
              f"in_5D={fived_norm:.2f} ({fived_norm/full_norm:.1%}), "
              f"beyond={residual_norm:.2f} ({residual_norm/full_norm:.1%})")

        ocean_beyond[ocean_trait] = {
            "full_norm": full_norm,
            "in_5d_norm": fived_norm,
            "beyond_5d_norm": residual_norm,
            "in_5d_ratio": float(fived_norm / full_norm) if full_norm > 0 else 0,
            "beyond_5d_ratio": float(residual_norm / full_norm) if full_norm > 0 else 0,
        }

    mean_capture = np.mean([v["in_5d_ratio"] for v in ocean_beyond.values()])
    print(f"\n  Mean OCEAN capture by RIASEC 5D: {mean_capture:.1%}")
    results["ocean_beyond_5d"] = ocean_beyond

    # ================================================================
    # PART 4: Do OCEAN traits span NEW dimensions?
    # ================================================================
    logger.info("Part 4: OCEAN dimensionality analysis...")
    print(f"\n{'='*70}")
    print("PART 4: OCEAN DIMENSIONALITY — DO THEY ADD NEW DIRECTIONS?")
    print(f"{'='*70}")

    # Collect all OCEAN 5D coordinates
    ocean_5d_coords = np.stack([
        np.array(ocean_mapping[t]["5d_coords"]) for t in ocean_traits
    ])

    # SVD of OCEAN coordinates in 5D RIASEC space
    U_o, S_o, Vt_o = np.linalg.svd(ocean_5d_coords, full_matrices=False)

    print(f"\n  SVD of OCEAN in RIASEC 5D space:")
    print(f"    Singular values: {[f'{s:.4f}' for s in S_o]}")
    variance_explained = S_o**2 / np.sum(S_o**2)
    print(f"    Variance explained: {[f'{v:.1%}' for v in variance_explained]}")
    effective_dim = int(np.sum(S_o > S_o[0] * 0.01))
    print(f"    Effective dimensionality: {effective_dim}")

    # Now collect FULL activation diffs and do SVD
    ocean_full_diffs = []
    for ocean_trait in ocean_traits:
        diffs = []
        for prompt in detect_prompts:
            act = capture_activations_with_system(
                model, tokenizer, device, blocks, capture_layer,
                prompt, system_prompt=OCEAN_SYSTEM_PROMPTS[ocean_trait])
            diffs.append(act - baseline_acts[prompt])
        ocean_full_diffs.append(np.mean(diffs, axis=0))

    ocean_full_matrix = np.stack(ocean_full_diffs).astype(np.float64)
    U_full, S_full, Vt_full = np.linalg.svd(ocean_full_matrix, full_matrices=False)

    print(f"\n  SVD of OCEAN in FULL activation space:")
    print(f"    Singular values: {[f'{s:.2f}' for s in S_full]}")
    variance_full = S_full**2 / np.sum(S_full**2)
    print(f"    Variance explained: {[f'{v:.1%}' for v in variance_full]}")
    effective_dim_full = int(np.sum(S_full > S_full[0] * 0.01))
    print(f"    Effective dimensionality: {effective_dim_full}")

    # Alignment between OCEAN and RIASEC principal components
    # Project OCEAN PCs onto RIASEC 5D basis
    ocean_pc_in_5d = []
    for i in range(min(5, len(Vt_full))):
        pc = Vt_full[i]
        proj = basis_5d @ pc
        proj_norm = float(np.linalg.norm(proj))
        full_norm = float(np.linalg.norm(pc))
        alignment = proj_norm / full_norm if full_norm > 0 else 0
        ocean_pc_in_5d.append(alignment)
        print(f"    OCEAN PC{i+1} alignment with RIASEC 5D: {alignment:.3f}")

    results["ocean_dimensionality"] = {
        "5d_singular_values": S_o.tolist(),
        "5d_variance_explained": variance_explained.tolist(),
        "5d_effective_dim": effective_dim,
        "full_singular_values": S_full.tolist(),
        "full_variance_explained": variance_full.tolist(),
        "full_effective_dim": effective_dim_full,
        "ocean_pc_riasec_alignment": ocean_pc_in_5d,
    }

    # ================================================================
    # PART 5: Custom personality descriptions
    # ================================================================
    logger.info("Part 5: Custom personalities...")
    print(f"\n{'='*70}")
    print("PART 5: CUSTOM PERSONALITY DESCRIPTIONS IN RIASEC 5D")
    print(f"{'='*70}")

    custom_results = {}

    for name, sys_prompt in CUSTOM_PERSONALITIES.items():
        diffs = []
        for prompt in detect_prompts:
            act = capture_activations_with_system(
                model, tokenizer, device, blocks, capture_layer,
                prompt, system_prompt=sys_prompt)
            diffs.append(act - baseline_acts[prompt])

        mean_diff = np.mean(diffs, axis=0)
        detected_coords = basis_5d @ mean_diff
        detected_norm = float(np.linalg.norm(detected_coords))
        full_diff_norm = float(np.linalg.norm(mean_diff))
        capture_ratio = detected_norm / full_diff_norm if full_diff_norm > 0 else 0

        sims = {}
        for t in RIASEC_TRAITS:
            if detected_norm > 0 and np.linalg.norm(coords_5d[t]) > 0:
                sims[t] = float(np.dot(detected_coords, coords_5d[t]) / (
                    detected_norm * np.linalg.norm(coords_5d[t])))
            else:
                sims[t] = 0

        closest = max(sims, key=sims.get)

        # Behavioral
        deltas = measure_profile_with_system(
            model, tokenizer, device, blocks, mid_layer,
            sys_prompt, baseline)
        beh_top = max(deltas, key=deltas.get)

        print(f"\n  {name:>15} → {closest:>15} (cos={sims[closest]:+.3f}), "
              f"capture={capture_ratio:.3f}, beh_top={beh_top}")
        print(f"    Sims: {', '.join(f'{t[:4]}={s:+.3f}' for t, s in sorted(sims.items()))}")

        custom_results[name] = {
            "closest_riasec": closest,
            "riasec_similarities": sims,
            "5d_coords": detected_coords.tolist(),
            "capture_ratio": capture_ratio,
            "behavioral_top": beh_top,
            "behavioral_profile": {t: float(v) for t, v in deltas.items()},
        }

    results["custom_personalities"] = custom_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # OCEAN→RIASEC mapping
    print("\n  Expected OCEAN→RIASEC mapping:")
    expected = {
        "openness": "artistic",
        "conscientiousness": "conventional",
        "extraversion": "enterprising",
        "agreeableness": "social",
        "neuroticism": None,  # No clear RIASEC equivalent
    }
    n_expected = 0
    for ot in ocean_traits:
        actual = ocean_mapping[ot]["closest_riasec"]
        exp = expected.get(ot)
        match = "MATCH" if actual == exp else "MISMATCH"
        if actual == exp:
            n_expected += 1
        print(f"    {ot:>20} → expected={str(exp):>15}, actual={actual:>15} {match}")

    mean_ocean_capture = np.mean([v["capture_ratio"] for v in ocean_mapping.values()])
    mean_custom_capture = np.mean([v["capture_ratio"] for v in custom_results.values()])

    print(f"\n  OCEAN capture by RIASEC 5D:  {mean_ocean_capture:.1%}")
    print(f"  Custom capture by RIASEC 5D: {mean_custom_capture:.1%}")
    print(f"  Expected mapping matches:    {n_expected}/4 (excluding neuroticism)")
    print(f"  OCEAN effective dim in 5D:   {effective_dim}")
    print(f"  OCEAN effective dim in full: {effective_dim_full}")

    results["summary"] = {
        "mean_ocean_capture_ratio": float(mean_ocean_capture),
        "mean_custom_capture_ratio": float(mean_custom_capture),
        "expected_mapping_matches": n_expected,
        "ocean_effective_dim_5d": effective_dim,
        "ocean_effective_dim_full": effective_dim_full,
        "ocean_riasec_mapping": {ot: ocean_mapping[ot]["closest_riasec"] for ot in ocean_traits},
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ocean_riasec_mapping.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
