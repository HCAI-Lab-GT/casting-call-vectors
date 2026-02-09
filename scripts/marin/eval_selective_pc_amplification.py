#!/usr/bin/env python
"""
Selective PC amplification: can we fix social's non-linearity?

The unified theory predicts that social's extrapolation failure comes from
PC1 contamination: social has a PC1 loading of -19.99 (41% fraction) which
aligns with artistic (-44.03). At high amplification, the PC1 behavioral
effect dominates social's PC3 identity.

HYPOTHESIS: If we amplify ONLY social's dominant PC3 component (zeroing PC1),
we should recover linear extrapolation.

EXPERIMENT:
1. Single-PC steering: activate each PC independently, measure behavioral profile
2. Selective amplification for social: PC3-only at 1×, 2×, 3×, 5×
3. Compare linearity: full-coord vs PC3-only for social
4. Same for artistic: full-coord vs PC1-only (should both be linear)
5. Cross-experiment: can we create NOVEL personalities by mixing PCs?

This tests whether the 5D personality space is a genuine 5-knob control panel.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="selective-pc")

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
    singular_values = S_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return residual, coords_5d, basis_5d, singular_values, mid_layer


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


def measure_trait_profile(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline):
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

    logger.info("Loading vectors and computing 5D basis...")
    residual, coords_5d, basis_5d, singular_values, mid_layer = load_residual_and_basis(
        target_id, riasec_dir)

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
    print(f"SELECTIVE PC AMPLIFICATION")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    variance_pct = (singular_values**2) / (singular_values**2).sum() * 100
    print(f"\n  Variance distribution: [{', '.join(f'{v:.1f}%' for v in variance_pct)}]")
    print(f"  Singular values: [{', '.join(f'{s:.1f}' for s in singular_values)}]")

    results = {}

    # ================================================================
    # PART 1: Single-PC steering
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 1: SINGLE-PC STEERING")
    print(f"  Activate each PC independently to map its behavioral meaning")
    print(f"{'='*70}")

    single_pc_results = {}

    for pc_idx in range(5):
        logger.info(f"Testing PC{pc_idx+1} only...")

        # Create a coordinate vector that is 1.0 on this PC and 0 on all others
        # Scale by average trait norm so the behavioral effect is comparable
        avg_norm = np.mean([np.linalg.norm(coords_5d[t]) for t in TRAITS])
        pc_coords = np.zeros(5)
        pc_coords[pc_idx] = -avg_norm  # Negative (following canonical sign convention)
        pc_vec = reconstruct_from_5d(pc_coords, basis_5d)

        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, pc_vec, alpha, baseline)

        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top_trait = sorted_prof[0][0]
        bot_trait = sorted_prof[-1][0]

        print(f"\n  PC{pc_idx+1} ({variance_pct[pc_idx]:.1f}% var):")
        print(f"    Behavioral profile: {' '.join(f'{t[:4]}={d:+.3f}' for t, d in sorted_prof)}")
        print(f"    Top: {top_trait}, Bottom: {bot_trait}")

        # Also test positive direction
        pc_coords_pos = np.zeros(5)
        pc_coords_pos[pc_idx] = +avg_norm
        pc_vec_pos = reconstruct_from_5d(pc_coords_pos, basis_5d)
        profile_pos = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, pc_vec_pos, alpha, baseline)
        sorted_pos = sorted(profile_pos.items(), key=lambda x: -x[1])

        print(f"    +PC{pc_idx+1} profile: {' '.join(f'{t[:4]}={d:+.3f}' for t, d in sorted_pos)}")
        print(f"    +PC{pc_idx+1} top: {sorted_pos[0][0]}, bottom: {sorted_pos[-1][0]}")

        # Check antisymmetry (positive should be mirror of negative)
        antisym_r, _ = pearsonr(
            [profile[t] for t in TRAITS],
            [profile_pos[t] for t in TRAITS]
        )
        print(f"    Antisymmetry: r = {antisym_r:.3f} (expect ≈ -1.0)")

        single_pc_results[f"PC{pc_idx+1}"] = {
            "variance_pct": float(variance_pct[pc_idx]),
            "negative_profile": {t: float(profile[t]) for t in TRAITS},
            "positive_profile": {t: float(profile_pos[t]) for t in TRAITS},
            "negative_top": top_trait,
            "negative_bottom": bot_trait,
            "positive_top": sorted_pos[0][0],
            "positive_bottom": sorted_pos[-1][0],
            "antisymmetry_r": float(antisym_r),
        }

    results["single_pc"] = single_pc_results

    # ================================================================
    # PART 2: Selective amplification for SOCIAL (the problem trait)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 2: SELECTIVE AMPLIFICATION — SOCIAL")
    print(f"  Social coords: [{', '.join(f'{c:+.1f}' for c in coords_5d['social'])}]")
    print(f"  PC1 fraction: {abs(coords_5d['social'][0])/np.linalg.norm(coords_5d['social']):.1%}")
    print(f"  PC3 fraction: {abs(coords_5d['social'][2])/np.linalg.norm(coords_5d['social']):.1%}")
    print(f"{'='*70}")

    amplifications = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Method A: Full coordinate amplification (baseline — known to fail)
    social_full_results = {}
    for amp in amplifications:
        logger.info(f"Social full-coord at {amp}×...")
        amp_coords = amp * coords_5d["social"]
        amp_vec = reconstruct_from_5d(amp_coords, basis_5d)
        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, amp_vec, alpha, baseline)
        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top = sorted_prof[0][0]
        print(f"\n  Full {amp}×: social={profile['social']:+.3f}, top={top}, "
              f"profile: {' '.join(f'{t[:4]}={d:+.2f}' for t, d in sorted_prof)}")
        social_full_results[str(amp)] = {
            "target_delta": float(profile["social"]),
            "top_trait": top,
            "is_target_top": top == "social",
            "profile": {t: float(profile[t]) for t in TRAITS},
        }

    # Method B: PC3-only amplification (zeroing PC1 contamination)
    social_pc3_results = {}
    for amp in amplifications:
        logger.info(f"Social PC3-only at {amp}×...")
        # Take social's coords, zero out PC1, amplify
        selective_coords = coords_5d["social"].copy()
        selective_coords[0] = 0.0  # Zero PC1
        selective_coords = amp * selective_coords
        amp_vec = reconstruct_from_5d(selective_coords, basis_5d)
        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, amp_vec, alpha, baseline)
        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top = sorted_prof[0][0]
        print(f"\n  PC3-only {amp}×: social={profile['social']:+.3f}, top={top}, "
              f"profile: {' '.join(f'{t[:4]}={d:+.2f}' for t, d in sorted_prof)}")
        social_pc3_results[str(amp)] = {
            "target_delta": float(profile["social"]),
            "top_trait": top,
            "is_target_top": top == "social",
            "profile": {t: float(profile[t]) for t in TRAITS},
        }

    # Method C: PC3+PC2 only (social's top 2 PCs, no PC1)
    social_no_pc1_results = {}
    for amp in amplifications:
        logger.info(f"Social no-PC1 at {amp}×...")
        selective_coords = coords_5d["social"].copy()
        selective_coords[0] = 0.0  # Zero only PC1
        selective_coords = amp * selective_coords
        amp_vec = reconstruct_from_5d(selective_coords, basis_5d)
        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, amp_vec, alpha, baseline)
        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top = sorted_prof[0][0]
        social_no_pc1_results[str(amp)] = {
            "target_delta": float(profile["social"]),
            "top_trait": top,
            "is_target_top": top == "social",
            "profile": {t: float(profile[t]) for t in TRAITS},
        }

    results["social_selective"] = {
        "full_coord": social_full_results,
        "pc3_only": social_pc3_results,
        "no_pc1": social_no_pc1_results,
    }

    # ================================================================
    # PART 3: Selective amplification for ARTISTIC (control — should stay linear)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 3: SELECTIVE AMPLIFICATION — ARTISTIC (CONTROL)")
    print(f"  Artistic coords: [{', '.join(f'{c:+.1f}' for c in coords_5d['artistic'])}]")
    print(f"  PC1 fraction: {abs(coords_5d['artistic'][0])/np.linalg.norm(coords_5d['artistic']):.1%}")
    print(f"{'='*70}")

    art_full_results = {}
    for amp in amplifications:
        logger.info(f"Artistic full-coord at {amp}×...")
        amp_coords = amp * coords_5d["artistic"]
        amp_vec = reconstruct_from_5d(amp_coords, basis_5d)
        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, amp_vec, alpha, baseline)
        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top = sorted_prof[0][0]
        art_full_results[str(amp)] = {
            "target_delta": float(profile["artistic"]),
            "top_trait": top,
            "is_target_top": top == "artistic",
            "profile": {t: float(profile[t]) for t in TRAITS},
        }

    art_pc1_results = {}
    for amp in amplifications:
        logger.info(f"Artistic PC1-only at {amp}×...")
        selective_coords = np.zeros(5)
        selective_coords[0] = amp * coords_5d["artistic"][0]  # Only PC1
        amp_vec = reconstruct_from_5d(selective_coords, basis_5d)
        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, amp_vec, alpha, baseline)
        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top = sorted_prof[0][0]
        art_pc1_results[str(amp)] = {
            "target_delta": float(profile["artistic"]),
            "top_trait": top,
            "is_target_top": top == "artistic",
            "profile": {t: float(profile[t]) for t in TRAITS},
        }

    results["artistic_selective"] = {
        "full_coord": art_full_results,
        "pc1_only": art_pc1_results,
    }

    # ================================================================
    # PART 4: Novel PC combinations — custom personalities
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 4: NOVEL PC COMBINATIONS")
    print(f"{'='*70}")

    avg_norm = np.mean([np.linalg.norm(coords_5d[t]) for t in TRAITS])

    # Create some novel personalities by combining PCs in non-standard ways
    novel_combos = {
        "pure_PC1": np.array([-avg_norm, 0, 0, 0, 0]),
        "pure_PC3": np.array([0, 0, -avg_norm, 0, 0]),
        "PC1+PC3": np.array([-avg_norm/np.sqrt(2), 0, -avg_norm/np.sqrt(2), 0, 0]),
        "PC1-PC3": np.array([-avg_norm/np.sqrt(2), 0, +avg_norm/np.sqrt(2), 0, 0]),
        "anti_artistic": -coords_5d["artistic"] / np.linalg.norm(coords_5d["artistic"]) * avg_norm,
        "social_purified": None,  # will compute below
    }

    # Social purified: social coords with PC1 zeroed and renormalized to original norm
    social_pure = coords_5d["social"].copy()
    social_pure[0] = 0.0
    social_pure = social_pure / np.linalg.norm(social_pure) * np.linalg.norm(coords_5d["social"])
    novel_combos["social_purified"] = social_pure

    novel_results = {}
    for name, coords in novel_combos.items():
        logger.info(f"Testing novel combo: {name}...")
        vec = reconstruct_from_5d(coords, basis_5d)
        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline)
        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top = sorted_prof[0][0]
        bot = sorted_prof[-1][0]
        print(f"\n  {name:>20}: top={top}, bot={bot}")
        print(f"    Profile: {' '.join(f'{t[:4]}={d:+.3f}' for t, d in sorted_prof)}")
        novel_results[name] = {
            "coords": coords.tolist(),
            "profile": {t: float(profile[t]) for t in TRAITS},
            "top_trait": top,
            "bottom_trait": bot,
        }

    results["novel_combinations"] = novel_results

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")

    # Linearity comparison: social full vs PC3-only
    for label, data in [("social_full", social_full_results),
                        ("social_pc3_only", social_pc3_results),
                        ("social_no_pc1", social_no_pc1_results)]:
        amps = [float(a) for a in data.keys()]
        deltas = [data[str(a)]["target_delta"] for a in amps]
        if len(amps) >= 3:
            r, p = pearsonr(amps, deltas)
            maintains_top = sum(1 for a in amps if data[str(a)]["is_target_top"])
            print(f"\n  {label:>20}: linearity r={r:.3f} (p={p:.4f}), "
                  f"top={maintains_top}/{len(amps)}")

    for label, data in [("artistic_full", art_full_results),
                        ("artistic_pc1_only", art_pc1_results)]:
        amps = [float(a) for a in data.keys()]
        deltas = [data[str(a)]["target_delta"] for a in amps]
        if len(amps) >= 3:
            r, p = pearsonr(amps, deltas)
            maintains_top = sum(1 for a in amps if data[str(a)]["is_target_top"])
            print(f"\n  {label:>20}: linearity r={r:.3f} (p={p:.4f}), "
                  f"top={maintains_top}/{len(amps)}")

    # PC semantic mapping
    print(f"\n  --- PC SEMANTIC MAP ---")
    for pc_idx in range(5):
        pcr = single_pc_results[f"PC{pc_idx+1}"]
        print(f"  PC{pc_idx+1} ({variance_pct[pc_idx]:.1f}%): "
              f"-dir→{pcr['negative_top']}, +dir→{pcr['positive_top']}, "
              f"antisym={pcr['antisymmetry_r']:.3f}")

    # Key question: did zeroing PC1 fix social's extrapolation?
    print(f"\n  --- KEY RESULT: SOCIAL NON-LINEARITY FIX ---")
    full_deltas = [social_full_results[str(a)]["target_delta"] for a in amplifications]
    pc3_deltas = [social_pc3_results[str(a)]["target_delta"] for a in amplifications]
    nop1_deltas = [social_no_pc1_results[str(a)]["target_delta"] for a in amplifications]

    full_r, _ = pearsonr(amplifications, full_deltas)
    pc3_r, _ = pearsonr(amplifications, pc3_deltas)
    nop1_r, _ = pearsonr(amplifications, nop1_deltas)

    full_top = sum(1 for a in amplifications if social_full_results[str(a)]["is_target_top"])
    pc3_top = sum(1 for a in amplifications if social_pc3_results[str(a)]["is_target_top"])
    nop1_top = sum(1 for a in amplifications if social_no_pc1_results[str(a)]["is_target_top"])

    print(f"  Full-coord: linearity={full_r:.3f}, top={full_top}/{len(amplifications)}")
    print(f"  PC3-only:   linearity={pc3_r:.3f}, top={pc3_top}/{len(amplifications)}")
    print(f"  No-PC1:     linearity={nop1_r:.3f}, top={nop1_top}/{len(amplifications)}")

    if pc3_r > full_r + 0.1 or nop1_r > full_r + 0.1:
        print(f"  CONCLUSION: Removing PC1 IMPROVES social's extrapolation!")
        print(f"  PC1 contamination is the cause of non-linearity (as predicted)")
    elif abs(pc3_r - full_r) < 0.1:
        print(f"  CONCLUSION: PC1 removal has NO effect on social's extrapolation")
        print(f"  Non-linearity is deeper than PC1 contamination")
    else:
        print(f"  CONCLUSION: PC1 removal WORSENS social's extrapolation")
        print(f"  PC1 component was actually helping")

    results["analysis"] = {
        "social_full_linearity": float(full_r),
        "social_pc3_linearity": float(pc3_r),
        "social_nop1_linearity": float(nop1_r),
        "social_full_top": full_top,
        "social_pc3_top": pc3_top,
        "social_nop1_top": nop1_top,
        "artistic_full_linearity": float(pearsonr(
            amplifications,
            [art_full_results[str(a)]["target_delta"] for a in amplifications])[0]),
        "artistic_pc1_linearity": float(pearsonr(
            amplifications,
            [art_pc1_results[str(a)]["target_delta"] for a in amplifications])[0]),
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selective_pc_amplification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
