#!/usr/bin/env python
"""
Isotropic extrapolation: fix non-linear scaling by whitening the 5D space.

Problem: social fails at 2× amplification because PC1 (33% variance) dominates
over PC3 (20% variance) where social's unique signature lives.

Solution: instead of amplifying in the original 5D space, amplify in a
WHITENED space where all PCs have equal variance. This should make
extrapolation equally linear for ALL traits.

Comparison:
1. Original extrapolation: k * coords_5d[trait] (what we tested before)
2. Isotropic extrapolation: coords / singular_values, then amplify, then de-whiten

If isotropic extrapolation fixes social's non-linearity, it proves the
variance structure is the causal explanation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="isotropic")

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


def get_5d_coords_and_basis(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d, S[:5]


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

    amplifications = [0.5, 1.0, 1.5, 2.0, 3.0]

    # Focus on social (the problematic trait) plus artistic (control)
    test_traits = ["artistic", "social"]

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    coords_5d, basis_5d, singular_values = get_5d_coords_and_basis(residual)

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
    print(f"ISOTROPIC VS ORIGINAL EXTRAPOLATION")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"Singular values: [{', '.join(f'{s:.1f}' for s in singular_values)}]")
    print(f"{'='*70}")

    results = {}

    for trait in test_traits:
        print(f"\n{'='*70}")
        print(f"TRAIT: {trait.upper()}")
        print(f"{'='*70}")

        orig_coords = coords_5d[trait]
        orig_norm = np.linalg.norm(orig_coords)

        # Whitened coordinates: divide by singular values
        whitened_coords = orig_coords / singular_values
        whitened_norm = np.linalg.norm(whitened_coords)

        print(f"  Original coords: [{', '.join(f'{c:+.2f}' for c in orig_coords)}]")
        print(f"  Whitened coords:  [{', '.join(f'{c:+.3f}' for c in whitened_coords)}]")
        print(f"  Original norm: {orig_norm:.2f}, Whitened norm: {whitened_norm:.4f}")

        # Show which PC dominates after whitening
        whitened_abs = np.abs(whitened_coords)
        dom_pc = np.argmax(whitened_abs)
        print(f"  Dominant PC (original): PC{np.argmax(np.abs(orig_coords))+1}")
        print(f"  Dominant PC (whitened): PC{dom_pc+1}")

        trait_results = {"original": {}, "isotropic": {}}

        for amp in amplifications:
            logger.info(f"Testing {trait} at {amp}× ({2} methods)...")

            # Method 1: Original (direct coordinate amplification)
            amp_coords_orig = amp * orig_coords
            amp_vec_orig = reconstruct_from_5d(amp_coords_orig, basis_5d)

            profile_orig = measure_trait_profile(
                model, tokenizer, device, blocks, mid_layer, amp_vec_orig, alpha, baseline)

            # Method 2: Isotropic (amplify in whitened space, then de-whiten)
            amp_whitened = amp * whitened_coords
            # De-whiten: multiply back by singular values
            amp_coords_iso = amp_whitened * singular_values
            amp_vec_iso = reconstruct_from_5d(amp_coords_iso, basis_5d)

            profile_iso = measure_trait_profile(
                model, tokenizer, device, blocks, mid_layer, amp_vec_iso, alpha, baseline)

            sorted_orig = sorted(profile_orig.items(), key=lambda x: -x[1])
            sorted_iso = sorted(profile_iso.items(), key=lambda x: -x[1])

            print(f"\n  {amp}× amplification:")
            print(f"    ORIGINAL: target={profile_orig[trait]:+.3f}, top={sorted_orig[0][0]}"
                  f" {'OK' if sorted_orig[0][0] == trait else 'WRONG'}")
            print(f"    ISOTROPIC: target={profile_iso[trait]:+.3f}, top={sorted_iso[0][0]}"
                  f" {'OK' if sorted_iso[0][0] == trait else 'WRONG'}")
            print(f"    Orig profile: " + "  ".join(f"{t[:3]}={d:+.2f}" for t, d in sorted_orig))
            print(f"    Iso  profile: " + "  ".join(f"{t[:3]}={d:+.2f}" for t, d in sorted_iso))

            trait_results["original"][str(amp)] = {
                "target_delta": float(profile_orig[trait]),
                "top_trait": sorted_orig[0][0],
                "correct": sorted_orig[0][0] == trait,
                "profile": {t: float(d) for t, d in profile_orig.items()},
                "vec_norm": float(np.linalg.norm(amp_vec_orig)),
            }
            trait_results["isotropic"][str(amp)] = {
                "target_delta": float(profile_iso[trait]),
                "top_trait": sorted_iso[0][0],
                "correct": sorted_iso[0][0] == trait,
                "profile": {t: float(d) for t, d in profile_iso.items()},
                "vec_norm": float(np.linalg.norm(amp_vec_iso)),
            }

        results[trait] = trait_results

    # Linearity comparison
    print(f"\n{'='*70}")
    print(f"LINEARITY COMPARISON")
    print(f"{'='*70}")

    from scipy.stats import pearsonr

    for trait in test_traits:
        for method in ["original", "isotropic"]:
            deltas = [results[trait][method][str(a)]["target_delta"] for a in amplifications]
            r, p = pearsonr(amplifications, deltas)
            correct = sum(1 for a in amplifications if results[trait][method][str(a)]["correct"])
            print(f"  {trait:>12} {method:>10}: r={r:.3f} (p={p:.3f}), correct top={correct}/{len(amplifications)}")

            results[trait][f"{method}_linearity"] = {"r": float(r), "p": float(p), "correct_top": correct}

    # Critical test: does isotropic fix social?
    print(f"\n{'='*70}")
    print(f"CRITICAL TEST: Does isotropic extrapolation fix social?")
    print(f"{'='*70}")

    social_orig_r = results["social"]["original_linearity"]["r"]
    social_iso_r = results["social"]["isotropic_linearity"]["r"]
    social_orig_top = results["social"]["original_linearity"]["correct_top"]
    social_iso_top = results["social"]["isotropic_linearity"]["correct_top"]

    print(f"  Social ORIGINAL: r={social_orig_r:.3f}, correct top = {social_orig_top}/5")
    print(f"  Social ISOTROPIC: r={social_iso_r:.3f}, correct top = {social_iso_top}/5")

    if social_iso_r > social_orig_r and social_iso_top > social_orig_top:
        print(f"  CONCLUSION: YES — isotropic extrapolation improves social linearity")
        print(f"  This confirms that non-uniform variance is the cause of social's failure")
    elif social_iso_top > social_orig_top:
        print(f"  CONCLUSION: PARTIAL — maintains correct top trait but linearity similar")
    else:
        print(f"  CONCLUSION: NO — isotropic extrapolation does NOT fix the issue")
        print(f"  The non-linearity has a different cause (e.g., saturation, model-specific)")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "isotropic_extrapolation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
