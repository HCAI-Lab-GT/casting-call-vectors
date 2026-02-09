#!/usr/bin/env python
"""
Zero-calibration transfer: Can we transfer personality vectors WITHOUT any
calibration data from the target model?

Current pipeline requires k=5 calibration vectors (5 trait correspondences)
to fit the Procrustes rotation. But what if the 5D personality simplex has
a CANONICAL orientation that's predictable without calibration?

Strategy:
1. Assume canonical simplex: 6 points in 5D (vertices of a regular simplex
   projected to 5D, labeled by Holland's RIASEC order)
2. Fit Procrustes from source's 5D coords to canonical coords
3. Fit Procrustes from canonical coords to target's 5D coords
   (NOTE: this second step still needs target — but we test whether
    a SINGLE canonical bridge works across ALL model pairs)
4. Alternative: use ONLY source coords + assumption about simplex
   geometry to predict target coords directly

If this works, it means: extract vectors on ANY model, apply a
canonical rotation, and steer ANY other model — no calibration needed.

Test multiple "zero-calibration" strategies:
A. Canonical simplex bridge (source → canonical → target)
B. Identity assumption (assume source PCA = target PCA, no rotation)
C. Sign-corrected identity (fix PC sign flips only)
D. Minimal calibration baseline (k=1,2,3 for comparison)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="zero-calibration")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Holland hexagonal order
HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]


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


def fit_procrustes(source_5d, target_5d):
    S = np.stack([source_5d[t] for t in TRAITS])
    T = np.stack([target_5d[t] for t in TRAITS])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))
    return R, scale


def build_canonical_simplex_5d():
    """Build a canonical regular simplex in 5D with 6 vertices.

    Place 6 points as a centered, unit-norm regular simplex in 5D,
    labeled in Holland RIASEC order. This gives a "universal" coordinate
    system that any model's 5D personality coords can be aligned to.
    """
    # Start with a regular simplex: 6 points in 5D
    # Use the standard construction: start with a regular simplex in R^6
    # and project to the 5D hyperplane orthogonal to the all-ones vector.
    n = 6  # number of points
    d = 5  # dimension

    # Standard simplex: identity rows minus centroid
    raw = np.eye(n) - np.ones((n, n)) / n

    # SVD to get the 5D representation
    U, S, Vt = np.linalg.svd(raw, full_matrices=False)
    coords = U[:, :d] * S[:d]  # (6, 5) matrix

    # Normalize to unit norm
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / norms

    # Scale to match typical 5D coordinate magnitudes
    coords = coords * 30.0  # approximate scale of real 5D coords

    return {trait: coords[i] for i, trait in enumerate(HOLLAND_ORDER)}


def transfer_via_canonical(source_5d, target_5d, source_basis, target_basis):
    """Transfer via canonical simplex bridge: source → canonical → target."""
    canonical = build_canonical_simplex_5d()

    # Source → Canonical
    R_sc, s_sc = fit_procrustes(source_5d, canonical)
    # Canonical → Target
    R_ct, s_ct = fit_procrustes(canonical, target_5d)

    transferred = {}
    for t in TRAITS:
        # Source 5D → Canonical 5D → Target 5D → Target full-dim
        canonical_coord = s_sc * (R_sc @ source_5d[t])
        target_coord = s_ct * (R_ct @ canonical_coord)
        transferred[t] = (target_basis.T @ target_coord).astype(np.float32)

    return transferred


def transfer_identity(source_5d, source_basis, target_basis):
    """Assume source PCA = target PCA (no rotation, just rescale)."""
    # Just project source 5D coords through target basis
    target_scale = np.mean([np.linalg.norm(target_basis.T @ np.ones(5))])
    source_scale = np.mean([np.linalg.norm(source_basis.T @ np.ones(5))])
    scale = target_scale / source_scale if source_scale > 0 else 1.0

    transferred = {}
    for t in TRAITS:
        # Assume source 5D coords ARE target 5D coords (up to scale)
        transferred[t] = (target_basis.T @ (scale * source_5d[t])).astype(np.float32)
    return transferred


def transfer_sign_corrected(source_5d, target_5d, source_basis, target_basis):
    """Fix PC sign flips (most common PCA ambiguity) but no rotation."""
    # For each PC, check if flipping the sign improves alignment
    S = np.stack([source_5d[t] for t in TRAITS])
    T = np.stack([target_5d[t] for t in TRAITS])

    signs = np.ones(5)
    for pc in range(5):
        # Check if this PC needs flipping
        corr = np.corrcoef(S[:, pc], T[:, pc])[0, 1]
        if corr < 0:
            signs[pc] = -1

    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))

    transferred = {}
    for t in TRAITS:
        corrected_coord = scale * (signs * source_5d[t])
        transferred[t] = (target_basis.T @ corrected_coord).astype(np.float32)
    return transferred


def transfer_standard(source_5d, target_5d, source_basis, target_basis):
    """Standard full Procrustes (k=6, using ALL traits)."""
    R, scale = fit_procrustes(source_5d, target_5d)
    transferred = {}
    for t in TRAITS:
        target_coord = scale * (R @ source_5d[t])
        transferred[t] = (target_basis.T @ target_coord).astype(np.float32)
    return transferred


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


def eval_accuracy(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    correct = 0
    total = 0
    total_delta = 0.0

    for steer_trait in TRAITS:
        vec = vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
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
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    if steer_trait not in (trait_a, trait_b):
                        continue
                    gap = pairwise_logprob_chat(model, tokenizer, device,
                                               TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                    base_gap = baseline[f"{trait_a}-{trait_b}"]
                    if steer_trait == trait_a:
                        d = gap - base_gap
                    else:
                        d = base_gap - gap
                    correct += int(d > 0)
                    total_delta += d
                    total += 1
        finally:
            hook_handle.remove()

    return correct / total if total else 0, total_delta / total if total else 0


def main():
    target_id = "marin-community/marin-8b-instruct"
    source_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 1.0

    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load vectors
    logger.info("Loading source (SmolLM3) vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    source_5d, source_basis, source_sv = get_5d_coords_and_basis(source_residual)

    logger.info("Loading target (Marin 8B) vectors...")
    target_residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    target_5d, target_basis, target_sv = get_5d_coords_and_basis(target_residual)

    # Build transfer strategies
    logger.info("Building transfer strategies...")
    strategies = {}

    # A. Self (positive control)
    strategies["Self (native)"] = target_residual

    # B. Full Procrustes (k=6 calibration, existing best)
    strategies["Full Procrustes (k=6)"] = transfer_standard(
        source_5d, target_5d, source_basis, target_basis)

    # C. Canonical simplex bridge
    strategies["Canonical simplex bridge"] = transfer_via_canonical(
        source_5d, target_5d, source_basis, target_basis)

    # D. Identity (no rotation)
    strategies["Identity (no rotation)"] = transfer_identity(
        source_5d, source_basis, target_basis)

    # E. Sign-corrected identity
    # NOTE: This technically uses target_5d for sign detection, so it's
    # not truly zero-calibration. But it only uses 5 bits of information.
    strategies["Sign-corrected (5 bits)"] = transfer_sign_corrected(
        source_5d, target_5d, source_basis, target_basis)

    # F. Random baseline
    rng = np.random.RandomState(42)
    random_vecs = {}
    target_norms = [np.linalg.norm(target_residual[t]) for t in TRAITS]
    mean_norm = np.mean(target_norms)
    for t in TRAITS:
        rv = rng.randn(target_residual[TRAITS[0]].shape[0]).astype(np.float32)
        random_vecs[t] = rv / np.linalg.norm(rv) * mean_norm
    strategies["Random"] = random_vecs

    # Also test: Canonical bridge from Llama 1B
    logger.info("Loading Llama 1B vectors for second source test...")
    llama_id = "meta-llama/Llama-3.2-1B-Instruct"
    llama_residual, _ = load_residual_vectors(llama_id, riasec_dir)
    llama_5d, llama_basis, _ = get_5d_coords_and_basis(llama_residual)

    strategies["Canonical bridge (Llama→Marin)"] = transfer_via_canonical(
        llama_5d, target_5d, llama_basis, target_basis)
    strategies["Full Procrustes (Llama→Marin)"] = transfer_standard(
        llama_5d, target_5d, llama_basis, target_basis)

    # Load target model
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
    print(f"ZERO-CALIBRATION TRANSFER STRATEGIES")
    print(f"Source: {source_id} → Target: {target_id}")
    print(f"{'='*70}")

    # Evaluate each strategy
    results = {}
    for strat_name, vectors in strategies.items():
        logger.info(f"Evaluating: {strat_name}")
        accuracy, mean_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline)
        results[strat_name] = {
            "accuracy": float(accuracy),
            "mean_delta": float(mean_delta),
        }
        print(f"  {strat_name:>35}: {accuracy:>5.0%}  mean_delta={mean_delta:>+.3f}")

    # Analyze canonical bridge quality
    print(f"\n--- Canonical simplex analysis ---")
    canonical = build_canonical_simplex_5d()
    print(f"  Canonical simplex coordinates:")
    for t in HOLLAND_ORDER:
        c = canonical[t]
        print(f"    {t:>14}: [{', '.join(f'{x:+.2f}' for x in c)}]")

    # How well does source align to canonical?
    R_sc, s_sc = fit_procrustes(source_5d, canonical)
    source_to_canonical = {}
    for t in TRAITS:
        source_to_canonical[t] = s_sc * (R_sc @ source_5d[t])
    cos_sc = []
    for t in TRAITS:
        c = np.dot(source_to_canonical[t], canonical[t]) / (
            np.linalg.norm(source_to_canonical[t]) * np.linalg.norm(canonical[t]))
        cos_sc.append(c)
    print(f"\n  Source→Canonical alignment: mean cosine = {np.mean(cos_sc):.3f}")

    # How well does target align to canonical?
    R_ct_inv, s_ct_inv = fit_procrustes(target_5d, canonical)
    target_to_canonical = {}
    for t in TRAITS:
        target_to_canonical[t] = s_ct_inv * (R_ct_inv @ target_5d[t])
    cos_tc = []
    for t in TRAITS:
        c = np.dot(target_to_canonical[t], canonical[t]) / (
            np.linalg.norm(target_to_canonical[t]) * np.linalg.norm(canonical[t]))
        cos_tc.append(c)
    print(f"  Target→Canonical alignment: mean cosine = {np.mean(cos_tc):.3f}")

    # The real question: does canonical bridge introduce more error than direct Procrustes?
    # Direct: source → target (1 rotation)
    # Canonical: source → canonical → target (2 rotations)
    direct_cos = []
    canonical_cos = []
    direct_vecs = strategies["Full Procrustes (k=6)"]
    canonical_vecs = strategies["Canonical simplex bridge"]
    for t in TRAITS:
        d_cos = np.dot(direct_vecs[t], target_residual[t]) / (
            np.linalg.norm(direct_vecs[t]) * np.linalg.norm(target_residual[t]))
        c_cos = np.dot(canonical_vecs[t], target_residual[t]) / (
            np.linalg.norm(canonical_vecs[t]) * np.linalg.norm(target_residual[t]))
        direct_cos.append(d_cos)
        canonical_cos.append(c_cos)

    print(f"\n  Vector cosine to native (direct vs canonical):")
    for i, t in enumerate(TRAITS):
        print(f"    {t:>14}: direct={direct_cos[i]:.3f}, canonical={canonical_cos[i]:.3f}, diff={canonical_cos[i]-direct_cos[i]:+.3f}")
    print(f"    {'MEAN':>14}: direct={np.mean(direct_cos):.3f}, canonical={np.mean(canonical_cos):.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zero_calibration_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
