#!/usr/bin/env python
"""
Comprehensive analysis of Marin 32B persona vectors, including:
1. Cross-model cosine matrix comparison (1B, 8B, 32B + Qwen 7B)
2. Shared-specific decomposition at 32B scale
3. Residual cross-model correlation (confound rebuttal at scale)
4. Scaling analysis: how does persona geometry change with model size?
5. Per-trait specificity comparison across scales

Run after run_all_riasec.py completes for marin-32b-base.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODELS = {
    "llama-1b": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "safe_id": "meta-llama__Llama-3.2-1B-Instruct",
        "size": "1B",
        "type": "instruct",
    },
    "qwen-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "safe_id": "Qwen__Qwen2.5-7B-Instruct",
        "size": "7B",
        "type": "instruct",
    },
    "marin-8b": {
        "model_id": "marin-community/marin-8b-instruct",
        "safe_id": "marin-community__marin-8b-instruct",
        "size": "8B",
        "type": "instruct",
    },
    "marin-32b": {
        "model_id": "marin-community/marin-32b-base",
        "safe_id": "marin-community__marin-32b-base",
        "size": "32B",
        "type": "base",
    },
}

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
SAFETENSORS_DIR = Path("./persona_data/model_inits/")
OUTPUT_DIR = Path("./outputs/analysis/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_vectors(model_key: str) -> dict[str, np.ndarray]:
    """Load response_persona_vector for all 6 RIASEC traits."""
    info = MODELS[model_key]
    vectors = {}
    for trait in TRAITS:
        path = SAFETENSORS_DIR / f"{trait}_persona_initialization" / f"{info['safe_id']}.safetensors"
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            return {}
        data = load_file(str(path))
        vectors[trait] = data["response_persona_vector"].numpy().flatten()
    return vectors


def cosine_matrix(vectors: dict[str, np.ndarray]) -> np.ndarray:
    """6x6 pairwise cosine similarity matrix for RIASEC traits."""
    vecs = np.array([vectors[t] for t in TRAITS])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = vecs / norms
    return normed @ normed.T


def shared_specific_decomposition(vectors: dict[str, np.ndarray]):
    """Decompose vectors into shared direction and trait-specific residuals."""
    vecs = np.array([vectors[t] for t in TRAITS])
    # Shared direction = first PC of the 6 vectors
    mean = vecs.mean(axis=0)
    centered = vecs - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # But shared direction is better computed as PC1 of the raw vectors
    U_raw, S_raw, Vt_raw = np.linalg.svd(vecs, full_matrices=False)
    shared_dir = Vt_raw[0]  # first right singular vector
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    # Project each vector onto shared direction
    projections = vecs @ shared_dir
    # Residuals
    residuals = vecs - np.outer(projections, shared_dir)

    # Variance explained by shared direction
    total_var = np.sum(np.var(vecs, axis=0))
    shared_var = np.var(projections) * 1  # projections is 1D
    # Better: fraction of total norm explained
    total_norm_sq = np.sum(vecs ** 2)
    shared_norm_sq = np.sum(projections ** 2)
    frac_shared = shared_norm_sq / total_norm_sq

    # PCA of residuals
    U_res, S_res, Vt_res = np.linalg.svd(residuals, full_matrices=False)
    res_var = S_res ** 2 / np.sum(S_res ** 2) * 100

    return {
        "shared_direction": shared_dir,
        "projections": {t: float(projections[i]) for i, t in enumerate(TRAITS)},
        "residuals": {t: residuals[i] for i, t in enumerate(TRAITS)},
        "frac_shared": float(frac_shared),
        "residual_pca_variance": res_var[:5].tolist(),
        "vector_norms": {t: float(np.linalg.norm(vectors[t])) for t in TRAITS},
        "residual_norms": {t: float(np.linalg.norm(residuals[i])) for i, t in enumerate(TRAITS)},
    }


def cross_model_cosine_correlation(cos_A: np.ndarray, cos_B: np.ndarray) -> float:
    """Pearson correlation between upper-triangle entries of two 6x6 cosine matrices."""
    idx = np.triu_indices(6, k=1)
    a = cos_A[idx]
    b = cos_B[idx]
    return float(np.corrcoef(a, b)[0, 1])


def residual_cosine_matrix(residuals: dict[str, np.ndarray]) -> np.ndarray:
    """Cosine matrix of residual (shared-removed) vectors."""
    vecs = np.array([residuals[t] for t in TRAITS])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = vecs / norms
    return normed @ normed.T


def main():
    print("=" * 70)
    print("MARIN 32B COMPREHENSIVE PERSONA VECTOR ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load all vectors
    # ------------------------------------------------------------------
    print("\n--- Loading vectors ---")
    all_vectors = {}
    for key in MODELS:
        vecs = load_vectors(key)
        if vecs:
            all_vectors[key] = vecs
            dim = len(next(iter(vecs.values())))
            print(f"  {key}: loaded {len(vecs)} traits, dim={dim}")
        else:
            print(f"  {key}: MISSING - skipping")

    if "marin-32b" not in all_vectors:
        print("\nERROR: Marin 32B vectors not found. Run extraction first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Cosine matrices per model
    # ------------------------------------------------------------------
    print("\n--- Cosine similarity matrices ---")
    cos_matrices = {}
    for key, vecs in all_vectors.items():
        cos_matrices[key] = cosine_matrix(vecs)
        print(f"\n  {key}:")
        mat = cos_matrices[key]
        for i, ti in enumerate(TRAITS):
            row = " ".join(f"{mat[i, j]:+.3f}" for j in range(6))
            print(f"    {ti[:4]:>4}: {row}")

    # ------------------------------------------------------------------
    # 3. Cross-model correlations (all pairs)
    # ------------------------------------------------------------------
    print("\n--- Cross-model cosine matrix correlations ---")
    model_keys = sorted(all_vectors.keys())
    cross_results = {}
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            a, b = model_keys[i], model_keys[j]
            r = cross_model_cosine_correlation(cos_matrices[a], cos_matrices[b])
            label = f"{a} vs {b}"
            cross_results[label] = r
            size_a = MODELS[a]["size"]
            size_b = MODELS[b]["size"]
            type_a = MODELS[a]["type"]
            type_b = MODELS[b]["type"]
            print(f"  {label:35s}: r = {r:.4f}  ({size_a} {type_a} vs {size_b} {type_b})")

    # ------------------------------------------------------------------
    # 4. Shared-specific decomposition per model
    # ------------------------------------------------------------------
    print("\n--- Shared-specific decomposition ---")
    decompositions = {}
    for key, vecs in all_vectors.items():
        dec = shared_specific_decomposition(vecs)
        decompositions[key] = dec
        size = MODELS[key]["size"]
        mtype = MODELS[key]["type"]
        print(f"\n  {key} ({size} {mtype}):")
        print(f"    Fraction shared: {dec['frac_shared']:.3f}")
        print(f"    Residual PCA var: {[f'{v:.1f}%' for v in dec['residual_pca_variance']]}")
        print(f"    Shared projections: {', '.join(f'{t[:4]}={v:.3f}' for t, v in dec['projections'].items())}")
        print(f"    Vector norms: {', '.join(f'{t[:4]}={v:.3f}' for t, v in dec['vector_norms'].items())}")
        print(f"    Residual norms: {', '.join(f'{t[:4]}={v:.3f}' for t, v in dec['residual_norms'].items())}")

    # ------------------------------------------------------------------
    # 5. Residual cross-model correlations (confound rebuttal at scale)
    # ------------------------------------------------------------------
    print("\n--- Residual cross-model correlations (shared direction removed) ---")
    res_cos_matrices = {}
    for key in all_vectors:
        res_cos_matrices[key] = residual_cosine_matrix(decompositions[key]["residuals"])

    res_cross = {}
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            a, b = model_keys[i], model_keys[j]
            r_orig = cross_results[f"{a} vs {b}"]
            r_res = cross_model_cosine_correlation(res_cos_matrices[a], res_cos_matrices[b])
            label = f"{a} vs {b}"
            res_cross[label] = {"original": r_orig, "residual": r_res}
            print(f"  {label:35s}: original r={r_orig:.4f}, residual r={r_res:.4f}")

    # ------------------------------------------------------------------
    # 6. Scaling analysis: how does geometry change with model size?
    # ------------------------------------------------------------------
    print("\n--- Scaling analysis ---")
    print("\n  Model size vs shared fraction:")
    for key in ["llama-1b", "qwen-7b", "marin-8b", "marin-32b"]:
        if key in decompositions:
            dec = decompositions[key]
            size = MODELS[key]["size"]
            mtype = MODELS[key]["type"]
            mean_norm = np.mean(list(dec["vector_norms"].values()))
            mean_res_norm = np.mean(list(dec["residual_norms"].values()))
            print(f"    {size:>4} ({mtype:8}): shared_frac={dec['frac_shared']:.3f}, "
                  f"mean_vec_norm={mean_norm:.3f}, mean_res_norm={mean_res_norm:.3f}")

    # ------------------------------------------------------------------
    # 7. Base vs Instruct comparison (Marin 8B instruct vs 32B base)
    # ------------------------------------------------------------------
    if "marin-8b" in all_vectors and "marin-32b" in all_vectors:
        print("\n--- Base vs Instruct: Marin 8B (instruct) vs 32B (base) ---")
        # This is interesting because it separates scale from instruction tuning
        r = cross_results.get("marin-32b vs marin-8b") or cross_results.get("marin-8b vs marin-32b")
        if r is None:
            # compute it
            r = cross_model_cosine_correlation(cos_matrices["marin-8b"], cos_matrices["marin-32b"])
        print(f"  Cosine matrix correlation: r = {r:.4f}")
        print(f"  Note: This combines scale (8B→32B) and training (instruct→base)")

    # ------------------------------------------------------------------
    # 8. Holland hexagonal structure at 32B
    # ------------------------------------------------------------------
    if "marin-32b" in res_cos_matrices:
        print("\n--- Holland hexagonal structure (Marin 32B residuals) ---")
        hex_order = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
        hex_idx = [TRAITS.index(t) for t in hex_order]
        mat = res_cos_matrices["marin-32b"]
        adjacent, alternate, opposite = [], [], []
        for i in range(6):
            for j in range(i + 1, 6):
                dist = min(abs(i - j), 6 - abs(i - j))
                ii, jj = hex_idx[i], hex_idx[j]
                val = mat[ii, jj]
                if dist == 1:
                    adjacent.append(val)
                elif dist == 2:
                    alternate.append(val)
                elif dist == 3:
                    opposite.append(val)
        print(f"  Adjacent mean: {np.mean(adjacent):.4f} ({len(adjacent)} pairs)")
        print(f"  Alternate mean: {np.mean(alternate):.4f} ({len(alternate)} pairs)")
        print(f"  Opposite mean: {np.mean(opposite):.4f} ({len(opposite)} pairs)")
        if np.mean(adjacent) > np.mean(alternate) > np.mean(opposite):
            print(f"  ✓ Ordering consistent with Holland's hexagon")
        else:
            print(f"  ✗ Ordering NOT consistent with Holland's hexagon")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "models": {k: {kk: vv for kk, vv in v.items()} for k, v in MODELS.items() if k in all_vectors},
        "cross_model_correlations": cross_results,
        "residual_cross_model_correlations": res_cross,
        "decompositions": {
            key: {
                "frac_shared": dec["frac_shared"],
                "residual_pca_variance": dec["residual_pca_variance"],
                "projections": dec["projections"],
                "vector_norms": dec["vector_norms"],
                "residual_norms": dec["residual_norms"],
            }
            for key, dec in decompositions.items()
        },
    }

    out_path = OUTPUT_DIR / "marin32b_comprehensive_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
