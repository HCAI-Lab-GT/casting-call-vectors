#!/usr/bin/env python
"""
Decompose RIASEC vectors into shared and trait-specific components.

Key question: RIASEC vectors all increase YES probability uniformly (no
cross-trait specificity). Are they all pointing in the same direction?
If so, we can subtract the shared direction and see if the residuals carry
trait-specific information.

Analysis:
1. Compute the mean RIASEC vector (shared direction)
2. Project each vector onto the shared direction and get residuals
3. Analyze: how much variance is shared vs trait-specific?
4. Compare cosine similarity structure of residuals vs originals

Usage:
  uv run python scripts/marin/analyze_vector_decomposition.py
"""

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
HEX_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]

RIASEC_DIR = Path("persona_data/model_inits")
OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def load_riasec_vectors(model_id: str) -> dict[str, np.ndarray]:
    safe_model = model_id.replace("/", "__")
    vectors = {}
    for trait in TRAITS:
        path = RIASEC_DIR / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            vectors[trait] = f.get_tensor("response_persona_vector").numpy().flatten()
    return vectors


def analyze_model(model_id: str):
    print(f"\n{'='*70}")
    print(f"VECTOR DECOMPOSITION: {model_id}")
    print(f"{'='*70}")

    vectors = load_riasec_vectors(model_id)
    hidden_dim = len(next(iter(vectors.values())))

    # Stack vectors: (6, hidden_dim)
    V = np.stack([vectors[t] for t in TRAITS])

    # 1. Compute mean vector (shared direction)
    mean_vec = V.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    mean_unit = mean_vec / mean_norm

    print(f"\nHidden dim: {hidden_dim}")
    print(f"Mean vector norm: {mean_norm:.4f}")

    # 2. For each vector, compute: projection onto mean, residual
    results = {"model_id": model_id, "hidden_dim": hidden_dim, "traits": TRAITS}
    results["mean_vector_norm"] = float(mean_norm)

    projections = {}
    residuals = {}
    for trait in TRAITS:
        v = vectors[trait]
        v_norm = np.linalg.norm(v)

        # Projection onto mean direction
        proj_scalar = np.dot(v, mean_unit)
        proj_vector = proj_scalar * mean_unit
        residual = v - proj_vector
        residual_norm = np.linalg.norm(residual)

        # Fraction of variance in shared direction
        shared_frac = proj_scalar**2 / (v_norm**2 + 1e-10)

        projections[trait] = proj_vector
        residuals[trait] = residual

        cos_with_mean = cosine_sim(v, mean_vec)

        print(f"\n  {trait}:")
        print(f"    vector norm: {v_norm:.4f}")
        print(f"    cosine with mean: {cos_with_mean:.4f}")
        print(f"    projection onto mean: {proj_scalar:.4f}")
        print(f"    residual norm: {residual_norm:.4f}")
        print(f"    shared variance fraction: {shared_frac:.1%}")

        results[trait] = {
            "vector_norm": float(v_norm),
            "cosine_with_mean": float(cos_with_mean),
            "projection_onto_mean": float(proj_scalar),
            "residual_norm": float(residual_norm),
            "shared_variance_fraction": float(shared_frac),
        }

    # 3. Compare cosine matrices: original vs residuals
    print(f"\n--- Original cosine matrix ---")
    for i, t1 in enumerate(TRAITS):
        row = f"  {t1[:6]:>6s}:"
        for j, t2 in enumerate(TRAITS):
            row += f" {cosine_sim(vectors[t1], vectors[t2]):+.3f}"
        print(row)

    print(f"\n--- Residual cosine matrix (shared direction removed) ---")
    residual_cosines = {}
    for t1 in TRAITS:
        residual_cosines[t1] = {}
        row = f"  {t1[:6]:>6s}:"
        for t2 in TRAITS:
            c = cosine_sim(residuals[t1], residuals[t2])
            residual_cosines[t1][t2] = c
            row += f" {c:+.3f}"
        print(row)

    results["original_cosine_matrix"] = {
        t1: {t2: cosine_sim(vectors[t1], vectors[t2]) for t2 in TRAITS} for t1 in TRAITS
    }
    results["residual_cosine_matrix"] = residual_cosines

    # 4. Check if residual cosine structure matches Holland hexagonal predictions
    print(f"\n--- Hexagonal distance analysis ---")
    for label, vecs in [("Original", vectors), ("Residual", residuals)]:
        for d_name, d_val in [("Adjacent(1)", 1), ("Alternate(2)", 2), ("Opposite(3)", 3)]:
            vals = []
            for i in range(6):
                for j in range(i + 1, 6):
                    if min(abs(i - j), 6 - abs(i - j)) == d_val:
                        vals.append(cosine_sim(vecs[HEX_ORDER[i]], vecs[HEX_ORDER[j]]))
            print(f"  {label:>8s} {d_name}: mean={np.mean(vals):.4f} (n={len(vals)})")

    # 5. PCA of residuals
    R = np.stack([residuals[t] for t in TRAITS])
    R_centered = R - R.mean(axis=0)
    # SVD since n < d
    U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
    explained = (S**2) / (S**2).sum()
    print(f"\n--- PCA of residual vectors ---")
    for i, (s, ev) in enumerate(zip(S, explained)):
        print(f"  PC{i+1}: singular={s:.4f}, explained={ev:.1%}")
    results["residual_pca_explained_variance"] = explained.tolist()

    # 6. Compute mean off-diagonal cosine for original and residual
    orig_off_diag = []
    resid_off_diag = []
    for i in range(len(TRAITS)):
        for j in range(i + 1, len(TRAITS)):
            orig_off_diag.append(cosine_sim(vectors[TRAITS[i]], vectors[TRAITS[j]]))
            resid_off_diag.append(cosine_sim(residuals[TRAITS[i]], residuals[TRAITS[j]]))

    print(f"\n--- Summary ---")
    print(f"  Original: mean off-diag cosine = {np.mean(orig_off_diag):.4f}")
    print(f"  Residual: mean off-diag cosine = {np.mean(resid_off_diag):.4f}")
    print(f"  (Lower residual means more trait-specific after removing shared direction)")

    results["original_mean_off_diag_cosine"] = float(np.mean(orig_off_diag))
    results["residual_mean_off_diag_cosine"] = float(np.mean(resid_off_diag))

    safe_model = model_id.replace("/", "__")
    out_path = OUTPUT_DIR / f"{safe_model}_vector_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    for model_id in [
        "meta-llama/Llama-3.2-1B-Instruct",
        "marin-community/marin-8b-instruct",
    ]:
        analyze_model(model_id)
