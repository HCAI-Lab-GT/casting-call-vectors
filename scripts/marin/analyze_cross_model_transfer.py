#!/usr/bin/env python
"""
Cross-model persona vector transfer analysis.

The cross-model cosine matrix correlation (r > 0.95) suggests the *structure*
of RIASEC vectors is preserved across architectures. But can we actually
TRANSFER a vector from one model to another?

This script tests:
1. Procrustes alignment of RIASEC vector spaces across models
2. Whether the alignment matrix generalizes (leave-one-out cross-validation)
3. Predicted vs actual specificity patterns after transfer

If this works, it means personality representations are not just structurally
similar -- they're functionally interchangeable up to a linear transformation.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

MODELS = {
    "llama-1b": "meta-llama__Llama-3.2-1B-Instruct",
    "marin-8b": "marin-community__marin-8b-instruct",
    "qwen-7b": "Qwen__Qwen2.5-7B-Instruct",
}

SAFETENSORS_DIR = Path("./persona_data/model_inits/")
OUTPUT_DIR = Path("./outputs/analysis/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_vectors(safe_id: str) -> dict[str, np.ndarray]:
    vectors = {}
    for trait in TRAITS:
        path = SAFETENSORS_DIR / f"{trait}_persona_initialization" / f"{safe_id}.safetensors"
        data = load_file(str(path))
        vectors[trait] = data["response_persona_vector"].numpy().flatten()
    return vectors


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def analyze_transfer(vectors_a: dict, vectors_b: dict, name_a: str, name_b: str):
    """Analyze transferability between two models' RIASEC vectors.

    Since models have different hidden dimensions, we can't do direct Procrustes.
    Instead, we work in the 6D RIASEC space (cosine similarity space).
    """
    # Build 6x6 cosine matrices
    def cos_matrix(vecs):
        V = np.array([vecs[t] for t in TRAITS])
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_normed = V / np.maximum(norms, 1e-8)
        return V_normed @ V_normed.T

    cos_a = cos_matrix(vectors_a)
    cos_b = cos_matrix(vectors_b)

    # Upper triangle correlation
    idx = np.triu_indices(6, k=1)
    raw_corr = float(np.corrcoef(cos_a[idx], cos_b[idx])[0, 1])

    # MDS to get 5D coordinates from cosine matrices
    # Convert cosine to distance: d = sqrt(2(1-cos))
    def cos_to_coords(cos_mat, n_dims=5):
        dist_sq = 2 * (1 - cos_mat)
        n = cos_mat.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ dist_sq @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort descending
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx][:n_dims]
        eigvecs = eigvecs[:, idx][:, :n_dims]
        # Clip negative eigenvalues
        eigvals = np.maximum(eigvals, 0)
        coords = eigvecs * np.sqrt(eigvals)
        return coords

    coords_a = cos_to_coords(cos_a)
    coords_b = cos_to_coords(cos_b)

    # Procrustes alignment in 5D
    R, scale = orthogonal_procrustes(coords_a, coords_b)

    # Leave-one-out cross-validation
    loo_cosine_errors = []
    loo_predictions = {}

    for leave_out_idx in range(6):
        leave_out_trait = TRAITS[leave_out_idx]
        train_idx = [i for i in range(6) if i != leave_out_idx]

        # Fit Procrustes on 5 training traits
        train_a = coords_a[train_idx]
        train_b = coords_b[train_idx]
        R_train, _ = orthogonal_procrustes(train_a, train_b)

        # Predict left-out trait
        predicted_b = coords_a[leave_out_idx] @ R_train
        actual_b = coords_b[leave_out_idx]

        # Cosine similarity of predicted vs actual
        cos = cosine_sim(predicted_b, actual_b)
        loo_cosine_errors.append(cos)

        # Also check: does the predicted vector's nearest neighbor match?
        distances = [np.linalg.norm(predicted_b - coords_b[j]) for j in range(6)]
        nearest = TRAITS[np.argmin(distances)]

        loo_predictions[leave_out_trait] = {
            "predicted_cosine": float(cos),
            "nearest_match": nearest,
            "correct": nearest == leave_out_trait,
        }

    # Perfect transfer would have all LOO predictions correct
    correct = sum(1 for v in loo_predictions.values() if v["correct"])

    return {
        "raw_correlation": raw_corr,
        "loo_mean_cosine": float(np.mean(loo_cosine_errors)),
        "loo_predictions": loo_predictions,
        "loo_correct": correct,
        "loo_total": 6,
    }


def analyze_within_model_stability(vectors: dict, name: str):
    """Analyze how stable the RIASEC structure is within a model using all-layers data."""
    safe_id = MODELS[name]
    all_layers = {}
    for trait in TRAITS:
        path = SAFETENSORS_DIR / f"{trait}_persona_initialization" / f"{safe_id}.safetensors"
        data = load_file(str(path))
        v = data["all_layers_response_persona_vector"].numpy()
        if v.ndim == 3:
            v = v.squeeze(1)
        all_layers[trait] = v

    num_layers = all_layers[TRAITS[0]].shape[0]
    mid_layer = num_layers // 2

    # Compare cosine matrix at extraction layer vs other layers
    def cos_matrix_at_layer(L):
        vecs = np.array([all_layers[t][L] for t in TRAITS])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / np.maximum(norms, 1e-8)
        return normed @ normed.T

    ref_cos = cos_matrix_at_layer(mid_layer)
    idx = np.triu_indices(6, k=1)
    ref_upper = ref_cos[idx]

    correlations = []
    for L in range(num_layers):
        cos = cos_matrix_at_layer(L)
        r = float(np.corrcoef(ref_upper, cos[idx])[0, 1])
        correlations.append({"layer": L, "correlation_with_mid": r})

    return correlations


def main():
    print("=" * 70)
    print("CROSS-MODEL PERSONA VECTOR TRANSFER ANALYSIS")
    print("=" * 70)

    # Load all vectors
    all_vectors = {}
    for key, safe_id in MODELS.items():
        all_vectors[key] = load_vectors(safe_id)

    # Analyze all pairs
    model_keys = sorted(MODELS.keys())
    results = {}
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            a, b = model_keys[i], model_keys[j]
            label = f"{a} -> {b}"
            res = analyze_transfer(all_vectors[a], all_vectors[b], a, b)
            results[label] = res

            print(f"\n--- {label} ---")
            print(f"  Raw cosine matrix correlation: {res['raw_correlation']:.4f}")
            print(f"  LOO mean cosine (in 5D): {res['loo_mean_cosine']:.4f}")
            print(f"  LOO correct predictions: {res['loo_correct']}/{res['loo_total']}")
            for trait, pred in res["loo_predictions"].items():
                check = "✓" if pred["correct"] else "✗"
                print(f"    {trait:15s}: predicted nearest = {pred['nearest_match']:15s} "
                      f"(cos={pred['predicted_cosine']:.3f}) {check}")

    # Within-model stability
    print(f"\n{'=' * 50}")
    print("Within-model layer stability")
    for name in model_keys:
        corrs = analyze_within_model_stability(all_vectors[name], name)
        results[f"{name}_layer_stability"] = corrs
        # Find range of layers with r > 0.9
        high_r = [c for c in corrs if c["correlation_with_mid"] > 0.9]
        print(f"  {name}: {len(high_r)}/{len(corrs)} layers have r > 0.9 with extraction layer")

    # Save
    out_path = OUTPUT_DIR / "cross_model_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
