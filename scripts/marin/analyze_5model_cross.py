#!/usr/bin/env python
"""Cross-model analysis across 5 models including SmolLM3-3B."""

import numpy as np
from safetensors.torch import load_file
from scipy.stats import pearsonr
from pathlib import Path

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
ROOT = Path(__file__).resolve().parents[2]


def load_mid_layer_vectors(model_id):
    safe = model_id.replace("/", "__")
    vecs = {}
    for trait in TRAITS:
        path = ROOT / f"persona_data/model_inits/{trait}_persona_initialization/{safe}.safetensors"
        data = load_file(str(path))
        v = data["all_layers_response_persona_vector"].numpy()
        if v.ndim == 3:
            v = v[:, 0, :]
        num_layers = v.shape[0] - 1
        mid = num_layers // 2
        vecs[trait] = v[mid + 1]
    return vecs


def get_residual_cos(vecs):
    V = np.stack([vecs[t] for t in TRAITS])
    V_normed = V / np.linalg.norm(V, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(V_normed, full_matrices=False)
    shared = Vt[0]
    R = V_normed - np.outer(V_normed @ shared, shared)
    R_normed = R / np.linalg.norm(R, axis=1, keepdims=True)
    cos_mat = R_normed @ R_normed.T
    return cos_mat[np.triu_indices(6, k=1)]


def procrustes_loo(vecs_a, vecs_b):
    def to_5d(vecs):
        V = np.stack([vecs[t] for t in TRAITS])
        V_normed = V / np.linalg.norm(V, axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(V_normed, full_matrices=False)
        return V_normed @ Vt[1:6].T

    A = to_5d(vecs_a)
    B = to_5d(vecs_b)

    correct = 0
    min_cos = 1.0
    for leave_out in range(6):
        mask = [i for i in range(6) if i != leave_out]
        U, _, Vt = np.linalg.svd(A[mask].T @ B[mask])
        R = U @ Vt
        predicted = A[leave_out] @ R
        actual = B[leave_out]
        cos = np.dot(predicted, actual) / (np.linalg.norm(predicted) * np.linalg.norm(actual))
        min_cos = min(min_cos, cos)
        dists = [np.dot(predicted, B[j]) / (np.linalg.norm(predicted) * np.linalg.norm(B[j]))
                 for j in range(6)]
        if np.argmax(dists) == leave_out:
            correct += 1
    return correct, min_cos


def simplex_analysis(vecs, name):
    V = np.stack([vecs[t] for t in TRAITS])
    V_normed = V / np.linalg.norm(V, axis=1, keepdims=True)
    _, s, Vt = np.linalg.svd(V_normed, full_matrices=False)

    shared = Vt[0]
    R = V_normed - np.outer(V_normed @ shared, shared)
    R_normed = R / np.linalg.norm(R, axis=1, keepdims=True)
    cos_mat = R_normed @ R_normed.T
    upper = cos_mat[np.triu_indices(6, k=1)]
    mean_cos = np.mean(upper)
    efficiency = (1.0 - mean_cos) / 1.2  # theoretical max = 1.2 for 6 in 5D

    return {
        "shared_fraction": float(s[0]**2 / np.sum(s**2)),
        "mean_residual_cos": float(mean_cos),
        "simplex_efficiency": float(efficiency),
        "s6_s1_ratio": float(s[-1] / s[0]),
    }


def main():
    models = {
        "Llama 1B": "meta-llama/Llama-3.2-1B-Instruct",
        "SmolLM3 3B": "HuggingFaceTB/SmolLM3-3B",
        "Qwen 7B": "Qwen/Qwen2.5-7B-Instruct",
        "Marin 8B": "marin-community/marin-8b-instruct",
        "Marin 32B": "marin-community/marin-32b-base",
    }

    all_vecs = {}
    for name, model_id in models.items():
        all_vecs[name] = load_mid_layer_vectors(model_id)

    names = list(models.keys())

    print("=" * 70)
    print("5-MODEL CROSS-MODEL COMPARISON (including SmolLM3-3B)")
    print("=" * 70)

    # Simplex analysis
    print(f"\n--- Simplex & Geometry ---")
    print(f"  {'Model':>12} {'Shared%':>8} {'MeanCos':>8} {'Efficiency':>10} {'s6/s1':>8}")
    print(f"  {'-'*50}")
    for name in names:
        sa = simplex_analysis(all_vecs[name], name)
        print(f"  {name:>12} {sa['shared_fraction']:>7.3f} {sa['mean_residual_cos']:>+7.3f} {sa['simplex_efficiency']:>9.1%} {sa['s6_s1_ratio']:>7.4f}")

    # Residual cosine correlation
    all_res_cos = {name: get_residual_cos(all_vecs[name]) for name in names}

    print(f"\n--- Residual Cosine Correlation (Pearson r) ---")
    print(f"  {'':>12}", end="")
    for n in names:
        print(f"{n:>12}", end="")
    print()
    for n1 in names:
        print(f"  {n1:>12}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"{'1.000':>12}", end="")
            else:
                r, _ = pearsonr(all_res_cos[n1], all_res_cos[n2])
                print(f"{r:>12.3f}", end="")
        print()

    # Procrustes LOO
    print(f"\n--- Procrustes LOO Transfer ---")
    print(f"  {'Pair':>25} {'LOO':>6} {'MinCos':>8}")
    print(f"  {'-'*45}")
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            correct, min_cos = procrustes_loo(all_vecs[n1], all_vecs[n2])
            print(f"  {n1:>10} → {n2:<10}: {correct}/6  {min_cos:.4f}")

    # Pairwise discrimination summary
    print(f"\n--- Pairwise Discrimination (from sweep results) ---")
    pairwise = {
        "Llama 1B": {"dim": 2048, "best_alpha": 1.0, "best_res": 0.63},
        "SmolLM3 3B": {"dim": 2048, "best_alpha": 1.0, "best_res": 0.73},
        "Qwen 7B": {"dim": 3584, "best_alpha": 5.0, "best_res": 0.83},
        "Marin 8B": {"dim": 4096, "best_alpha": 1.0, "best_res": 0.97},
    }

    print(f"  {'Model':>12} {'HidDim':>8} {'BestAlpha':>10} {'BestΔ%':>8}")
    print(f"  {'-'*42}")
    for name, d in pairwise.items():
        print(f"  {name:>12} {d['dim']:>8} {d['best_alpha']:>10.1f} {d['best_res']:>7.0%}")

    from scipy.stats import spearmanr
    dims = [d["dim"] for d in pairwise.values()]
    accs = [d["best_res"] for d in pairwise.values()]
    rho, p = spearmanr(dims, accs)
    print(f"\n  Spearman(hidden_dim, best_delta_acc): ρ={rho:.3f}, p={p:.3f}")


if __name__ == "__main__":
    main()
