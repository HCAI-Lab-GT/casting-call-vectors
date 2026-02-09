#!/usr/bin/env python
"""
Use simplex deviations to predict and explain observed specificity patterns.

The simplex deviations encode which trait pairs are more/less distinguishable.
Can we use this geometric structure to:
1. Predict which traits should be most/least specific in evaluation?
2. Explain why Investigative is consistently the most specific?
3. Predict the full 6x6 specificity matrix from geometry alone?

This bridges geometry (which we understand) with evaluation (which we observe).
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_all_layer_vectors(model_id):
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"
    all_vecs = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_vecs[trait] = vecs
    return all_vecs


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def get_residual_cosine_matrix(all_vecs, layer_idx):
    """Get 6x6 cosine matrix of residual vectors."""
    V = np.stack([all_vecs[t][layer_idx + 1] for t in TRAITS])
    V_normed = V / np.linalg.norm(V, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(V_normed, full_matrices=False)
    shared_dir = Vt[0]

    residuals = {}
    for i, t in enumerate(TRAITS):
        r = V_normed[i] - np.dot(V_normed[i], shared_dir) * shared_dir
        residuals[t] = r / max(np.linalg.norm(r), 1e-10)

    cos_mat = np.zeros((6, 6))
    for i, ti in enumerate(TRAITS):
        for j, tj in enumerate(TRAITS):
            cos_mat[i, j] = cosine_sim(residuals[ti], residuals[tj])
    return cos_mat


def predict_specificity_from_cosine(cos_mat):
    """Predict per-trait specificity from cosine matrix.

    If steering with trait i adds alpha * v_i to the residual stream,
    the effect on trait j's evaluation is proportional to cos(v_i, v_j).

    Per-trait specificity = self_cos - mean(other_cos) = 1.0 - mean(off_diag_row)
    """
    per_trait = {}
    for i, trait in enumerate(TRAITS):
        self_cos = cos_mat[i, i]  # Always 1.0 for normalized
        other_cos = [cos_mat[i, j] for j in range(6) if j != i]
        mean_other = np.mean(other_cos)
        per_trait[trait] = {
            "predicted_specificity": float(self_cos - mean_other),
            "mean_cross_cos": float(mean_other),
            "min_cross_cos": float(np.min(other_cos)),
            "max_cross_cos": float(np.max(other_cos)),
        }
    return per_trait


def load_observed_specificity(model_id):
    """Load observed specificity matrices from previous experiments."""
    safe_model = model_id.replace("/", "__")

    # Try multiple possible locations
    paths = [
        _repo_root() / f"outputs/specificity/{safe_model}_residual_specificity.json",
        _repo_root() / f"outputs/analysis/residual_optimal_layer_{safe_model}.json",
        _repo_root() / f"outputs/analysis/optimal_layer_specificity_{safe_model}.json",
    ]

    for path in paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data, str(path)
    return None, None


def main():
    print("="*70)
    print("SIMPLEX DEVIATION → SPECIFICITY PREDICTIONS")
    print("="*70)

    for model_id in [
        "meta-llama/Llama-3.2-1B-Instruct",
        "marin-community/marin-8b-instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]:
        try:
            all_vecs = load_all_layer_vectors(model_id)
        except FileNotFoundError:
            continue

        num_layers = all_vecs[TRAITS[0]].shape[0] - 1
        mid_layer = num_layers // 2

        print(f"\n{'='*70}")
        print(f"MODEL: {model_id} (L{mid_layer})")
        print(f"{'='*70}")

        # Get residual cosine matrix
        cos_mat = get_residual_cosine_matrix(all_vecs, mid_layer)
        ideal = -1.0 / 5

        # Per-trait predictions from geometry
        predictions = predict_specificity_from_cosine(cos_mat)

        print(f"\n--- Geometric Predictions ---")
        print(f"  {'Trait':>15} {'Pred Spec':>10} {'MeanCross':>10} {'MinCross':>10} {'MaxCross':>10}")
        print(f"  {'-'*60}")
        for trait in TRAITS:
            p = predictions[trait]
            print(f"  {trait:>15} {p['predicted_specificity']:>+10.4f} "
                  f"{p['mean_cross_cos']:>+10.4f} "
                  f"{p['min_cross_cos']:>+10.4f} "
                  f"{p['max_cross_cos']:>+10.4f}")

        # Rank by predicted specificity
        ranked = sorted(predictions.items(), key=lambda x: x[1]["predicted_specificity"], reverse=True)
        print(f"\n  Predicted specificity ranking:")
        for rank, (trait, info) in enumerate(ranked, 1):
            print(f"    {rank}. {trait} ({info['predicted_specificity']:+.4f})")

        # The key geometric predictor: how "extreme" is each trait?
        print(f"\n--- Geometric Extremity (deviation from simplex center) ---")
        # The simplex center is the mean of all 6 residual directions
        # Traits further from center should be more distinct

        V = np.stack([all_vecs[t][mid_layer + 1] for t in TRAITS])
        V_normed = V / np.linalg.norm(V, axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(V_normed, full_matrices=False)
        shared_dir = Vt[0]

        residuals = {}
        for i, t in enumerate(TRAITS):
            r = V_normed[i] - np.dot(V_normed[i], shared_dir) * shared_dir
            residuals[t] = r

        # Centroid of residual directions
        centroid = np.mean([residuals[t] for t in TRAITS], axis=0)
        centroid_norm = np.linalg.norm(centroid)
        print(f"  Centroid norm: {centroid_norm:.6f} (should be ~0 for perfect simplex)")

        for t in TRAITS:
            dist = np.linalg.norm(residuals[t] - centroid)
            norm = np.linalg.norm(residuals[t])
            print(f"  {t:>15}: norm={norm:.4f}, dist_from_center={dist:.4f}")

        # Load and compare with observed specificity
        obs_data, obs_path = load_observed_specificity(model_id)
        if obs_data:
            print(f"\n--- Comparison with Observed Specificity ---")
            print(f"  Source: {obs_path}")

            # Extract observed per-trait specificity
            if "residual_mid" in obs_data:
                # From residual_optimal_layer experiment
                matrix = np.array(obs_data["residual_mid"]["matrix"])
                obs_per_trait = {}
                for i, trait in enumerate(TRAITS):
                    diag = matrix[i, i]
                    off_diag = np.mean([matrix[i, j] for j in range(6) if j != i])
                    obs_per_trait[trait] = diag - off_diag
            elif "mid" in obs_data:
                # From optimal_layer_specificity experiment (full vectors)
                matrix = np.array(obs_data["mid"]["matrix"])
                obs_per_trait = {}
                for i, trait in enumerate(TRAITS):
                    diag = matrix[i, i]
                    off_diag = np.mean([matrix[i, j] for j in range(6) if j != i])
                    obs_per_trait[trait] = diag - off_diag
            else:
                obs_per_trait = None

            if obs_per_trait:
                print(f"\n  {'Trait':>15} {'Predicted':>10} {'Observed':>10} {'Match':>7}")
                print(f"  {'-'*45}")
                pred_ranks = []
                obs_ranks = []
                for trait in TRAITS:
                    pred = predictions[trait]["predicted_specificity"]
                    obs = obs_per_trait.get(trait, float('nan'))
                    match = "✓" if (pred > 1.2 and obs > 0) or (pred < 1.2 and obs < 0) else "✗"
                    print(f"  {trait:>15} {pred:>+10.4f} {obs:>+10.3f} {match:>7}")
                    pred_ranks.append(pred)
                    obs_ranks.append(obs)

                # Rank correlation
                from scipy.stats import spearmanr
                rho, pval = spearmanr(pred_ranks, obs_ranks)
                print(f"\n  Spearman rank correlation: ρ = {rho:.3f} (p = {pval:.3f})")
                print(f"  → {'Strong' if abs(rho) > 0.7 else 'Moderate' if abs(rho) > 0.4 else 'Weak'} "
                      f"{'positive' if rho > 0 else 'negative'} rank correlation")

        # Full vectors: predict which traits have highest shared loading
        print(f"\n--- Shared Direction Loading (explains agree-all bias) ---")
        for t in TRAITS:
            v = V_normed[TRAITS.index(t)]
            shared_proj = np.dot(v, shared_dir)
            residual_norm = np.linalg.norm(residuals[t])
            print(f"  {t:>15}: shared_proj={shared_proj:+.4f}, "
                  f"residual_norm={residual_norm:.4f}, "
                  f"ratio={residual_norm/abs(shared_proj):.4f}")


if __name__ == "__main__":
    main()
