#!/usr/bin/env python
"""
Löwdin (symmetric) orthogonalization of RIASEC residual vectors.

Unlike Gram-Schmidt, Löwdin orthogonalization is order-independent and
minimizes the total change to the vectors (closest orthogonal set in
Frobenius norm). This makes it the natural choice for orthogonalizing
personality vectors where no trait should be privileged.

Key questions:
1. How much do Löwdin-orthogonalized vectors differ from originals?
2. What does the overlap matrix look like? (Diagnoses cross-trait contamination)
3. Can we predict the specificity improvement from geometric properties alone?
4. Is the artistic-conventional anti-correlation explainable by Holland hexagonal
   structure?

No GPU needed - pure vector geometry analysis.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

HEXAGON_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
ADJACENT_PAIRS = [(HEXAGON_ORDER[i], HEXAGON_ORDER[(i+1) % 6]) for i in range(6)]
OPPOSITE_PAIRS = [(HEXAGON_ORDER[i], HEXAGON_ORDER[(i+3) % 6]) for i in range(3)]


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


def decompose_at_layer(all_vecs, layer_idx):
    V = np.stack([all_vecs[t][layer_idx + 1] for t in TRAITS])
    _, s, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residuals = {}
    for t in TRAITS:
        vec = all_vecs[t][layer_idx + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residuals[t] = vec - proj
    return residuals, shared_dir, s


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def lowdin_orthogonalize(vectors_dict, trait_order=TRAITS):
    """Löwdin (symmetric) orthogonalization.

    Given matrix V (n_traits x dim), computes V_orth = S^{-1/2} V
    where S = V V^T is the overlap matrix. This is the closest
    orthonormal set to the original vectors in Frobenius norm.

    Returns orthogonalized vectors (as dict) and the overlap matrix.
    """
    # Normalize first
    normed = {}
    norms = {}
    for t in trait_order:
        n = np.linalg.norm(vectors_dict[t])
        norms[t] = n
        normed[t] = vectors_dict[t] / max(n, 1e-10)

    V = np.stack([normed[t] for t in trait_order])  # (6, dim)

    # Overlap matrix S = V @ V^T
    S = V @ V.T  # (6, 6)

    # S^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-10)  # numerical stability
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Orthogonalized vectors
    V_orth = S_inv_sqrt @ V  # (6, dim)

    ortho_dict = {}
    for i, t in enumerate(trait_order):
        ortho_dict[t] = V_orth[i] * norms[t]  # restore original scale

    return ortho_dict, S, S_inv_sqrt, norms


def analyze_overlap_matrix(S, trait_order=TRAITS):
    """Analyze the overlap matrix structure."""
    trait_idx = {t: i for i, t in enumerate(trait_order)}

    # Adjacent, alternate, opposite means
    adj_vals = [S[trait_idx[a], trait_idx[b]] for a, b in ADJACENT_PAIRS]
    opp_vals = [S[trait_idx[a], trait_idx[b]] for a, b in OPPOSITE_PAIRS]

    off_diag = S[~np.eye(6, dtype=bool)]

    return {
        "mean_off_diagonal": float(np.mean(off_diag)),
        "mean_abs_off_diagonal": float(np.mean(np.abs(off_diag))),
        "adjacent_mean": float(np.mean(adj_vals)),
        "opposite_mean": float(np.mean(opp_vals)),
        "condition_number": float(np.linalg.cond(S)),
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(S))),
        "max_eigenvalue": float(np.max(np.linalg.eigvalsh(S))),
    }


def compute_direction_change(original, orthogonalized):
    """How much did each vector's direction change?"""
    changes = {}
    for t in TRAITS:
        cos = cosine_sim(original[t], orthogonalized[t])
        angle = np.degrees(np.arccos(np.clip(abs(cos), 0, 1)))
        norm_ratio = np.linalg.norm(orthogonalized[t]) / max(np.linalg.norm(original[t]), 1e-10)
        changes[t] = {
            "cosine_with_original": float(cos),
            "angle_degrees": float(angle),
            "norm_ratio": float(norm_ratio),
        }
    return changes


def predict_specificity_matrix(vectors_dict, trait_order=TRAITS):
    """Predict the 6x6 specificity matrix from vector geometry.

    If steering adds alpha * v_i to the residual stream, and the model's
    response to trait j's questions depends on the projection of the
    perturbation onto the "natural" trait j direction, then:

    predicted_effect(steer_i, eval_j) ∝ cos(v_i, v_j) * ||v_i|| * ||v_j||

    For normalized vectors, this is just the cosine similarity.
    """
    V = np.stack([vectors_dict[t] for t in trait_order])
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_normed = V / np.maximum(norms, 1e-10)

    cos_mat = V_normed @ V_normed.T

    diag = np.mean(np.diag(cos_mat))
    off_diag = np.mean(cos_mat[~np.eye(6, dtype=bool)])

    return {
        "cosine_matrix": cos_mat.tolist(),
        "diagonal_mean": float(diag),
        "off_diagonal_mean": float(off_diag),
        "predicted_specificity": float(diag - off_diag),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "meta-llama/Llama-3.2-1B-Instruct",
        "marin-community/marin-8b-instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ])
    args = ap.parse_args()

    all_results = {}

    for model_id in args.models:
        safe_model = model_id.replace("/", "__")
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        try:
            all_vecs = load_all_layer_vectors(model_id)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        num_layers = all_vecs[TRAITS[0]].shape[0] - 1
        mid_layer = num_layers // 2

        # Decompose at mid layer
        residuals, shared_dir, sv = decompose_at_layer(all_vecs, mid_layer)

        print(f"\n--- Original Residual Vectors (L{mid_layer}) ---")

        # Original cosine structure
        orig_pred = predict_specificity_matrix(residuals)
        print(f"  Predicted specificity: {orig_pred['predicted_specificity']:+.4f}")
        print(f"  Diagonal mean: {orig_pred['diagonal_mean']:.4f}")
        print(f"  Off-diagonal mean: {orig_pred['off_diagonal_mean']:.4f}")

        # Löwdin orthogonalization
        ortho, S, S_inv_sqrt, norms = lowdin_orthogonalize(residuals)

        print(f"\n--- Overlap Matrix (before orthogonalization) ---")
        print(f"  {'':>15}", end="")
        for t in TRAITS:
            print(f"{t[:5]:>8}", end="")
        print()
        for i, ti in enumerate(TRAITS):
            print(f"  {ti:>15}", end="")
            for j in range(6):
                print(f"{S[i,j]:>8.3f}", end="")
            print()

        overlap_analysis = analyze_overlap_matrix(S)
        print(f"\n  Mean |off-diag|: {overlap_analysis['mean_abs_off_diagonal']:.4f}")
        print(f"  Condition number: {overlap_analysis['condition_number']:.2f}")
        print(f"  Eigenvalues: [{overlap_analysis['min_eigenvalue']:.3f}, {overlap_analysis['max_eigenvalue']:.3f}]")

        # Direction changes
        print(f"\n--- Direction Changes After Löwdin ---")
        changes = compute_direction_change(residuals, ortho)
        for t in TRAITS:
            c = changes[t]
            print(f"  {t:>15}: cos={c['cosine_with_original']:.4f}, "
                  f"angle={c['angle_degrees']:.1f}°, "
                  f"norm_ratio={c['norm_ratio']:.4f}")

        mean_cos = np.mean([changes[t]["cosine_with_original"] for t in TRAITS])
        mean_angle = np.mean([changes[t]["angle_degrees"] for t in TRAITS])
        print(f"\n  Mean cos with original: {mean_cos:.4f}")
        print(f"  Mean angle change: {mean_angle:.1f}°")

        # Verify orthogonality
        ortho_pred = predict_specificity_matrix(ortho)
        print(f"\n--- Löwdin-Orthogonalized Vectors ---")
        print(f"  Predicted specificity: {ortho_pred['predicted_specificity']:+.4f}")
        print(f"  Diagonal mean: {ortho_pred['diagonal_mean']:.4f}")
        print(f"  Off-diagonal mean: {ortho_pred['off_diagonal_mean']:.4f}")

        # Verify: orthogonalized cosine matrix should be identity
        print(f"\n  Verification (should be ~identity):")
        ortho_cos = np.array(ortho_pred["cosine_matrix"])
        print(f"  Max |off-diag|: {np.max(np.abs(ortho_cos[~np.eye(6, dtype=bool)])):.6f}")

        # Specificity improvement
        orig_spec = orig_pred["predicted_specificity"]
        ortho_spec = ortho_pred["predicted_specificity"]
        print(f"\n--- Specificity Improvement ---")
        print(f"  Original:       {orig_spec:+.4f}")
        print(f"  Orthogonalized: {ortho_spec:+.4f}")
        print(f"  Improvement:    {ortho_spec - orig_spec:+.4f}")

        # Now do it for ALL layers
        print(f"\n--- Layer-by-Layer Löwdin Analysis ---")
        print(f"  {'Layer':>5} {'Orig Spec':>10} {'Orth Spec':>10} {'MeanAngle':>10} {'CondNum':>8}")
        print(f"  {'-'*50}")

        layer_results = []
        for layer in range(num_layers):
            res, _, _ = decompose_at_layer(all_vecs, layer)
            o_pred = predict_specificity_matrix(res)

            orth_vecs, S_l, _, _ = lowdin_orthogonalize(res)
            orth_pred = predict_specificity_matrix(orth_vecs)

            ch = compute_direction_change(res, orth_vecs)
            mean_ang = np.mean([ch[t]["angle_degrees"] for t in TRAITS])
            cond = np.linalg.cond(S_l)

            layer_results.append({
                "layer": layer,
                "orig_specificity": float(o_pred["predicted_specificity"]),
                "orth_specificity": float(orth_pred["predicted_specificity"]),
                "mean_angle_change": float(mean_ang),
                "condition_number": float(cond),
            })

            if layer % max(1, num_layers // 10) == 0 or layer == mid_layer:
                marker = " <-- mid" if layer == mid_layer else ""
                print(f"  L{layer:>3} {o_pred['predicted_specificity']:>10.4f} "
                      f"{orth_pred['predicted_specificity']:>10.4f} "
                      f"{mean_ang:>9.1f}° {cond:>8.1f}{marker}")

        # Find the layer where original specificity is maximized
        best_orig = max(layer_results, key=lambda x: x["orig_specificity"])
        print(f"\n  Best original specificity: L{best_orig['layer']} = {best_orig['orig_specificity']:+.4f}")
        print(f"  Mid layer specificity:    L{mid_layer} = "
              f"{next(l['orig_specificity'] for l in layer_results if l['layer'] == mid_layer):+.4f}")

        # What's the relationship between condition number and specificity?
        conds = [l["condition_number"] for l in layer_results]
        orig_specs = [l["orig_specificity"] for l in layer_results]
        corr = np.corrcoef(conds, orig_specs)[0, 1]
        print(f"\n  Correlation(condition_number, orig_specificity): {corr:.3f}")

        all_results[safe_model] = {
            "num_layers": num_layers,
            "mid_layer": mid_layer,
            "mid_layer_analysis": {
                "original_specificity": orig_spec,
                "orthogonalized_specificity": ortho_spec,
                "overlap_analysis": overlap_analysis,
                "direction_changes": changes,
                "mean_cos_with_original": float(mean_cos),
                "mean_angle_change": float(mean_angle),
            },
            "layer_results": layer_results,
            "cond_spec_correlation": float(corr),
            "best_orig_specificity_layer": best_orig["layer"],
        }

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"\n  {'Model':>40} {'OrigSpec':>9} {'OrthSpec':>9} {'MeanAngle':>10} {'CondNum':>8}")
        print(f"  {'-'*80}")
        for name, r in all_results.items():
            mid = r["mid_layer_analysis"]
            ov = mid["overlap_analysis"]
            print(f"  {name:>40} {mid['original_specificity']:>+9.4f} "
                  f"{mid['orthogonalized_specificity']:>+9.4f} "
                  f"{mid['mean_angle_change']:>9.1f}° {ov['condition_number']:>8.1f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lowdin_orthogonalization.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
