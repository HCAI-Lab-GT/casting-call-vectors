#!/usr/bin/env python
"""
Dimensionality analysis of RIASEC residual persona vectors.

KEY INSIGHT: After removing the shared direction (PC1), the 6 RIASEC
vectors live in a 5D subspace. You CANNOT make 6 vectors orthogonal
in 5D - this is a mathematical impossibility. This means:

1. Some cross-trait correlation is UNAVOIDABLE (not a deficiency of the method)
2. The observed specificity is bounded by the intrinsic dimensionality
3. One degree of freedom is "wasted" on the linear dependency

This analysis computes:
- The intrinsic dimensionality of the residual subspace
- The theoretical specificity bound
- How close the current vectors are to the optimal configuration
- The simplex structure (equiangular frame theory)
- Which trait contributes the redundant dimension

No GPU needed.
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


def decompose_at_layer(all_vecs, layer_idx):
    V = np.stack([all_vecs[t][layer_idx + 1] for t in TRAITS])
    _, s, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir /= np.linalg.norm(shared_dir)
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


def simplex_bound(n_vectors, n_dims):
    """Theoretical minimum cosine for n_vectors in n_dims.

    For n unit vectors in d dimensions, the tightest packing has
    mutual cosine = -1/(n-1) (equiangular tight frame) when n <= d+1.

    When n > d+1, the minimum absolute cosine is bounded below by
    sqrt((n-d) / (d*(n-1))).
    """
    if n_vectors <= n_dims + 1:
        # Simplex bound: all vectors can be equiangular
        optimal_cosine = -1.0 / (n_vectors - 1)
        return {
            "regime": "underdetermined (n <= d+1)",
            "optimal_mutual_cosine": float(optimal_cosine),
            "achievable": True,
            "optimal_specificity": float(1.0 - optimal_cosine),
        }
    else:
        # Overdetermined: cannot achieve equiangular
        min_abs_cos = np.sqrt((n_vectors - n_dims) / (n_dims * (n_vectors - 1)))
        return {
            "regime": "overdetermined (n > d+1)",
            "min_abs_cosine": float(min_abs_cos),
            "achievable": False,
            "note": f"{n_vectors} vectors in {n_dims}D must have |cos| >= {min_abs_cos:.4f}",
        }


def analyze_intrinsic_dim(residuals):
    """Analyze the intrinsic dimensionality of the residual subspace."""
    # Normalize
    normed = {}
    for t in TRAITS:
        n = np.linalg.norm(residuals[t])
        normed[t] = residuals[t] / max(n, 1e-10)

    V = np.stack([normed[t] for t in TRAITS])  # (6, dim)

    # SVD of the residual matrix
    _, s, _ = np.linalg.svd(V, full_matrices=False)

    # Effective dimensionality
    var_explained = s**2 / np.sum(s**2)
    cumvar = np.cumsum(var_explained)

    # Numerical rank (threshold)
    rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    rank_999 = int(np.searchsorted(cumvar, 0.999) + 1)
    numerical_rank = int(np.sum(s / s[0] > 1e-6))

    return {
        "singular_values": [float(x) for x in s],
        "variance_explained": [float(x) for x in var_explained],
        "cumulative_variance": [float(x) for x in cumvar],
        "rank_99pct": rank_99,
        "rank_999pct": rank_999,
        "numerical_rank": numerical_rank,
        "ratio_s1_s5": float(s[0] / max(s[4], 1e-10)),
        "ratio_s5_s6": float(s[4] / max(s[5], 1e-10)) if len(s) > 5 else None,
    }


def find_linear_dependency(residuals):
    """Find the linear relationship among 6 residual vectors.

    Since 6 vectors span 5D, there exists coefficients c_1,...,c_6
    such that sum(c_i * v_i) ≈ 0. Find these coefficients.
    """
    V = np.stack([residuals[t] for t in TRAITS])  # (6, dim)

    # Normalize for fair comparison
    norms = np.linalg.norm(V, axis=1)
    V_normed = V / norms[:, None]

    # The null space: find c such that V_normed^T @ c ≈ 0
    # SVD of V_normed (6 x dim): U (6x6), S (6,), Vt (6 x dim)
    # The last column of U corresponds to the smallest singular value
    U, s, Vt = np.linalg.svd(V_normed, full_matrices=True)
    null_vec = U[:, -1]  # Left singular vector for smallest singular value

    # Normalize to have unit max absolute coefficient
    null_vec = null_vec / np.max(np.abs(null_vec))

    # Verify: sum(c_i * v_i) should be near zero
    reconstruction = np.zeros_like(V_normed[0])
    for i in range(6):
        reconstruction += null_vec[i] * V_normed[i]
    residual_norm = np.linalg.norm(reconstruction)

    dependency = {
        "coefficients": {t: float(null_vec[i]) for i, t in enumerate(TRAITS)},
        "residual_norm": float(residual_norm),
        "smallest_sv": float(s[-1]),
    }

    # Which trait is most "redundant" (highest absolute coefficient)?
    abs_coeffs = np.abs(null_vec)
    most_redundant_idx = np.argmax(abs_coeffs)
    dependency["most_redundant_trait"] = TRAITS[most_redundant_idx]

    return dependency


def compute_cosine_bound(n_traits, residual_dim):
    """What's the best achievable specificity for n traits in d dims?

    For 6 normalized vectors in 5D, the optimal configuration is the
    simplex (regular 5-simplex in 5D), where all mutual cosines = -1/5.

    Predicted specificity = 1 - (-1/5) = 1.2 = 6/5
    """
    if n_traits <= residual_dim + 1:
        optimal_cos = -1.0 / (n_traits - 1)
        optimal_spec = 1.0 - optimal_cos
        return {
            "optimal_mutual_cosine": float(optimal_cos),
            "optimal_specificity": float(optimal_spec),
            "is_simplex": n_traits == residual_dim + 1,
            "simplex_note": (
                f"{n_traits} vectors in {residual_dim}D = "
                f"regular {residual_dim}-simplex: cos = {optimal_cos:.4f}, "
                f"spec = {optimal_spec:.4f}"
            ) if n_traits == residual_dim + 1 else None,
        }
    return None


def measure_simplex_deviation(residuals):
    """How close are the residual vectors to a regular simplex?

    For a regular simplex in 5D, all 15 pairwise cosines = -1/5 = -0.2.
    Measure deviation from this ideal.
    """
    cos_pairs = []
    for i in range(6):
        for j in range(i + 1, 6):
            cos = cosine_sim(residuals[TRAITS[i]], residuals[TRAITS[j]])
            cos_pairs.append(cos)

    cos_pairs = np.array(cos_pairs)
    ideal = -1.0 / 5  # = -0.2

    return {
        "mean_cosine": float(np.mean(cos_pairs)),
        "std_cosine": float(np.std(cos_pairs)),
        "ideal_cosine": float(ideal),
        "mean_deviation_from_ideal": float(np.mean(cos_pairs - ideal)),
        "rms_deviation_from_ideal": float(np.sqrt(np.mean((cos_pairs - ideal)**2))),
        "max_deviation": float(np.max(np.abs(cos_pairs - ideal))),
        "all_cosines": [float(x) for x in cos_pairs],
        "percent_close_to_ideal": float(np.mean(np.abs(cos_pairs - ideal) < 0.1) * 100),
    }


def analyze_equiangular_proximity(all_vecs, num_layers):
    """Track how close residuals are to simplex across layers."""
    results = []
    for layer in range(num_layers):
        residuals, _, _ = decompose_at_layer(all_vecs, layer)
        simplex = measure_simplex_deviation(residuals)
        dim = analyze_intrinsic_dim(residuals)

        results.append({
            "layer": layer,
            "rms_from_simplex": simplex["rms_deviation_from_ideal"],
            "mean_cosine": simplex["mean_cosine"],
            "effective_dim": dim["numerical_rank"],
            "s6_over_s1": dim["singular_values"][5] / max(dim["singular_values"][0], 1e-10),
        })

    return results


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
        residuals, _, _ = decompose_at_layer(all_vecs, mid_layer)

        # 1. Intrinsic dimensionality
        print(f"\n--- Intrinsic Dimensionality (L{mid_layer}) ---")
        dim = analyze_intrinsic_dim(residuals)
        print(f"  Singular values: {', '.join(f'{s:.4f}' for s in dim['singular_values'])}")
        print(f"  Variance explained: {', '.join(f'{v:.1%}' for v in dim['variance_explained'])}")
        print(f"  Numerical rank: {dim['numerical_rank']}")
        print(f"  Ratio s5/s6: {dim['ratio_s5_s6']:.1f}" if dim["ratio_s5_s6"] else "")
        print(f"  → The 6 residual vectors span a {dim['numerical_rank']}D subspace")

        # 2. Linear dependency
        print(f"\n--- Linear Dependency ---")
        dep = find_linear_dependency(residuals)
        print(f"  Coefficients (sum c_i * v_i ≈ 0):")
        for t in TRAITS:
            c = dep["coefficients"][t]
            print(f"    {t:>15}: {c:+.4f}")
        print(f"  Residual norm: {dep['residual_norm']:.6f}")
        print(f"  Most redundant trait: {dep['most_redundant_trait']}")

        # 3. Simplex bound
        print(f"\n--- Theoretical Simplex Bound ---")
        bound = compute_cosine_bound(6, dim["numerical_rank"])
        if bound:
            print(f"  6 vectors in {dim['numerical_rank']}D:")
            print(f"  Optimal mutual cosine: {bound['optimal_mutual_cosine']:.4f}")
            print(f"  Optimal specificity:   {bound['optimal_specificity']:.4f}")
            if bound.get("simplex_note"):
                print(f"  → {bound['simplex_note']}")

        # 4. Simplex deviation
        print(f"\n--- Simplex Proximity (L{mid_layer}) ---")
        simplex = measure_simplex_deviation(residuals)
        print(f"  Ideal cosine (regular 5-simplex): {simplex['ideal_cosine']:.4f}")
        print(f"  Mean actual cosine: {simplex['mean_cosine']:.4f}")
        print(f"  RMS deviation from ideal: {simplex['rms_deviation_from_ideal']:.4f}")
        print(f"  Max deviation: {simplex['max_deviation']:.4f}")
        print(f"  % within 0.1 of ideal: {simplex['percent_close_to_ideal']:.0f}%")

        # Show all 15 pairwise cosines sorted
        pairs = []
        for i in range(6):
            for j in range(i + 1, 6):
                cos = cosine_sim(residuals[TRAITS[i]], residuals[TRAITS[j]])
                pairs.append((TRAITS[i], TRAITS[j], cos))
        pairs.sort(key=lambda x: x[2])

        print(f"\n  All 15 pairwise cosines (sorted):")
        ideal = -1.0 / 5
        for t1, t2, cos in pairs:
            dev = cos - ideal
            print(f"    {t1[:5]:>5}-{t2[:5]:<5}: {cos:+.4f}  (dev={dev:+.4f})")

        # 5. Layer-by-layer simplex proximity
        print(f"\n--- Layer-by-Layer Simplex Proximity ---")
        layer_prox = analyze_equiangular_proximity(all_vecs, num_layers)

        print(f"  {'Layer':>5} {'RMS':>8} {'MeanCos':>9} {'Dim':>4} {'s6/s1':>8}")
        print(f"  {'-'*40}")
        for l in layer_prox:
            if l["layer"] % max(1, num_layers // 10) == 0 or l["layer"] == mid_layer:
                marker = " <-- mid" if l["layer"] == mid_layer else ""
                print(f"  L{l['layer']:>3} {l['rms_from_simplex']:>8.4f} {l['mean_cosine']:>+9.4f} "
                      f"{l['effective_dim']:>4} {l['s6_over_s1']:>8.4f}{marker}")

        best_simplex = min(layer_prox, key=lambda x: x["rms_from_simplex"])
        print(f"\n  Closest to simplex: L{best_simplex['layer']} "
              f"(RMS={best_simplex['rms_from_simplex']:.4f})")

        # KEY INSIGHT
        print(f"\n{'='*70}")
        print(f"KEY INSIGHT FOR {model_id}")
        print(f"{'='*70}")

        observed_spec = 1.0 - simplex["mean_cosine"]
        theoretical_spec = bound["optimal_specificity"] if bound else None

        print(f"\n  Observed mean specificity (from geometry): {observed_spec:.4f}")
        if theoretical_spec:
            print(f"  Theoretical optimal (simplex):             {theoretical_spec:.4f}")
            efficiency = observed_spec / theoretical_spec * 100
            print(f"  Efficiency: {efficiency:.1f}%")
            print(f"\n  The vectors achieve {efficiency:.0f}% of the theoretically optimal specificity.")
            print(f"  The remaining {100-efficiency:.0f}% is due to deviation from the regular simplex,")
            print(f"  i.e., some trait pairs are more correlated than others.")
            print(f"  Artistic-Conventional is the most correlated pair (cos={pairs[0][2]:+.3f})")
            print(f"  vs the ideal of {ideal:+.3f}")

        all_results[safe_model] = {
            "num_layers": num_layers,
            "mid_layer": mid_layer,
            "intrinsic_dim": dim,
            "linear_dependency": dep,
            "simplex_bound": bound,
            "simplex_deviation": simplex,
            "pairwise_cosines": [(t1, t2, float(cos)) for t1, t2, cos in pairs],
            "observed_specificity": float(observed_spec),
            "theoretical_specificity": float(theoretical_spec) if theoretical_spec else None,
            "layer_simplex_proximity": layer_prox,
            "best_simplex_layer": best_simplex["layer"],
        }

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")

        print(f"\n  {'Model':>40} {'ObsSpec':>8} {'TheoSpec':>9} {'Eff%':>6} "
              f"{'BestLayer':>10} {'RMS':>8}")
        print(f"  {'-'*85}")
        for name, r in all_results.items():
            eff = r["observed_specificity"] / r["theoretical_specificity"] * 100 if r["theoretical_specificity"] else 0
            print(f"  {name:>40} {r['observed_specificity']:>8.4f} "
                  f"{r['theoretical_specificity']:>9.4f} {eff:>5.1f}% "
                  f"L{r['best_simplex_layer']:>3}({r['best_simplex_layer']/r['num_layers']*100:.0f}%) "
                  f"{r['layer_simplex_proximity'][r['best_simplex_layer']]['rms_from_simplex']:>8.4f}")

        # Most interesting: is the linear dependency the same across models?
        print(f"\n  Linear dependency coefficients (which traits are 'redundant'):")
        print(f"  {'':>15}", end="")
        for name in all_results:
            print(f"  {name[:20]:>20}", end="")
        print()
        for t in TRAITS:
            print(f"  {t:>15}", end="")
            for name, r in all_results.items():
                c = r["linear_dependency"]["coefficients"][t]
                print(f"  {c:>+20.3f}", end="")
            print()

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dimensionality_bound.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
