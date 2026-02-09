#!/usr/bin/env python
"""
Random baselines for two claims:
1) RIASEC vectors vs assistant-axis PCA components orthogonality.
2) RIASEC hexagon ordering (adjacent > alternate > opposite).

Run:
  .venv/bin/python scripts/marin/analysis/random_baselines.py
"""

from __future__ import annotations

import json

import numpy as np
import random_baselines_common as c
import random_baselines_hexagon as hex_base
import random_baselines_orthogonality as ortho


def main() -> None:
    traits_unit = c.load_riasec_unit_vectors()
    pcs, pca_artifact = c.load_pcs()

    print("PCA artifact keys:", list(pca_artifact.keys()))
    print("PCA components shape:", tuple(pca_artifact["components"].shape))

    cmp_npz = np.load(c.COMPARISON_NPZ_PATH)
    print("Comparison arrays keys:", list(cmp_npz.keys()))
    for k in cmp_npz.keys():
        print(" ", k, tuple(cmp_npz[k].shape), str(cmp_npz[k].dtype))

    obs_ortho = ortho.orthogonality_observed(traits_unit, pcs)
    print("\nObserved orthogonality:")
    print(" max_abs_cos:", obs_ortho["max_abs_cos"])
    print(" mean_abs_cos:", obs_ortho["mean_abs_cos"])
    print(" mean_subspace_r2:", obs_ortho["mean_subspace_r2"])

    mc, n_hex_monotonic = ortho.run_monte_carlo(pcs)
    null_summary = mc["summary"]

    n_comp = len(c.TRAITS) * c.N_PCS
    p_max_emp = float((np.sum(mc["max_abs_cos"] >= obs_ortho["max_abs_cos"]) + 1) / (c.N_TRIALS + 1))
    p_mean_emp = float((np.sum(mc["mean_abs_cos"] >= obs_ortho["mean_abs_cos"]) + 1) / (c.N_TRIALS + 1))
    p_r2_emp = float((np.sum(mc["mean_subspace_r2"] >= obs_ortho["mean_subspace_r2"]) + 1) / (c.N_TRIALS + 1))
    p_max_ana = ortho.analytic_p_max_abs_cos(obs_ortho["max_abs_cos"], c.DIMENSIONS, n_comp)

    print("\nOrthogonality Monte Carlo null quantiles:")
    print(" max_abs_cos:", null_summary["max_abs_cos"])
    print(" mean_abs_cos:", null_summary["mean_abs_cos"])
    print(" mean_subspace_r2:", null_summary["mean_subspace_r2"])
    print("\nOrthogonality p-values:")
    print(" max_abs_cos empirical:", p_max_emp)
    print(" max_abs_cos analytic:", p_max_ana)
    print(" mean_abs_cos empirical:", p_mean_emp)
    print(" mean_subspace_r2 empirical:", p_r2_emp)

    obs_hex = hex_base.hexagon_observed(traits_unit)
    n_perm_monotonic, p_perm = hex_base.hexagon_label_permutation(obs_hex["sim_matrix"])
    p_hex_mc = float(n_hex_monotonic / c.N_TRIALS)

    print("\nObserved hexagon category means:")
    print(" adjacent_mean:", obs_hex["adjacent_mean"])
    print(" alternate_mean:", obs_hex["alternate_mean"])
    print(" opposite_mean:", obs_hex["opposite_mean"])
    print(" is_monotonic:", obs_hex["is_monotonic"])
    print("\nHexagon permutation null:")
    print(" n_monotonic:", n_perm_monotonic, "p_value:", p_perm)
    print("\nHexagon random-vector null:")
    print(" n_monotonic:", n_hex_monotonic, "p_value:", p_hex_mc)

    out = {
        "orthogonality_baseline": {
            "dimensions": c.DIMENSIONS,
            "n_traits": len(c.TRAITS),
            "n_pcs": c.N_PCS,
            "n_comparisons": n_comp,
            "observed": {
                "max_abs_cos": obs_ortho["max_abs_cos"],
                "mean_abs_cos": obs_ortho["mean_abs_cos"],
                "mean_subspace_r2": obs_ortho["mean_subspace_r2"],
                "per_trait_subspace_r2": obs_ortho["per_trait_subspace_r2"],
            },
            "null_monte_carlo": {
                "n_trials": c.N_TRIALS,
                "max_abs_cos": null_summary["max_abs_cos"],
                "mean_abs_cos": null_summary["mean_abs_cos"],
                "mean_subspace_r2": null_summary["mean_subspace_r2"],
            },
            "p_values": {
                "max_abs_cos_empirical": p_max_emp,
                "max_abs_cos_analytic": p_max_ana,
                "mean_abs_cos_empirical": p_mean_emp,
                "mean_subspace_r2_empirical": p_r2_emp,
            },
        },
        "hexagon_baseline": {
            "observed": {
                "adjacent_mean": obs_hex["adjacent_mean"],
                "alternate_mean": obs_hex["alternate_mean"],
                "opposite_mean": obs_hex["opposite_mean"],
                "is_monotonic": obs_hex["is_monotonic"],
            },
            "label_permutation": {
                "n_permutations": 720,
                "n_monotonic": int(n_perm_monotonic),
                "p_value": float(p_perm),
            },
            "random_vector": {
                "n_trials": c.N_TRIALS,
                "n_monotonic": int(n_hex_monotonic),
                "p_value": float(p_hex_mc),
            },
        },
    }

    c.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(c.OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved JSON to:", str(c.OUTPUT_PATH))


if __name__ == "__main__":
    main()
