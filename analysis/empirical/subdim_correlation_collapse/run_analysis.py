"""
Subdimension inter-correlation collapse analysis.

At each alpha, compute the 5x5 Spearman correlation matrix between subdimension
scores *across roles* for each method. The hypothesis:

  - At high alpha, AA's 5 subdimensions become perfectly correlated — they all
    collapse together driven by the same axis failure. The behavioural space
    collapses to a single dimension.
  - Steered maintains independent variation across subdimensions — different roles
    remain distinct on different dimensions.

This is a new angle on collapse: not just "scores go down" but "all behavioural
dimensions fuse into one."

Outputs (data/):
  subdim_corr_matrices.csv   per-method, per-alpha: full 5x5 correlation matrices (long form)
  subdim_mean_intercorr.csv  per-method, per-alpha: mean off-diagonal |r|
  subdim_pca_variance.csv    fraction of variance explained by PC1 (collapse indicator)
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
SUBDIM_CSV = (
    REPO
    / "analysis"
    / "empirical"
    / "subdimension_advantage"
    / "data"
    / "subdim_long.csv"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]
SUBDIMS = ["emotional_register", "vocab_choice", "social_dynamic",
           "motivation", "worldview_alignment"]


def mean_off_diagonal_r(corr_mat: np.ndarray) -> float:
    n = corr_mat.shape[0]
    off_diag = [abs(corr_mat[i, j]) for i, j in combinations(range(n), 2)]
    return float(np.mean(off_diag))


def pc1_variance_fraction(mat: np.ndarray) -> float:
    mat_c = mat - mat.mean(axis=0)
    _, s, _ = np.linalg.svd(mat_c, full_matrices=False)
    var = s ** 2
    return float(var[0] / var.sum()) if var.sum() > 0 else float("nan")


def main() -> None:
    df = pd.read_csv(SUBDIM_CSV)
    print(f"Loaded {df['role'].nunique()} roles x {df['alpha'].nunique()} alphas "
          f"x {df['subdim'].nunique()} subdims")

    corr_rows = []
    mean_rows = []
    pca_rows = []

    for method, score_col in [("steered", "steered_mean"), ("assistant_axis", "aa_mean")]:
        for alpha in ALPHAS:
            sub = df[df["alpha"] == alpha].copy()

            # Build roles x subdims matrix
            pivot = sub.pivot(index="role", columns="subdim", values=score_col)[SUBDIMS]
            mat = pivot.to_numpy(dtype=float)

            # Spearman correlation between subdimensions (across roles)
            corr_mat = np.zeros((5, 5))
            for i in range(5):
                for j in range(5):
                    if i == j:
                        corr_mat[i, j] = 1.0
                    else:
                        r, _ = stats.spearmanr(mat[:, i], mat[:, j])
                        corr_mat[i, j] = float(r)

            # Save long-form correlations
            for i, sd_i in enumerate(SUBDIMS):
                for j, sd_j in enumerate(SUBDIMS):
                    corr_rows.append(dict(
                        method=method,
                        alpha=alpha,
                        subdim_i=sd_i,
                        subdim_j=sd_j,
                        spearman_r=corr_mat[i, j],
                    ))

            mean_r = mean_off_diagonal_r(corr_mat)
            pc1_frac = pc1_variance_fraction(mat)

            mean_rows.append(dict(
                method=method,
                alpha=alpha,
                mean_abs_intercorr=mean_r,
                pc1_variance_fraction=pc1_frac,
            ))
            pca_rows.append(dict(method=method, alpha=alpha, pc1_var=pc1_frac))

            print(f"  {method:14s}  α={alpha}:  mean|r|={mean_r:.3f}  "
                  f"PC1 var={pc1_frac:.3f}")

    pd.DataFrame(corr_rows).to_csv(DATA_DIR / "subdim_corr_matrices.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(DATA_DIR / "subdim_mean_intercorr.csv", index=False)
    print(f"\nSaved to {DATA_DIR}")


if __name__ == "__main__":
    main()
