"""
Best-vs-best comparison: each method at its natural peak alpha.

For each of 275 roles, take the maximum score achieved across all 4 alphas for
steered and AA separately. Compare these peaks. This is the fairest single
comparison — AA is not penalised for high-alpha collapse; it gets to use α=1.5
if that's when it peaks. If steered wins even here, the result is airtight.

Outputs (data/):
  best_vs_best.csv        per-role peak scores, at which alpha, and advantage
  best_vs_best_stats.csv  aggregate statistics and paired test
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
CURVES_CSV = (
    REPO
    / "analysis"
    / "empirical"
    / "score_trajectory_comparison"
    / "data"
    / "alpha_curves_both.csv"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    curves = pd.read_csv(CURVES_CSV)
    print(f"Loaded {curves['role'].nunique()} roles")

    # Per-role peak scores
    steered_peak = (
        curves.groupby("role")
        .apply(lambda df: df.loc[df["steered_score"].idxmax(),
                                  ["alpha", "steered_score"]])
        .rename(columns={"alpha": "steered_peak_alpha",
                         "steered_score": "steered_peak_score"})
        .reset_index()
    )
    aa_peak = (
        curves.groupby("role")
        .apply(lambda df: df.loc[df["aa_score"].idxmax(),
                                  ["alpha", "aa_score"]])
        .rename(columns={"alpha": "aa_peak_alpha", "aa_score": "aa_peak_score"})
        .reset_index()
    )

    bvb = steered_peak.merge(aa_peak, on="role")
    bvb["peak_advantage"] = bvb["steered_peak_score"] - bvb["aa_peak_score"]
    bvb["steered_wins"] = bvb["peak_advantage"] > 0

    bvb.to_csv(DATA_DIR / "best_vs_best.csv", index=False)

    adv = bvb["peak_advantage"].to_numpy(dtype=float)
    stat, p = stats.wilcoxon(adv, alternative="greater")
    p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
    z = stats.norm.ppf(1 - p_clipped)
    effect_r = float(z / np.sqrt(len(adv)))

    print(f"\nBest-vs-best results (n={len(bvb)}):")
    print(f"  Steered mean peak score : {bvb['steered_peak_score'].mean():.2f}")
    print(f"  AA     mean peak score  : {bvb['aa_peak_score'].mean():.2f}")
    print(f"  Mean advantage          : {adv.mean():+.2f}")
    print(f"  Median advantage        : {float(np.median(adv)):+.2f}")
    print(f"  Win rate (steered > AA) : {bvb['steered_wins'].mean():.1%}")
    print(f"  Wilcoxon p              : {p:.2e}")
    print(f"  Effect size r           : {effect_r:.3f}")

    print(f"\n  Steered peak alpha dist : "
          + "  ".join(f"α={a}: {v}" for a, v in
                      bvb["steered_peak_alpha"].value_counts().sort_index().items()))
    print(f"  AA peak alpha dist      : "
          + "  ".join(f"α={a}: {v}" for a, v in
                      bvb["aa_peak_alpha"].value_counts().sort_index().items()))

    stats_df = pd.DataFrame([dict(
        n=len(bvb),
        steered_mean_peak=bvb["steered_peak_score"].mean(),
        aa_mean_peak=bvb["aa_peak_score"].mean(),
        mean_advantage=adv.mean(),
        median_advantage=float(np.median(adv)),
        win_rate=bvb["steered_wins"].mean(),
        wilcoxon_stat=float(stat),
        p_value=float(p),
        effect_r=effect_r,
    )])
    stats_df.to_csv(DATA_DIR / "best_vs_best_stats.csv", index=False)
    print(f"\nSaved to {DATA_DIR}")


if __name__ == "__main__":
    main()
