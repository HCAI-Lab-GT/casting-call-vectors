"""
Gap-closing trajectory: how much of the distance to baseline does each method close?

Baseline (gold-standard prompt engineering) scores ~89.0 on the role alignment metric.
For each role and alpha, we compute the fraction of the gap to baseline that has been
closed relative to the starting point (α=1.0):

  gap_closed(role, alpha, method) =
      (score(role, alpha, method) - score(role, α=1.0, method))
      / (BASELINE - score(role, α=1.0, method))

Positive = moving toward baseline.
Negative = moving away from baseline (getting worse as alpha increases).

Steered closes the gap monotonically.
AA closes it slightly at α=1.5 then reverses — at α=2.5 it is further from
baseline than it was at α=1.0 for most roles.

Outputs (data/):
  gap_closing.csv        per-role x alpha: gap_closed fraction for both methods
  gap_closing_summary.csv  mean gap_closed per alpha per method
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

BASELINE = 89.0
ALPHAS = [1.0, 1.5, 2.0, 2.5]


def main() -> None:
    curves = pd.read_csv(CURVES_CSV)
    print(f"Loaded {curves['role'].nunique()} roles, baseline={BASELINE}")

    # Get each role's starting score at α=1.0
    start = curves[curves["alpha"] == 1.0][["role", "steered_score", "aa_score"]].copy()
    start = start.rename(columns={"steered_score": "steered_start",
                                   "aa_score": "aa_start"})
    merged = curves.merge(start, on="role")

    rows = []
    for _, row in merged.iterrows():
        for method, score_col, start_col in [
            ("steered", "steered_score", "steered_start"),
            ("assistant_axis", "aa_score", "aa_start"),
        ]:
            score = row[score_col]
            start_score = row[start_col]
            gap_total = BASELINE - start_score
            if abs(gap_total) < 1e-6:
                gap_closed = float("nan")
            else:
                gap_closed = (score - start_score) / gap_total

            rows.append(dict(
                role=row["role"],
                alpha=row["alpha"],
                method=method,
                score=float(score),
                start_score=float(start_score),
                gap_to_baseline=float(BASELINE - score),
                gap_closed=float(gap_closed),
            ))

    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(DATA_DIR / "gap_closing.csv", index=False)

    # Summary
    summary = gap_df.groupby(["method", "alpha"])["gap_closed"].agg(
        mean="mean", median="median", std="std",
        pct_positive=lambda x: (x > 0).mean(),
        pct_exceeds_one=lambda x: (x > 1.0).mean(),
    ).reset_index()
    summary.to_csv(DATA_DIR / "gap_closing_summary.csv", index=False)

    print("\nGap-closing summary (mean fraction of gap to baseline closed):")
    for _, row in summary.iterrows():
        print(f"  {row['method']:14s}  α={row['alpha']}:  "
              f"mean={row['mean']:+.3f}  median={row['median']:+.3f}  "
              f"pct_moving_toward_baseline={row['pct_positive']:.1%}")

    # Paired test at α=2.5: who closes more gap?
    s25 = gap_df[(gap_df["alpha"] == 2.5) & (gap_df["method"] == "steered")].set_index("role")["gap_closed"]
    a25 = gap_df[(gap_df["alpha"] == 2.5) & (gap_df["method"] == "assistant_axis")].set_index("role")["gap_closed"]
    common = s25.index.intersection(a25.index)
    diff = (s25.loc[common] - a25.loc[common]).dropna().to_numpy()
    stat, p = stats.wilcoxon(diff, alternative="greater")
    print(f"\nPaired Wilcoxon (steered gap_closed > AA gap_closed) at α=2.5: "
          f"p={p:.2e}, n={len(diff)}")

    print(f"\nSaved to {DATA_DIR}")


if __name__ == "__main__":
    main()
