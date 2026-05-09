"""
Paired statistical tests and bootstrap confidence intervals on steered vs AA advantage.

For each of the 4 alpha values, across 275 roles:
  - Wilcoxon signed-rank test on (steered_score - aa_score) vs zero
  - Effect size r = Z / sqrt(N)
  - Bootstrap CI (10,000 resamples) on mean advantage

This directly addresses the generalization concern: the sampling unit is roles (n=275),
not alpha values (n=4). At each alpha, we have a fully powered paired test.

Outputs (data/):
  paired_tests.csv       Wilcoxon statistic, p-value, effect size per alpha
  bootstrap_ci.csv       mean advantage + 95% CI per alpha (bootstrap)
  per_role_advantage.csv steered - aa score for every role x alpha
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

ALPHAS = [1.0, 1.5, 2.0, 2.5]
N_BOOT = 10_000
RNG_SEED = 42


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    curves = pd.read_csv(CURVES_CSV)
    curves["advantage"] = curves["steered_score"] - curves["aa_score"]
    print(f"Loaded {curves['role'].nunique()} roles x {curves['alpha'].nunique()} alphas")

    paired_rows = []
    boot_rows = []

    for alpha in ALPHAS:
        sub = curves[curves["alpha"] == alpha].copy()
        adv = sub["advantage"].to_numpy(dtype=float)
        adv = adv[np.isfinite(adv)]
        n = len(adv)

        stat, p = stats.wilcoxon(adv, alternative="greater")
        # Effect size r = Z / sqrt(N); approximate Z from normal.
        # Clip p away from 0/1 to avoid ppf returning ±inf.
        p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
        z = stats.norm.ppf(1 - p_clipped)
        effect_r = float(z / np.sqrt(n))

        win_rate = float((adv > 0).mean())
        mean_adv = float(adv.mean())
        median_adv = float(np.median(adv))

        paired_rows.append(dict(
            alpha=alpha,
            n=n,
            mean_advantage=mean_adv,
            median_advantage=median_adv,
            win_rate=win_rate,
            wilcoxon_stat=float(stat),
            p_value=float(p),
            effect_r=effect_r,
        ))

        mean_ci, lo, hi = bootstrap_mean_ci(adv, N_BOOT, RNG_SEED)
        boot_rows.append(dict(
            alpha=alpha,
            n=n,
            mean_advantage=mean_ci,
            ci_low=lo,
            ci_high=hi,
            ci_entirely_positive=int(lo > 0),
        ))

        print(f"  α={alpha}: mean={mean_adv:+.2f}  win_rate={win_rate:.1%}  "
              f"p={p:.2e}  r={effect_r:.3f}  95%CI=[{lo:+.2f}, {hi:+.2f}]")

    pd.DataFrame(paired_rows).to_csv(DATA_DIR / "paired_tests.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(DATA_DIR / "bootstrap_ci.csv", index=False)

    per_role = curves[["role", "alpha", "steered_score", "aa_score", "advantage"]]
    per_role.to_csv(DATA_DIR / "per_role_advantage.csv", index=False)

    print(f"\nSaved to {DATA_DIR}")


if __name__ == "__main__":
    main()
