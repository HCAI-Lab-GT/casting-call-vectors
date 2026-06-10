"""
Figures for paired significance analysis.

  Fig 1: Bootstrap CI on mean advantage per alpha — error bars showing CIs entirely > 0
  Fig 2: Advantage distribution per alpha — violin/box plots for all 4 alphas
  Fig 3: Win rate per alpha — fraction of roles where steered > AA
  Fig 4: Effect size (r) per alpha — bar chart

Usage:
    python analysis/empirical/paired_significance/make_figures.py
"""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
plot_style.apply_style()

from plot_style import STEERED, AA, BASELINE, WIN, ACCENT, REPR, ALPHA_COLORS
STEERED_COLOR = STEERED
AXIS_COLOR = AA
ADV_COLOR = ACCENT

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]


def _save(fig, name: str) -> None:
    stem = name.replace(".pdf", "").replace(".png", "")
    plot_style.save_fig(fig, FIG_DIR, stem)


def fig1_bootstrap_ci(boot: pd.DataFrame, tests: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    alphas = boot["alpha"].tolist()
    means = boot["mean_advantage"].tolist()
    los = boot["ci_low"].tolist()
    his = boot["ci_high"].tolist()
    errs_lo = [m - l for m, l in zip(means, los)]
    errs_hi = [h - m for h, m in zip(his, means)]

    ax.bar(alphas, means, width=0.3, color=ADV_COLOR, alpha=0.8, edgecolor="white", zorder=3)
    ax.errorbar(alphas, means, yerr=[errs_lo, errs_hi],
                fmt="none", color="black", capsize=6, linewidth=2, zorder=4)

    for alpha, lo, hi, mean in zip(alphas, los, his, means):
        color = WIN if lo > 0 else "#B2182B"
        ax.text(alpha, hi + 1.5, f"[{lo:+.1f}, {hi:+.1f}]",
                ha="center", va="bottom", fontsize=8, color=color)

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Mean advantage (steered − AA score)")
    ax.set_title("Bootstrap 95% CI on mean steered − AA advantage (n=275 roles)")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"α={a}" for a in alphas])

    p_vals = tests.set_index("alpha")["p_value"]
    for alpha, mean in zip(alphas, means):
        p = p_vals.get(alpha, 1.0)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(alpha, mean / 2, sig, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white")

    plt.tight_layout()
    _save(fig, "fig1_bootstrap_ci.pdf")


def fig2_advantage_distributions(per_role: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    data = [per_role[per_role["alpha"] == a]["advantage"].dropna().tolist() for a in ALPHAS]
    positions = ALPHAS
    parts = ax.violinplot(data, positions=positions, widths=0.3,
                          showmedians=True, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor(ADV_COLOR)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    for i, (alpha, d) in enumerate(zip(ALPHAS, data)):
        arr = np.array(d)
        ax.scatter([alpha] * len(arr), arr, s=4, color="gray", alpha=0.3, zorder=2)
        pct_pos = (arr > 0).mean()
        y_min = min(arr.min(), -5)
        ax.text(alpha, y_min - 3, f"{pct_pos:.0%}\nwin",
                ha="center", va="top", fontsize=8)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Advantage (steered − AA score, 0–100)")
    ax.set_title("Per-role advantage distribution at each alpha (275 roles)")
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels([f"α={a}" for a in ALPHAS])
    plt.tight_layout()
    _save(fig, "fig2_advantage_distributions.pdf")


def fig3_win_rate(tests: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    alphas = tests["alpha"].tolist()
    win_rates = tests["win_rate"].tolist()

    bars = ax.bar(alphas, win_rates, width=0.3, color=WIN,
                  alpha=0.85, edgecolor="white")
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="gray", linewidth=1.2, linestyle="--", label="50% (chance)")
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Fraction of roles where steered > AA")
    ax.set_title("Win rate: fraction of 275 roles where steered beats AA")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_ylim(0, 1.1)
    plot_style.legend_above(ax, ncol=1)
    plt.tight_layout()
    _save(fig, "fig3_win_rate.pdf")


def fig4_effect_size(tests: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    alphas = tests["alpha"].tolist()
    rs = tests["effect_r"].tolist()
    p_vals = tests["p_value"].tolist()

    bars = ax.bar(alphas, rs, width=0.3, color=ADV_COLOR, alpha=0.85, edgecolor="white")

    thresholds = [(0.5, "large", WIN), (0.3, "medium", "olive"), (0.1, "small", "gray")]
    for thresh, label, color in thresholds:
        ax.axhline(thresh, color=color, linewidth=1, linestyle="--",
                   label=f"r={thresh} ({label} effect)")

    for bar, r, p in zip(bars, rs, p_vals):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"r={r:.2f}\n{sig}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Effect size r (Wilcoxon signed-rank)")
    ax.set_title("Effect size of steered > AA advantage per alpha (n=275 roles)")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_ylim(0, max(rs) * 1.35)
    plot_style.legend_above(ax, ncol=3)
    plt.tight_layout()
    _save(fig, "fig4_effect_size.pdf")


def main() -> None:
    boot = pd.read_csv(DATA_DIR / "bootstrap_ci.csv")
    tests = pd.read_csv(DATA_DIR / "paired_tests.csv")
    per_role = pd.read_csv(DATA_DIR / "per_role_advantage.csv")

    fig1_bootstrap_ci(boot, tests)
    fig2_advantage_distributions(per_role)
    fig3_win_rate(tests)
    fig4_effect_size(tests)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
