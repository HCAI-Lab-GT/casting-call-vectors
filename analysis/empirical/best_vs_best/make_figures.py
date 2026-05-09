"""
Figures for best-vs-best comparison.

  Fig 1: Scatter — steered peak score vs AA peak score per role (diagonal = tie)
  Fig 2: Advantage histogram — distribution of (steered_peak - aa_peak)
  Fig 3: Peak alpha heatmap — which alpha each method peaks at, per role (sorted)

Usage:
    python analysis/empirical/best_vs_best/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STEERED_COLOR = "#2ecc71"
AXIS_COLOR = "#3498db"
ADV_COLOR = "#e67e22"


def _save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {path}")


def fig1_peak_scatter(bvb: pd.DataFrame, stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    win = bvb[bvb["steered_wins"]]
    lose = bvb[~bvb["steered_wins"]]

    ax.scatter(win["aa_peak_score"], win["steered_peak_score"],
               color=STEERED_COLOR, s=18, alpha=0.6, label=f"Steered wins (n={len(win)})")
    ax.scatter(lose["aa_peak_score"], lose["steered_peak_score"],
               color=AXIS_COLOR, s=18, alpha=0.6, label=f"AA wins (n={len(lose)})")

    lims = [0, 100]
    ax.plot(lims, lims, color="gray", linewidth=1.2, linestyle="--", label="Tie line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    row = stats.iloc[0]
    ax.text(0.05, 0.95,
            f"Mean advantage: {row['mean_advantage']:+.1f}\n"
            f"Win rate: {row['win_rate']:.1%}\n"
            f"p = {row['p_value']:.1e}  r = {row['effect_r']:.2f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    ax.set_xlabel("AA peak score (best α for AA, 0–100)")
    ax.set_ylabel("Steered peak score (best α for steered, 0–100)")
    ax.set_title("Best-vs-best: each method at its natural peak alpha\n"
                 "(AA free to use α=1.5 if that's when it peaks)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig1_peak_scatter.pdf")


def fig2_advantage_histogram(bvb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    adv = bvb["peak_advantage"].values
    bins = np.linspace(adv.min() - 1, adv.max() + 1, 35)
    ax.hist(adv, bins=bins, color=ADV_COLOR, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="No advantage")
    ax.axvline(adv.mean(), color="darkred", linewidth=2,
               label=f"Mean = {adv.mean():+.1f}")

    win_pct = (adv > 0).mean()
    ax.text(0.97, 0.95, f"{win_pct:.1%} of roles:\nsteered peak > AA peak",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel("Advantage: steered peak score − AA peak score")
    ax.set_ylabel("Number of roles")
    ax.set_title("Distribution of best-vs-best advantage across 275 roles\n"
                 "(both methods using their optimal alpha)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig2_advantage_histogram.pdf")


def fig3_peak_alpha_comparison(bvb: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    alphas = [1.0, 1.5, 2.0, 2.5]
    alpha_labels = [f"α={a}" for a in alphas]

    for ax, col, color, label in [
        (axes[0], "steered_peak_alpha", STEERED_COLOR, "Steered"),
        (axes[1], "aa_peak_alpha", AXIS_COLOR, "Assistant Axis"),
    ]:
        counts = bvb[col].value_counts().reindex(alphas, fill_value=0)
        bars = ax.bar(alpha_labels, counts.values, color=color, alpha=0.85, edgecolor="white")
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{count}\n({count/len(bvb):.0%})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{label}: which alpha is its peak?")
        ax.set_ylabel("Number of roles")
        ax.set_ylim(0, len(bvb) * 1.1)

    fig.suptitle("Peak alpha distribution — where does each method score highest?",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig3_peak_alpha_distribution.pdf")


def main() -> None:
    bvb = pd.read_csv(DATA_DIR / "best_vs_best.csv")
    stats_df = pd.read_csv(DATA_DIR / "best_vs_best_stats.csv")
    print(f"Loaded {len(bvb)} roles")

    fig1_peak_scatter(bvb, stats_df)
    fig2_advantage_histogram(bvb)
    fig3_peak_alpha_comparison(bvb)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
