"""
Figures for subdimension inter-correlation collapse.

  Fig 1: Mean inter-subdimension correlation vs alpha — steered vs AA line plot
  Fig 2: PC1 variance fraction vs alpha — steered vs AA (collapse = PC1 → 1.0)
  Fig 3: Correlation matrices at alpha 1.0 and 2.5 — 2x2 heatmap grid
  Fig 4: Combined panel summary

Usage:
    python analysis/empirical/subdim_correlation_collapse/make_figures.py
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

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]
SUBDIMS = ["emotional_register", "vocab_choice", "social_dynamic",
           "motivation", "worldview_alignment"]
SUBDIM_LABELS = ["Emotional\nRegister", "Vocab\nChoice", "Social\nDynamic",
                 "Motivation", "Worldview\nAlignment"]


def _save(fig, name: str) -> None:
    stem = name.replace(".pdf", "").replace(".png", "")
    plot_style.save_fig(fig, FIG_DIR, stem)


def fig1_mean_intercorr(mean_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, color, label, ls in [
        ("steered", STEERED_COLOR, "Steered (proposed)", "-o"),
        ("assistant_axis", AXIS_COLOR, "Assistant Axis", "--s"),
    ]:
        sub = mean_df[mean_df["method"] == method].sort_values("alpha")
        ax.plot(sub["alpha"], sub["mean_abs_intercorr"], ls, color=color,
                linewidth=2.5, markersize=8, label=label)
        for _, row in sub.iterrows():
            ax.text(row["alpha"], row["mean_abs_intercorr"] + 0.008,
                    f'{row["mean_abs_intercorr"]:.3f}',
                    ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Mean |Spearman r| between subdimensions")
    ax.set_title("Subdimension inter-correlation vs steering strength")
    ax.set_xticks(ALPHAS)
    ax.set_ylim(0, 1.05)
    plot_style.legend_above(ax, ncol=2)
    plt.tight_layout()
    _save(fig, "fig1_mean_intercorrelation.pdf")


def fig2_pc1_variance(mean_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, color, label, ls in [
        ("steered", STEERED_COLOR, "Steered (proposed)", "-o"),
        ("assistant_axis", AXIS_COLOR, "Assistant Axis", "--s"),
    ]:
        sub = mean_df[mean_df["method"] == method].sort_values("alpha")
        ax.plot(sub["alpha"], sub["pc1_variance_fraction"], ls, color=color,
                linewidth=2.5, markersize=8, label=label)
        for _, row in sub.iterrows():
            ax.text(row["alpha"], row["pc1_variance_fraction"] + 0.008,
                    f'{row["pc1_variance_fraction"]:.3f}',
                    ha="center", va="bottom", fontsize=8, color=color)

    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", label="Total collapse (PC1=100%)")
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Fraction of variance in PC1")
    ax.set_title("PC1 variance fraction: how 1-dimensional is the behavioral space?")
    ax.set_xticks(ALPHAS)
    ax.set_ylim(0, 1.1)
    plot_style.legend_above(ax, ncol=3)
    plt.tight_layout()
    _save(fig, "fig2_pc1_variance.pdf")


def fig3_corr_matrices(corr_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    pairs = [
        (0, 0, "steered", 1.0, "Steered — α=1.0"),
        (0, 1, "steered", 2.5, "Steered — α=2.5"),
        (1, 0, "assistant_axis", 1.0, "Assistant Axis — α=1.0"),
        (1, 1, "assistant_axis", 2.5, "Assistant Axis — α=2.5"),
    ]

    for row, col, method, alpha, title in pairs:
        ax = axes[row][col]
        sub = corr_df[(corr_df["method"] == method) & (corr_df["alpha"] == alpha)]
        mat = sub.pivot(index="subdim_i", columns="subdim_j", values="spearman_r")
        mat = mat.reindex(index=SUBDIMS, columns=SUBDIMS).to_numpy(dtype=float)

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(SUBDIM_LABELS, fontsize=8)
        ax.set_yticklabels(SUBDIM_LABELS, fontsize=8)
        ax.set_title(title)

        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if abs(mat[i, j]) < 0.7 else "white")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Subdimension Spearman correlation matrices across 275 roles",
                 y=1.01)
    plt.tight_layout()
    _save(fig, "fig3_corr_matrices.pdf")


def fig4_summary_panel(mean_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title_suffix in [
        (axes[0], "mean_abs_intercorr",
         "Mean |Spearman r| between subdimensions",
         "Inter-subdimension correlation"),
        (axes[1], "pc1_variance_fraction",
         "Fraction of variance explained by PC1",
         "PC1 dominance (behavioural dimensionality)"),
    ]:
        for method, color, label, ls in [
            ("steered", STEERED_COLOR, "Steered (proposed)", "-o"),
            ("assistant_axis", AXIS_COLOR, "Assistant Axis", "--s"),
        ]:
            sub = mean_df[mean_df["method"] == method].sort_values("alpha")
            ax.plot(sub["alpha"], sub[metric], ls, color=color,
                    linewidth=2.5, markersize=8, label=label)

        ax.set_xlabel(r"Steering strength ($\alpha$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)
        ax.set_xticks(ALPHAS)
        plot_style.legend_above(ax, ncol=2)

    fig.suptitle("Subdimension correlation collapse: does AA's behavioural space "
                 "fuse into a single dimension?", y=1.02)
    plt.tight_layout()
    _save(fig, "fig4_summary_panel.pdf")


def main() -> None:
    mean_df = pd.read_csv(DATA_DIR / "subdim_mean_intercorr.csv")
    corr_df = pd.read_csv(DATA_DIR / "subdim_corr_matrices.csv")

    fig1_mean_intercorr(mean_df)
    fig2_pc1_variance(mean_df)
    fig3_corr_matrices(corr_df)
    fig4_summary_panel(mean_df)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
