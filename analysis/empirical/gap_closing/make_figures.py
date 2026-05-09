"""
Figures for gap-closing trajectory analysis.

  Fig 1: Mean gap-closed fraction vs alpha — steered vs AA line plot
  Fig 2: Gap-closed distribution at α=2.5 — side-by-side violins/histograms
  Fig 3: Absolute score vs baseline — both methods + baseline line
  Fig 4: Per-role gap-closed heatmap — roles sorted by steered gap_closed

Usage:
    python analysis/empirical/gap_closing/make_figures.py
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

ALPHAS = [1.0, 1.5, 2.0, 2.5]
BASELINE = 89.0
STEERED_COLOR = "#2ecc71"
AXIS_COLOR = "#3498db"
BASELINE_COLOR = "#e74c3c"


def _save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {path}")


def fig1_mean_gap_closed(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, color, label, ls in [
        ("steered", STEERED_COLOR, "Steered (proposed)", "-o"),
        ("assistant_axis", AXIS_COLOR, "Assistant Axis", "--s"),
    ]:
        sub = summary[summary["method"] == method].sort_values("alpha")
        ax.plot(sub["alpha"], sub["mean"], ls, color=color,
                linewidth=2.5, markersize=8, label=label)
        for _, row in sub.iterrows():
            ax.text(row["alpha"], row["mean"] + 0.015,
                    f'{row["mean"]:+.2f}', ha="center", va="bottom",
                    fontsize=8, color=color)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--",
               label="No progress (α=1.0 baseline)")
    ax.axhline(1.0, color=BASELINE_COLOR, linewidth=1.2, linestyle=":",
               label="Gap fully closed (= gold standard)")

    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Mean fraction of gap to baseline closed")
    ax.set_title("Gap-closing trajectory: how much of the gap to\n"
                 "gold-standard performance does each method close?")
    ax.set_xticks(ALPHAS)
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig1_mean_gap_closed.pdf")


def fig2_distribution_at_peak(gap_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, method, color, label in [
        (axes[0], "steered", STEERED_COLOR, "Steered at α=2.5"),
        (axes[1], "assistant_axis", AXIS_COLOR, "Assistant Axis at α=2.5"),
    ]:
        vals = gap_df[(gap_df["alpha"] == 2.5) & (gap_df["method"] == method)]["gap_closed"].dropna()
        bins = np.linspace(vals.min() - 0.05, max(vals.max() + 0.05, 1.5), 35)
        ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="No change")
        ax.axvline(1.0, color=BASELINE_COLOR, linewidth=1.5, linestyle=":",
                   label="Baseline reached")
        ax.axvline(vals.mean(), color="black", linewidth=2,
                   label=f"Mean = {vals.mean():+.2f}")
        pct_pos = (vals > 0).mean()
        ax.text(0.97, 0.95, f"{pct_pos:.1%} moving\ntoward baseline",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_title(label)
        ax.set_xlabel("Fraction of gap to baseline closed")
        ax.set_ylabel("Number of roles")
        ax.legend(fontsize=8)

    fig.suptitle("Gap-closing distribution at α=2.5 across 275 roles\n"
                 "(0 = no progress, 1 = reached baseline, negative = moved away)",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig2_gap_distribution_at_peak.pdf")


def fig3_absolute_score_trajectory(gap_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, color, label, ls in [
        ("steered", STEERED_COLOR, "Steered (proposed)", "-o"),
        ("assistant_axis", AXIS_COLOR, "Assistant Axis", "--s"),
    ]:
        sub = (
            gap_df[gap_df["method"] == method]
            .groupby("alpha")["score"]
            .agg(mean="mean", sem=lambda x: x.std() / len(x) ** 0.5)
            .reset_index()
        )
        ax.plot(sub["alpha"], sub["mean"], ls, color=color,
                linewidth=2.5, markersize=8, label=label)
        ax.fill_between(sub["alpha"],
                        sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"],
                        color=color, alpha=0.15)

    ax.axhline(BASELINE, color=BASELINE_COLOR, linewidth=2, linestyle="--",
               label=f"Baseline (gold standard) = {BASELINE:.0f}")

    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Mean role alignment score (0–100)")
    ax.set_title("Score trajectory relative to gold-standard baseline\n"
                 "(shading = ±1 SEM; 275 roles)")
    ax.set_xticks(ALPHAS)
    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig3_absolute_score_vs_baseline.pdf")


def fig4_per_role_heatmap(gap_df: pd.DataFrame) -> None:
    # Pivot to roles x alphas, separately for each method, side by side
    steered_pivot = gap_df[gap_df["method"] == "steered"].pivot(
        index="role", columns="alpha", values="gap_closed"
    )
    aa_pivot = gap_df[gap_df["method"] == "assistant_axis"].pivot(
        index="role", columns="alpha", values="gap_closed"
    )

    # Sort by steered gap_closed at α=2.5
    order = steered_pivot[2.5].sort_values(ascending=False).index
    steered_pivot = steered_pivot.loc[order]
    aa_pivot = aa_pivot.loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    vmin, vmax = -1.0, 1.5

    for ax, data, title in [
        (axes[0], steered_pivot, "Steered (proposed)"),
        (axes[1], aa_pivot, "Assistant Axis"),
    ]:
        im = ax.imshow(data.to_numpy(), aspect="auto", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"α={a}" for a in ALPHAS])
        ax.set_yticks([])
        ax.set_ylabel("Roles (sorted by steered gap-closed at α=2.5)" if ax == axes[0] else "")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.5,
                     label="Gap-closed fraction\n(1=baseline, 0=no change, <0=worse)")

    fig.suptitle("Per-role gap-closing fraction across all alphas\n"
                 "(green = closing gap toward baseline; red = moving away)",
                 y=1.01)
    plt.tight_layout()
    _save(fig, "fig4_per_role_heatmap.pdf")


def main() -> None:
    gap_df = pd.read_csv(DATA_DIR / "gap_closing.csv")
    summary = pd.read_csv(DATA_DIR / "gap_closing_summary.csv")
    print(f"Loaded {gap_df['role'].nunique()} roles")

    fig1_mean_gap_closed(summary)
    fig2_distribution_at_peak(gap_df)
    fig3_absolute_score_trajectory(gap_df)
    fig4_per_role_heatmap(gap_df)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
