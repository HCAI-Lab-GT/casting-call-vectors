"""
Figures for score trajectory comparison (steered vs assistant axis).

  Fig 1: AUC distribution — overlapping histograms steered vs AA
  Fig 2: Monotonicity and peak-alpha bar chart
  Fig 3: Mean score curves by norm quartile — steered vs AA side-by-side
  Fig 4: Score gain (score_2.5 − score_1.0) distribution — steered vs AA
  Fig 5: AUC advantage vs role vector norm

Usage:
    python analysis/empirical/score_trajectory_comparison/make_figures.py
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
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]


def _save(fig, name: str) -> None:
    stem = name.replace(".pdf", "").replace(".png", "")
    plot_style.save_fig(fig, OUT_DIR, stem)


# ── Figure 1: AUC distribution ─────────────────────────────────────────────────

def fig1_auc_distribution(traj: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(traj["steered_auc"].min(), traj["aa_auc"].min()),
        max(traj["steered_auc"].max(), traj["aa_auc"].max()),
        30,
    )
    ax.hist(traj["steered_auc"], bins=bins, color=STEERED_COLOR, alpha=0.6,
            label=f"Steered  (mean={traj['steered_auc'].mean():.1f})", edgecolor="white")
    ax.hist(traj["aa_auc"], bins=bins, color=AXIS_COLOR, alpha=0.6,
            label=f"Assistant axis  (mean={traj['aa_auc'].mean():.1f})", edgecolor="white")
    ax.axvline(traj["steered_auc"].mean(), color=STEERED_COLOR, linewidth=2, linestyle="--")
    ax.axvline(traj["aa_auc"].mean(), color=AXIS_COLOR, linewidth=2, linestyle="--")
    ax.set_xlabel("AUC of score vs alpha curve (normalised by alpha range)")
    ax.set_ylabel("Number of roles")
    ax.set_title("Score trajectory AUC distribution across 275 roles")
    plot_style.legend_above(ax, ncol=2)
    plt.tight_layout()
    _save(fig, "fig1_auc_distribution.pdf")


# ── Figure 2: Monotonicity and peak alpha ──────────────────────────────────────

def fig2_monotonicity_and_peak(traj: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: monotonicity rates
    ax = axes[0]
    methods = ["Steered", "Assistant axis"]
    rates = [traj["steered_monotonic"].mean(), traj["aa_monotonic"].mean()]
    colors = [STEERED_COLOR, AXIS_COLOR]
    bars = ax.bar(methods, rates, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--")
    ax.set_ylabel("Fraction of roles with monotonically increasing score")
    ax.set_title("Monotonic improvement rate")
    ax.set_ylim(0, 1)

    # Right: peak alpha distribution stacked bar
    ax = axes[1]
    x = np.arange(len(ALPHAS))
    w = 0.35
    for offset, method, col, color in [
        (-w / 2, "Steered", "steered_peak_alpha", STEERED_COLOR),
        (w / 2, "AA", "aa_peak_alpha", AXIS_COLOR),
    ]:
        counts = traj[col].value_counts()
        vals = [counts.get(a, 0) / len(traj) for a in ALPHAS]
        ax.bar(x + offset, vals, w, color=color, alpha=0.85, edgecolor="white", label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([f"α={a}" for a in ALPHAS])
    ax.set_ylabel("Fraction of roles")
    ax.set_title("Peak alpha distribution")
    plot_style.legend_above(ax, ncol=2)

    fig.suptitle("Score trajectory shape: steered vs assistant axis", y=1.02)
    plt.tight_layout()
    _save(fig, "fig2_monotonicity_and_peak.pdf")


# ── Figure 3: Mean score curves by AA peak-alpha category ─────────────────────

def fig3_mean_curves_by_peak(traj: pd.DataFrame, curves: pd.DataFrame) -> None:
    # Split roles into "AA peaks early (< 2.5)" vs "AA peaks at 2.5"
    early_peak = traj[traj["aa_peak_alpha"] < 2.5]["role"].values
    late_peak = traj[traj["aa_peak_alpha"] >= 2.5]["role"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, roles, label in [
        (axes[0], early_peak, f"AA peaks early (n={len(early_peak)})"),
        (axes[1], late_peak,  f"AA peaks at α=2.5 (n={len(late_peak)})"),
    ]:
        sub = curves[curves["role"].isin(roles)]
        s_mean = sub.groupby("alpha")["steered_score"].mean()
        a_mean = sub.groupby("alpha")["aa_score"].mean()
        s_sem = sub.groupby("alpha")["steered_score"].sem()
        a_sem = sub.groupby("alpha")["aa_score"].sem()

        ax.plot(s_mean.index, s_mean.values, "-o", color=STEERED_COLOR,
                linewidth=2, markersize=6, label="Steered")
        ax.fill_between(s_mean.index, s_mean - s_sem, s_mean + s_sem,
                        color=STEERED_COLOR, alpha=0.15)
        ax.plot(a_mean.index, a_mean.values, "--s", color=AXIS_COLOR,
                linewidth=2, markersize=6, label="Assistant axis")
        ax.fill_between(a_mean.index, a_mean - a_sem, a_mean + a_sem,
                        color=AXIS_COLOR, alpha=0.15)

        ax.set_xlabel(r"Steering strength ($\alpha$)")
        ax.set_ylabel("Mean role alignment score (0–100)")
        ax.set_title(label)
        ax.set_xticks(ALPHAS)
        plot_style.legend_above(ax, ncol=2)

    fig.suptitle("Mean score trajectories grouped by AA peak-alpha behaviour (shading = ±1 SEM)",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig3_mean_curves_by_peak_category.pdf")


# ── Figure 4: Score gain distribution ─────────────────────────────────────────

def fig4_score_gain(traj: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(traj["steered_gain"].min(), traj["aa_gain"].min()),
        max(traj["steered_gain"].max(), traj["aa_gain"].max()),
        30,
    )
    ax.hist(traj["steered_gain"], bins=bins, color=STEERED_COLOR, alpha=0.6,
            label=f"Steered  (mean={traj['steered_gain'].mean():+.1f})", edgecolor="white")
    ax.hist(traj["aa_gain"], bins=bins, color=AXIS_COLOR, alpha=0.6,
            label=f"Assistant axis  (mean={traj['aa_gain'].mean():+.1f})", edgecolor="white")
    ax.axvline(traj["steered_gain"].mean(), color=STEERED_COLOR, linewidth=2, linestyle="--")
    ax.axvline(traj["aa_gain"].mean(), color=AXIS_COLOR, linewidth=2, linestyle="--")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Score gain: score(α=2.5) − score(α=1.0)")
    ax.set_ylabel("Number of roles")
    ax.set_title("Score gain from low to high alpha: steered vs assistant axis")
    plot_style.legend_above(ax, ncol=2)
    plt.tight_layout()
    _save(fig, "fig4_score_gain_distribution.pdf")


# ── Figure 5: Slope comparison scatter ────────────────────────────────────────

def fig5_slope_comparison(traj: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(traj["aa_slope"], traj["steered_slope"],
                    c=traj["auc_advantage"], cmap="RdYlGn",
                    s=25, alpha=0.7, edgecolors="none",
                    vmin=traj["auc_advantage"].quantile(0.05),
                    vmax=traj["auc_advantage"].quantile(0.95))
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("AUC advantage (steered − AA)")

    lims = [
        min(traj["aa_slope"].min(), traj["steered_slope"].min()) - 1,
        max(traj["aa_slope"].max(), traj["steered_slope"].max()) + 1,
    ]
    ax.plot(lims, lims, color="gray", linewidth=1, linestyle="--", label="y = x (tie)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("AA score slope (Δscore / Δα)")
    ax.set_ylabel("Steered score slope (Δscore / Δα)")
    ax.set_title("Per-role score slope: steered vs assistant axis")
    plot_style.legend_above(ax, ncol=1)
    pct_above = float((traj["steered_slope"] > traj["aa_slope"]).mean())
    ax.text(0.05, 0.95, f"{pct_above:.1%} of roles:\nsteered slope > AA slope",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.tight_layout()
    _save(fig, "fig5_slope_comparison.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    traj = pd.read_csv(DATA_DIR / "trajectory_comparison.csv")
    curves = pd.read_csv(DATA_DIR / "alpha_curves_both.csv")
    print(f"Loaded {len(traj)} roles")

    fig1_auc_distribution(traj)
    fig2_monotonicity_and_peak(traj)
    fig3_mean_curves_by_peak(traj, curves)
    fig4_score_gain(traj)
    fig5_slope_comparison(traj)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
