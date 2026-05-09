"""
Figures for sub-dimension advantage analysis.

  Fig 1: Mean steered vs AA score per sub-dimension at each alpha (grouped bar)
  Fig 2: Mean advantage (steered − AA) per sub-dimension × alpha heatmap
  Fig 3: Sub-dimension variance collapse — steered vs AA std per subdim per alpha
  Fig 4: Scatter — perp_frac vs advantage, one panel per subdim (at alpha=2.0)

Usage:
    python analysis/empirical/subdimension_advantage/make_figures.py
"""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

import plot_style
plot_style.apply_style()

from plot_style import STEERED, AA, BASELINE, WIN, ACCENT, REPR, ALPHA_COLORS
STEERED_COLOR = STEERED
AXIS_COLOR = AA

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]

SUBDIM_LABELS = {
    "emotional_register":  "Emotional\nRegister",
    "vocab_choice":        "Vocab\nChoice",
    "social_dynamic":      "Social\nDynamic",
    "motivation":          "Motivation",
    "worldview_alignment": "Worldview\nAlignment",
}
SUBDIM_ORDER = list(SUBDIM_LABELS.keys())


def _save(fig, name: str) -> None:
    stem = name.replace(".pdf", "").replace(".png", "")
    plot_style.save_fig(fig, OUT_DIR, stem)


# ── Figure 1: Grouped bar — steered vs AA mean score per subdim at α=2.5 ──────

def fig1_mean_scores_at_peak(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, alpha in zip(axes, [1.0, 2.5]):
        sub = summary[summary["alpha"] == alpha].set_index("subdim").loc[SUBDIM_ORDER]
        x = np.arange(len(SUBDIM_ORDER))
        w = 0.35
        ax.bar(x - w / 2, sub["steered_mean"], w, color=STEERED_COLOR,
               alpha=0.85, label="Steered", edgecolor="white")
        ax.bar(x + w / 2, sub["aa_mean"], w, color=AXIS_COLOR,
               alpha=0.85, label="Assistant axis", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([SUBDIM_LABELS[s] for s in SUBDIM_ORDER], fontsize=9)
        ax.set_ylabel("Mean judge score (0–100)")
        ax.set_title(f"α = {alpha}")
        plot_style.legend_above(ax, ncol=2)
        ax.set_ylim(0, 80)

    fig.suptitle("Mean sub-dimension scores: steered vs assistant axis",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig1_mean_scores_per_subdim.pdf")


# ── Figure 2: Heatmap — mean advantage per subdim × alpha ─────────────────────

def fig2_advantage_heatmap(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="alpha", columns="subdim", values="advantage_mean")
    pivot = pivot[SUBDIM_ORDER]

    fig, ax = plt.subplots(figsize=(9, 4))
    vmax = np.abs(pivot.values).max()
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean advantage (steered − AA)")

    ax.set_xticks(range(len(SUBDIM_ORDER)))
    ax.set_xticklabels([SUBDIM_LABELS[s] for s in SUBDIM_ORDER], fontsize=9)
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_yticklabels([f"α={a}" for a in ALPHAS])
    ax.set_title("Sub-dimension advantage heatmap: steered − assistant axis")

    for i, a in enumerate(ALPHAS):
        for j, sd in enumerate(SUBDIM_ORDER):
            val = pivot.loc[a, sd] if a in pivot.index else float("nan")
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.tight_layout()
    _save(fig, "fig2_advantage_heatmap.pdf")


# ── Figure 3: Sub-dimension variance collapse ──────────────────────────────────

def fig3_variance_collapse(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(SUBDIM_ORDER), figsize=(15, 4.5), sharey=True)

    for ax, subdim in zip(axes, SUBDIM_ORDER):
        sub = summary[summary["subdim"] == subdim].set_index("alpha")
        ax.plot(ALPHAS, [sub.loc[a, "steered_std"] if a in sub.index else float("nan")
                         for a in ALPHAS],
                "-o", color=STEERED_COLOR, linewidth=2, markersize=6, label="Steered")
        ax.plot(ALPHAS, [sub.loc[a, "aa_std"] if a in sub.index else float("nan")
                         for a in ALPHAS],
                "--s", color=AXIS_COLOR, linewidth=2, markersize=6, label="AA")
        ax.set_xticks(ALPHAS)
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(SUBDIM_LABELS[subdim])
        if ax is axes[0]:
            ax.set_ylabel("Std across roles")
            plot_style.legend_above(ax, ncol=2)

    fig.suptitle("Sub-dimension score variance across roles: steered vs assistant axis",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig3_variance_collapse_per_subdim.pdf")


# ── Figure 4: Scatter — perp_frac vs advantage at α=2.0 ───────────────────────

def fig4_geo_vs_subdim_advantage(long_df: pd.DataFrame, geo_corr: pd.DataFrame,
                                  decomp: pd.DataFrame) -> None:
    alpha = 2.0
    merged = long_df[long_df["alpha"] == alpha].merge(
        decomp[["role", "perp_frac"]], on="role", how="inner"
    )

    fig, axes = plt.subplots(1, len(SUBDIM_ORDER), figsize=(15, 4), sharey=False)

    for ax, subdim in zip(axes, SUBDIM_ORDER):
        sub = merged[merged["subdim"] == subdim].dropna(subset=["perp_frac", "advantage"])
        x = sub["perp_frac"].to_numpy(dtype=float)
        y = sub["advantage"].to_numpy(dtype=float)
        ax.scatter(x, y, s=15, alpha=0.55, color=STEERED_COLOR, edgecolors="none")
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, m * xr + b, color="black", linewidth=1.2, alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")

        # Annotate r
        row = geo_corr[(geo_corr["alpha"] == alpha) & (geo_corr["subdim"] == subdim)]
        if not row.empty:
            r = row.iloc[0]["pearson_r"]
            p = row.iloc[0]["pearson_p"]
            pstr = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            ax.text(0.05, 0.95, f"r={r:+.3f}\n{pstr}", transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_xlabel("perp_frac")
        ax.set_title(SUBDIM_LABELS[subdim])
        if ax is axes[0]:
            ax.set_ylabel("Advantage (steered − AA)")

    fig.suptitle(f"Geometric role-specificity vs sub-dimension advantage at α={alpha}",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig4_geo_vs_subdim_advantage.pdf")


# ── Figure 5: Win-rate per subdim — what fraction of roles does steered win? ──

def fig5_win_rate(long_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SUBDIM_ORDER))
    width = 0.18
    offsets = np.linspace(-(len(ALPHAS) - 1) / 2, (len(ALPHAS) - 1) / 2, len(ALPHAS)) * width

    for i, alpha in enumerate(ALPHAS):
        rates = []
        for subdim in SUBDIM_ORDER:
            sub = long_df[(long_df["alpha"] == alpha) & (long_df["subdim"] == subdim)]
            if len(sub) == 0:
                rates.append(float("nan"))
            else:
                rates.append(float((sub["advantage"] > 0).mean()))
        ax.bar(x + offsets[i], rates, width * 0.9, label=f"α={alpha}",
               color=ALPHA_COLORS[alpha], edgecolor="white", alpha=0.9)

    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", label="50% (tie)")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBDIM_LABELS[s] for s in SUBDIM_ORDER])
    ax.set_ylabel("Fraction of roles where steered > AA")
    ax.set_ylim(0, 1)
    ax.set_title("Win rate per sub-dimension: fraction of roles where steered beats AA")
    plot_style.legend_above(ax, ncol=5)
    plt.tight_layout()
    _save(fig, "fig5_win_rate_per_subdim.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    long_df = pd.read_csv(DATA_DIR / "subdim_long.csv")
    summary = pd.read_csv(DATA_DIR / "subdim_summary.csv")
    print(f"Loaded {len(long_df)} long records, {len(summary)} summary rows")

    fig1_mean_scores_at_peak(summary)
    fig2_advantage_heatmap(summary)
    fig3_variance_collapse(summary)
    fig5_win_rate(long_df)

    geo_corr_path = DATA_DIR / "subdim_geo_corr.csv"
    decomp_path = (
        Path(__file__).resolve().parents[3]
        / "analysis" / "empirical" / "role_vector_decomposition" / "data" / "decomposition.csv"
    )
    if geo_corr_path.exists() and decomp_path.exists():
        geo_corr = pd.read_csv(geo_corr_path)
        decomp = pd.read_csv(decomp_path)
        fig4_geo_vs_subdim_advantage(long_df, geo_corr, decomp)
    else:
        print("Skipping fig4: geo_corr or decomposition data not found")

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
