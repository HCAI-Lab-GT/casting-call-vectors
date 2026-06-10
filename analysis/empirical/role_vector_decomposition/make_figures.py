"""
Figures for the role vector decomposition analysis.

  Fig 1: Distribution of cosine similarities between proposed vectors and d_aa
          (+ d_aa internal consistency histogram as inset)
  Fig 2: Per-role scatter — angle_deg vs score_advantage at each alpha (2×2)
  Fig 3: Per-role scatter — perp_frac vs score_advantage at each alpha (2×2)
  Fig 4: perp_norm vs parallel_norm scatter (colored by score_advantage at alpha=2.5)
  Fig 5: Correlation summary bar chart — r values per feature per alpha

Usage:
    python analysis/empirical/role_vector_decomposition/make_figures.py
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

GEO_FEATURE_LABELS = {
    "angle_deg": "Angle between v_i and d_aa (°)",
    "perp_frac": "Perpendicular fraction ‖v_⊥‖ / ‖v_i‖",
    "perp_norm": "Perpendicular component norm ‖v_⊥‖",
    "role_norm": "Role vector norm ‖v_i‖",
}


def _annotate_r(ax, x, y, fontsize=9):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    r, p = stats.pearsonr(x[mask], y[mask])
    rho, _ = stats.spearmanr(x[mask], y[mask])
    pstr = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.text(0.05, 0.95, f"r = {r:+.3f}  ρ = {rho:+.3f}  {pstr}",
            transform=ax.transAxes, fontsize=fontsize, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))


def _save(fig, name: str) -> None:
    stem = name.replace(".pdf", "").replace(".png", "")
    plot_style.save_fig(fig, OUT_DIR, stem)


# ── Figure 1: cos(v_i, d_aa) distribution ─────────────────────────────────────

def fig1_cos_distribution(df: pd.DataFrame, consistency: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: |cos(proposed_v_i, d_aa)| — use absolute value because proposed vectors
    # are contrastive (role − base) while d_aa points toward assistant, so the sign
    # is a direction-convention artifact; the magnitude is what reflects alignment.
    ax = axes[0]
    cos_vals = df["cos_with_daa"].dropna().abs()
    ax.hist(cos_vals, bins=30, color=STEERED_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(cos_vals.mean(), color=BASELINE, linewidth=1.8, linestyle="--",
               label=f"Mean = {cos_vals.mean():.3f}")
    ax.set_xlabel("|cos(v_i, d_aa)|  [proposed vector vs assistant axis direction]")
    ax.set_ylabel("Number of roles")
    ax.set_title("Alignment of proposed vectors with d_aa")
    plot_style.legend_above(ax, ncol=1)

    # Right: cos(v_aa_i, d_aa) — internal consistency check
    ax = axes[1]
    cos_aa = consistency["cos_with_daa"].dropna()
    ax.hist(cos_aa, bins=30, color=AXIS_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(cos_aa.mean(), color=BASELINE, linewidth=1.8, linestyle="--",
               label=f"Mean = {cos_aa.mean():.3f}")
    ax.set_xlabel("cos(v_aa_i, d_aa)  [per-role AA vector vs d_aa direction]")
    ax.set_ylabel("Number of roles")
    ax.set_title("AA vector consistency with global d_aa")
    plot_style.legend_above(ax, ncol=1)

    fig.suptitle(
        "Cosine similarity distributions: proposed vectors vs assistant axis",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig1_cos_distribution.pdf")


# ── Figure 2: angle_deg vs score_advantage (2×2 across alphas) ────────────────

def _scatter_vs_advantage(df: pd.DataFrame, feature: str, feat_label: str,
                          title_prefix: str, fname: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True)
    axes = axes.flatten()
    for i, a in enumerate(ALPHAS):
        ax = axes[i]
        adv_col = f"advantage_alpha_{a}"
        if adv_col not in df.columns:
            ax.set_visible(False)
            continue
        sub = df.dropna(subset=[feature, adv_col])
        x = sub[feature].to_numpy(dtype=float)
        y = sub[adv_col].to_numpy(dtype=float)
        ax.scatter(x, y, s=22, alpha=0.65, color=ALPHA_COLORS[a], edgecolors="none")
        # OLS line
        if np.isfinite(x).sum() > 2:
            m, b = np.polyfit(x[np.isfinite(x) & np.isfinite(y)],
                              y[np.isfinite(x) & np.isfinite(y)], 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, m * xr + b, color="black", linewidth=1.2, alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", zorder=1)
        _annotate_r(ax, x, y)
        ax.set_title(f"α = {a}  (n={len(sub)} roles)")
        ax.set_xlabel(feat_label)
        ax.set_ylabel("Score advantage\n(steered − assistant axis)")

    fig.suptitle(title_prefix, y=1.01)
    plt.tight_layout()
    _save(fig, fname)


def fig2_angle_vs_advantage(df: pd.DataFrame) -> None:
    _scatter_vs_advantage(
        df, "angle_deg", GEO_FEATURE_LABELS["angle_deg"],
        "Role-vector misalignment angle vs score advantage",
        "fig2_angle_vs_advantage.pdf",
    )


def fig3_perpfrac_vs_advantage(df: pd.DataFrame) -> None:
    _scatter_vs_advantage(
        df, "perp_frac", GEO_FEATURE_LABELS["perp_frac"],
        "Perpendicular fraction (role-specific component) vs score advantage",
        "fig3_perpfrac_vs_advantage.pdf",
    )


# ── Figure 4: perp_norm vs parallel_norm, colored by advantage at alpha=2.5 ───

def fig4_decomp_scatter(df: pd.DataFrame) -> None:
    adv_col = "advantage_alpha_2.5"
    sub = df.dropna(subset=["perp_norm", "parallel_norm", adv_col])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(
        sub["parallel_norm"], sub["perp_norm"],
        c=sub[adv_col], cmap="RdYlGn",
        s=40, alpha=0.80, edgecolors="none",
        vmin=sub[adv_col].quantile(0.05), vmax=sub[adv_col].quantile(0.95),
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Score advantage at α=2.5\n(steered − assistant axis)")

    # Diagonal lines showing constant ‖v_i‖ (perp² + par² = norm²)
    max_norm = max(sub["parallel_norm"].max(), sub["perp_norm"].max()) * 1.05
    for norm_val in np.percentile(sub["role_norm"], [25, 50, 75]):
        theta = np.linspace(0, np.pi / 2, 200)
        ax.plot(norm_val * np.cos(theta), norm_val * np.sin(theta),
                color="lightgray", linewidth=0.8, linestyle=":", zorder=1)
        ax.text(norm_val * np.cos(np.pi / 8) * 0.95,
                norm_val * np.sin(np.pi / 8) * 0.95,
                f"‖v‖={norm_val:.1f}", fontsize=7, color="gray", ha="center")

    ax.set_xlabel("Parallel component ‖v_∥‖  (along d_aa, captured by assistant axis)")
    ax.set_ylabel("Perpendicular component ‖v_⊥‖  (role-specific, discarded by assistant axis)")
    ax.set_title("Decomposition of role vectors: v_i = v_∥ + v_⊥")
    ax.set_xlim(0, max_norm)
    ax.set_ylim(0, max_norm)
    plt.tight_layout()
    _save(fig, "fig4_decomp_perp_vs_parallel.pdf")


# ── Figure 5: Correlation summary bar chart ────────────────────────────────────

def fig5_correlation_summary(corr_df: pd.DataFrame) -> None:
    features = [f for f in GEO_FEATURE_LABELS if f in corr_df["feature"].unique()]
    n_feats = len(features)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, metric, ylabel in [
        (axes[0], "pearson_r", "Pearson r"),
        (axes[1], "spearman_r", "Spearman ρ"),
    ]:
        x = np.arange(len(ALPHAS))
        width = 0.18
        offsets = np.linspace(-(n_feats - 1) / 2, (n_feats - 1) / 2, n_feats) * width

        for j, feat in enumerate(features):
            sub = corr_df[corr_df["feature"] == feat].set_index("alpha")
            rs = [sub.loc[a, metric] if a in sub.index else float("nan") for a in ALPHAS]
            ax.bar(x + offsets[j], rs, width * 0.92, label=GEO_FEATURE_LABELS[feat],
                   alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a}" for a in ALPHAS])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} of geometric features vs score advantage")
        ax.set_ylim(-0.5, 0.7)
        if ax is axes[0]:
            plot_style.legend_above(ax, ncol=2)

    fig.suptitle("Geometric decomposition correlates with score advantage", y=1.02)
    plt.tight_layout()
    _save(fig, "fig5_correlation_summary.pdf")


# ── Figure 6: perp_frac distribution ──────────────────────────────────────────

def fig6_perpfrac_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = df["perp_frac"].dropna()
    ax.hist(vals, bins=30, color=STEERED_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color=BASELINE, linewidth=1.8, linestyle="--",
               label=f"Mean = {vals.mean():.3f}")
    ax.axvline(vals.median(), color=BASELINE, linewidth=1.2, linestyle=":",
               label=f"Median = {vals.median():.3f}")
    ax.set_xlabel("Perpendicular fraction ‖v_⊥‖ / ‖v_i‖")
    ax.set_ylabel("Number of roles")
    ax.set_title("Role-specific fraction of each proposed vector")
    plot_style.legend_above(ax, ncol=2)
    plt.tight_layout()
    _save(fig, "fig6_perpfrac_distribution.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    decomp = pd.read_csv(DATA_DIR / "decomposition.csv")
    consistency = pd.read_csv(DATA_DIR / "d_aa_consistency.csv")
    corr_df = pd.read_csv(DATA_DIR / "correlations.csv")

    print(f"Loaded {len(decomp)} roles")

    fig1_cos_distribution(decomp, consistency)
    fig2_angle_vs_advantage(decomp)
    fig3_perpfrac_vs_advantage(decomp)
    fig4_decomp_scatter(decomp)
    fig5_correlation_summary(corr_df)
    fig6_perpfrac_distribution(decomp)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
