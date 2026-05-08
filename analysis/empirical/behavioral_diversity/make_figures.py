"""
Figures for the behavioral diversity and cross-method RSA analysis.

  Fig 1: Effective rank vs alpha — steered vs assistant axis (diversity collapse)
  Fig 2: Inter-role behavioral variance vs alpha — steered vs assistant axis
  Fig 3: Score std across roles vs alpha — steered vs assistant axis
  Fig 4: RSA comparison — repr_cos vs beh_steered vs beh_aa (grouped bar)
  Fig 5: Per-alpha RSA line chart — which method's behavior does geometry predict better?
  Fig 6: RSA(beh_steered, beh_aa) vs alpha — how similar are the two behavioral structures?

Usage:
    python analysis/empirical/behavioral_diversity/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]
STEERED_COLOR = "#2ecc71"
AXIS_COLOR = "#3498db"
REPR_COLOR = "#e67e22"


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 1: Effective rank vs alpha ─────────────────────────────────────────

def fig1_effective_rank(div: pd.DataFrame) -> None:
    s = div[div["method"] == "steered"].set_index("alpha")
    a = div[div["method"] == "assistant_axis"].set_index("alpha")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s.index, s["effective_rank"], "-o", color=STEERED_COLOR,
            linewidth=2, markersize=7, label="Steered (proposed)")
    ax.plot(a.index, a["effective_rank"], "--s", color=AXIS_COLOR,
            linewidth=2, markersize=7, label="Assistant axis")
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Effective rank of behavioral matrix")
    ax.set_xticks(ALPHAS)
    ax.set_title(
        "Behavioral diversity: effective rank of the 275-role behavioral matrix\n"
        r"(higher = roles more diverse in behavior space)"
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig1_effective_rank_vs_alpha.pdf")


# ── Figure 2: Inter-role mean pairwise L2 distance ────────────────────────────

def fig2_mean_pairwise_l2(div: pd.DataFrame) -> None:
    s = div[div["method"] == "steered"].set_index("alpha")
    a = div[div["method"] == "assistant_axis"].set_index("alpha")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s.index, s["mean_pairwise_l2"], "-o", color=STEERED_COLOR,
            linewidth=2, markersize=7, label="Steered (proposed)")
    ax.plot(a.index, a["mean_pairwise_l2"], "--s", color=AXIS_COLOR,
            linewidth=2, markersize=7, label="Assistant axis")
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Mean pairwise L2 distance between role profiles")
    ax.set_xticks(ALPHAS)
    ax.set_title(
        "Inter-role behavioral diversity: mean pairwise distance\n"
        "(z-scored per feature; higher = roles more spread out)"
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig2_mean_pairwise_l2_vs_alpha.pdf")


# ── Figure 3: Score std across roles ──────────────────────────────────────────

def fig3_score_std(div: pd.DataFrame) -> None:
    s = div[div["method"] == "steered"].set_index("alpha")
    a = div[div["method"] == "assistant_axis"].set_index("alpha")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s.index, s["score_std"], "-o", color=STEERED_COLOR,
            linewidth=2, markersize=7, label="Steered (proposed)")
    ax.plot(a.index, a["score_std"], "--s", color=AXIS_COLOR,
            linewidth=2, markersize=7, label="Assistant axis")
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Std of role alignment score across roles (0–100)")
    ax.set_xticks(ALPHAS)
    ax.set_title(
        "Inter-role score variance: std of role alignment scores across 275 roles\n"
        "(lower = roles converging; higher = roles preserving distinct signatures)"
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig3_score_std_vs_alpha.pdf")


# ── Figure 4: Diversity summary 3-panel ───────────────────────────────────────

def fig4_diversity_panel(div: pd.DataFrame) -> None:
    s = div[div["method"] == "steered"].set_index("alpha")
    a = div[div["method"] == "assistant_axis"].set_index("alpha")

    metrics = [
        ("effective_rank", "Effective rank", "Higher = more behavioral dimensions"),
        ("mean_pairwise_l2", "Mean pairwise L2", "Higher = more inter-role spread"),
        ("score_std", "Score std across roles", "Higher = roles more distinct"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, ylabel, subtitle) in zip(axes, metrics):
        if col not in s.columns or col not in a.columns:
            ax.set_visible(False)
            continue
        ax.plot(s.index, s[col], "-o", color=STEERED_COLOR, linewidth=2,
                markersize=7, label="Steered")
        ax.plot(a.index, a[col], "--s", color=AXIS_COLOR, linewidth=2,
                markersize=7, label="Assistant axis")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ALPHAS)
        ax.set_title(f"{ylabel}\n{subtitle}", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Behavioral diversity: steered vs. assistant axis across steering strengths",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig4_diversity_panel.pdf")


# ── Figure 5: Cross-method RSA grouped bar ────────────────────────────────────

def fig5_rsa_comparison(rsa_df: pd.DataFrame) -> None:
    # Select same-metric pairs only (corr vs corr, l2 vs l2) to avoid
    # mixing distance metrics across the two axes of the comparison.
    SAME_METRIC_PAIRS = [
        "beh_steered_corr_vs_beh_aa_corr",
        "beh_steered_l2_vs_beh_aa_l2",
        "repr_cos_vs_beh_steered_corr",
        "repr_cos_vs_beh_aa_corr",
        "repr_cos_vs_beh_steered_l2",
        "repr_cos_vs_beh_aa_l2",
    ]
    target_rows = rsa_df[rsa_df["comparison"].isin(SAME_METRIC_PAIRS)].copy()
    # Preserve display order
    target_rows = target_rows.set_index("comparison").loc[
        [p for p in SAME_METRIC_PAIRS if p in target_rows["comparison"].values]
    ].reset_index()

    LABEL_MAP = {
        "beh_steered_corr_vs_beh_aa_corr": "beh_steered\nvs\nbeh_aa\n(corr)",
        "beh_steered_l2_vs_beh_aa_l2":     "beh_steered\nvs\nbeh_aa\n(L2)",
        "repr_cos_vs_beh_steered_corr":     "repr_cos\nvs\nbeh_steered\n(corr)",
        "repr_cos_vs_beh_aa_corr":          "repr_cos\nvs\nbeh_aa\n(corr)",
        "repr_cos_vs_beh_steered_l2":       "repr_cos\nvs\nbeh_steered\n(L2)",
        "repr_cos_vs_beh_aa_l2":            "repr_cos\nvs\nbeh_aa\n(L2)",
    }

    labels, rs, ps = [], [], []
    for _, row in target_rows.iterrows():
        labels.append(LABEL_MAP.get(row["comparison"], row["comparison"]))
        rs.append(row["rsa_spearman"])
        ps.append(row["mantel_p"])

    colors = []
    for comp in target_rows["comparison"]:
        if "repr_cos" in comp and "steered" in comp:
            colors.append(STEERED_COLOR)
        elif "repr_cos" in comp and "beh_aa" in comp:
            colors.append(AXIS_COLOR)
        else:
            colors.append(REPR_COLOR)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(range(len(labels)), rs, color=colors, edgecolor="white", alpha=0.85)
    for bar, r_val, p_val in zip(bars, rs, ps):
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 if r_val >= 0 else bar.get_height() - 0.02,
                f"{r_val:+.3f}{sig}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mantel RSA (Spearman r)")
    ax.set_title("Cross-method behavioral RSA\n"
                 "(* p<0.05, ** p<0.01, *** p<0.001, Mantel permutation)")
    plt.tight_layout()
    _save(fig, "fig5_rsa_comparison_bar.pdf")


# ── Figure 6: Per-alpha RSA line chart ────────────────────────────────────────

def fig6_per_alpha_rsa(per_alpha_df: pd.DataFrame) -> None:
    sub = per_alpha_df[per_alpha_df["beh_distance"] == "corr"].copy()
    if sub.empty:
        print("Fig 6 skipped: no corr-distance per-alpha RSA data")
        return

    s = sub[sub["method"] == "steered"].set_index("alpha")
    a = sub[sub["method"] == "assistant_axis"].set_index("alpha")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not s.empty:
        ax.plot(s.index, s["rsa_spearman"], "-o", color=STEERED_COLOR,
                linewidth=2, markersize=7, label="RSA(repr_cos, beh_steered)")
    if not a.empty:
        ax.plot(a.index, a["rsa_spearman"], "--s", color=AXIS_COLOR,
                linewidth=2, markersize=7, label="RSA(repr_cos, beh_aa)")

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Mantel RSA (Spearman r)")
    ax.set_xticks(ALPHAS)
    ax.set_title(
        "Geometry–behavior link per alpha: steered vs assistant axis\n"
        r"RSA(repr_cos, beh_$\cdot$) — higher = geometry better predicts behavior"
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig6_per_alpha_rsa_comparison.pdf")


# ── Figure 7: Score distribution shift (box plots) ────────────────────────────

def fig7_score_distributions(aa_profiles: pd.DataFrame, steered_profiles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, df, method, color, score_feat in [
        (axes[0], steered_profiles, "Steered (proposed)", STEERED_COLOR, "steered_score"),
        (axes[1], aa_profiles, "Assistant axis", AXIS_COLOR, "assistant_axis_score"),
    ]:
        data_per_alpha = []
        for a in ALPHAS:
            col = f"{score_feat}__alpha_{a}"
            if col in df.columns:
                data_per_alpha.append(df[col].dropna().to_numpy())
            else:
                data_per_alpha.append(np.array([]))

        bp = ax.boxplot(
            [d for d in data_per_alpha if len(d) > 0],
            positions=[a for a, d in zip(ALPHAS, data_per_alpha) if len(d) > 0],
            widths=0.2,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker=".", markersize=3, alpha=0.3),
        )
        ax.set_xlabel(r"Steering strength ($\alpha$)")
        ax.set_ylabel("Role alignment score (0–100)")
        ax.set_title(f"{method}\n(distribution across 275 roles)")
        ax.set_xticks(ALPHAS)

    fig.suptitle("Score distribution across roles: does spread collapse at high α?",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig7_score_distributions_by_alpha.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    div = pd.read_csv(DATA_DIR / "diversity_by_alpha.csv")
    rsa_df = pd.read_csv(DATA_DIR / "rsa_cross_method.csv")
    per_alpha_df = pd.read_csv(DATA_DIR / "rsa_per_alpha_method_comparison.csv")
    aa_profiles = pd.read_csv(DATA_DIR / "aa_behavioral_profiles.csv")

    steered_profiles_path = (
        Path(__file__).resolve().parents[3] /
        "analysis" / "empirical" / "rsa_geometry_behavior" / "data" / "behavioral_profiles.csv"
    )
    if steered_profiles_path.exists():
        steered_profiles = pd.read_csv(steered_profiles_path)
    else:
        print(f"Warning: steered behavioral profiles not found at {steered_profiles_path}")
        steered_profiles = pd.DataFrame()

    print(f"Loaded diversity data: {len(div)} rows")
    print(f"Loaded RSA results: {len(rsa_df)} comparisons")

    fig1_effective_rank(div)
    fig2_mean_pairwise_l2(div)
    fig3_score_std(div)
    fig4_diversity_panel(div)
    fig5_rsa_comparison(rsa_df)
    fig6_per_alpha_rsa(per_alpha_df)

    if not steered_profiles.empty:
        fig7_score_distributions(aa_profiles, steered_profiles)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
