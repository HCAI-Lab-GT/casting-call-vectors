"""
Figures for the magnitude-vs-stability hypothesis.

Reads:
    data/vector_magnitude_stability.csv
    data/alpha_curves_long.csv
    data/correlations.csv

Writes (figures/):
    fig1_norm_vs_score_per_alpha.png
    fig2_alpha_curves_by_norm_quartile.png
    fig3_norm_vs_slope_and_delta.png
    fig4_norm_vs_late_alpha_drop.png
    fig5_correlation_heatmap.png
    fig6_subdimensions_at_alpha2_5.png
    fig7_summary_panel.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]
ALPHA_COLS = [f"score_at_alpha_{str(a).replace('.', '_')}" for a in ALPHAS]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def annotated_scatter(ax, x, y, *, color, label_prefix=""):
    ax.scatter(x, y, s=14, alpha=0.55, color=color, edgecolors="white", linewidths=0.4)
    if len(x) >= 3:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(np.min(x), np.max(x), 50)
        ax.plot(xs, slope * xs + intercept, color=color, lw=1.6)
        rho, p_s = stats.spearmanr(x, y)
        ax.text(
            0.03,
            0.96,
            f"{label_prefix}Pearson r = {r:+.2f}  (p = {p:.1e})\n"
            f"{label_prefix}Spearman ρ = {rho:+.2f}  (p = {p_s:.1e})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.85, pad=2.5),
        )


def fig1_norm_vs_score_per_alpha(merged: pd.DataFrame) -> None:
    """One panel per alpha: norm vs steered_score@alpha. Tests whether the
    magnitude–score relationship reverses or strengthens with alpha."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, (a, col) in enumerate(zip(ALPHAS, ALPHA_COLS)):
        ax = axes[i]
        annotated_scatter(
            ax,
            merged["norm"].values,
            merged[col].values,
            color=cmap(i / max(1, len(ALPHAS) - 1)),
        )
        ax.set_title(f"alpha = {a}")
        ax.set_xlabel("Vector L2 norm")
        if i == 0:
            ax.set_ylabel("Mean steered_score (judge)")
    fig.suptitle(
        "Vector magnitude vs. steered-score, per alpha\n"
        "Negative at low alpha → positive at high alpha: high-norm vectors are slower-acting but stronger when fully engaged.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_norm_vs_score_per_alpha.png")
    plt.close(fig)


def fig2_alpha_curves_by_norm_quartile(long: pd.DataFrame, merged: pd.DataFrame) -> None:
    """Bin roles into norm quartiles and plot mean steered_score across alpha
    for each quartile, with bootstrapped 95% CIs."""
    quartiles = pd.qcut(merged["norm"], q=4, labels=["Q1 (low norm)", "Q2", "Q3", "Q4 (high norm)"])
    role_to_q = dict(zip(merged["role"], quartiles))
    long = long.copy()
    long["quartile"] = long["role"].map(role_to_q)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
    cmap = plt.get_cmap("plasma")
    qlist = ["Q1 (low norm)", "Q2", "Q3", "Q4 (high norm)"]
    for i, q in enumerate(qlist):
        sub = long[long["quartile"] == q]
        means = sub.groupby("alpha")["steered_score_mean"].mean()
        # bootstrap 95% CI
        cis = []
        for a in ALPHAS:
            vals = sub[sub["alpha"] == a]["steered_score_mean"].values
            boots = np.random.default_rng(0).choice(vals, size=(2000, len(vals)), replace=True).mean(axis=1)
            cis.append((np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
        cis = np.array(cis)
        color = cmap(i / 3)
        ax.plot(ALPHAS, means.values, "-o", lw=2, color=color, label=f"{q}  (n={len(sub)//len(ALPHAS)})")
        ax.fill_between(ALPHAS, cis[:, 0], cis[:, 1], color=color, alpha=0.15)
    ax.set_xlabel("Steering strength alpha")
    ax.set_ylabel("Mean steered_score (judge)")
    ax.set_title("Alpha–score curves grouped by vector-norm quartile")
    ax.legend(frameon=False)
    ax.set_xticks(ALPHAS)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_alpha_curves_by_norm_quartile.png")
    plt.close(fig)


def fig3_norm_vs_slope_and_delta(merged: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    annotated_scatter(
        axes[0],
        merged["norm"].values,
        merged["score_slope"].values,
        color="#1f77b4",
    )
    axes[0].axhline(0, color="black", lw=0.6, ls="--")
    axes[0].set_xlabel("Vector L2 norm")
    axes[0].set_ylabel("OLS slope of mean steered_score vs alpha")
    axes[0].set_title("Norm predicts steering responsiveness")

    annotated_scatter(
        axes[1],
        merged["norm"].values,
        merged["delta_high_minus_low"].values,
        color="#d62728",
    )
    axes[1].axhline(0, color="black", lw=0.6, ls="--")
    axes[1].set_xlabel("Vector L2 norm")
    axes[1].set_ylabel("Score(alpha=2.5) − Score(alpha=1.0)")
    axes[1].set_title("High-norm vectors gain more from larger alpha")

    fig.suptitle(
        "Magnitude predicts how strongly a role vector responds to higher alpha",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_norm_vs_slope_and_delta.png")
    plt.close(fig)


def fig4_norm_vs_late_alpha_drop(merged: pd.DataFrame) -> None:
    """The cleanest direct test of the stability hypothesis: when does the score
    curve peak before alpha=2.5, and does that 'collapse' depend on norm?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    annotated_scatter(
        axes[0],
        merged["norm"].values,
        merged["late_alpha_drop"].values,
        color="#2ca02c",
    )
    axes[0].axhline(0, color="black", lw=0.6, ls="--")
    axes[0].set_xlabel("Vector L2 norm")
    axes[0].set_ylabel("max(score) − score(alpha=2.5)")
    axes[0].set_title("Late-alpha collapse vs norm")

    # Bar chart: fraction of roles whose score peaks BEFORE alpha=2.5,
    # split by norm quartile.
    q = pd.qcut(merged["norm"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    peak_before = (merged["peak_alpha"] < 2.5).astype(int)
    by_q = pd.DataFrame({"q": q, "peak_before": peak_before}).groupby("q")["peak_before"].mean()
    axes[1].bar(range(4), by_q.values, color=plt.get_cmap("plasma")(np.linspace(0.1, 0.85, 4)))
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(by_q.index)
    axes[1].set_ylabel("Fraction of roles with score peak < alpha=2.5")
    axes[1].set_xlabel("Norm quartile")
    axes[1].set_title("Early peak (= instability at high alpha)\nis more common for low-norm vectors")
    for i, v in enumerate(by_q.values):
        axes[1].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_norm_vs_late_alpha_drop.png")
    plt.close(fig)


def fig5_correlation_heatmap(cor: pd.DataFrame) -> None:
    """Compact heatmap of (magnitude metric) × (stability metric) Pearson r."""
    pivot_r = cor.pivot(index="magnitude_metric", columns="stability_metric", values="pearson_r")
    pivot_p = cor.pivot(index="magnitude_metric", columns="stability_metric", values="pearson_p")
    fig, ax = plt.subplots(1, 1, figsize=(11, 2.6))
    im = ax.imshow(pivot_r.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(pivot_r.shape[1]))
    ax.set_xticklabels(pivot_r.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(pivot_r.shape[0]))
    ax.set_yticklabels(pivot_r.index, fontsize=9)
    for i in range(pivot_r.shape[0]):
        for j in range(pivot_r.shape[1]):
            r = pivot_r.values[i, j]
            p = pivot_p.values[i, j]
            star = "*" if p < 0.05 else ""
            ax.text(j, i, f"{r:+.2f}{star}", ha="center", va="center",
                    color="white" if abs(r) > 0.35 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Pearson r")
    ax.set_title("Correlation: magnitude metrics × stability metrics  (* = p<0.05)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_correlation_heatmap.png")
    plt.close(fig)


def fig6_subdimensions_at_alpha2_5(merged: pd.DataFrame) -> None:
    SUBDIMS = [
        "sub25_cmp_emotional_register",
        "sub25_cmp_vocab_choice",
        "sub25_cmp_social_dynamic",
        "sub25_cmp_motivation",
        "sub25_cmp_worldview_alignment",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6), sharey=True)
    for i, c in enumerate(SUBDIMS):
        annotated_scatter(axes[i], merged["norm"].values, merged[c].values, color="#9467bd")
        axes[i].set_title(c.replace("sub25_cmp_", ""))
        axes[i].set_xlabel("Vector L2 norm")
        if i == 0:
            axes[i].set_ylabel("Mean sub-dim score @ alpha=2.5")
    fig.suptitle("Per-subdimension: vector norm vs judge sub-score at alpha=2.5", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_subdimensions_at_alpha2_5.png")
    plt.close(fig)


def fig7_summary_panel(merged: pd.DataFrame, long: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

    # Panel A — slope vs norm
    ax = fig.add_subplot(gs[0, 0])
    annotated_scatter(ax, merged["norm"].values, merged["score_slope"].values, color="#1f77b4")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Norm")
    ax.set_ylabel("Slope of score vs alpha")
    ax.set_title("A.  Norm → score slope")

    # Panel B — score@2.5 vs norm
    ax = fig.add_subplot(gs[0, 1])
    annotated_scatter(ax, merged["norm"].values, merged["score_at_alpha_2_5"].values, color="#2ca02c")
    ax.set_xlabel("Norm")
    ax.set_ylabel("Steered score @ alpha=2.5")
    ax.set_title("B.  Norm → score at alpha=2.5")

    # Panel C — score@1.0 vs norm
    ax = fig.add_subplot(gs[0, 2])
    annotated_scatter(ax, merged["norm"].values, merged["score_at_alpha_1_0"].values, color="#ff7f0e")
    ax.set_xlabel("Norm")
    ax.set_ylabel("Steered score @ alpha=1.0")
    ax.set_title("C.  Norm → score at alpha=1.0")

    # Panel D — alpha curves by quartile
    ax = fig.add_subplot(gs[1, :2])
    quartiles = pd.qcut(merged["norm"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    role_to_q = dict(zip(merged["role"], quartiles))
    long2 = long.copy()
    long2["quartile"] = long2["role"].map(role_to_q)
    cmap = plt.get_cmap("plasma")
    for i, q in enumerate(["Q1 (low)", "Q2", "Q3", "Q4 (high)"]):
        sub = long2[long2["quartile"] == q]
        means = sub.groupby("alpha")["steered_score_mean"].mean()
        ax.plot(ALPHAS, means.values, "-o", lw=2.2, color=cmap(i / 3), label=q)
    ax.set_xlabel("Steering strength alpha")
    ax.set_ylabel("Mean steered_score (judge)")
    ax.set_title("D.  Alpha–score curves by norm quartile")
    ax.set_xticks(ALPHAS)
    ax.legend(frameon=False, fontsize=9)

    # Panel E — late-alpha drop bar
    ax = fig.add_subplot(gs[1, 2])
    q = pd.qcut(merged["norm"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    peak_before = (merged["peak_alpha"] < 2.5).astype(int)
    by_q = pd.DataFrame({"q": q, "peak_before": peak_before}).groupby("q")["peak_before"].mean()
    ax.bar(range(4), by_q.values, color=cmap(np.linspace(0.1, 0.85, 4)))
    ax.set_xticks(range(4))
    ax.set_xticklabels(by_q.index, rotation=20)
    ax.set_ylabel("Frac. roles peaking before alpha=2.5")
    ax.set_title("E.  Early peak (instability)\nvs norm quartile")
    for i, v in enumerate(by_q.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)

    fig.suptitle(
        "Does role-vector internal magnitude predict steering stability at high alpha?",
        fontsize=13,
        y=0.995,
    )
    fig.savefig(FIG_DIR / "fig7_summary_panel.png")
    plt.close(fig)


def main() -> None:
    merged = pd.read_csv(DATA_DIR / "vector_magnitude_stability.csv")
    long = pd.read_csv(DATA_DIR / "alpha_curves_long.csv")
    cor = pd.read_csv(DATA_DIR / "correlations.csv")

    fig1_norm_vs_score_per_alpha(merged)
    fig2_alpha_curves_by_norm_quartile(long, merged)
    fig3_norm_vs_slope_and_delta(merged)
    fig4_norm_vs_late_alpha_drop(merged)
    fig5_correlation_heatmap(cor)
    fig6_subdimensions_at_alpha2_5(merged)
    fig7_summary_panel(merged, long)

    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
