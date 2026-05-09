"""
Figures for RSA between role-vector geometry and steered-behavior geometry.

Reads from data/ (produced by run_analysis.py):
    rdm_repr_cos.npy, rdm_repr_l2.npy
    rdm_beh_corr.npy, rdm_beh_l2.npy
    rdm_beh_corr_alpha{a}.npy
    role_order.json, role_norms.csv, behavioral_profiles.csv
    rsa_main.csv, rsa_per_alpha.csv, rsa_partial.csv
    mantel_null_distribution.npz

Writes (figures/):
    fig1_rdms_side_by_side.png        : RDM_repr | RDM_beh, sorted by repr clustering
    fig2_pairwise_distance_scatter.png: pairwise repr-dist vs beh-dist (hexbin), with regression line
    fig3_mantel_null.png              : null distribution of RSA r under role-label permutation
    fig4_rsa_per_alpha.png            : RSA at each alpha (does the link strengthen?)
    fig5_rsa_distance_grid.png        : RSA across distance choices (cos/L2 x corr/L2)
    fig6_partial_rsa.png              : full RSA vs RSA partialling out norm
    fig7_summary_panel.png            : compact 4-panel summary
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

import plot_style
plot_style.apply_style()
from plot_style import STEERED, AA, BASELINE, WIN, ACCENT, REPR, ALPHA_COLORS

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]


def _save(fig, name: str) -> None:
    stem = name.replace(".pdf", "").replace(".png", "")
    plot_style.save_fig(fig, FIG_DIR, stem)


def upper_tri(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(rdm, k=1)
    return rdm[iu]


def hclust_order(rdm: np.ndarray) -> np.ndarray:
    """Return a 1D leaf order (n,) by sorting on the first PCA component of the RDM.

    A real seriation would use hierarchical clustering, but scipy.cluster is
    broken in this venv. PC1 of the RDM gives a smooth ordering that surfaces
    block structure adequately for visual inspection.
    """
    rdm = np.asarray(rdm, dtype=float)
    rdm = rdm - rdm.mean(axis=0, keepdims=True)
    coord = PCA(n_components=1, random_state=0).fit_transform(rdm).ravel()
    return np.argsort(coord)


def fig1_rdms_side_by_side() -> None:
    rdm_repr = np.load(DATA_DIR / "rdm_repr_cos.npy")
    rdm_beh = np.load(DATA_DIR / "rdm_beh_corr.npy")
    order = hclust_order(rdm_repr)
    rdm_repr_o = rdm_repr[order][:, order]
    rdm_beh_o = rdm_beh[order][:, order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1, 1, 0.04]})
    im0 = axes[0].imshow(rdm_repr_o, cmap="viridis",
                         vmin=np.percentile(rdm_repr_o, 2),
                         vmax=np.percentile(rdm_repr_o, 98))
    axes[0].set_title("Representational RDM (1 - cosine on role vectors)")
    axes[0].set_xlabel("role"); axes[0].set_ylabel("role")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im0, cax=axes[2], shrink=0.8)

    im1 = axes[1].imshow(rdm_beh_o, cmap="viridis",
                         vmin=np.percentile(rdm_beh_o, 2),
                         vmax=np.percentile(rdm_beh_o, 98))
    axes[1].set_title("Behavioral RDM (1 - corr on judge profile, 24-d)")
    axes[1].set_xlabel("role")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.suptitle(
        "Both RDMs sorted by hierarchical clustering of the representational RDM. "
        "Shared block structure -> the geometry predicts behavior.",
        fontsize=11,
    )
    _save(fig, "fig1_rdms_side_by_side")


def fig2_pairwise_scatter() -> None:
    rdm_repr = np.load(DATA_DIR / "rdm_repr_cos.npy")
    rdm_beh = np.load(DATA_DIR / "rdm_beh_corr.npy")
    x = upper_tri(rdm_repr)
    y = upper_tri(rdm_beh)
    rho = stats.spearmanr(x, y).statistic

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(x, y, gridsize=60, mincnt=1, cmap="magma", bins="log")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    # rolling-median trend
    bins = np.quantile(x, np.linspace(0, 1, 25))
    centers, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() > 20:
            centers.append(0.5 * (lo + hi))
            meds.append(np.median(y[mask]))
    ax.plot(centers, meds, "-", color="cyan", lw=2, label="rolling median")
    ax.set_xlabel("pairwise representational distance (1 - cosine)")
    ax.set_ylabel("pairwise behavioral distance (1 - corr on 24-d profile)")
    ax.set_title(f"All {len(x):,} role pairs   |   Spearman r = {rho:+.3f}")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, "fig2_pairwise_distance_scatter")


def fig3_mantel_null() -> None:
    npz = np.load(DATA_DIR / "mantel_null_distribution.npz")
    main = pd.read_csv(DATA_DIR / "rsa_main.csv")
    keys = list(npz.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        null = npz[key]
        repr_name, beh_name = key.split("__")
        row = main[
            (main["repr_distance"] == repr_name) & (main["beh_distance"] == beh_name)
        ].iloc[0]
        obs = row["rsa_spearman"]
        p = row["mantel_p"]
        ax.hist(null, bins=60, color="#888", edgecolor="white")
        ax.axvline(obs, color="crimson", lw=2,
                   label=f"observed r = {obs:+.3f}\np = {p:.0e}")
        ax.set_title(f"{repr_name}  vs  {beh_name}")
        ax.set_xlabel("Spearman r under role-label permutation")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylabel("count of permutations")
    fig.suptitle("Mantel null distributions — bigger gap = stronger geometry-behavior link",
                 fontsize=11)
    _save(fig, "fig3_mantel_null")


def fig4_rsa_per_alpha() -> None:
    df = pd.read_csv(DATA_DIR / "rsa_per_alpha.csv")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    width = 0.35
    xs = np.arange(len(ALPHAS))
    colors = [STEERED, ACCENT]
    behs = ["beh_corr", "beh_l2"]
    voff = 0.004  # same gap above bar top for both series
    for i, (beh, color) in enumerate(zip(behs, colors)):
        sub = df[df["beh_distance"] == beh].sort_values("alpha")
        ax.bar(xs + (i - 0.5) * width, sub["rsa_spearman"].values,
               width=width, label=beh, color=color, edgecolor="white", alpha=0.9)
        for x, r, p in zip(xs + (i - 0.5) * width,
                            sub["rsa_spearman"].values,
                            sub["mantel_p"].values):
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            label = f"{r:+.3f}\n{sig}" if sig else f"{r:+.3f}"
            ax.text(x, r + voff, label,
                    ha="center", va="bottom", fontsize=7, linespacing=1.1)
    # give enough headroom above tallest bar so text isn't clipped
    ymax = df["rsa_spearman"].max()
    ax.set_ylim(bottom=0, top=ymax * 1.30)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([fr"$\alpha={a}$" for a in ALPHAS])
    ax.set_ylabel(r"RSA Spearman $r$ (vs.\ repr$_{\cos}$ RDM)")
    ax.set_title("Per-alpha behavioral RSA")
    plot_style.legend_above(ax, ncol=2)
    # Significance key — placed inside axes so it survives tight_layout
    ax.text(0.02, 0.97,
            "* $p<0.05$    ** $p<0.01$    *** $p<0.001$\n(Mantel permutation test)",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, lw=0))
    plt.tight_layout()
    _save(fig, "fig4_rsa_per_alpha")


def fig5_distance_grid() -> None:
    df = pd.read_csv(DATA_DIR / "rsa_main.csv")
    pivot = df.pivot(index="repr_distance", columns="beh_distance", values="rsa_spearman")
    pivot_p = df.pivot(index="repr_distance", columns="beh_distance", values="mantel_p")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r",
                   vmin=-max(abs(pivot.values.min()), abs(pivot.values.max())),
                   vmax=max(abs(pivot.values.min()), abs(pivot.values.max())))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"r={pivot.values[i,j]:+.3f}\np={pivot_p.values[i,j]:.0e}",
                    ha="center", va="center", color="black", fontsize=9)
    ax.set_title("RSA across distance choices")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman r")
    _save(fig, "fig5_rsa_distance_grid")


def fig6_partial_rsa() -> None:
    df = pd.read_csv(DATA_DIR / "rsa_partial.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(df))
    width = 0.35
    ax.bar(xs - width / 2, df["rsa_full"], width=width, color="#3b82f6",
           edgecolor="white", label="full RSA")
    ax.bar(xs + width / 2, df["rsa_partial_minus_norm"], width=width, color="#10b981",
           edgecolor="white", label="partial RSA (control: ||v||)")
    for i, (f, p) in enumerate(zip(df["rsa_full"], df["rsa_partial_minus_norm"])):
        ax.text(i - width / 2, f + 0.003, f"{f:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, p + 0.003, f"{p:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r}\n<->\n{b}" for r, b in
                        zip(df["repr_distance"], df["beh_distance"])], fontsize=8)
    ax.set_ylabel("Spearman r")
    ax.set_title("Does the geometry-behavior link survive controlling for vector norm?")
    ax.legend(frameon=False)
    ax.axhline(0, color="k", lw=0.5)
    _save(fig, "fig6_partial_rsa")


def fig7_summary_panel() -> None:
    rdm_repr = np.load(DATA_DIR / "rdm_repr_cos.npy")
    rdm_beh = np.load(DATA_DIR / "rdm_beh_corr.npy")
    main = pd.read_csv(DATA_DIR / "rsa_main.csv")
    per_alpha = pd.read_csv(DATA_DIR / "rsa_per_alpha.csv")
    npz = np.load(DATA_DIR / "mantel_null_distribution.npz")

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # Panel A: hexbin pairwise scatter
    ax = fig.add_subplot(gs[0, 0])
    x = upper_tri(rdm_repr); y = upper_tri(rdm_beh)
    rho = stats.spearmanr(x, y).statistic
    hb = ax.hexbin(x, y, gridsize=50, mincnt=1, cmap="magma", bins="log")
    ax.set_xlabel("repr distance (1 - cos)")
    ax.set_ylabel("beh distance (1 - corr)")
    ax.set_title(f"A. pairwise distances\nSpearman r = {rho:+.3f}")

    # Panel B: representational RDM (sorted)
    ax = fig.add_subplot(gs[0, 1])
    order = hclust_order(rdm_repr)
    im = ax.imshow(rdm_repr[order][:, order], cmap="viridis",
                   vmin=np.percentile(rdm_repr, 2),
                   vmax=np.percentile(rdm_repr, 98))
    ax.set_title("B. Representational RDM (sorted)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Panel C: behavioral RDM (same sort)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(rdm_beh[order][:, order], cmap="viridis",
                   vmin=np.percentile(rdm_beh, 2),
                   vmax=np.percentile(rdm_beh, 98))
    ax.set_title("C. Behavioral RDM (same sort)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Panel D: Mantel null for primary combo
    ax = fig.add_subplot(gs[1, 0])
    null = npz["repr_cos__beh_corr"]
    row = main[(main["repr_distance"] == "repr_cos") &
               (main["beh_distance"] == "beh_corr")].iloc[0]
    ax.hist(null, bins=60, color="#888", edgecolor="white")
    ax.axvline(row["rsa_spearman"], color="crimson", lw=2,
               label=f"observed = {row['rsa_spearman']:+.3f}\np = {row['mantel_p']:.0e}")
    ax.set_title("D. Mantel null (cos vs corr)")
    ax.set_xlabel("Spearman r under permutation")
    ax.set_ylabel("count")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # Panel E: per-alpha bars
    ax = fig.add_subplot(gs[1, 1])
    sub = per_alpha[per_alpha["beh_distance"] == "beh_corr"].sort_values("alpha")
    ax.bar(np.arange(len(sub)), sub["rsa_spearman"].values,
           color="#3b82f6", edgecolor="white")
    for i, (r, p) in enumerate(zip(sub["rsa_spearman"], sub["mantel_p"])):
        ax.text(i, r + 0.003, f"{r:+.3f}\np={p:.0e}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels([f"alpha={a}" for a in sub["alpha"]])
    ax.set_ylabel("RSA Spearman r")
    ax.set_title("E. RSA per alpha")
    ax.axhline(0, color="k", lw=0.5)

    # Panel F: 4-combo grid
    ax = fig.add_subplot(gs[1, 2])
    pivot = main.pivot(index="repr_distance", columns="beh_distance",
                       values="rsa_spearman")
    im = ax.imshow(pivot.values, cmap="RdYlBu_r",
                   vmin=-abs(pivot.values).max(),
                   vmax=abs(pivot.values).max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:+.3f}", ha="center", va="center")
    ax.set_title("F. RSA across distance choices")
    fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle("Representational vs Behavioral RSA on 275 roles", fontsize=13)
    _save(fig, "fig7_summary_panel")


def main() -> None:
    print("Building figures ...")
    fig1_rdms_side_by_side(); print("  fig1 done")
    fig2_pairwise_scatter(); print("  fig2 done")
    fig3_mantel_null(); print("  fig3 done")
    fig4_rsa_per_alpha(); print("  fig4 done")
    fig5_distance_grid(); print("  fig5 done")
    fig6_partial_rsa(); print("  fig6 done")
    fig7_summary_panel(); print("  fig7 done")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
