"""
Figures for the intrinsic-dimensionality analysis.

Reads:
    data/spectrum_repr_raw.csv
    data/spectrum_repr_centered.csv
    data/spectrum_beh.csv
    data/spectrum_random.csv
    data/explained_variance_curves.csv
    data/summary.csv

Writes (figures/):
    fig1_singular_value_spectrum.png    : sigma_i vs i, log-y, all four matrices
    fig2_explained_variance_curves.png  : cumulative variance vs d
    fig3_effective_rank_summary.png     : bar chart of effective rank / PR
    fig4_pairwise_cosine_distribution.png : anisotropy diagnostic
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "repr_raw": "#1d4ed8",
    "repr_centered": "#3b82f6",
    "behavioral": "#10b981",
    "repr_random_null": "#9ca3af",
}
LABELS = {
    "repr_raw": "role vectors (raw)",
    "repr_centered": "role vectors (centered)",
    "behavioral": "behavioral profiles (24-d)",
    "repr_random_null": "iid Gaussian null (same shape)",
}


def fig1_spectrum() -> None:
    files = {
        "repr_raw": "spectrum_repr_raw.csv",
        "repr_centered": "spectrum_repr_centered.csv",
        "behavioral": "spectrum_beh.csv",
        "repr_random_null": "spectrum_random.csv",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, fn in files.items():
        df = pd.read_csv(DATA_DIR / fn)
        ax.plot(df["rank_index"], df["singular_value"],
                color=COLORS[k], label=LABELS[k], lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("rank index i")
    ax.set_ylabel("singular value $\\sigma_i$  (log scale)")
    ax.set_title("Singular-value spectra")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "fig1_singular_value_spectrum.png")
    plt.close(fig)


def fig2_explained_variance() -> None:
    df = pd.read_csv(DATA_DIR / "explained_variance_curves.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in ["repr_raw", "repr_centered", "behavioral", "repr_random_null"]:
        sub = df[df["matrix"] == k].sort_values("rank_index")
        ax.plot(sub["rank_index"], sub["cumulative_variance"],
                color=COLORS[k], label=LABELS[k], lw=2)
    for thr in [0.5, 0.9, 0.99]:
        ax.axhline(thr, color="k", lw=0.4, ls=":")
        ax.text(1, thr + 0.005, f"{int(thr*100)}%", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of components d")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title("How few directions explain the cloud?  (zoom: first 100)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(FIG_DIR / "fig2_explained_variance_curves.png")
    plt.close(fig)


def fig3_effective_rank_summary() -> None:
    s = pd.read_csv(DATA_DIR / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics = [
        ("effective_rank", "Effective rank  exp(-sum p log p)"),
        ("participation_ratio", "Participation ratio  $(\\sum\\sigma^2)^2/\\sum\\sigma^4$"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        xs = np.arange(len(s))
        for x, mat, v in zip(xs, s["matrix"], s[col]):
            ax.bar(x, v, color=COLORS.get(mat, "#888"), edgecolor="white")
            ax.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS.get(m, m) for m in s["matrix"]],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
    axes[0].set_ylabel("dimensions")
    fig.suptitle("Effective dimensionality of the role cloud", fontsize=12)
    fig.savefig(FIG_DIR / "fig3_effective_rank_summary.png")
    plt.close(fig)


def fig4_pairwise_cosine() -> None:
    s = pd.read_csv(DATA_DIR / "summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(s))
    for x, mat, v in zip(xs, s["matrix"], s["mean_pairwise_cosine"]):
        ax.bar(x, v, color=COLORS.get(mat, "#888"), edgecolor="white")
        ax.text(x, v + 0.005, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS.get(m, m) for m in s["matrix"]],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean pairwise cosine")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Anisotropy: high mean cosine => global bias direction "
                 "(less room for role-specific structure)")
    fig.savefig(FIG_DIR / "fig4_pairwise_cosine_distribution.png")
    plt.close(fig)


def main() -> None:
    fig1_spectrum(); print("  fig1 done")
    fig2_explained_variance(); print("  fig2 done")
    fig3_effective_rank_summary(); print("  fig3 done")
    fig4_pairwise_cosine(); print("  fig4 done")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
