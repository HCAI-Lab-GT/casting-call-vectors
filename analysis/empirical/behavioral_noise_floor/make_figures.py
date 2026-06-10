"""
Figures for the behavioral-noise-floor / RSA-ceiling analysis.

Reads:
    data/noise_floor_split_half.csv
    data/noise_floor_summary.csv
    data/noise_corrected_rsa.csv

Writes (figures/):
    fig1_split_half_distribution.png : per-seed split-half reliability histogram
    fig2_observed_vs_ceiling.png     : observed RSA bars next to ceiling lines
    fig3_noise_corrected_rsa.png     : raw RSA / ceiling (fraction of explainable)
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


def fig1_distribution() -> None:
    df = pd.read_csv(DATA_DIR / "noise_floor_split_half.csv")
    summary = pd.read_csv(DATA_DIR / "noise_floor_summary.csv").set_index("metric")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pairs = [
        ("split_half_corr", "full_corr_spearmanbrown",
         "behavioral RDM = corr-distance"),
        ("split_half_l2", "full_l2_spearmanbrown",
         "behavioral RDM = L2 (z-scored)"),
    ]
    for ax, (half_col, full_col, title) in zip(axes, pairs):
        ax.hist(df[half_col], bins=40, color="#888", edgecolor="white",
                alpha=0.7, label="half-data reliability")
        ax.hist(df[full_col], bins=40, color="#3b82f6", edgecolor="white",
                alpha=0.7, label="full-data (Spearman-Brown)")
        for col, color, ls in [(half_col, "#444", ":"), (full_col, "#1d4ed8", "--")]:
            m = summary.loc[col, "mean"]
            ax.axvline(m, color=color, lw=1.6, ls=ls)
        ax.set_xlabel("Spearman r between RDM_A and RDM_B")
        ax.set_ylabel("count of random splits")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Split-half reliability of the behavioral RDM (200 seeds)", fontsize=12)
    fig.savefig(FIG_DIR / "fig1_split_half_distribution.png")
    plt.close(fig)


def fig2_observed_vs_ceiling() -> None:
    nc = pd.read_csv(DATA_DIR / "noise_corrected_rsa.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(nc))
    bars = ax.bar(
        xs, nc["rsa_observed"], color="#3b82f6", edgecolor="white",
        label="observed RSA",
    )
    # Plot ceiling as a horizontal line per bar
    for x, c in zip(xs, nc["ceiling"]):
        ax.hlines(c, x - 0.4, x + 0.4, color="crimson", lw=2.2, zorder=3)
    # crimson legend handle
    ax.plot([], [], color="crimson", lw=2.2, label="reliability ceiling (S-B)")
    for x, r, c, p in zip(xs, nc["rsa_observed"], nc["ceiling"], nc["mantel_p"]):
        frac = r / c if c > 0 else float("nan")
        ax.text(
            x, max(r, c) + 0.01,
            f"r={r:+.3f}\np={p:.0e}\n{frac*100:.0f}% of ceiling",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r}\n<->\n{b}" for r, b in zip(nc["repr_distance"], nc["beh_distance"])],
        fontsize=8,
    )
    ax.set_ylabel("Spearman r")
    ax.set_ylim(0, max(nc["ceiling"].max(), nc["rsa_observed"].max()) * 1.30)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Observed geometry-behavior RSA vs the achievable ceiling")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(FIG_DIR / "fig2_observed_vs_ceiling.png")
    plt.close(fig)


def fig3_noise_corrected() -> None:
    nc = pd.read_csv(DATA_DIR / "noise_corrected_rsa.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = np.arange(len(nc))
    nc_vals = nc["rsa_noise_corrected"].to_numpy()
    bars = ax.bar(xs, nc_vals, color="#10b981", edgecolor="white")
    for x, v in zip(xs, nc_vals):
        ax.text(x, v + 0.01, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r}\n<->\n{b}" for r, b in zip(nc["repr_distance"], nc["beh_distance"])],
        fontsize=8,
    )
    ax.set_ylabel("noise-corrected RSA  (observed / ceiling)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="crimson", lw=1, ls="--",
               label="theoretical maximum = 1.0")
    ax.set_ylim(0, max(nc_vals.max() * 1.4, 0.4))
    ax.set_title("Fraction of explainable behavioral variance captured by geometry")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "fig3_noise_corrected_rsa.png")
    plt.close(fig)


def main() -> None:
    fig1_distribution(); print("  fig1 done")
    fig2_observed_vs_ceiling(); print("  fig2 done")
    fig3_noise_corrected(); print("  fig3 done")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
