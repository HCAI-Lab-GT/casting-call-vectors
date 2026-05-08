"""
Figures for perp-component RSA analysis.

  Fig 1: RSA grouped bar — perp_rdm vs {beh_steered, beh_aa, repr_cos}
                         — parallel_rdm vs {beh_steered, beh_aa, repr_cos}
  Fig 2: Difference bars — Δr = RSA(perp, beh_steered) − RSA(perp, beh_aa)
                             and Δr = RSA(parallel, beh_aa) − RSA(parallel, beh_steered)
  Fig 3: Scatter — pairwise perp cosine similarity vs behavioral distance
                   (steered vs AA side-by-side)

Usage:
    python analysis/empirical/perp_component_rsa/make_figures.py
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
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEERED_COLOR = "#2ecc71"
AXIS_COLOR = "#3498db"
PERP_COLOR = "#9b59b6"
PAR_COLOR = "#e67e22"
REPR_COLOR = "#95a5a6"


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {path}")


def _get(rsa_df: pd.DataFrame, rdm_a: str, rdm_b: str) -> tuple[float, float]:
    row = rsa_df[(rsa_df["rdm_a"] == rdm_a) & (rsa_df["rdm_b"] == rdm_b)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row.iloc[0]["rsa_spearman"]), float(row.iloc[0]["mantel_p"])


def _sig(p: float) -> str:
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))


# ── Figure 1: Full RSA comparison bar ─────────────────────────────────────────

def fig1_rsa_bars(rsa_df: pd.DataFrame) -> None:
    # Group: (rdm_a, rdm_b, label, color)
    entries = [
        ("perp_cos",     "beh_steered_corr", "perp\nvs beh_steered",  STEERED_COLOR),
        ("perp_cos",     "beh_aa_corr",      "perp\nvs beh_aa",       AXIS_COLOR),
        ("perp_cos",     "repr_cos",          "perp\nvs repr_cos",     PERP_COLOR),
        ("parallel_cos", "beh_steered_corr", "parallel\nvs beh_steered", STEERED_COLOR),
        ("parallel_cos", "beh_aa_corr",      "parallel\nvs beh_aa",   AXIS_COLOR),
        ("parallel_cos", "repr_cos",          "parallel\nvs repr_cos", PAR_COLOR),
        ("repr_cos",     "beh_steered_corr", "repr_cos\nvs beh_steered", REPR_COLOR),
        ("repr_cos",     "beh_aa_corr",      "repr_cos\nvs beh_aa",   REPR_COLOR),
    ]

    labels, rs, ps, colors = [], [], [], []
    for a, b, lbl, col in entries:
        r, p = _get(rsa_df, a, b)
        labels.append(lbl)
        rs.append(r)
        ps.append(p)
        colors.append(col)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(labels)), rs, color=colors, alpha=0.85, edgecolor="white")
    for bar, r_val, p_val in zip(bars, rs, ps):
        sig = _sig(p_val)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003 if r_val >= 0 else bar.get_height() - 0.015,
                f"{r_val:+.3f}{sig}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)

    # Separator between perp / parallel / repr groups
    for x in [2.5, 5.5]:
        ax.axvline(x, color="lightgray", linewidth=1.2, linestyle=":")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mantel RSA (Spearman r)")
    ax.set_title("RSA between geometric components and behavioral / representational RDMs\n"
                 "(* p<0.05, ** p<0.01, *** p<0.001, Mantel permutation)")

    ax.text(1, ax.get_ylim()[1] * 0.97, "perp (v_⊥)", ha="center", fontsize=9,
            color=PERP_COLOR, fontweight="bold")
    ax.text(4, ax.get_ylim()[1] * 0.97, "parallel (v_∥)", ha="center", fontsize=9,
            color=PAR_COLOR, fontweight="bold")
    ax.text(6.5, ax.get_ylim()[1] * 0.97, "repr_cos\n(reference)", ha="center", fontsize=9,
            color="gray")

    plt.tight_layout()
    _save(fig, "fig1_rsa_component_bars.pdf")


# ── Figure 2: Differential RSA ─────────────────────────────────────────────────

def fig2_differential_rsa(rsa_df: pd.DataFrame) -> None:
    perp_s, _ = _get(rsa_df, "perp_cos", "beh_steered_corr")
    perp_a, _ = _get(rsa_df, "perp_cos", "beh_aa_corr")
    par_s,  _ = _get(rsa_df, "parallel_cos", "beh_steered_corr")
    par_a,  _ = _get(rsa_df, "parallel_cos", "beh_aa_corr")

    diffs = [
        ("RSA(perp, beh_steered)\n− RSA(perp, beh_aa)\n(positive → perp drives steered more)",
         perp_s - perp_a, STEERED_COLOR),
        ("RSA(parallel, beh_aa)\n− RSA(parallel, beh_steered)\n(positive → parallel drives AA more)",
         par_a - par_s, AXIS_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels, vals, colors = zip(*diffs)
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85,
                  edgecolor="white", width=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001 if val >= 0 else bar.get_height() - 0.005,
                f"{val:+.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ΔRSA (Spearman r)")
    ax.set_title("Geometric subspace selectivity:\nwhich component predicts which method's behavior?")
    plt.tight_layout()
    _save(fig, "fig2_differential_rsa.pdf")


# ── Figure 3: Scatter — pairwise perp sim vs behavioral distance ───────────────

def fig3_perp_vs_behavioral(rsa_df: pd.DataFrame) -> None:
    perp_rdm = np.load(DATA_DIR / "perp_rdm.npy")
    beh_s_rdm_path = (
        Path(__file__).resolve().parents[3]
        / "analysis" / "empirical" / "rsa_geometry_behavior" / "data" / "rdm_beh_corr.npy"
    )
    beh_aa_rdm_path = (
        Path(__file__).resolve().parents[3]
        / "analysis" / "empirical" / "behavioral_diversity" / "data" / "rdm_beh_aa_corr.npy"
    )
    if not beh_s_rdm_path.exists() or not beh_aa_rdm_path.exists():
        print("Skipping fig3: behavioral RDMs not found")
        return

    beh_s_rdm = np.load(beh_s_rdm_path)
    beh_aa_rdm = np.load(beh_aa_rdm_path)

    n = perp_rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    perp_vec = perp_rdm[idx]

    # Subsample for scatter readability
    rng = np.random.default_rng(0)
    sample = rng.choice(len(perp_vec), size=min(5000, len(perp_vec)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, beh_rdm, label, color in [
        (axes[0], beh_s_rdm, "Steered behavioral distance", STEERED_COLOR),
        (axes[1], beh_aa_rdm, "AA behavioral distance", AXIS_COLOR),
    ]:
        beh_vec = beh_rdm[idx]
        ax.scatter(perp_vec[sample], beh_vec[sample], s=4, alpha=0.3,
                   color=color, edgecolors="none")
        # Trend line
        m, b = np.polyfit(perp_vec[sample], beh_vec[sample], 1)
        xr = np.linspace(perp_vec.min(), perp_vec.max(), 200)
        ax.plot(xr, m * xr + b, color="black", linewidth=1.5)

        # Annotate RSA
        rdm_a_name = "perp_cos"
        rdm_b_name = "beh_steered_corr" if "Steered" in label else "beh_aa_corr"
        r, p = _get(rsa_df, rdm_a_name, rdm_b_name)
        ax.text(0.05, 0.95, f"RSA r = {r:+.3f}\n{_sig(p) or 'n.s.'}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel("Pairwise perp-component cosine distance (1 − cos(v⊥_i, v⊥_j))")
        ax.set_ylabel(label)
        ax.set_title(f"perp_rdm vs {label.split()[0].lower()} behavioral RDM")

    fig.suptitle("Do roles with similar role-specific residuals (v_⊥) show similar behavior?\n"
                 "(subsample of 5 000 role pairs shown)", y=1.02)
    plt.tight_layout()
    _save(fig, "fig3_perp_vs_behavioral_scatter.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rsa_df = pd.read_csv(DATA_DIR / "rsa_perp_components.csv")
    print(f"Loaded {len(rsa_df)} RSA comparisons")

    fig1_rsa_bars(rsa_df)
    fig2_differential_rsa(rsa_df)
    fig3_perp_vs_behavioral(rsa_df)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
