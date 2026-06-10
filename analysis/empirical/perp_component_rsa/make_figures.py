"""
Figures for perp-component RSA analysis.

  Fig 1: RSA grouped bar — perp_rdm and aa_vec_rdm vs behavioral/repr RDMs
  Fig 2: Differential RSA — perp favours steered, aa_vec favours AA?
  Fig 3: Scatter — pairwise perp distance vs behavioral distance (steered vs AA)

Usage:
    python analysis/empirical/perp_component_rsa/make_figures.py
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
PERP_COLOR = REPR
AA_VEC_COLOR = ACCENT
REPR_COLOR = BASELINE

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    stem = name.replace(".pdf", "").replace(".png", "")
    plot_style.save_fig(fig, OUT_DIR, stem)


def _get(rsa_df: pd.DataFrame, rdm_a: str, rdm_b: str) -> tuple[float, float]:
    row = rsa_df[(rsa_df["rdm_a"] == rdm_a) & (rsa_df["rdm_b"] == rdm_b)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row.iloc[0]["rsa_spearman"]), float(row.iloc[0]["mantel_p"])


def _sig(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))


# ── Figure 1: Full RSA comparison bar ─────────────────────────────────────────

def fig1_rsa_bars(rsa_df: pd.DataFrame) -> None:
    entries = [
        ("perp_cos",   "beh_steered_corr", "v_⊥\nvs beh_steered",   STEERED_COLOR),
        ("perp_cos",   "beh_aa_corr",      "v_⊥\nvs beh_aa",         AXIS_COLOR),
        ("perp_cos",   "repr_cos",          "v_⊥\nvs repr_cos",       PERP_COLOR),
        ("aa_vec_cos", "beh_aa_corr",      "aa_vec\nvs beh_aa",      AXIS_COLOR),
        ("aa_vec_cos", "beh_steered_corr", "aa_vec\nvs beh_steered", STEERED_COLOR),
        ("aa_vec_cos", "repr_cos",          "aa_vec\nvs repr_cos",    AA_VEC_COLOR),
        ("perp_cos",   "aa_vec_cos",        "v_⊥\nvs aa_vec",         REPR_COLOR),
        ("repr_cos",   "beh_steered_corr", "repr_cos\nvs beh_steered", REPR_COLOR),
        ("repr_cos",   "beh_aa_corr",      "repr_cos\nvs beh_aa",    REPR_COLOR),
    ]

    labels, rs, ps, colors = [], [], [], []
    for a, b, lbl, col in entries:
        r, p = _get(rsa_df, a, b)
        labels.append(lbl)
        rs.append(r)
        ps.append(p)
        colors.append(col)

    fig, ax = plt.subplots(figsize=(15, 5))
    bars = ax.bar(range(len(labels)), rs, color=colors, alpha=0.85, edgecolor="white")
    for bar, r_val, p_val in zip(bars, rs, ps):
        if not np.isfinite(r_val):
            continue
        sig = _sig(p_val)
        y = bar.get_height() + 0.003 if r_val >= 0 else bar.get_height() - 0.015
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{r_val:+.3f}{sig}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)

    for x in [2.5, 5.5, 6.5]:
        ax.axvline(x, color="lightgray", linewidth=1.2, linestyle=":")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mantel RSA (Spearman r)")
    ax.set_title("RSA between geometric components and behavioral / representational RDMs"
                 " (* p<0.05, ** p<0.01, *** p<0.001)")

    ylim = ax.get_ylim()
    ax.text(1,   ylim[1] * 0.96, "v_⊥ (role-specific)", ha="center",
            fontsize=9, color=PERP_COLOR, fontweight="bold")
    ax.text(4.5, ylim[1] * 0.96, "aa_vec (AA repr.)",   ha="center",
            fontsize=9, color=AA_VEC_COLOR, fontweight="bold")
    ax.text(7.5, ylim[1] * 0.96, "reference",           ha="center",
            fontsize=9, color="gray")

    plt.tight_layout()
    _save(fig, "fig1_rsa_component_bars.pdf")


# ── Figure 2: Differential RSA ────────────────────────────────────────────────

def fig2_differential_rsa(rsa_df: pd.DataFrame) -> None:
    perp_s, _ = _get(rsa_df, "perp_cos",   "beh_steered_corr")
    perp_a, _ = _get(rsa_df, "perp_cos",   "beh_aa_corr")
    aa_a,   _ = _get(rsa_df, "aa_vec_cos", "beh_aa_corr")
    aa_s,   _ = _get(rsa_df, "aa_vec_cos", "beh_steered_corr")

    diffs = [
        ("RSA(v_⊥, beh_steered) − RSA(v_⊥, beh_aa)\n"
         "(positive → role-specific component drives steered more)",
         perp_s - perp_a, STEERED_COLOR),
        ("RSA(aa_vec, beh_aa) − RSA(aa_vec, beh_steered)\n"
         "(positive → AA vector structure drives AA behavior more)",
         aa_a - aa_s, AXIS_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels, vals, colors = zip(*diffs)
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85,
                  edgecolor="white", width=0.45)
    for bar, val in zip(bars, vals):
        if not np.isfinite(val):
            continue
        y = bar.get_height() + 0.001 if val >= 0 else bar.get_height() - 0.006
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:+.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ΔRSA (Spearman r)")
    ax.set_title("Geometric subspace selectivity: does each component predict its method's behavior better?")
    plt.tight_layout()
    _save(fig, "fig2_differential_rsa.pdf")


# ── Figure 3: Scatter — pairwise perp distance vs behavioral distance ──────────

def fig3_perp_vs_behavioral() -> None:
    perp_rdm   = np.load(DATA_DIR / "perp_rdm.npy")
    beh_s_rdm  = np.load(DATA_DIR / "beh_steered_rdm.npy")
    beh_aa_rdm = np.load(DATA_DIR / "beh_aa_rdm.npy")

    n = perp_rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    perp_vec = perp_rdm[idx]

    rng = np.random.default_rng(0)
    sample = rng.choice(len(perp_vec), size=min(5_000, len(perp_vec)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    rsa_df = pd.read_csv(DATA_DIR / "rsa_perp_components.csv")

    for ax, beh_rdm, beh_name, beh_rdm_key, color in [
        (axes[0], beh_s_rdm,  "Steered behavioral distance", "beh_steered_corr", STEERED_COLOR),
        (axes[1], beh_aa_rdm, "AA behavioral distance",      "beh_aa_corr",      AXIS_COLOR),
    ]:
        beh_vec = beh_rdm[idx]
        ax.scatter(perp_vec[sample], beh_vec[sample],
                   s=4, alpha=0.3, color=color, edgecolors="none")
        m, b = np.polyfit(perp_vec[sample], beh_vec[sample], 1)
        xr = np.linspace(perp_vec.min(), perp_vec.max(), 200)
        ax.plot(xr, m * xr + b, color="black", linewidth=1.5)

        r, p = _get(rsa_df, "perp_cos", beh_rdm_key)
        ax.text(0.05, 0.95, f"RSA r = {r:+.3f}\n{_sig(p) or 'n.s.'}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel("Pairwise v_⊥ cosine distance")
        ax.set_ylabel(beh_name)
        ax.set_title(f"v_⊥ RDM vs {beh_name.split()[0].lower()} behavioral RDM")

    fig.suptitle(f"Do roles with similar role-specific residuals (v_⊥) behave similarly?"
                 f"  ({min(5_000, len(perp_vec)):,} sampled role pairs)",
                 y=1.02)
    plt.tight_layout()
    _save(fig, "fig3_perp_vs_behavioral_scatter.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rsa_df = pd.read_csv(DATA_DIR / "rsa_perp_components.csv")
    print(f"Loaded {len(rsa_df)} RSA comparisons")

    fig1_rsa_bars(rsa_df)
    fig2_differential_rsa(rsa_df)
    fig3_perp_vs_behavioral()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
