"""Regenerate the Sec 5 / geometry-appendix figures (Workstream C).

Data comes from the empirical-geometry branch analyses (PR #14 merges them
to main as analysis/empirical/...). Until that merge lands, files are read
via `git show origin/empirical-geometry:<path>` fallback, so this script
works on any checkout either way.

Figures (added one at a time, each behind its verification gate):
  out/fig_rsa_grid.pdf  -> fig:rsa-per-alpha (right panel)

Conventions / corrections discovered here:
  - run_analysis.py builds 275x275 RDMs (assistant INCLUDED); the paper's
    Sec 5.3 said "274x274" -- prose corrected to 275 in the same commit,
    since the published RSA values (0.137; 0.085->0.179) reproduce only
    from the 275-role matrices.
  - rsa_partial.csv: the Sec 4.1 "Delta-rho = -0.015" is the (repr_cos,
    beh_l2) cell; the convention-consistent (repr_cos, beh_corr) cell is
    -0.005. Pending Glenn's call on which to cite.
"""

import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pvx_fig_style as style

REPO = Path(__file__).resolve().parents[2]
BRANCH = "origin/empirical-geometry"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def read_branch_csv(rel_path: str) -> pd.DataFrame:
    """Read a repo CSV from disk if present, else from the geometry branch."""
    local = REPO / rel_path
    if local.exists():
        return pd.read_csv(local)
    raw = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BRANCH}:{rel_path}"],
        check=True, capture_output=True, text=True).stdout
    return pd.read_csv(io.StringIO(raw))


def verify(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name}: got {got:.4f}, paper {want}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def fig_rsa_grid():
    main = read_branch_csv(
        "analysis/empirical/rsa_geometry_behavior/data/rsa_main.csv")
    grid = main.pivot(index="repr_distance", columns="beh_distance",
                      values="rsa_spearman")
    grid = grid.reindex(index=["repr_cos", "repr_l2"],
                        columns=["beh_corr", "beh_l2"])
    pvals = main.pivot(index="repr_distance", columns="beh_distance",
                       values="mantel_p").reindex_like(grid)

    fig, ax = plt.subplots(figsize=(style.HALF_PANEL_W_IN, 1.45))
    ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0, vmax=0.3)
    for i in range(2):
        for j in range(2):
            v = grid.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if v > 0.18 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["corr.", "L2"], fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["cosine", "L2"], fontsize=7)
    ax.set_xlabel("behavioral distance", fontsize=7)
    ax.set_ylabel("repr. distance", fontsize=7)
    fig.savefig(OUT / "fig_rsa_grid.pdf")
    plt.close(fig)
    print("  Mantel p per cell:", {f"{r}x{c}": float(pvals.loc[r, c])
          for r in pvals.index for c in pvals.columns})
    return grid


def fig_effective_rank():
    curves = read_branch_csv(
        "analysis/empirical/intrinsic_dimensionality/data/explained_variance_curves.csv")
    summary = read_branch_csv(
        "analysis/empirical/intrinsic_dimensionality/data/summary.csv"
    ).set_index("matrix")
    series = {
        "repr_raw": ("raw (ER 3)", style.GREY, "-"),
        "repr_centered": ("centered (ER 50)", style.BLUE, "-"),
        "behavioral": ("behavior (ER 2)", style.VERMILLION, "-"),
        "repr_random_null": ("null (ER 266)", style.GREEN, ":"),
    }
    fig, ax = plt.subplots(figsize=(style.HALF_PANEL_W_IN, 1.45))
    for key, (label, color, ls) in series.items():
        sub = curves[curves["matrix"] == key]
        assert len(sub) > 0, f"no curve rows for matrix key {key!r}"
        ax.plot(sub["rank_index"], sub["cumulative_variance"], color=color,
                ls=ls, lw=1.1, label=label)
    ax.axhline(0.9, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("rank")
    ax.set_ylabel("cumul. variance")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5, loc="lower right", handlelength=1.0,
              labelspacing=0.25, borderaxespad=0.2, frameon=True,
              framealpha=0.95, edgecolor="none", facecolor="white")
    fig.savefig(OUT / "fig_effective_rank.pdf")
    plt.close(fig)
    return summary


def fig_noise_corrected_rsa():
    nc = read_branch_csv(
        "analysis/empirical/behavioral_noise_floor/data/noise_corrected_rsa.csv")
    labels = [f"{r.replace('repr_', '').replace('l2', 'L2')}·"
              f"{c.replace('beh_', '').replace('l2', 'L2')}"
              for r, c in zip(nc["repr_distance"], nc["beh_distance"])]
    x = np.arange(len(nc))
    fig, ax = plt.subplots(figsize=(style.HALF_PANEL_W_IN, 1.45))
    ax.bar(x - 0.19, nc["rsa_observed"], width=0.36, color=style.BLUE,
           label="observed")
    ax.bar(x + 0.19, nc["rsa_noise_corrected"], width=0.36,
           color=style.SKY, label="noise-corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5, rotation=20, ha="right")
    ax.set_ylabel("RSA (Spearman)")
    ax.set_ylim(0, 0.32)
    ax.text(0.03, 0.97, "ceiling 0.97–0.99", transform=ax.transAxes,
            fontsize=5.5, color=style.GREY, va="top")
    ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, 0.88),
              handlelength=1.0)
    fig.savefig(OUT / "fig_noise_corrected_rsa.pdf")
    plt.close(fig)
    return nc


def fig_norm_curves():
    stab = read_branch_csv(
        "analysis/empirical/vector_magnitude_effect_on_stability/data/vector_magnitude_stability.csv")
    stab["quartile"] = pd.qcut(stab["norm"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    alphas = [1.0, 1.5, 2.0, 2.5]
    cols = [f"score_at_alpha_{str(a).replace('.', '_')}" for a in alphas]
    palette = {"Q1": style.SKY, "Q2": style.GREEN, "Q3": style.ORANGE,
               "Q4": style.BLUE}
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 1.6))
    for q, sub in stab.groupby("quartile", observed=True):
        mean = sub[cols].mean()
        sem = sub[cols].sem()
        ax.plot(alphas, mean, color=palette[str(q)], marker="o", ms=2,
                label=f"{q} (n={len(sub)})")
        ax.fill_between(alphas, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=palette[str(q)], alpha=0.15, lw=0)
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel("mean judge score")
    ax.set_xticks(alphas)
    ax.legend(fontsize=5.5, loc="upper left", title="norm quartile",
              title_fontsize=5.5, handlelength=1.2)
    fig.savefig(OUT / "fig_norm_alpha_curves.pdf")
    plt.close(fig)
    peak_early = (stab["peak_alpha"] < 2.5).groupby(
        stab["quartile"], observed=True).mean()
    return stab, peak_early


def main():
    style.apply()
    print("verification gate (RSA):")
    grid = fig_rsa_grid()
    verify("RSA cos x corr (headline)", float(grid.loc["repr_cos", "beh_corr"]),
           0.137, 0.001)
    verify("RSA cos x L2", float(grid.loc["repr_cos", "beh_l2"]), 0.233, 0.001)
    verify("RSA L2 x corr", float(grid.loc["repr_l2", "beh_corr"]), 0.068, 0.001)
    verify("RSA L2 x L2", float(grid.loc["repr_l2", "beh_l2"]), 0.263, 0.001)

    per_alpha = read_branch_csv(
        "analysis/empirical/rsa_geometry_behavior/data/rsa_per_alpha.csv")
    cc = per_alpha[(per_alpha.repr_distance == "repr_cos")
                   & (per_alpha.beh_distance == "beh_corr")]
    for a, want in [(1.0, 0.085), (1.5, 0.092), (2.0, 0.121), (2.5, 0.179)]:
        got = float(cc[cc.alpha == a]["rsa_spearman"].iloc[0])
        verify(f"per-alpha RSA cos x corr @ {a}", got, want, 0.001)

    print("verification gate (noise floor / dimensionality):")
    summary = fig_effective_rank()
    verify("centered cloud effective rank",
           float(summary.loc["repr_centered", "effective_rank"]), 49.6, 0.1)
    verify("centered cloud PR",
           float(summary.loc["repr_centered", "participation_ratio"]),
           15.1, 0.05)
    verify("behavioral effective rank",
           float(summary.loc["behavioral", "effective_rank"]), 2.0, 0.05)
    verify("behavioral PR",
           float(summary.loc["behavioral", "participation_ratio"]), 1.45, 0.05)
    nf = read_branch_csv(
        "analysis/empirical/behavioral_noise_floor/data/noise_floor_summary.csv"
    ).set_index("metric")
    verify("split-half Spearman-Brown (corr)",
           float(nf.loc["full_corr_spearmanbrown", "mean"]), 0.97, 0.005)
    nc = fig_noise_corrected_rsa()
    max_delta = float((nc["rsa_noise_corrected"] - nc["rsa_observed"]).abs().max())
    verify("max noise-correction delta (paper: within 0.02)", max_delta,
           0.005, 0.005)

    print("verification gate (norm-slope):")
    corr = read_branch_csv(
        "analysis/empirical/vector_magnitude_effect_on_stability/data/correlations.csv")
    corr = corr[corr["magnitude_metric"] == "norm"].set_index(
        "stability_metric")
    verify("norm x alpha-slope Pearson r",
           float(corr.loc["score_slope", "pearson_r"]), 0.49, 0.005)
    verify("norm x (2.5-1.0) delta Pearson r",
           float(corr.loc["delta_high_minus_low", "pearson_r"]), 0.49, 0.005)
    stab, peak_early = fig_norm_curves()
    assert len(stab) == 275
    # branch README records 28/38/22/12% by quartile; the paper's old
    # "28-38% of bottom-three-quartile roles" mis-scoped Q3 (22%) --
    # Sec 5.1 prose corrected to 22-38% in the same commit.
    for q, want in [("Q1", 0.275), ("Q2", 0.377), ("Q3", 0.221),
                    ("Q4", 0.116)]:
        verify(f"{q} early-peak fraction", float(peak_early[q]), want, 0.005)

    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
