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

    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
