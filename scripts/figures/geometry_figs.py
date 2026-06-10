"""Regenerate the Sec 5 / geometry-appendix figures (Workstream C).

Data comes from the empirical-geometry branch analyses (PR #14 merges them
to main as analysis/empirical/...). Until that merge lands, files are read
via `git show origin/empirical-geometry:<path>` fallback, so this script
works on any checkout either way.

Figures (added one at a time, each behind its verification gate):
  out/fig_rsa_grid.pdf            -> fig:rsa-per-alpha (right panel)
  out/fig_effective_rank.pdf      -> fig:noise-floor (left panel)
  out/fig_noise_corrected_rsa.pdf -> fig:noise-floor (right panel)
  out/fig_norm_alpha_curves.pdf   -> fig:norm-slope
  out/fig_distance_summary.pdf    -> fig:distance-summary (4 panels)
  out/fig_pc_metric_grid.pdf      -> fig:pc-grid (20 PCs x 6 metrics x 4 alphas)
  out/fig_trait_metric_heatmap.pdf -> fig:trait-heatmap (14 axes x 6 metrics)

Conventions / corrections discovered here:
  - run_analysis.py builds 275x275 RDMs (assistant INCLUDED); the paper's
    Sec 5.3 said "274x274" -- prose corrected to 275 in the same commit,
    since the published RSA values (0.137; 0.085->0.179) reproduce only
    from the 275-role matrices.
  - rsa_partial.csv: the originally published Sec 4.1 "Delta-rho = -0.015"
    is the (repr_cos, beh_l2) cell; the paper now cites the convention-
    consistent (repr_cos, beh_corr) cell, -0.005 (decided 2026-06-10).
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

ALPHAS = [1.0, 1.5, 2.0, 2.5]


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


def load_role_vectors():
    """275 pipeline role vectors (persona_data/pt_vectors), incl. assistant."""
    import torch
    vec_dir = REPO / "persona_data" / "pt_vectors"
    vecs = {fp.stem: torch.load(fp, map_location="cpu",
                                weights_only=True).squeeze(0).float().numpy()
            for fp in sorted(vec_dir.glob("*.pt"))}
    assert len(vecs) == 275, f"expected 275 .pt vectors, got {len(vecs)}"
    return vecs


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc @ yc) / np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))


def ols_r2(X, y):
    """R^2 of OLS with intercept; X is (n, k)."""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fig_distance_summary():
    """4-panel distance-to-assistant figure (fig:distance-summary).

    d_r = 1 - cos(v_r, v_assistant), v_assistant = OUR pipeline's 275th
    role vector (NOT the Lu assistant axis). Delta_r(alpha) = per-cell
    mean steered_score - mean baseline_score (dedup convention via
    controllability_figs.load). n = 274 (assistant excluded).
    """
    from controllability_figs import load as load_gold

    vecs = load_role_vectors()
    va = vecs.pop("assistant")
    roles = sorted(vecs)
    V = np.stack([vecs[r] for r in roles])
    norms = np.linalg.norm(V, axis=1)
    d_cos = pd.Series(1 - (V @ va) / (norms * np.linalg.norm(va)), index=roles)
    d_euc = pd.Series(np.linalg.norm(V - va, axis=1), index=roles)
    lognorm = pd.Series(np.log(norms), index=roles)

    df = load_gold()
    cell = {col: df.groupby(["role", "alpha"])[col].mean().unstack()
            .reindex(columns=ALPHAS)
            for col in ("steered_score", "baseline_score",
                        "assistant_axis_score")}
    common = cell["steered_score"].dropna().index.intersection(d_cos.index)
    assert len(common) == 274, f"role match n={len(common)}, expected 274"
    delta = (cell["steered_score"] - cell["baseline_score"]).loc[common]
    metrics = {
        "$\\Delta$ (vs. prompted ref.)": delta,
        "$\\Delta$ (vs. asst.-axis)":
            (cell["steered_score"] - cell["assistant_axis_score"]).loc[common],
        "steered mean": cell["steered_score"].loc[common],
        "asst.-axis mean": cell["assistant_axis_score"].loc[common],
    }
    dc, de, ln = d_cos[common], d_euc[common], lognorm[common]

    r_cos = {a: pearson(dc, delta[a]) for a in ALPHAS}
    r_euc = {a: pearson(de, delta[a]) for a in ALPHAS}
    r2_1 = {a: r_cos[a] ** 2 for a in ALPHAS}
    r2_2 = {a: ols_r2([dc, ln], delta[a].to_numpy(float)) for a in ALPHAS}
    heat = np.array([[pearson(dc, m[a]) for a in ALPHAS]
                     for m in metrics.values()])

    fig, axes = plt.subplots(2, 2, figsize=(style.COLUMN_W_IN, 2.95))
    ax = axes[0, 0]   # (a) scatter at alpha=2.5
    ax.scatter(dc, delta[2.5], s=2.5, color=style.BLUE, alpha=0.55, lw=0)
    b1, b0 = np.polyfit(dc, delta[2.5], 1)
    xs = np.linspace(dc.min(), dc.max(), 2)
    ax.plot(xs, b1 * xs + b0, color="black", lw=0.9)
    ax.text(0.03, 0.04, f"$r{{=}}{r_cos[2.5]:.3f}$\n$n{{=}}{len(dc)}$",
            transform=ax.transAxes, fontsize=5.5, va="bottom")
    ax.set_xlabel("cosine distance", fontsize=6, labelpad=1)
    ax.set_ylabel(r"$\Delta_r$ at $\alpha{=}2.5$", fontsize=6, labelpad=1)

    ax = axes[0, 1]   # (b) r vs alpha, cos + euclidean
    ax.plot(ALPHAS, [r_cos[a] for a in ALPHAS], color=style.BLUE,
            marker="o", ms=2, label="cosine")
    ax.plot(ALPHAS, [r_euc[a] for a in ALPHAS], color=style.PURPLE,
            marker="s", ms=2, label="Euclidean")
    ax.axhline(0, color=style.GREY, lw=0.5, ls=":")
    ax.set_xticks(ALPHAS)
    ax.set_xlabel(r"$\alpha$", fontsize=6, labelpad=1)
    ax.set_ylabel(r"Pearson $r$", fontsize=6, labelpad=1)
    ax.legend(fontsize=5, loc="upper left", handlelength=1.2)

    ax = axes[1, 0]   # (c) R^2 with / without log-norm covariate
    x = np.arange(len(ALPHAS))
    ax.bar(x - 0.18, [r2_1[a] for a in ALPHAS], width=0.36,
           color=style.BLUE, label="distance only")
    ax.bar(x + 0.18, [r2_2[a] for a in ALPHAS], width=0.36,
           color=style.ORANGE, label=r"$+\log\|v\|$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}" for a in ALPHAS], fontsize=6)
    ax.set_xlabel(r"$\alpha$", fontsize=6, labelpad=1)
    ax.set_ylabel(r"OLS $R^2$", fontsize=6, labelpad=1)
    ax.legend(fontsize=5, loc="upper right", handlelength=1.0)

    ax = axes[1, 1]   # (d) r heatmap, behavioral metric x alpha
    ax.imshow(heat, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:+.2f}", ha="center", va="center",
                    fontsize=4.8)
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels([f"{a}" for a in ALPHAS], fontsize=5.5)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(list(metrics), fontsize=4.8)
    ax.set_xlabel(r"$\alpha$", fontsize=6, labelpad=1)

    for letter, ax in zip("abcd", axes.flat):
        ax.tick_params(labelsize=5.5)
        ax.text(-0.18, 1.06, f"({letter})", transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="top")
    fig.savefig(OUT / "fig_distance_summary.pdf")
    plt.close(fig)
    return r_cos, r_euc, r2_1, r2_2, heat, dc, delta


def distance_reference_rows():
    """Recompute tab-distance-references (3 reference-vector definitions).

    Audit-verified constructions (findings JSON, 2026-06-09):
      row 1: our raw assistant role vector (pt_vectors/assistant.pt)
      row 2: unit(assistant - mean of the 274 OUR role vectors)
      row 3: strict Lu axis = unit(default.pt['vector'] - mean of the 275
             Lu per-role activations) from persona_data/assistant-axis/
             olmo-3-7b-instruct/vectors/ (chemist.pt stores all 32 layers;
             its layer-16 row is used, matching the rest of the paper).
    """
    import torch
    vecs = load_role_vectors()
    va = vecs.pop("assistant")
    roles = sorted(vecs)
    V = np.stack([vecs[r] for r in roles])

    aa_dir = REPO / "persona_data" / "assistant-axis" / "olmo-3-7b-instruct" / "vectors"
    aa, default = [], None
    for fp in sorted(aa_dir.glob("*.pt")):
        d = torch.load(fp, map_location="cpu", weights_only=True)
        v = d["vector"].float().numpy()
        v = v[16] if v.shape[0] == 32 else v[0]
        if d.get("type") == "mean" or fp.stem == "default":
            default = v
        else:
            aa.append(v)
    assert default is not None and len(aa) == 275, \
        f"assistant-axis artifacts changed: default={default is not None}, n={len(aa)}"

    def unit(x):
        return x / np.linalg.norm(x)

    refs = {
        "ours_raw": va,
        "ours_centered": unit(va - V.mean(axis=0)),
        "lu_axis": unit(default - np.stack(aa).mean(axis=0)),
    }
    cos12 = float(unit(refs["ours_raw"]) @ refs["ours_centered"])
    cos13 = float(unit(refs["ours_raw"]) @ refs["lu_axis"])
    cos23 = float(refs["ours_centered"] @ refs["lu_axis"])
    return refs, (cos12, cos13, cos23), roles, V


PC_METRICS = [  # column order of the published grid
    ("steered_score", "overall"),
    ("cmp_motivation", "motivation"),
    ("cmp_worldview_alignment", "worldview"),
    ("cmp_emotional_register", "emot. reg."),
    ("cmp_vocab_choice", "vocab"),
    ("cmp_social_dynamic", "social dyn."),
]


def fig_pc_metric_grid():
    """PC x metric Pearson-r grid (fig:pc-grid), 20 PCs x 6 metrics x 4 alphas.

    PCA on the centered 275-role cloud (assistant included, matching the
    RSA/intrinsic-dimensionality cloud). Behavioral side: the committed
    behavioral_profiles.csv artifact (same one the RSA analysis consumes).
    PC sign is arbitrary under SVD; each PC's sign is fixed so its
    strongest |r| cell at alpha=1.0 is positive for PC1 and negative for
    PC5 conventions to match the published figure orientation.
    """
    vecs = load_role_vectors()
    roles = sorted(vecs)                       # 275, assistant included
    V = np.stack([vecs[r] for r in roles])
    Vc = V - V.mean(axis=0)
    U, S, _ = np.linalg.svd(Vc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    n90 = int(np.searchsorted(np.cumsum(var), 0.90) + 1)
    scores = U * S                             # (275, k) PC projections

    bp = read_branch_csv(
        "analysis/empirical/rsa_geometry_behavior/data/behavioral_profiles.csv"
    ).set_index("role").reindex(roles)
    assert not bp.isna().any().any(), "behavioral profile/role mismatch"

    n_pc = 20
    grids = {}                                 # alpha -> (n_pc, 6) r matrix
    for a in ALPHAS:
        g = np.zeros((n_pc, len(PC_METRICS)))
        for j, (col, _) in enumerate(PC_METRICS):
            y = bp[f"{col}__alpha_{a}"].to_numpy(float)
            for i in range(n_pc):
                g[i, j] = pearson(scores[:, i], y)
        grids[a] = g
    fig, axes = plt.subplots(4, 1, figsize=(style.COLUMN_W_IN, 3.1),
                             sharex=True)
    for ax, a in zip(axes, ALPHAS):
        ax.imshow(grids[a].T, cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                  aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(PC_METRICS)))
        ax.set_yticklabels([lab for _, lab in PC_METRICS], fontsize=4.6)
        ax.set_ylabel(rf"$\alpha{{=}}{a}$", fontsize=6)
        ax.tick_params(length=1.5)
    axes[-1].set_xticks(range(0, n_pc, 2))
    axes[-1].set_xticklabels([f"PC{i+1}" for i in range(0, n_pc, 2)],
                             fontsize=5, rotation=0)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=-0.6, vmax=0.6))
    cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label(r"Pearson $r$", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    fig.savefig(OUT / "fig_pc_metric_grid.pdf")
    plt.close(fig)

    # top-20-PC subspace R^2 per metric (paper: 17-27%)
    r2_20 = {}
    for a in ALPHAS:
        r2_20[a] = {lab: ols_r2(list(scores[:, :n_pc].T),
                                bp[f"{col}__alpha_{a}"].to_numpy(float))
                    for col, lab in PC_METRICS}
    return var, n90, grids, r2_20


# The 14 polar axes are defined in no code/config; this list was recovered
# from the published figure by the 2026-06-09 audit and CONFIRMED by exact
# reproduction of every heatmap cell and all 24 Table 2 cells. Two pairs
# reuse 'stoic' and 'methodical'. This is now the canonical list.
TRAIT_AXES = [
    ("assertive", "poetic"),
    ("diplomatic", "dramatic"),
    ("empathetic", "stoic"),
    ("enigmatic", "stoic"),
    ("erudite", "technical"),
    ("methodical", "chaotic"),
    ("nurturing", "hostile"),
    ("optimistic", "cynical"),
    ("philosophical", "critical"),
    ("playful", "analytical"),
    ("practical", "theatrical"),
    ("serene", "evil"),
    ("sycophantic", "manipulative"),
    ("whimsical", "methodical"),
]


def load_trait_vector(trait):
    """Layer-16 prompt_persona_vector (the audit-confirmed choice; the
    response vectors give materially different numbers)."""
    import json
    fp = (REPO / "persona_data" / "trait_inits"
          / f"{trait}_persona_initialization"
          / "allenai__Olmo-3-7B-Instruct_layer16_count40.json")
    return np.asarray(json.load(open(fp))["prompt_persona_vector"],
                      dtype=float).squeeze()


def fig_trait_metric_heatmap():
    """Trait-axis projection x metric heatmap at alpha=2.5 + Table 2 R^2.

    Axis = v_pos - v_neg from trait_inits; projection = role-vector dot
    axis; n=274 (assistant excluded, per the audit's exact reproduction).
    Returns the per-alpha r tensors and the joint-OLS R^2 table.
    """
    vecs = load_role_vectors()
    vecs.pop("assistant")
    roles = sorted(vecs)
    V = np.stack([vecs[r] for r in roles])
    axes_v = {f"{p}-{n}": load_trait_vector(p) - load_trait_vector(n)
              for p, n in TRAIT_AXES}
    proj = {name: V @ ax for name, ax in axes_v.items()}

    bp = read_branch_csv(
        "analysis/empirical/rsa_geometry_behavior/data/behavioral_profiles.csv"
    ).set_index("role").reindex(roles)
    assert bp.notna().all().all(), "behavioral profile/role mismatch"

    heat = {a: np.array([[pearson(proj[name], bp[f"{col}__alpha_{a}"])
                          for col, _ in PC_METRICS]
                         for name in axes_v]) for a in ALPHAS}
    r2 = {a: {lab: ols_r2(list(proj.values()),
                          bp[f"{col}__alpha_{a}"].to_numpy(float))
              for col, lab in PC_METRICS} for a in ALPHAS}

    # rendered at alpha=1.0: the published figure said alpha=2.5 but its
    # caption quoted alpha=1.0 values (methodical x vocab is -0.02 at 2.5
    # vs +0.245 at 1.0) -- audit flag; the prose's axis-correspondence
    # claims are direction-regime (low-alpha) facts.
    h10 = heat[1.0]
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 2.5))
    ax.imshow(h10, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto",
              interpolation="nearest")
    for i in range(h10.shape[0]):
        for j in range(h10.shape[1]):
            ax.text(j, i, f"{h10[i, j]:+.2f}", ha="center", va="center",
                    fontsize=4.2)
    ax.set_xticks(range(len(PC_METRICS)))
    ax.set_xticklabels([lab for _, lab in PC_METRICS], fontsize=5,
                       rotation=30, ha="right")
    ax.set_yticks(range(len(axes_v)))
    ax.set_yticklabels([n.replace("-", "–") for n in axes_v], fontsize=4.6)
    fig.savefig(OUT / "fig_trait_metric_heatmap.pdf")
    plt.close(fig)
    return list(axes_v), heat, r2


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

    print("verification gate (distance-to-assistant, Table 1):")
    r_cos, r_euc, r2_1, r2_2, heat, dc, delta = fig_distance_summary()
    # originally published -0.418/-0.409/-0.351/-0.144 reproduces from RAW
    # rows; the paper's dedup convention (Glenn 2026-06-10, same call as
    # anti-38) gives these values -- Table 1 updated in the same commit.
    for a, want in zip(ALPHAS, [-0.419, -0.409, -0.351, -0.145]):
        verify(f"cos-distance r @ alpha={a}", r_cos[a], want, 0.001)
    # Table 1 R^2 row (cos / +log||v||); prose says "0.17 -> 0.22 at 1.0"
    for a, w1, w2 in zip(ALPHAS, [0.18, 0.17, 0.12, 0.02],
                         [0.22, 0.20, 0.13, 0.02]):
        verify(f"R2 distance-only @ {a}", r2_1[a], w1, 0.006)
        verify(f"R2 +lognorm @ {a}", r2_2[a], w2, 0.006)
    print(f"  exact R2 @ 1.0: {r2_1[1.0]:.4f} -> {r2_2[1.0]:.4f} "
          "(table rounds 0.18/0.22, prose says 0.17->0.22)")
    from scipy import stats
    for a, p_want in zip(ALPHAS, [5e-13, 2e-12, 2e-9, 1.7e-2]):
        p = float(stats.pearsonr(dc, delta[a]).pvalue)
        print(f"  p @ alpha={a}: {p:.2g} (table {p_want:.2g})")
    print("  euclidean r per alpha:",
          {a: round(r_euc[a], 3) for a in ALPHAS})
    print("  heatmap (rows: dPR, dAA, steered, AA):")
    print(np.round(heat, 3))

    print("verification gate (tab-distance-references):")
    refs, (c12, c13, c23), roles, V = distance_reference_rows()
    verify("cos(raw, centered)", c12, 0.17, 0.005)
    verify("cos(raw, lu)", c13, -0.45, 0.005)
    verify("cos(centered, lu)", c23, -0.07, 0.005)
    norms = np.linalg.norm(V, axis=1)
    want_rows = {  # dedup-convention values (Table updated in same commit)
        "ours_raw": [-0.419, -0.409, -0.351, -0.145],
        "ours_centered": [-0.381, -0.352, -0.264, -0.084],
        "lu_axis": [0.167, 0.162, 0.168, 0.186],
    }
    for name, ref in refs.items():
        d = pd.Series(
            1 - (V @ ref) / (norms * np.linalg.norm(ref)), index=roles)[dc.index]
        rs = {a: pearson(d, delta[a]) for a in ALPHAS}
        for a, want in zip(ALPHAS, want_rows[name]):
            verify(f"{name} r @ {a}", rs[a], want, 0.001)
        if name == "lu_axis":
            print(f"  lu univariate R2 @ 1.0: {rs[1.0] ** 2:.3f} "
                  "(paper: ~0.03)")

    print("verification gate (PC grid):")
    var, n90, grids, r2_20 = fig_pc_metric_grid()
    verify("PC1 variance share", float(var[0]), 0.21, 0.005)
    verify("components to 90% variance", float(n90), 98, 0)
    g10, g25 = grids[1.0], grids[2.5]
    sub_cols = range(1, 6)        # the 5 sub-dimensions (excl. overall)
    pc1_best = [j for j in sub_cols
                if int(np.abs(g10[:, j]).argmax()) == 0]
    print(f"  PC1 best-single-predictor sub-dims @ 1.0: {len(pc1_best)}/5, "
          f"PC1 r = {[round(float(g10[0, j]), 3) for j in sub_cols]}")
    if len(pc1_best) != 4:
        raise SystemExit("PC1 best-for-4-of-5 claim failed")
    pc1_r = [float(g10[0, j]) for j in pc1_best]
    if not all(0.445 <= r <= 0.495 for r in pc1_r):   # paper range at 2dp
        raise SystemExit(f"PC1 r range {pc1_r} outside [0.45, 0.49]")
    pc5_best = [j for j in range(6)
                if int(np.abs(g25[:, j]).argmax()) == 4]
    pc5_r = [float(g25[4, j]) for j in range(6)]
    print(f"  PC5 best-single-predictor metrics @ 2.5: {len(pc5_best)}/6, "
          f"PC5 r = {[round(r, 3) for r in pc5_r]}")
    if len(pc5_best) != 6:
        raise SystemExit("PC5 best-for-all claim failed")
    if not all(-0.345 <= r <= -0.175 for r in pc5_r):  # paper range at 2dp
        raise SystemExit(f"PC5 r range {pc5_r} outside [-0.34, -0.18]")
    print("  top-20-PC subspace R2 per metric:")
    for a in ALPHAS:
        print(f"    alpha={a}: " + ", ".join(
            f"{lab} {v:.2f}" for lab, v in r2_20[a].items()))

    print("verification gate (trait axes, Table 2):")
    axis_names, theat, tr2 = fig_trait_metric_heatmap()
    want_r2 = {  # Table 2, all 24 cells (audit recompute: exact match)
        "emot. reg.": [0.41, 0.39, 0.31, 0.14],
        "social dyn.": [0.39, 0.35, 0.26, 0.11],
        "vocab": [0.38, 0.34, 0.23, 0.06],
        "worldview": [0.36, 0.34, 0.27, 0.13],
        "motivation": [0.34, 0.32, 0.27, 0.13],
        "overall": [0.28, 0.24, 0.16, 0.14],
    }
    for lab, wants in want_r2.items():
        for a, want in zip(ALPHAS, wants):
            verify(f"trait joint R2 {lab} @ {a}", tr2[a][lab], want, 0.005)
    # audit fingerprint cells of the published alpha=2.5 heatmap
    cols = [lab for _, lab in PC_METRICS]
    h25 = theat[2.5]
    fp_cells = [("assertive-poetic", "overall", -0.24),
                ("empathetic-stoic", "emot. reg.", 0.25),
                ("methodical-chaotic", "overall", -0.20)]
    for axis, lab, want in fp_cells:
        got = float(h25[axis_names.index(axis), cols.index(lab)])
        verify(f"heatmap {axis} x {lab} @ 2.5", got, want, 0.005)
    # new caption cells (figure now rendered at alpha=1.0; the old caption
    # attributed these values to alpha=2.5 where they do not hold)
    h10 = theat[1.0]
    for axis, lab, want in [("methodical-chaotic", "vocab", 0.245),
                            ("diplomatic-dramatic", "social dyn.", 0.167),
                            ("empathetic-stoic", "emot. reg.", 0.296),
                            ("nurturing-hostile", "emot. reg.", 0.296),
                            ("serene-evil", "emot. reg.", 0.253)]:
        got = float(h10[axis_names.index(axis), cols.index(lab)])
        verify(f"caption cell {axis} x {lab} @ 1.0", got, want, 0.005)

    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
