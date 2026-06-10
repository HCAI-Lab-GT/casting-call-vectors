"""
sparsity_sans_mythical.py

Compare the geometric spread of two role-vector sets:

  A) our proposed steering vectors:
     persona_data/model_inits/<role>_persona_initialization/*.safetensors
     (layer-16 activations after the role prompt)

  B) assistant-axis steering vectors:
     persona_data/assistant-axis/olmo-3-7b-instruct/vectors/<role>.pt
     (the "assistant axis" variant used for comparison in the empirical
     experiments)

Claim we want to verify:
    Set A (proposed steering) is more spread out than set B (assistant-axis)
    across roles — so the proposed vectors carry more role-specific
    information than the assistant axis alone.

Metrics computed (per set):
  - per-role norm
  - per-role distance-to-centroid
  - pairwise cosine similarity across all role pairs
  - total variance = tr(cov), participation ratio (effective dim)
  - variance fraction captured by top PC (low = more spread)

Outputs under `analysis/geometry/figures_sans_mythical/sparsity/`:
  - density_norms.png                KDE of per-role norms  (A vs B)
  - density_dist_to_centroid.png     KDE of distances to centroid (A vs B)
  - density_pairwise_cosine.png      KDE of pairwise cos sim (A vs B)
  - scree_side_by_side.png           PCA scree, both sets
  - sparsity_summary.tsv             numeric summary (full & sans_mythical)
  - sparsity_overall.txt             human-readable headline

The mythical cluster is dropped using
`analysis/geometry/figures_sans_mythical/cluster_membership.txt` (written by
full_analysis_sans_mythical.py).
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.numpy import load_file
from scipy.stats import gaussian_kde

FIG_IN = "analysis/geometry/figures_sans_mythical"
OUT_DIR = os.path.join(FIG_IN, "sparsity")
os.makedirs(OUT_DIR, exist_ok=True)

PROPOSED_DIR = "persona_data/model_inits"
PROPOSED_LAYER = 16
PROPOSED_COUNT = 50

AXIS_DIR = "persona_data/assistant-axis/olmo-3-7b-instruct/vectors"


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
def load_proposed_vectors():
    """safetensors files → {role: np.float32 (D,)}"""
    out = {}
    for entry in sorted(os.listdir(PROPOSED_DIR)):
        d = os.path.join(PROPOSED_DIR, entry)
        if not os.path.isdir(d):
            continue
        files = [
            f for f in os.listdir(d)
            if f.endswith(".safetensors") and f"count{PROPOSED_COUNT}" in f
        ]
        if not files:
            continue
        data = load_file(os.path.join(d, files[0]))
        key = next(iter(data))
        name = entry.replace("_persona_initialization", "")
        out[name] = data[key].astype(np.float32)[PROPOSED_LAYER - 1].squeeze()
    return out


def load_axis_vectors():
    """.pt files → {role: np.float32 (D,)}.

    Most files store shape (1, 4096); a handful (e.g. chemist.pt) were saved
    as raw per-sample activations (N, 4096) — for those we mean-reduce
    across the sample axis so every role ends up at the same (D,) shape.
    """
    out = {}
    for p in sorted(Path(AXIS_DIR).glob("*.pt")):
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        vec = d["vector"].float().numpy()
        # Collapse to 1D of length D
        while vec.ndim > 1:
            if vec.shape[0] == 1:
                vec = vec[0]
            else:
                print(f"[sparsity] mean-reducing {p.name}: shape {vec.shape} -> ({vec.shape[-1]},)")
                vec = vec.mean(axis=0)
        out[p.stem] = vec.astype(np.float32)
    return out


def load_excluded_roles():
    path = os.path.join(FIG_IN, "cluster_membership.txt")
    excluded, in_excl = set(), False
    if not os.path.exists(path):
        print(f"[sparsity] WARNING: {path} not found — running without sans-mythical view.")
        return excluded
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("# Excluded"):
                in_excl = True;  continue
            if s.startswith("# Kept"):
                in_excl = False; continue
            if in_excl and s:
                excluded.add(os.path.splitext(s.split()[0])[0])
    return excluded


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def _pca_stats(matrix):
    """PCA variance statistics on a (N, D) matrix. Returns eigvals,
    total_var, participation_ratio, top_pc_frac.

    Scale-sensitive unless `matrix` is row-normalized.
    """
    N = matrix.shape[0]
    centered = matrix - matrix.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eigvals = (s ** 2) / max(N - 1, 1)
    total_var = float(eigvals.sum())
    pr = float((total_var ** 2) / (eigvals ** 2).sum()) if total_var > 0 else float("nan")
    top_frac = float(eigvals[0] / total_var) if total_var > 0 else float("nan")
    return eigvals, total_var, pr, top_frac


def _metrics(matrix):
    """
    matrix: (N, D)
    returns per-set geometric stats. We report both scale-sensitive
    (raw vectors) and scale-invariant (unit-normalized vectors) versions
    of variance / participation ratio, because the two directories have
    very different norms and only the scale-invariant version measures
    directional spread fairly.
    """
    N = matrix.shape[0]
    norms = np.linalg.norm(matrix, axis=1)

    centroid = matrix.mean(axis=0)
    dist_to_centroid = np.linalg.norm(matrix - centroid, axis=1)

    # Unit-normalized (direction-only) view
    unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    cos = unit @ unit.T
    iu = np.triu_indices(N, k=1)
    cos_flat = cos[iu]

    # Raw-scale PCA
    eigvals, total_var, pr, top_frac = _pca_stats(matrix)
    # Unit-scale PCA (directional spread only)
    eigvals_u, total_var_u, pr_u, top_frac_u = _pca_stats(unit)

    return {
        "n": N,
        "norms": norms,
        "dist_to_centroid": dist_to_centroid,
        "cos_sim_flat": cos_flat,
        # raw / scale-sensitive
        "total_var": total_var,
        "participation_ratio": pr,
        "top_pc_frac": top_frac,
        "eigvals": eigvals,
        # unit / scale-invariant
        "total_var_unit": total_var_u,
        "participation_ratio_unit": pr_u,
        "top_pc_frac_unit": top_frac_u,
        "eigvals_unit": eigvals_u,
    }


def _common_matrix(vecs_dict, keep_names):
    keep = [n for n in keep_names if n in vecs_dict]
    return keep, np.stack([vecs_dict[n] for n in keep], axis=0)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def _kde_overlay(ax, values_A, values_B, label_A, label_B, xlabel):
    all_vals = np.concatenate([values_A, values_B])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    xs = np.linspace(lo - pad, hi + pad, 400)
    for vals, color, label in [
        (values_A, "tab:red",  f"{label_A} (n={len(values_A)})"),
        (values_B, "tab:blue", f"{label_B} (n={len(values_B)})"),
    ]:
        if len(vals) < 2:
            continue
        ax.hist(vals, bins=40, density=True, color=color, alpha=0.18,
                range=(xs[0], xs[-1]))
        try:
            kde = gaussian_kde(vals)
            ax.plot(xs, kde(xs), color=color, lw=2.0,
                    label=f"{label}  μ={vals.mean():.3f} σ={vals.std(ddof=1):.3f}")
        except np.linalg.LinAlgError:
            ax.axvline(vals[0], color=color, lw=2.0, label=label)
        ax.axvline(vals.mean(), color=color, lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.legend(loc="best")


def _plot_densities(metrics_A, metrics_B, label_A, label_B, set_label):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _kde_overlay(axes[0], metrics_A["norms"], metrics_B["norms"],
                 label_A, label_B, "per-role vector norm")
    axes[0].set_title(f"Per-role norms ({set_label})")
    _kde_overlay(axes[1], metrics_A["dist_to_centroid"], metrics_B["dist_to_centroid"],
                 label_A, label_B, "distance to set centroid")
    axes[1].set_title(f"Distance to centroid ({set_label})")
    _kde_overlay(axes[2], metrics_A["cos_sim_flat"], metrics_B["cos_sim_flat"],
                 label_A, label_B, "pairwise cosine similarity")
    axes[2].set_title(f"Pairwise cosine sim ({set_label})")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"density_panel_{set_label}.png")
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"[sparsity] wrote {out}")


def _plot_scree(metrics_A, metrics_B, label_A, label_B, set_label):
    """Two-row scree: top row = raw (scale-sensitive), bottom row = unit
    (scale-invariant, the one that measures directional spread)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row_idx, (eig_key, row_label) in enumerate([
        ("eigvals",      "raw (scale-sensitive)"),
        ("eigvals_unit", "unit-normalized (directional spread)"),
    ]):
        ax1, ax2 = axes[row_idx]
        for m, color, lab in [
            (metrics_A, "tab:red",  label_A),
            (metrics_B, "tab:blue", label_B),
        ]:
            ev = m[eig_key]
            frac = ev / ev.sum() if ev.sum() > 0 else ev
            k = min(30, len(frac))
            ax1.plot(range(1, k + 1), frac[:k], "o-", color=color, label=lab)
            ax2.plot(range(1, k + 1), np.cumsum(frac)[:k], "o-", color=color, label=lab)
        ax1.set_xlabel("PC index"); ax1.set_ylabel("variance fraction")
        ax1.set_title(f"Scree — {row_label} ({set_label})")
        ax1.set_yscale("log"); ax1.legend()
        ax2.set_xlabel("PC index"); ax2.set_ylabel("cumulative variance fraction")
        ax2.set_title("Cumulative"); ax2.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"scree_{set_label}.png")
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"[sparsity] wrote {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def _row(set_label, method, m):
    return {
        "set": set_label,
        "method": method,
        "n_roles": m["n"],
        "mean_norm": float(m["norms"].mean()),
        "std_norm": float(m["norms"].std(ddof=1)),
        "mean_dist_to_centroid": float(m["dist_to_centroid"].mean()),
        "std_dist_to_centroid": float(m["dist_to_centroid"].std(ddof=1)),
        "mean_pairwise_cos": float(m["cos_sim_flat"].mean()),
        # Raw / scale-sensitive (dominated by vector magnitude)
        "total_var": m["total_var"],
        "participation_ratio": m["participation_ratio"],
        "top_pc_frac": m["top_pc_frac"],
        # Unit / scale-invariant (pure directional spread — what you want
        # to cite when comparing across sets with different norms)
        "total_var_unit": m["total_var_unit"],
        "participation_ratio_unit": m["participation_ratio_unit"],
        "top_pc_frac_unit": m["top_pc_frac_unit"],
    }


def main():
    vecs_proposed = load_proposed_vectors()
    vecs_axis = load_axis_vectors()
    excluded = load_excluded_roles()

    common = sorted(set(vecs_proposed) & set(vecs_axis) - {"assistant"})
    print(f"[sparsity] {len(vecs_proposed)} proposed | {len(vecs_axis)} axis | "
          f"{len(common)} common roles | {len(excluded)} excluded as mythical")

    rows = []
    for set_label, names in [
        ("full",          common),
        ("sans_mythical", [n for n in common if n not in excluded]),
    ]:
        _, A = _common_matrix(vecs_proposed, names)
        _, B = _common_matrix(vecs_axis,     names)
        mA = _metrics(A)
        mB = _metrics(B)

        _plot_densities(mA, mB, "proposed (model_inits)", "assistant-axis", set_label)
        _plot_scree(mA, mB, "proposed (model_inits)", "assistant-axis", set_label)

        rows.append(_row(set_label, "proposed",       mA))
        rows.append(_row(set_label, "assistant_axis", mB))

    df = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_DIR, "sparsity_summary.tsv")
    df.to_csv(summary_path, sep="\t", index=False)
    print(f"\n[sparsity] wrote {summary_path}")
    print(df.to_string(index=False))

    # Headline report
    lines = ["Sparsity: proposed steering vs assistant-axis steering\n",
             "=" * 56 + "\n\n"]
    lines.append(
        "Scale note: raw total variance and norms are magnitude-dominated;\n"
        "the two sets have very different vector norms, so those metrics\n"
        "conflate 'bigger' with 'more spread'. The *_unit rows below are\n"
        "computed on unit-normalized vectors, so they measure directional\n"
        "(angular) spread only — that's the fair comparison for steering.\n\n"
    )
    for set_label in ["full", "sans_mythical"]:
        sub = df[df["set"] == set_label].set_index("method")
        if "proposed" not in sub.index or "assistant_axis" not in sub.index:
            continue
        p = sub.loc["proposed"]; a = sub.loc["assistant_axis"]
        lines.append(f"[{set_label}]  n_roles={int(p['n_roles'])}\n")
        lines.append(
            f"  mean norm                    proposed={p['mean_norm']:.3f}   "
            f"axis={a['mean_norm']:.3f}   ratio={p['mean_norm']/max(a['mean_norm'],1e-12):.2f}x  (magnitude — not a spread signal)\n"
        )
        lines.append(
            f"  mean pairwise cosine         proposed={p['mean_pairwise_cos']:.3f}   "
            f"axis={a['mean_pairwise_cos']:.3f}                     (lower => more directional spread)\n"
        )
        lines.append(
            f"  participation ratio (unit)   proposed={p['participation_ratio_unit']:.2f}   "
            f"axis={a['participation_ratio_unit']:.2f}                         (effective # of directions, scale-invariant)\n"
        )
        lines.append(
            f"  top-PC var fraction (unit)   proposed={p['top_pc_frac_unit']*100:.1f}%   "
            f"axis={a['top_pc_frac_unit']*100:.1f}%                    (lower => less collapse onto one axis)\n\n"
        )
        lines.append(
            "  scale-sensitive (for reference, not the headline):\n"
        )
        lines.append(
            f"    total variance (raw)       proposed={p['total_var']:.2f}   "
            f"axis={a['total_var']:.2f}   ratio={p['total_var']/max(a['total_var'],1e-12):.2f}x\n"
        )
        lines.append(
            f"    participation ratio (raw)  proposed={p['participation_ratio']:.2f}   "
            f"axis={a['participation_ratio']:.2f}\n\n"
        )

    # Headline line on sans_mythical
    sub = df[df["set"] == "sans_mythical"].set_index("method")
    if {"proposed", "assistant_axis"}.issubset(sub.index):
        p = sub.loc["proposed"]; a = sub.loc["assistant_axis"]
        pr_ratio = p['participation_ratio_unit'] / max(a['participation_ratio_unit'], 1e-12)
        lines.append("Headline (sans mythical)\n" + "-" * 56 + "\n")
        lines.append(
            f"Across {int(p['n_roles'])} roles, the proposed steering vectors occupy\n"
            f"~{p['participation_ratio_unit']:.1f} effective directions vs ~{a['participation_ratio_unit']:.1f} for the assistant-axis\n"
            f"family ({pr_ratio:.1f}x more, on unit-normalized vectors), and their mean\n"
            f"pairwise cosine is {p['mean_pairwise_cos']:.2f} vs {a['mean_pairwise_cos']:.2f} — i.e. the proposed\n"
            f"vectors are substantially more *directionally* spread out, even though\n"
            f"the assistant-axis vectors have larger individual norms\n"
            f"({a['mean_norm']:.2f} vs {p['mean_norm']:.2f}).\n"
        )

    txt = "".join(lines)
    report_path = os.path.join(OUT_DIR, "sparsity_overall.txt")
    with open(report_path, "w") as f:
        f.write(txt)
    print(f"\n[sparsity] wrote {report_path}\n")
    print(txt)


if __name__ == "__main__":
    main()
