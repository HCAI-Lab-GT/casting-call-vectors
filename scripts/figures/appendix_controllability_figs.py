"""Regenerate the seven Appendix F/G controllability figures from CSVs.

Outputs (vector PDF, true printed size):
  out/fig_per_role_heatmap.pdf          -> fig:per_role_heatmap
  out/fig_monotonicity_bars.pdf         -> fig:monotonicity_bars
  out/fig_alpha_curves_all_metrics.pdf  -> fig:alpha_curves_all_metrics
  out/fig_individual_curves.pdf         -> fig:individual_curves
  out/fig_deterioration_dist.pdf        -> fig:deterioration_dist
  out/fig_dim_deterioration_heatmap.pdf -> fig:dim_deterioration_heatmap
  out/fig_baseline_vs_steered.pdf       -> fig:baseline_vs_steered

Same dedup-38 convention as controllability_figs.py (imported). The
baseline-vs-steered scatter is REFRAMED to the artifact-backed prompted
reference (baseline_score column) instead of the nonexistent "unsteered
baseline"; under that frame 37/38 roles fall below the diagonal.

Confidence bands are 1.96*SEM of the group mean (matching Fig 4), not the
old caption's bootstrap claim; captions updated accordingly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pvx_fig_style as style
from controllability_figs import (
    ALPHAS, CMP_DIMS, DIM_LABELS, OUT, classify_anti, load, per_role_r, verify,
)

ALL_DIMS = ["steered_score"] + CMP_DIMS
ALL_LABELS = {"steered_score": "overall", **DIM_LABELS}


def cell_means(df, col):
    """role x alpha matrix of per-cell means for one score column."""
    return (df.groupby(["role", "alpha"])[col].mean()
            .unstack().reindex(columns=ALPHAS))


def ols_slopes(cell):
    a = np.array(ALPHAS)
    ac = a - a.mean()
    return cell.apply(
        lambda row: float(np.dot(row - row.mean(), ac) / np.dot(ac, ac)),
        axis=1)


def fig_per_role_heatmap(df):
    r = pd.DataFrame({ALL_LABELS[c]: per_role_r(df, c) for c in ALL_DIMS})
    r = r.dropna().sort_values("overall", ascending=False)
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 3.4))
    im = ax.imshow(r.to_numpy(), aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(r.columns)))
    ax.set_xticklabels(r.columns, rotation=35, ha="right", fontsize=6)
    ax.set_yticks([])
    ax.set_ylabel(f"roles (n={len(r)}, sorted by overall $r$)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(r"Pearson $r$ (score vs. $\alpha$)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(OUT / "fig_per_role_heatmap.pdf")
    plt.close(fig)
    return r


def fig_monotonicity_bars(df):
    metrics = {}
    for col in ALL_DIMS:
        r = per_role_r(df, col)
        cell = cell_means(df, col).dropna()
        mono = (cell.diff(axis=1).iloc[:, 1:] > 0).all(axis=1)
        metrics[ALL_LABELS[col]] = {
            r"$r>0$": float((r > 0).mean()),
            r"$r>0.6$": float((r > 0.6).mean()),
            r"$r>0.8$": float((r > 0.8).mean()),
            r"$r>0.95$": float((r > 0.95).mean()),
            "monotone": float(mono.mean()),
        }
    m = pd.DataFrame(metrics).T  # axes x thresholds
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 1.7))
    x = np.arange(len(m.index))
    w = 0.16
    colors = [style.BLUE, style.SKY, style.GREEN, style.ORANGE, style.PURPLE]
    for i, col in enumerate(m.columns):
        ax.bar(x + (i - 2) * w, m[col], width=w, color=colors[i], label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(m.index, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("fraction of roles")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=5, fontsize=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), columnspacing=0.8,
              handlelength=1.0, handletextpad=0.4)
    fig.savefig(OUT / "fig_monotonicity_bars.pdf")
    plt.close(fig)
    return m


def fig_alpha_curves(df, anti):
    ctl = df[~df["role"].isin(anti)]
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 1.7))
    palette = [style.BLUE, style.VERMILLION, style.GREEN, style.ORANGE,
               style.SKY, style.PURPLE]
    n_ctl = None
    for color, col in zip(palette, ALL_DIMS):
        cell = cell_means(ctl, col).dropna()
        n_ctl = len(cell)
        mean, sem = cell.mean(axis=0), cell.sem(axis=0)
        ax.plot(ALPHAS, mean, color=color, marker="o", ms=2,
                label=ALL_LABELS[col],
                lw=1.4 if col == "steered_score" else 1.0)
        ax.fill_between(ALPHAS, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.15, lw=0)
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel("mean judge score")
    ax.set_xticks(ALPHAS)
    ax.set_ylim(28, 76)
    ax.legend(ncol=3, fontsize=5.5, loc="lower center", handlelength=1.2,
              columnspacing=0.9)
    fig.savefig(OUT / "fig_alpha_curves_all_metrics.pdf")
    plt.close(fig)
    return n_ctl


def fig_individual_curves(df, anti):
    cell = cell_means(df[df["role"].isin(anti)], "steered_score").dropna()
    slopes = ols_slopes(cell)
    fig, ax = plt.subplots(figsize=(0.62 * style.COLUMN_W_IN, 1.7))
    for _, row in cell.iterrows():
        ax.plot(ALPHAS, row, color=style.VERMILLION, alpha=0.35, lw=0.7)
    ax.plot(ALPHAS, cell.mean(axis=0), color="black", lw=1.4,
            label=f"mean (n={len(cell)})")
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel("judge score")
    ax.set_xticks(ALPHAS)
    ax.legend(fontsize=5.5, loc="lower left")
    fig.savefig(OUT / "fig_individual_curves.pdf")
    plt.close(fig)
    return float(slopes.min()), float(slopes.max())


def fig_deterioration_dist(df, anti):
    # drop is alpha=1.0 -> 2.5 (the convention behind the paper's 7.0/6.0;
    # the old caption SAID peak->2.5 but the numbers never matched that
    # definition -- caption fixed to alpha=1.0 in the same commit)
    cell = cell_means(df[df["role"].isin(anti)], "steered_score").dropna()
    drop = cell[1.0] - cell[2.5]
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 1.5))
    ax.hist(drop, bins=14, color=style.VERMILLION, alpha=0.85)
    ax.axvline(drop.mean(), color="black", lw=0.8, ls="--")
    ax.text(drop.mean() + 0.3, ax.get_ylim()[1] * 0.9,
            f"mean {drop.mean():.1f}", fontsize=6)
    ax.annotate(f"supervisor ({drop.max():.1f})",
                xy=(drop.max(), 0.4), xytext=(drop.max() - 1.5, 3.2),
                fontsize=6, ha="right",
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel(r"score drop, $\alpha\,1.0 \to 2.5$")
    ax.set_ylabel("roles")
    fig.savefig(OUT / "fig_deterioration_dist.pdf")
    plt.close(fig)
    return drop


def fig_dim_heatmap(df, anti):
    sub = df[df["role"].isin(anti)]
    drops = pd.DataFrame({
        ALL_LABELS[c]: (lambda cell: cell[1.0] - cell[2.5])(
            cell_means(sub, c).dropna())
        for c in ALL_DIMS})
    drops = drops.sort_values("overall", ascending=False)
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 2.6))
    lim = float(np.abs(drops.to_numpy()).max())
    im = ax.imshow(drops.to_numpy(), aspect="auto", cmap="RdBu_r",
                   vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks(range(len(drops.columns)))
    ax.set_xticklabels(drops.columns, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(drops)))
    ax.set_yticklabels(drops.index, fontsize=4.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(r"score drop, $\alpha\,1.0 \to 2.5$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(OUT / "fig_dim_deterioration_heatmap.pdf")
    plt.close(fig)


def fig_baseline_vs_steered(df, anti):
    sub = df[df["role"].isin(anti)]
    ref = sub.groupby("role")["baseline_score"].mean()
    steered25 = cell_means(sub, "steered_score").dropna()[2.5]
    fig, ax = plt.subplots(figsize=(0.62 * style.COLUMN_W_IN, 1.7))
    lo, hi = 55, 100
    ax.plot([lo, hi], [lo, hi], color=style.GREY, lw=0.8, ls="--")
    ax.scatter(ref.reindex(steered25.index), steered25, s=8,
               color=style.VERMILLION, alpha=0.8, lw=0)
    ax.set_xlabel("prompted-reference score")
    ax.set_ylabel(r"steered score at $\alpha{=}2.5$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.savefig(OUT / "fig_baseline_vs_steered.pdf")
    plt.close(fig)
    below = int((steered25 < ref.reindex(steered25.index)).sum())
    return below, len(steered25)


def main():
    style.apply()
    df = load()
    anti = classify_anti(df)
    assert len(anti) == 38, f"anti set drifted: {len(anti)}"

    print("verification gate:")
    m = fig_monotonicity_bars(df)
    verify("overall r>0", m.loc["overall", r"$r>0$"], 0.82, 0.01)
    verify("overall r>0.8", m.loc["overall", r"$r>0.8$"], 0.79, 0.01)
    verify("overall strict monotone", m.loc["overall", "monotone"], 0.74, 0.01)
    weakest = m.drop(index="overall").idxmin()
    print(f"  weakest axis per metric: {dict(weakest)}")

    drop = fig_deterioration_dist(df, anti)
    verify("anti mean drop", float(drop.mean()), 7.0, 0.1)
    verify("anti median drop", float(drop.median()), 6.0, 0.1)
    verify("anti max drop (supervisor)", float(drop.max()), 23.4, 0.1)

    smin, smax = fig_individual_curves(df, anti)
    print(f"  anti OLS slope range: {smin:.1f} to {smax:.1f} "
          f"(caption claims -8.9 to -1.8 on the old 37-role set)")

    below, n = fig_baseline_vs_steered(df, anti)
    print(f"  below diagonal: {below}/{n}")
    if below != 37 or n != 38:
        raise SystemExit("baseline-vs-steered count mismatch")

    n_ctl = fig_alpha_curves(df, anti)
    print(f"  controllable-curve n = {n_ctl} (expect 237)")

    fig_per_role_heatmap(df)
    fig_dim_heatmap(df, anti)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
