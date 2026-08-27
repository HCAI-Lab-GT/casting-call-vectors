"""Regenerate the two Section-4 body figures from committed CSVs.

Outputs (vector PDF, true printed size):
  out/fig_pearson_r_distribution.pdf   -> fig:pearson_r_dist
  out/fig_trajectory_contrast.pdf      -> fig:anticontrollability (left)
  out/fig_dim_deterioration_bars.pdf   -> fig:anticontrollability (right)

Replaces the fabricated "unsteered baseline (alpha=0)" point of the old
trajectory figure with the artifact-backed prompted-reference level
(baseline_score column), matching the rewritten Sec 4.2 prose.

Every statistic the paper claims is recomputed and asserted before any
figure is written; a mismatch aborts with a report.

Conventions (must match the paper's tables):
  - duplicate (role, alpha, question) rows dropped, keep first
  - per-role controllability r = Pearson over the n=4 (alpha, mean score)
    points (Eq. 2 of the paper)
  - anti-controllable: r < 0 simultaneously for steered_score and all
    five cmp_* dimensions

Convention decision (Glenn, 2026-06-10): the classification uses the SAME
dedup convention as Tables 12-13 and the headline means. This yields 38
anti-controllable roles (13.8%), not the originally published 37: 'coach'
flips in (its cmp_emotional_register r is +0.29 on raw rows, -0.10 after
dedup). The paper's Sec 4.2 / intro / contributions were updated to n=38
the same day. Raw-row classification (the pre-2026-06-10 numbers) is NOT
what this script asserts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pvx_fig_style as style

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiment_data" / "gold_prompt_experiments"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

CMP_DIMS = [
    "cmp_emotional_register",
    "cmp_vocab_choice",
    "cmp_social_dynamic",
    "cmp_motivation",
    "cmp_worldview_alignment",
]
DIM_LABELS = {
    "cmp_vocab_choice": "Vocabulary Choice",
    "cmp_social_dynamic": "Social Dynamic",
    "cmp_emotional_register": "Emotional Register",
    "cmp_worldview_alignment": "Worldview Align.",
    "cmp_motivation": "Motivation",
}
ALPHAS = [1.0, 1.5, 2.0, 2.5]


def load() -> pd.DataFrame:
    frames = []
    usecols = (
        ["role", "alpha", "question", "steered_score",
         "assistant_axis_score", "baseline_score"] + CMP_DIMS
    )
    for fp in sorted(DATA.glob("Comparison_GoldStandard_*.csv")):
        frames.append(pd.read_csv(fp, usecols=usecols))
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["role", "alpha", "question"], keep="first")
    print(f"rows: {n_raw} raw, {len(df)} after dedup "
          f"({n_raw - len(df)} duplicates dropped)")
    return df


def per_role_r(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Pearson r over the n=4 (alpha, per-alpha mean) points, per role."""
    cell = df.groupby(["role", "alpha"])[score_col].mean().unstack()
    cell = cell.reindex(columns=ALPHAS).dropna()
    a = np.array(ALPHAS)
    a_c = a - a.mean()
    x = cell.to_numpy()
    x_c = x - x.mean(axis=1, keepdims=True)
    denom = np.sqrt((x_c ** 2).sum(axis=1)) * np.sqrt((a_c ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (x_c @ a_c) / denom
    out = pd.Series(r, index=cell.index).dropna()
    n_dropped = len(df["role"].unique()) - len(out)
    if n_dropped:
        print(f"  note: {score_col}: r defined for {len(out)} roles "
              f"({n_dropped} dropped: incomplete alpha coverage or zero variance)")
    return out


def classify_anti(df: pd.DataFrame) -> pd.Index:
    """Roles with negative alpha-trend on all six behavioral axes."""
    neg = pd.Series(True, index=pd.Index(df["role"].unique(), name="role"))
    for col in ["steered_score"] + CMP_DIMS:
        r = per_role_r(df, col)
        neg &= (r < 0).reindex(neg.index, fill_value=False)
    return neg[neg].index


def verify(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name}: got {got:.3f}, paper {want}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def fig_histogram(r_steered: pd.Series, r_aa: pd.Series):
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 2.0))
    bins = np.linspace(-1.05, 1.05, 43)
    ax.hist(r_steered, bins=bins, color=style.BLUE, alpha=0.85,
            edgecolor="white", linewidth=0.25, label="CastVectors")
    ax.hist(r_aa, bins=bins, color=style.VERMILLION, alpha=0.65,
            edgecolor=style.VERMILLION, linewidth=0.3, hatch="//",
            label="Assistant Axis")
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel(r"Per-Role Pearson $r$ (Score vs. $\alpha$)")
    ax.set_ylabel("Roles")
    ax.legend(loc="upper left")
    ax.annotate(f"median ${np.median(r_aa):+.2f}$", xy=(-0.89, 0),
                xytext=(-0.6, ax.get_ylim()[1] * 0.68),
                color=style.VERMILLION, ha="left", fontsize=7)
    ax.annotate(f"median ${np.median(r_steered):+.2f}$", xy=(0.98, 0),
                xytext=(0.82, ax.get_ylim()[1] * 0.92),
                color=style.BLUE, ha="right", fontsize=7)
    fig.savefig(OUT / "fig_pearson_r_distribution.pdf")
    plt.close(fig)


def fig_trajectory(df: pd.DataFrame, anti: pd.Index):
    is_anti = df["role"].isin(anti)
    groups = {
        "anti-controllable": (df[is_anti], style.SATURATION, "--", "s"),
        "controllable": (df[~is_anti], style.BLUE, "-", "o"),
    }
    fig, ax = plt.subplots(figsize=(style.HALF_PANEL_W_IN, 1.65))
    curve_means = {}
    for label, (sub, color, ls, marker) in groups.items():
        per_role = sub.groupby(["role", "alpha"])["steered_score"].mean().unstack()
        per_role = per_role.reindex(columns=ALPHAS)
        mean = per_role.mean(axis=0)
        curve_means[label] = (mean, color, len(per_role))
        sem = per_role.sem(axis=0)
        ax.plot(ALPHAS, mean, color=color, marker=marker, ms=2.5, ls=ls)
        ax.fill_between(ALPHAS, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=color, alpha=0.2, lw=0)
        ref = sub.groupby("role")["baseline_score"].mean().mean()
        ax.axhline(ref, color=color, lw=0.7, ls=":")
    ax.text(1.02, groups["controllable"][0].groupby("role")["baseline_score"]
            .mean().mean() + 1.2, "Prompted Reference", fontsize=6.0,
            color=style.GREY, va="bottom")
    # direct curve labels (legend would collide with the rising blue curve)
    # Labels follow the paper's category names: "anti-controllable over
    # the tested range" is the published term ("saturated" appears in the
    # paper only as one unidentified mechanism), and the non-anti bucket
    # includes the partial-deterioration roles, so it is not all
    # "controllable" — keep it neutral.
    m_anti, c_anti, n_anti = curve_means["anti-controllable"]
    ax.text(1.72, m_anti[1.5] - 4.5, f"Anti-controllable Roles\n(n={n_anti})",
            color=c_anti, fontsize=6.0, ha="center", va="top")
    m_ctl, c_ctl, n_ctl = curve_means["controllable"]
    ax.text(2.45, 50.5, f"Other Roles (n={n_ctl})",
            color=c_ctl, fontsize=6.0, ha="right", va="bottom")
    ax.set_xlabel(r"Steering Coefficient $\alpha$")
    ax.set_ylabel("Mean Judge Score")
    ax.set_xticks(ALPHAS)
    ax.set_ylim(45, 95)
    fig.savefig(OUT / "fig_trajectory_contrast.pdf")
    plt.close(fig)


def fig_dim_bars(df: pd.DataFrame, anti: pd.Index):
    sub = df[df["role"].isin(anti)]
    drops = {}
    for col in CMP_DIMS:
        cell = sub.groupby(["role", "alpha"])[col].mean().unstack()
        drops[DIM_LABELS[col]] = float(
            (cell[1.0] - cell[2.5]).mean()
        )
    order = sorted(drops, key=drops.get)
    vals = [drops[k] for k in order]
    fig, ax = plt.subplots(figsize=(style.HALF_PANEL_W_IN, 1.65))
    ax.barh(order, vals, color=style.SATURATION, alpha=0.85, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=6)
    ax.set_xlabel("Score Drop")  # alpha window stated in the caption
    ax.set_xlim(0, max(vals) * 1.22)
    fig.savefig(OUT / "fig_dim_deterioration_bars.pdf")
    plt.close(fig)
    return drops


def main():
    style.apply()
    df = load()

    print("verification gate:")
    r_steered = per_role_r(df, "steered_score")
    r_aa = per_role_r(df, "assistant_axis_score")
    verify("median r (role vectors)", float(np.median(r_steered)), 0.98, 0.005)
    verify("share r>0 (role vectors)", float((r_steered > 0).mean()), 0.82, 0.01)
    verify("share r>0.8 (role vectors)", float((r_steered > 0.8).mean()), 0.79, 0.01)
    verify("median r (assistant axis)", float(np.median(r_aa)), -0.89, 0.005)
    verify("share r<0 (assistant axis)", float((r_aa < 0).mean()), 0.981, 0.005)

    anti = classify_anti(df)
    print(f"  {'OK ' if len(anti) == 38 else 'FAIL'} anti-controllable count: "
          f"{len(anti)} (paper: 38, dedup convention)")
    if len(anti) != 38:
        raise SystemExit("anti count mismatch")

    traj = (df[df["role"].isin(anti)]
            .groupby(["role", "alpha"])["steered_score"].mean().unstack()
            .reindex(columns=ALPHAS).mean(axis=0))
    for a, want in zip(ALPHAS, [81.1, 80.7, 78.9, 74.2]):
        verify(f"anti mean @ alpha={a}", float(traj[a]), want, 0.1)
    anti_ref = df[df["role"].isin(anti)].groupby("role")["baseline_score"].mean().mean()
    verify("anti prompted-reference mean", float(anti_ref), 87.2, 0.1)

    fig_histogram(r_steered, r_aa)
    fig_trajectory(df, anti)
    drops = fig_dim_bars(df, anti)
    print("dimension drops (alpha 1.0->2.5, anti roles):")
    for k, v in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"   {k}: {v:.1f}")
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
