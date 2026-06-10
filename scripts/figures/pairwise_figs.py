"""Regenerate the four Appendix I pairwise-evaluation figures from CSVs.

Outputs (vector PDF, true printed size):
  out/fig_pw_overall_win_rate.pdf   -> fig:overall_win_rate
  out/fig_pw_score_dist.pdf         -> fig:score_dist
  out/fig_pw_per_role_advantage.pdf -> fig:per_role_advantage
  out/fig_pw_per_role_win_rate.pdf  -> fig:per_role_win_rate

Data: experiment_data/pairwise_judge_experiments_gpt/ (39 roles, alpha=2.5,
sample_count=50; judged rows = pw_debiased_winner notnull; 7,666 pairs).

Conventions established 2026-06-11:
  - advantage "mean 48.7" = mean of per-role means; pooled-pairs mean is
    47.9 (the audit's number) -- caption now states the convention.
  - per-role minimum advantage is 11.4 (absurdist), not the old 11.2.
  - win rate = steered wins / all judged pairs (ties in denominator):
    ALL 39 roles have a majority, absurdist at bare parity 50.4% -- the
    old "38 of 39" claim does not hold; caption + appendix prose fixed.

NOTE: these figures describe Judge 3. Its gold-anchoring description in
Appendix I is pending a team decision and is NOT touched here.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pvx_fig_style as style

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiment_data" / "pairwise_judge_experiments_gpt"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def load():
    cols = ["role", "alpha", "question", "pw_debiased_steered_score",
            "pw_debiased_baseline_score", "pw_steered_advantage",
            "pw_debiased_winner"]
    frames = [pd.read_csv(fp, usecols=cols)
              for fp in sorted(DATA.glob("pairwise_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["pw_debiased_winner"].notna()]
    df = df.drop_duplicates(["role", "alpha", "question"])
    return df


def verify(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name}: got {got:.3f}, paper {want}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def fig_overall(df):
    shares = df["pw_debiased_winner"].value_counts(normalize=True)
    order = ["steered", "tie", "baseline"]
    labels = {"steered": "steered wins", "tie": "ties",
              "baseline": "assistant-axis wins"}
    colors = {"steered": style.BLUE, "tie": style.GREY,
              "baseline": style.VERMILLION}
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 0.95))
    left = 0.0
    small_above = True
    for k in order:
        v = float(shares[k])
        ax.barh([0], [v], left=left, color=colors[k], height=0.55)
        if v > 0.08:
            ax.text(left + v / 2, 0, f"{labels[k]}\n{v:.1%}", ha="center",
                    va="center", fontsize=6, color="white")
        elif small_above:
            ax.text(left + v / 2, 0.42, f"{labels[k]} {v:.1%}", ha="center",
                    va="bottom", fontsize=5.5, color=colors[k])
            small_above = False
        else:
            ax.text(left + v / 2, -0.42, f"{labels[k]} {v:.1%}", ha="center",
                    va="top", fontsize=5.5, color=colors[k])
        left += v
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("share of 7,666 judged pairs")
    fig.savefig(OUT / "fig_pw_overall_win_rate.pdf")
    plt.close(fig)
    return shares


def fig_score_dist(df):
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 1.5))
    bins = np.linspace(0, 100, 41)
    ax.hist(df["pw_debiased_steered_score"], bins=bins, color=style.BLUE,
            alpha=0.8, label=f"steered (mean {df['pw_debiased_steered_score'].mean():.1f})")
    ax.hist(df["pw_debiased_baseline_score"], bins=bins,
            color=style.VERMILLION, alpha=0.6,
            label=f"assistant axis (mean {df['pw_debiased_baseline_score'].mean():.1f})")
    ax.set_xlabel("debiased judge score")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=6, loc="upper left")
    fig.savefig(OUT / "fig_pw_score_dist.pdf")
    plt.close(fig)


def fig_per_role_advantage(df):
    adv = df.groupby("role")["pw_steered_advantage"].mean().sort_values(
        ascending=False)
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 1.9))
    ax.bar(range(len(adv)), adv, color=style.BLUE, width=0.75)
    ax.axhline(adv.mean(), color="black", lw=0.7, ls="--")
    ax.text(len(adv) - 1, adv.mean() + 1.5, f"mean {adv.mean():.1f}",
            fontsize=6, ha="right")
    ax.set_xticks(range(len(adv)))
    ax.set_xticklabels(adv.index, rotation=90, fontsize=3.8)
    ax.set_ylabel("mean steered advantage")
    ax.set_xlim(-0.6, len(adv) - 0.4)
    fig.savefig(OUT / "fig_pw_per_role_advantage.pdf")
    plt.close(fig)
    return adv


def fig_per_role_win_rate(df):
    win = (df.assign(w=df["pw_debiased_winner"] == "steered")
           .groupby("role")["w"].mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(style.COLUMN_W_IN, 1.9))
    colors = [style.VERMILLION if r == "absurdist" else style.BLUE
              for r in win.index]
    ax.bar(range(len(win)), win, color=colors, width=0.75)
    ax.axhline(0.5, color="black", lw=0.7, ls="--")
    ax.text(0.4, 0.515, "parity", fontsize=5.5)
    ax.set_xticks(range(len(win)))
    ax.set_xticklabels(win.index, rotation=90, fontsize=3.8)
    ax.set_ylabel("steered win rate")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.6, len(win) - 0.4)
    fig.savefig(OUT / "fig_pw_per_role_win_rate.pdf")
    plt.close(fig)
    return win


def main():
    style.apply()
    df = load()

    print("verification gate:")
    verify("judged pairs", float(len(df)), 7666, 0)
    verify("roles", float(df["role"].nunique()), 39, 0)
    shares = fig_overall(df)
    verify("steered win share", float(shares["steered"]), 0.929, 0.001)
    verify("tie share", float(shares["tie"]), 0.040, 0.001)
    verify("baseline win share", float(shares["baseline"]), 0.032, 0.001)
    verify("steered debiased mean",
           float(df["pw_debiased_steered_score"].mean()), 78.7, 0.1)
    verify("baseline debiased mean",
           float(df["pw_debiased_baseline_score"].mean()), 30.8, 0.1)
    verify("pooled advantage", float(df["pw_steered_advantage"].mean()),
           47.9, 0.1)

    adv = fig_per_role_advantage(df)
    verify("per-role advantage mean-of-means", float(adv.mean()), 48.7, 0.1)
    verify("per-role advantage min (absurdist)", float(adv.min()), 11.4, 0.1)
    assert adv.idxmin() == "absurdist"
    verify("all advantages positive", float((adv > 0).mean()), 1.0, 0)

    win = fig_per_role_win_rate(df)
    n_majority = int((win > 0.5).sum())
    print(f"  majority-win roles: {n_majority}/39 "
          f"(lowest: {win.idxmin()} {win.min():.1%})")
    if n_majority != 39 or win.idxmin() != "absurdist":
        raise SystemExit("per-role win-rate pattern changed")

    fig_score_dist(df)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
