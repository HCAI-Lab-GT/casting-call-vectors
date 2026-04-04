"""
Visualize pairwise judge results from pairwise_judge_experiments.py output.

Usage:
    python scripts/evaluation/plot_pairwise_results.py \
        --input_dir experiment_data/pairwise_judge_experiments_gpt/ \
        --output_dir analysis/empirical/pairwise_graphs/ \
        --sample_count 50 --alpha 2.5
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

WINNER_COLORS = {
    "steered":  "#2ecc71",
    "tie":      "#95a5a6",
    "baseline": "#e74c3c",
}


def load_results(input_dir: str, sample_count: int, alpha: float) -> pd.DataFrame:
    files = sorted(glob.glob(f"{input_dir}/pairwise_*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        raise FileNotFoundError(f"No valid pairwise CSVs found in {input_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    return combined[
        (combined["sample_count"] == sample_count) &
        (combined["alpha"] == alpha)
    ].copy()


def plot_overall_win_rate(df: pd.DataFrame, label: str, output_dir: Path) -> None:
    counts = df["pw_debiased_winner"].value_counts()
    total = len(df)
    categories = ["steered", "tie", "baseline"]
    values = [counts.get(c, 0) for c in categories]
    pcts = [v / total * 100 for v in values]
    colors = [WINNER_COLORS[c] for c in categories]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, pcts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, pct, count in zip(bars, pcts, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{pct:.1f}%\n(n={count})", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("% of comparisons")
    ax.set_title(f"Overall Pairwise Win Rate\n{label}", fontsize=10)
    ax.set_ylim(0, max(pcts) * 1.2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"overall_win_rate_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_role_win_rate(df: pd.DataFrame, label: str, output_dir: Path) -> None:
    roles = sorted(df["role"].unique())
    steered_pcts, tie_pcts, baseline_pcts = [], [], []

    for role in roles:
        rdf = df[df["role"] == role]
        total = len(rdf)
        counts = rdf["pw_debiased_winner"].value_counts()
        steered_pcts.append(counts.get("steered", 0) / total * 100)
        tie_pcts.append(counts.get("tie", 0) / total * 100)
        baseline_pcts.append(counts.get("baseline", 0) / total * 100)

    x = range(len(roles))
    fig, ax = plt.subplots(figsize=(max(10, len(roles) * 0.4), 5))
    ax.bar(x, steered_pcts, label="steered", color=WINNER_COLORS["steered"])
    ax.bar(x, tie_pcts, bottom=steered_pcts, label="tie", color=WINNER_COLORS["tie"])
    ax.bar(x, baseline_pcts,
           bottom=[s + t for s, t in zip(steered_pcts, tie_pcts)],
           label="baseline", color=WINNER_COLORS["baseline"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of comparisons")
    ax.set_title(f"Per-Role Pairwise Win Rate\n{label}", fontsize=10)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"per_role_win_rate_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_role_advantage(df: pd.DataFrame, label: str, output_dir: Path) -> None:
    role_avg = (
        df.groupby("role")["pw_steered_advantage"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(max(10, len(role_avg) * 0.4), 5))
    colors = [WINNER_COLORS["steered"] if v >= 0 else WINNER_COLORS["baseline"]
              for v in role_avg.values]
    ax.bar(range(len(role_avg)), role_avg.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(role_avg.mean(), color="#2c3e50", linewidth=1.2, linestyle="--",
               label=f"Mean: {role_avg.mean():.1f}")

    ax.set_xticks(list(range(len(role_avg))))
    ax.set_xticklabels(role_avg.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Avg steered advantage (0-100 scale)")
    ax.set_title(f"Per-Role Average Steered Advantage\n{label}", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"per_role_advantage_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_score_distribution(df: pd.DataFrame, label: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["pw_debiased_steered_score"], bins=30, alpha=0.7,
            color=WINNER_COLORS["steered"], label="Our method (steered)")
    ax.hist(df["pw_debiased_baseline_score"], bins=30, alpha=0.7,
            color=WINNER_COLORS["baseline"], label="Assistant axis (baseline)")
    ax.axvline(df["pw_debiased_steered_score"].mean(), color="#27ae60",
               linestyle="--", linewidth=1.5,
               label=f'Steered mean: {df["pw_debiased_steered_score"].mean():.1f}')
    ax.axvline(df["pw_debiased_baseline_score"].mean(), color="#c0392b",
               linestyle="--", linewidth=1.5,
               label=f'Baseline mean: {df["pw_debiased_baseline_score"].mean():.1f}')
    ax.set_xlabel("Debiased score (0-100)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution\n{label}", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"score_distribution_{label}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="./experiment_data/pairwise_judge_experiments_gpt/")
    ap.add_argument("--output_dir", default="./analysis/empirical/pairwise_graphs/")
    ap.add_argument("--sample_count", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=2.5)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = f"sample_count_{args.sample_count}_alpha_{args.alpha}"

    df = load_results(args.input_dir, args.sample_count, args.alpha)
    print(f"Loaded {len(df)} rows for {label} across {df['role'].nunique()} roles")

    plot_overall_win_rate(df, label, output_dir)
    plot_per_role_win_rate(df, label, output_dir)
    plot_per_role_advantage(df, label, output_dir)
    plot_score_distribution(df, label, output_dir)

    print(f"\nAll plots saved to {output_dir}")
