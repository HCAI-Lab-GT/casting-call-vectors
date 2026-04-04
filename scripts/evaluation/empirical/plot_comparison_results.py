"""
Visualize gold-standard comparison results across alphas.

Usage:
    python scripts/evaluation/plot_comparison_results.py \
        --input_dir experiment_data/gold_prompt_experiments/ \
        --output_dir analysis/empirical/comparison_plots/ \
        --best_alpha 2.5
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Colors ────────────────────────────────────────────────────────────────────
STEERED_COLOR   = "#2ecc71"
AXIS_COLOR      = "#3498db"
BASELINE_COLOR  = "#e74c3c"
TIE_COLOR       = "#95a5a6"

ALPHA_COLORS = {1.0: "#aed6f1", 1.5: "#5dade2", 2.0: "#2e86c1", 2.5: "#1a5276"}
ALPHA_MARKERS = {1.0: "o", 1.5: "s", 2.0: "^", 2.5: "D"}

# ── Column groups ─────────────────────────────────────────────────────────────
STYLE_DIMS       = ["cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic"]
CONTENT_DIMS     = ["cmp_motivation", "cmp_worldview_alignment"]
AXIS_STYLE_DIMS  = ["assistant_axis_cmp_emotional_register",
                    "assistant_axis_cmp_vocab_choice",
                    "assistant_axis_cmp_social_dynamic"]
AXIS_CONTENT_DIMS = ["assistant_axis_cmp_motivation",
                     "assistant_axis_cmp_worldview_alignment"]
DIM_LABELS       = ["Emotional\nRegister", "Vocab\nChoice", "Social\nDynamic",
                    "Motivation", "Worldview\nAlignment"]
DIM_KEYS         = ["emotional_register", "vocab_choice", "social_dynamic",
                    "motivation", "worldview_alignment"]


# ── Data loading ──────────────────────────────────────────────────────────────
def _parse_score(val) -> float:
    """Extract a numeric score from a value that may be a serialized pandas Series string."""
    if pd.isna(val):
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        # Handles strings like "0    10\nName: assistant_axis_score, dtype: int64"
        m = re.search(r"\b(\d+)\s*\n", str(val))
        if m:
            return float(m.group(1))
        m = re.search(r"\d+", str(val))
        if m:
            return float(m.group())
        return float("nan")


SCORE_COLS = [
    "steered_score", "assistant_axis_score", "baseline_score",
    "cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic",
    "cmp_motivation", "cmp_worldview_alignment",
    "assistant_axis_cmp_emotional_register", "assistant_axis_cmp_vocab_choice",
    "assistant_axis_cmp_social_dynamic", "assistant_axis_cmp_motivation",
    "assistant_axis_cmp_worldview_alignment",
]


def load_results(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{input_dir}/Comparison_GoldStandard_*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        raise FileNotFoundError(f"No valid Comparison CSVs found in {input_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    # Coerce score columns — some rows contain serialized Series strings
    for col in SCORE_COLS:
        if col in combined.columns:
            combined[col] = combined[col].apply(_parse_score)
    return combined


def enrich(df: pd.DataFrame, tie_threshold: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    diff = df["steered_score"] - df["assistant_axis_score"]
    df["winner"] = np.where(diff > tie_threshold, "steered",
                   np.where(diff < -tie_threshold, "assistant_axis", "tie"))
    df["advantage"] = diff
    df["steered_style"]   = df[STYLE_DIMS].mean(axis=1)
    df["steered_content"] = df[CONTENT_DIMS].mean(axis=1)
    # axis_cmp columns may be NaN for most rows — computed only where available
    df["axis_style"]   = df[AXIS_STYLE_DIMS].mean(axis=1)
    df["axis_content"] = df[AXIS_CONTENT_DIMS].mean(axis=1)
    return df


# ── Plot 1: Alpha vs. Score ───────────────────────────────────────────────────
def plot1_alpha_vs_score(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby("alpha").agg(
        steered      =("steered_score",       "mean"),
        assistant_axis=("assistant_axis_score","mean"),
        baseline     =("baseline_score",      "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["alpha"], grouped["steered"], marker="o", color=STEERED_COLOR,
            linewidth=2, label="Steered (ours)")
    ax.plot(grouped["alpha"], grouped["assistant_axis"], marker="s", color=AXIS_COLOR,
            linewidth=2, linestyle="--", label="Assistant axis")
    ax.plot(grouped["alpha"], grouped["baseline"], marker="^", color=BASELINE_COLOR,
            linewidth=2, linestyle=":", label="Baseline (prompted)")

    ax.set_xlabel("Steering strength (α)")
    ax.set_ylabel("Avg role alignment score (0–100)")
    ax.set_title("Role Alignment Score vs. Steering Strength")
    ax.set_xticks(grouped["alpha"].tolist())
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / "plot1_alpha_vs_score.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: Per-Role Win Rate Stacked Bar ─────────────────────────────────────
def plot2_per_role_win_rate(df: pd.DataFrame, best_alpha: float,
                            output_dir: Path) -> None:
    sub   = df[df["alpha"] == best_alpha]
    roles = sorted(sub["role"].unique())
    steered_pcts, tie_pcts, axis_pcts = [], [], []

    for role in roles:
        rdf   = sub[sub["role"] == role]
        total = len(rdf)
        counts = rdf["winner"].value_counts()
        steered_pcts.append(counts.get("steered",        0) / total * 100)
        tie_pcts.append(    counts.get("tie",            0) / total * 100)
        axis_pcts.append(   counts.get("assistant_axis", 0) / total * 100)

    x = range(len(roles))
    fig, ax = plt.subplots(figsize=(max(14, len(roles) * 0.45), 5))
    ax.bar(x, steered_pcts, color=STEERED_COLOR, label="Steered wins")
    ax.bar(x, tie_pcts, bottom=steered_pcts, color=TIE_COLOR, label="Tie")
    ax.bar(x, axis_pcts,
           bottom=[s + t for s, t in zip(steered_pcts, tie_pcts)],
           color=AXIS_COLOR, label="Assistant axis wins")

    ax.set_xticks(list(x))
    ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of comparisons")
    ax.set_title(f"Per-Role Win Rate: Steered vs. Assistant Axis  (α={best_alpha})")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"plot2_per_role_win_rate_alpha{best_alpha}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: Per-Role Advantage Bar ────────────────────────────────────────────
def plot3_per_role_advantage(df: pd.DataFrame, best_alpha: float,
                             output_dir: Path) -> None:
    sub      = df[df["alpha"] == best_alpha]
    role_avg = sub.groupby("role")["advantage"].mean().sort_values(ascending=False)

    fig, ax  = plt.subplots(figsize=(max(14, len(role_avg) * 0.45), 5))
    colors   = [STEERED_COLOR if v >= 0 else AXIS_COLOR for v in role_avg.values]
    ax.bar(range(len(role_avg)), role_avg.values, color=colors,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(role_avg.mean(), color="#2c3e50", linewidth=1.2, linestyle="--",
               label=f"Mean: {role_avg.mean():.1f}")

    ax.set_xticks(list(range(len(role_avg))))
    ax.set_xticklabels(role_avg.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Avg score advantage (steered − assistant axis)")
    ax.set_title(f"Per-Role Steered Advantage  (α={best_alpha})")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"plot3_per_role_advantage_alpha{best_alpha}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 4: Score Distribution Histogram ──────────────────────────────────────
def plot4_score_distribution(df: pd.DataFrame, best_alpha: float,
                             output_dir: Path) -> None:
    sub = df[df["alpha"] == best_alpha]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sub["steered_score"],       bins=30, alpha=0.7,
            color=STEERED_COLOR, label="Steered (ours)")
    ax.hist(sub["assistant_axis_score"], bins=30, alpha=0.7,
            color=AXIS_COLOR,    label="Assistant axis")
    ax.axvline(sub["steered_score"].mean(), color="#27ae60",
               linestyle="--", linewidth=1.5,
               label=f'Steered mean: {sub["steered_score"].mean():.1f}')
    ax.axvline(sub["assistant_axis_score"].mean(), color="#1a5276",
               linestyle="--", linewidth=1.5,
               label=f'Axis mean: {sub["assistant_axis_score"].mean():.1f}')

    ax.set_xlabel("Role alignment score (0–100)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution: Steered vs. Assistant Axis  (α={best_alpha})")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"plot4_score_distribution_alpha{best_alpha}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 5: Style vs. Content Scatter ─────────────────────────────────────────
def plot5_style_content_scatter(df: pd.DataFrame, best_alpha: float,
                                output_dir: Path) -> None:
    sub = df[(df["alpha"] == best_alpha) &
             df["assistant_axis_cmp_emotional_register"].notna()]

    if sub.empty:
        print("Plot 5 skipped: no assistant_axis_cmp data at alpha={best_alpha}")
        return

    role_df = sub.groupby("role").agg(
        steered_style  =("steered_style",   "mean"),
        steered_content=("steered_content", "mean"),
        axis_style     =("axis_style",      "mean"),
        axis_content   =("axis_content",    "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(role_df["steered_style"], role_df["steered_content"],
               color=STEERED_COLOR, s=65, alpha=0.9, label="Steered (ours)", zorder=3)
    ax.scatter(role_df["axis_style"], role_df["axis_content"],
               color=AXIS_COLOR, s=65, alpha=0.9, marker="s",
               label="Assistant axis", zorder=3)

    for _, row in role_df.iterrows():
        ax.plot([row["steered_style"], row["axis_style"]],
                [row["steered_content"], row["axis_content"]],
                color="gray", alpha=0.25, linewidth=0.8, zorder=1)
        ax.annotate(row["role"],
                    (row["steered_style"], row["steered_content"]),
                    fontsize=5, alpha=0.65, ha="center", va="bottom")

    ax.axhline(50, color="lightgray", linewidth=0.8, linestyle="--")
    ax.axvline(50, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Style alignment  (emotional register + vocab choice + social dynamic) / 3")
    ax.set_ylabel("Content alignment  (motivation + worldview alignment) / 2")
    ax.set_title(
        f"Style vs. Content Alignment by Role  (α={best_alpha}, n={len(role_df)} roles)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"plot5_style_content_scatter_alpha{best_alpha}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 6: 5-Dimension Grouped Bar ───────────────────────────────────────────
def plot6_dimension_comparison(df: pd.DataFrame, best_alpha: float,
                               output_dir: Path) -> None:
    sub = df[(df["alpha"] == best_alpha) &
             df["assistant_axis_cmp_emotional_register"].notna()]

    if sub.empty:
        print(f"Plot 6 skipped: no assistant_axis_cmp data at alpha={best_alpha}")
        return

    steered_means = [
        sub["cmp_emotional_register"].mean(),
        sub["cmp_vocab_choice"].mean(),
        sub["cmp_social_dynamic"].mean(),
        sub["cmp_motivation"].mean(),
        sub["cmp_worldview_alignment"].mean(),
    ]
    axis_means = [
        sub["assistant_axis_cmp_emotional_register"].mean(),
        sub["assistant_axis_cmp_vocab_choice"].mean(),
        sub["assistant_axis_cmp_social_dynamic"].mean(),
        sub["assistant_axis_cmp_motivation"].mean(),
        sub["assistant_axis_cmp_worldview_alignment"].mean(),
    ]

    x     = np.arange(len(DIM_LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, steered_means, width,
                   color=STEERED_COLOR, label="Steered (ours)", edgecolor="white")
    bars2 = ax.bar(x + width / 2, axis_means,    width,
                   color=AXIS_COLOR,   label="Assistant axis",  edgecolor="white")

    for bar, val in zip(bars1, steered_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, axis_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(DIM_LABELS, fontsize=9)
    ax.set_ylabel("Avg alignment to gold label (0–100)")
    ax.set_title(
        f"Gold-Label Alignment by Dimension  (α={best_alpha}, n={len(sub['role'].unique())} roles)"
    )
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = output_dir / f"plot6_dimension_comparison_alpha{best_alpha}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",     default="./experiment_data/gold_prompt_experiments/")
    ap.add_argument("--output_dir",    default="./analysis/empirical/comparison_plots/")
    ap.add_argument("--best_alpha",    type=float, default=2.5)
    ap.add_argument("--tie_threshold", type=float, default=2.0,
                    help="Score diff below which result is a tie (default: 2.0)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_dir)
    print(f"Loaded {len(df):,} rows | {df['role'].nunique()} roles | "
          f"alphas: {sorted(df['alpha'].unique())}")

    df = enrich(df, tie_threshold=args.tie_threshold)

    plot1_alpha_vs_score(df, output_dir)
    plot2_per_role_win_rate(df, args.best_alpha, output_dir)
    plot3_per_role_advantage(df, args.best_alpha, output_dir)
    plot4_score_distribution(df, args.best_alpha, output_dir)
    plot5_style_content_scatter(df, args.best_alpha, output_dir)
    plot6_dimension_comparison(df, args.best_alpha, output_dir)

    print(f"\nAll plots saved to {output_dir}")
