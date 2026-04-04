"""
Generate publication-ready figures for persona vector steering paper.

Figures produced:
  figure1_alpha_vs_score.pdf         -- Line chart: alpha vs avg score (3 methods)
  figure2_score_distribution.pdf     -- 2x2 grid: score distributions at each alpha
  figure3_style_content_scatter.pdf  -- 2x2 grid: style vs content scatter at each alpha
  figure4_dimension_comparison.pdf   -- 2x2 grid: 5-dim grouped bar at each alpha
  figure5_3d_style_scatter.pdf       -- 3D scatter: 3 style dimensions
  figure6_2d_content_scatter.pdf     -- 2D scatter: 2 content dimensions

Usage:
    python scripts/evaluation/plot_paper_figures.py \
        --input_dir experiment_data/gold_prompt_experiments/ \
        --output_dir analysis/empirical/paper_figures/
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── LaTeX / publication style ─────────────────────────────────────────────────
matplotlib.rcParams.update({
    "text.usetex":          False,   # set True if LaTeX is installed
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size":            11,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "figure.dpi":           150,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.3,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "savefig.bbox":         "tight",
    "savefig.dpi":          300,
})

ALPHAS = [1.0, 1.5, 2.0, 2.5]

# ── Colors ────────────────────────────────────────────────────────────────────
STEERED_COLOR  = "#2ecc71"
AXIS_COLOR     = "#3498db"
BASELINE_COLOR = "#e74c3c"
TIE_COLOR      = "#95a5a6"

ALPHA_STYLES = {
    1.0: {"color": "#aed6f1", "marker": "o"},
    1.5: {"color": "#5dade2", "marker": "s"},
    2.0: {"color": "#2e86c1", "marker": "^"},
    2.5: {"color": "#1a5276", "marker": "D"},
}

# ── Dimension column groups ───────────────────────────────────────────────────
STYLE_DIMS      = ["cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic"]
CONTENT_DIMS    = ["cmp_motivation", "cmp_worldview_alignment"]
AXIS_STYLE_DIMS = ["assistant_axis_cmp_emotional_register",
                   "assistant_axis_cmp_vocab_choice",
                   "assistant_axis_cmp_social_dynamic"]
AXIS_CONTENT_DIMS = ["assistant_axis_cmp_motivation",
                     "assistant_axis_cmp_worldview_alignment"]
DIM_LABELS      = ["Emotional\nRegister", "Vocab\nChoice", "Social\nDynamic",
                   "Motivation", "Worldview\nAlignment"]

SCORE_COLS = [
    "steered_score", "assistant_axis_score", "baseline_score",
    "cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic",
    "cmp_motivation", "cmp_worldview_alignment",
    "assistant_axis_cmp_emotional_register", "assistant_axis_cmp_vocab_choice",
    "assistant_axis_cmp_social_dynamic", "assistant_axis_cmp_motivation",
    "assistant_axis_cmp_worldview_alignment",
]


# ── Data loading ──────────────────────────────────────────────────────────────
def _parse_score(val) -> float:
    if pd.isna(val):
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        m = re.search(r"\b(\d+)\s*\n", str(val))
        if m:
            return float(m.group(1))
        m = re.search(r"\d+", str(val))
        if m:
            return float(m.group())
        return float("nan")


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
    combined = combined[combined["sample_count"] == 50].copy()
    for col in SCORE_COLS:
        if col in combined.columns:
            combined[col] = combined[col].apply(_parse_score)
    combined["steered_style"]    = combined[STYLE_DIMS].mean(axis=1)
    combined["steered_content"]  = combined[CONTENT_DIMS].mean(axis=1)
    combined["axis_style"]       = combined[AXIS_STYLE_DIMS].mean(axis=1)
    combined["axis_content"]     = combined[AXIS_CONTENT_DIMS].mean(axis=1)
    return combined


# ── Figure 1: Alpha vs. Score ─────────────────────────────────────────────────
def figure1_alpha_vs_score(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby("alpha").agg(
        steered       =("steered_score",        "mean"),
        assistant_axis=("assistant_axis_score",  "mean"),
        baseline      =("baseline_score",        "mean"),
        steered_std   =("steered_score",         "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))

    # Shaded std band for steered
    ax.fill_between(
        grouped["alpha"],
        grouped["steered"] - grouped["steered_std"],
        grouped["steered"] + grouped["steered_std"],
        color=STEERED_COLOR, alpha=0.08,
    )
    ax.plot(grouped["alpha"], grouped["steered"], marker="o",
            color=STEERED_COLOR, linewidth=2, label="Steered")
    ax.plot(grouped["alpha"], grouped["assistant_axis"], marker="s",
            color=AXIS_COLOR, linewidth=2, linestyle="--", label="Assistant axis")
    ax.plot(grouped["alpha"], grouped["baseline"], marker="^",
            color=BASELINE_COLOR, linewidth=2, linestyle=":", label="Baseline")

    ax.set_xlabel(r"Steering strength ($\alpha$)")
    ax.set_ylabel("Mean role alignment score (0–100)")
    ax.set_title("Role Alignment vs. Steering Strength")
    ax.set_xticks(ALPHAS)
    ax.legend()
    plt.tight_layout()
    path = output_dir / "figure1_alpha_vs_score.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


# ── Figure 2: Score Distribution 2×2 ─────────────────────────────────────────
def figure2_score_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, alpha in enumerate(ALPHAS):
        ax  = axes[i]
        sub = df[df["alpha"] == alpha]

        ax.hist(sub["steered_score"],        bins=30, alpha=0.7,
                color=STEERED_COLOR, label="Steered")
        ax.hist(sub["assistant_axis_score"], bins=30, alpha=0.7,
                color=AXIS_COLOR,    label="Assistant axis")
        ax.axvline(sub["steered_score"].mean(),        color="#27ae60",
                   linestyle="--", linewidth=1.4,
                   label=f'Steered $\\mu$={sub["steered_score"].mean():.1f}')
        ax.axvline(sub["assistant_axis_score"].mean(), color="#1a5276",
                   linestyle="--", linewidth=1.4,
                   label=f'Axis $\\mu$={sub["assistant_axis_score"].mean():.1f}')

        ax.set_title(f"$\\alpha = {alpha}$")
        ax.set_xlabel("Role alignment score (0–100)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Score Distributions: Steered vs. Assistant Axis", y=1.01)
    plt.tight_layout()
    path = output_dir / "figure2_score_distribution.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


# ── Figure 3: Style vs. Content Scatter 2×2 ──────────────────────────────────
def figure3_style_content_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    cmp_df = df.dropna(subset=["assistant_axis_cmp_emotional_register"])
    if cmp_df.empty:
        print("Figure 3 skipped: no assistant_axis_cmp data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, alpha in enumerate(ALPHAS):
        ax  = axes[i]
        sub = cmp_df[cmp_df["alpha"] == alpha]

        role_df = sub.groupby("role").agg(
            steered_style  =("steered_style",   "mean"),
            steered_content=("steered_content", "mean"),
            axis_style     =("axis_style",      "mean"),
            axis_content   =("axis_content",    "mean"),
        ).reset_index()

        ax.scatter(role_df["steered_style"], role_df["steered_content"],
                   color=STEERED_COLOR, s=45, alpha=0.85, label="Steered", zorder=3)
        ax.scatter(role_df["axis_style"], role_df["axis_content"],
                   color=AXIS_COLOR, s=45, alpha=0.85, marker="s",
                   label="Assistant axis", zorder=3)

        ax.axhline(50, color="lightgray", linewidth=0.8, linestyle="--", zorder=1)
        ax.axvline(50, color="lightgray", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(f"$\\alpha = {alpha}$  (n={len(role_df)} roles)")
        ax.set_xlabel("Style alignment (0–100)")
        ax.set_ylabel("Content alignment (0–100)")
        ax.legend(fontsize=8)

    fig.suptitle("Style vs. Content Alignment by Role", y=1.01)
    plt.tight_layout()
    path = output_dir / "figure3_style_content_scatter.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


# ── Figure 4: 5-Dimension Comparison 2×2 ─────────────────────────────────────
def figure4_dimension_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    cmp_df = df.dropna(subset=["assistant_axis_cmp_emotional_register"])
    if cmp_df.empty:
        print("Figure 4 skipped: no assistant_axis_cmp data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    x     = np.arange(len(DIM_LABELS))
    width = 0.35

    for i, alpha in enumerate(ALPHAS):
        ax  = axes[i]
        sub = cmp_df[cmp_df["alpha"] == alpha]

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

        bars1 = ax.bar(x - width / 2, steered_means, width,
                       color=STEERED_COLOR, label="Steered", edgecolor="white")
        bars2 = ax.bar(x + width / 2, axis_means,    width,
                       color=AXIS_COLOR,   label="Assistant axis",  edgecolor="white")

        for bar, val in zip(bars1, steered_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars2, axis_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS, fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Avg alignment to gold label (0–100)")
        ax.set_title(f"$\\alpha = {alpha}$  (n={sub['role'].nunique()} roles)")
        ax.legend(fontsize=8)

    fig.suptitle("Gold-Label Alignment by Dimension: Steered vs. Assistant Axis", y=1.01)
    plt.tight_layout()
    path = output_dir / "figure4_dimension_comparison.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


# ── Figure 5: 3D Style Scatter 2×2 ───────────────────────────────────────────
def figure5_3d_style_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    cmp_df = df.dropna(subset=["assistant_axis_cmp_emotional_register"])
    if cmp_df.empty:
        print("Figure 5 skipped: no assistant_axis_cmp data")
        return

    fig = plt.figure(figsize=(14, 11))

    for i, alpha in enumerate(ALPHAS):
        sub = cmp_df[cmp_df["alpha"] == alpha]
        role_df = sub.groupby("role").agg(
            s_er=("cmp_emotional_register",               "mean"),
            s_vc=("cmp_vocab_choice",                     "mean"),
            s_sd=("cmp_social_dynamic",                   "mean"),
            a_er=("assistant_axis_cmp_emotional_register","mean"),
            a_vc=("assistant_axis_cmp_vocab_choice",      "mean"),
            a_sd=("assistant_axis_cmp_social_dynamic",    "mean"),
        ).reset_index()

        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(role_df["s_er"], role_df["s_vc"], role_df["s_sd"],
                   color=STEERED_COLOR, s=45, alpha=0.85,
                   label="Steered", depthshade=True)
        ax.scatter(role_df["a_er"], role_df["a_vc"], role_df["a_sd"],
                   color=AXIS_COLOR, s=45, alpha=0.85, marker="s",
                   label="Assistant axis", depthshade=True)

        ax.set_xlabel("Emotional\nRegister", labelpad=6, fontsize=8)
        ax.set_ylabel("Vocab\nChoice",       labelpad=6, fontsize=8)
        ax.set_zlabel("Social\nDynamic",     labelpad=6, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)
        ax.set_title(f"$\\alpha = {alpha}$  (n={len(role_df)} roles)", fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("Style Dimension Alignment by Role: Steered vs. Assistant Axis", y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    path = output_dir / "figure5_3d_style_scatter.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


# ── Figure 6: 2D Content Scatter 2×2 ─────────────────────────────────────────
def figure6_2d_content_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    cmp_df = df.dropna(subset=["assistant_axis_cmp_motivation"])
    if cmp_df.empty:
        print("Figure 6 skipped: no assistant_axis_cmp data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, alpha in enumerate(ALPHAS):
        ax  = axes[i]
        sub = cmp_df[cmp_df["alpha"] == alpha]

        role_df = sub.groupby("role").agg(
            s_mot=("cmp_motivation",                        "mean"),
            s_wv =("cmp_worldview_alignment",               "mean"),
            a_mot=("assistant_axis_cmp_motivation",         "mean"),
            a_wv =("assistant_axis_cmp_worldview_alignment","mean"),
        ).reset_index()

        ax.scatter(role_df["s_mot"], role_df["s_wv"],
                   color=STEERED_COLOR, s=50, alpha=0.85,
                   label="Steered", zorder=3)
        ax.scatter(role_df["a_mot"], role_df["a_wv"],
                   color=AXIS_COLOR, s=50, alpha=0.85, marker="s",
                   label="Assistant axis", zorder=3)

        ax.axhline(50, color="lightgray", linewidth=0.8, linestyle="--", zorder=1)
        ax.axvline(50, color="lightgray", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Motivation alignment (0–100)")
        ax.set_ylabel("Worldview alignment (0–100)")
        ax.set_title(f"$\\alpha = {alpha}$  (n={len(role_df)} roles)")
        ax.legend(fontsize=8)

    fig.suptitle("Content Dimension Alignment by Role: Steered vs. Assistant Axis", y=1.01)
    plt.tight_layout()
    path = output_dir / "figure6_2d_content_scatter.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  default="./experiment_data/gold_prompt_experiments/")
    ap.add_argument("--output_dir", default="./analysis/empirical/paper_figures/")
    ap.add_argument("--best_alpha", type=float, default=2.5,
                    help="Alpha used for single-alpha figures (5, 6)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_dir)
    print(f"Loaded {len(df):,} rows | {df['role'].nunique()} roles | "
          f"alphas: {sorted(df['alpha'].unique())}")

    figure1_alpha_vs_score(df, output_dir)
    figure2_score_distribution(df, output_dir)
    figure3_style_content_scatter(df, output_dir)
    figure4_dimension_comparison(df, output_dir)
    figure5_3d_style_scatter(df, output_dir)
    figure6_2d_content_scatter(df, output_dir)

    print(f"\nAll figures saved to {output_dir}")
