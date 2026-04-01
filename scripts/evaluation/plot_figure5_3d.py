"""
Generate Figure 5: 3D Style Scatter (2x2 grid across all 4 alphas).

Usage:
    python scripts/evaluation/plot_figure5_3d.py \
        --input_dir experiment_data/regenerated_modal/gold_prompt_experiments/ \
        --output_dir experiment_data/regenerated_modal/paper_figures/
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams.update({
    "text.usetex":          False,
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size":            11,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "figure.dpi":           150,
    "savefig.bbox":         "tight",
    "savefig.dpi":          300,
})

ALPHAS        = [1.0, 1.5, 2.0, 2.5]
STEERED_COLOR = "#2ecc71"
AXIS_COLOR    = "#3498db"

SCORE_COLS = [
    "cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic",
    "assistant_axis_cmp_emotional_register", "assistant_axis_cmp_vocab_choice",
    "assistant_axis_cmp_social_dynamic",
]


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
        raise FileNotFoundError(f"No valid CSVs found in {input_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["sample_count"] == 50].copy()
    for col in SCORE_COLS:
        if col in combined.columns:
            combined[col] = combined[col].apply(_parse_score)
    return combined


def plot(df: pd.DataFrame, output_dir: Path) -> None:
    cmp_df = df.dropna(subset=["assistant_axis_cmp_emotional_register"])
    if cmp_df.empty:
        print("Figure 5 skipped: no assistant_axis_cmp data")
        return

    fig = plt.figure(figsize=(14, 11))

    for i, alpha in enumerate(ALPHAS):
        sub = cmp_df[cmp_df["alpha"] == alpha]
        role_df = sub.groupby("role").agg(
            s_er=("cmp_emotional_register",                "mean"),
            s_vc=("cmp_vocab_choice",                      "mean"),
            s_sd=("cmp_social_dynamic",                    "mean"),
            a_er=("assistant_axis_cmp_emotional_register", "mean"),
            a_vc=("assistant_axis_cmp_vocab_choice",       "mean"),
            a_sd=("assistant_axis_cmp_social_dynamic",     "mean"),
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
        
        # Reduced labelpad for the Z-axis to pull the label inward
        ax.set_zlabel("Social\nDynamic",     labelpad=0, fontsize=8) 
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)
        ax.set_title(f"$\\alpha = {alpha}$  (n={len(role_df)} roles)", fontsize=10)
        ax.legend(fontsize=7)

    # Lowered the suptitle slightly so it doesn't clash with the top margin
    fig.suptitle("Style Dimension Alignment by Role: Steered vs. Assistant Axis", y=0.98)
    
    # Added padding between subplots
    plt.tight_layout(w_pad=4.0, h_pad=2.0)
    
    # Increased the right margin space (lowered the 'right' value)
    plt.subplots_adjust(right=0.75)
    
    path = output_dir / "figure5_3d_style_scatter.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  default="./experiment_data/regenerated_modal/gold_prompt_experiments/")
    ap.add_argument("--output_dir", default="./experiment_data/regenerated_modal/paper_figures/")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_dir)
    print(f"Loaded {len(df):,} rows | {df['role'].nunique()} roles | "
          f"alphas: {sorted(df['alpha'].unique())}")

    plot(df, output_dir)