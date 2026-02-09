#!/usr/bin/env python
"""
Generate visualization plots for the Marin personality vector experiments.

Produces:
  1. PCA scree plot (explained variance)
  2. Role projections scatter (PC1 vs PC2, colored by category)
  3. RIASEC vectors projected into PCA space
  4. Cosine similarity heatmap (RIASEC vs PCA components)
  5. Steering effectiveness bar charts (from RIASEC eval)

Usage:
  python scripts/marin/analysis/visualize.py
  python scripts/marin/analysis/visualize.py --model_id marin-community/marin-8b-instruct
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

from pvx import setup_logging

logger = setup_logging(name="marin-visualize")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Color palette for role categories
CATEGORY_COLORS = {
    "scientist": "#1f77b4",
    "artist": "#ff7f0e",
    "professional": "#2ca02c",
    "thinker": "#d62728",
    "adventurer": "#9467bd",
    "leader": "#8c564b",
    "craftsperson": "#e377c2",
    "communicator": "#7f7f7f",
    "unconventional": "#bcbd22",
    "assistant": "#17becf",
}


def plot_scree(explained_variance: np.ndarray, output_path: Path):
    """Plot PCA scree plot with explained and cumulative variance."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    n = len(explained_variance)
    x = np.arange(1, n + 1)

    ax1.bar(x, explained_variance * 100, alpha=0.7, color="#1f77b4", label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("PCA Scree Plot: Role Vector Variance")

    ax2 = ax1.twinx()
    cumulative = np.cumsum(explained_variance) * 100
    ax2.plot(x, cumulative, "r-o", markersize=4, label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_ylim(0, 105)

    # Add reference lines
    for threshold in [50, 70, 90]:
        ax2.axhline(y=threshold, color="gray", linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scree plot: %s", output_path)


def plot_role_scatter(
    projections: np.ndarray, role_names: list, role_categories: dict,
    output_path: Path,
):
    """Scatter plot of roles in PC1 vs PC2 space, colored by category."""
    fig, ax = plt.subplots(figsize=(12, 8))

    categories = set(role_categories.values())
    for cat in sorted(categories):
        indices = [i for i, name in enumerate(role_names) if role_categories.get(name) == cat]
        if not indices:
            continue
        color = CATEGORY_COLORS.get(cat, "#333333")
        ax.scatter(
            projections[indices, 0], projections[indices, 1],
            c=color, label=cat, s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
        )
        # Label points
        for idx in indices:
            name = role_names[idx].replace("_", " ")
            ax.annotate(
                name, (projections[idx, 0], projections[idx, 1]),
                fontsize=6, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel("PC1 (Assistant Axis)")
    ax.set_ylabel("PC2")
    ax.set_title("Role Projections in PCA Space")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved role scatter: %s", output_path)


def plot_cosine_heatmap(
    cosine_matrix: np.ndarray, trait_names: list, pca_labels: list,
    output_path: Path,
):
    """Heatmap of cosine similarities between RIASEC vectors and PCA components."""
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(cosine_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pca_labels)))
    ax.set_xticklabels(pca_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(trait_names)))
    ax.set_yticklabels([t.capitalize() for t in trait_names])
    ax.set_title("Cosine Similarity: RIASEC Vectors vs PCA Components")

    # Add text annotations
    for i in range(len(trait_names)):
        for j in range(len(pca_labels)):
            val = cosine_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cosine heatmap: %s", output_path)


def plot_steering_effectiveness(eval_path: Path, output_path: Path):
    """Bar chart of RIASEC steering effectiveness (YES counts by alpha)."""
    with open(eval_path) as f:
        eval_data = json.load(f)

    results = eval_data["results"]
    traits = sorted(results.keys())
    alphas = eval_data["alphas"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, trait in enumerate(traits):
        ax = axes[i]
        trait_data = results[trait]

        alpha_labels = []
        target_counts = []
        total_counts = []

        for alpha in alphas:
            a_key = str(alpha)
            if a_key not in trait_data:
                continue
            counts = trait_data[a_key]["counts"]
            target = counts.get(trait, 0)
            total = trait_data[a_key]["total_yes"]
            alpha_labels.append(f"a={alpha:+.0f}")
            target_counts.append(target)
            total_counts.append(total)

        x = np.arange(len(alpha_labels))
        width = 0.35

        ax.bar(x - width/2, target_counts, width, label=f"{trait} YES", color="#1f77b4")
        ax.bar(x + width/2, total_counts, width, label="Total YES", color="#aec7e8")
        ax.set_xticks(x)
        ax.set_xticklabels(alpha_labels)
        ax.set_title(trait.capitalize())
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    plt.suptitle("RIASEC Steering Effectiveness", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved steering effectiveness: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--pca_dir", type=str, default="./data/assistant_axis/pca/")
    parser.add_argument("--comparison_dir", type=str, default="./outputs/analysis/")
    parser.add_argument("--eval_dir", type=str, default="./outputs/riasec_eval/")
    parser.add_argument("--output_dir", type=str, default="./outputs/figures/")
    args = parser.parse_args()

    safe_model = args.model_id.replace("/", "__")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scree plot
    pca_path = Path(args.pca_dir) / f"{safe_model}_pca.pt"
    if pca_path.exists():
        pca_data = torch.load(pca_path, weights_only=False)
        explained_variance = pca_data["explained_variance_ratio"].numpy()
        plot_scree(explained_variance, output_dir / f"{safe_model}_scree.png")

        # 2. Role scatter
        projections = pca_data["projections"].numpy()
        role_names = pca_data["role_names"]
        role_categories = pca_data["role_categories"]
        plot_role_scatter(projections, role_names, role_categories, output_dir / f"{safe_model}_role_scatter.png")
    else:
        logger.warning("PCA data not found at %s, skipping scree/scatter plots.", pca_path)

    # 3. Cosine heatmap
    comparison_path = Path(args.comparison_dir) / f"{safe_model}_comparison_arrays.npz"
    if comparison_path.exists():
        arrays = np.load(comparison_path, allow_pickle=True)
        cosine_matrix = arrays["cosine_riasec_vs_pca"]
        trait_names = arrays["trait_names"].tolist()
        n_pca = cosine_matrix.shape[1]
        pca_labels = [f"PC{i+1}" for i in range(n_pca)]
        plot_cosine_heatmap(cosine_matrix, trait_names, pca_labels, output_dir / f"{safe_model}_cosine_heatmap.png")
    else:
        logger.warning("Comparison data not found at %s, skipping heatmap.", comparison_path)

    # 4. Steering effectiveness
    eval_path = Path(args.eval_dir) / f"{safe_model}_riasec_eval.json"
    if eval_path.exists():
        plot_steering_effectiveness(eval_path, output_dir / f"{safe_model}_steering_effectiveness.png")
    else:
        logger.warning("Eval data not found at %s, skipping steering plot.", eval_path)

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
