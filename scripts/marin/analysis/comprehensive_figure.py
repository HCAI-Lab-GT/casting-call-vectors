#!/usr/bin/env python
"""
Generate comprehensive multi-panel figure summarizing all findings.

Produces a publication-quality figure with:
  Panel A: PCA role scatter (PC1 vs PC2) for Llama 1B
  Panel B: RIASEC inter-trait cosine heatmap (cross-model)
  Panel C: RIASEC vs PCA cosine alignment heatmap
  Panel D: PCA scree plot
  Panel E: PC1 projections bar chart (roles sorted by PC1)
  Panel F: Cross-model RIASEC geometry correlation

Usage:
  python scripts/marin/analysis/comprehensive_figure.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from safetensors import safe_open
from scipy import stats

from pvx import setup_logging
from pvx.utils.riasec_utils import RIASECHelpers

matplotlib.use("Agg")

logger = setup_logging(name="marin-comprehensive-figure")

CLEAN_DPI = 300

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

TRAIT_COLORS = {
    "realistic": "#e41a1c",
    "investigative": "#377eb8",
    "artistic": "#4daf4a",
    "social": "#984ea3",
    "enterprising": "#ff7f00",
    "conventional": "#a65628",
}


def load_riasec_cosine_matrix(model_id: str, vectors_dir: str):
    """Load RIASEC vectors and compute cosine similarity matrix."""
    safe_model = model_id.replace("/", "__")
    traits = sorted(RIASECHelpers.RIASEC_TRAITS)
    vectors = {}

    for trait in traits:
        path = Path(vectors_dir) / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        if not path.exists():
            return None, None
        with safe_open(str(path), framework="pt") as f:
            vec = f.get_tensor("response_persona_vector").numpy().flatten()
            vectors[trait] = vec

    n = len(traits)
    matrix = np.zeros((n, n))
    for i, t1 in enumerate(traits):
        for j, t2 in enumerate(traits):
            dot = np.dot(vectors[t1], vectors[t2])
            norm1 = np.linalg.norm(vectors[t1])
            norm2 = np.linalg.norm(vectors[t2])
            matrix[i, j] = dot / (norm1 * norm2 + 1e-10)

    return matrix, traits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--riasec_dir", default="./persona_data/model_inits/")
    parser.add_argument("--pca_dir", default="./data/assistant_axis/pca/")
    parser.add_argument("--comparison_dir", default="./outputs/analysis/")
    parser.add_argument("--output_dir", default="./outputs/figures/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    llama_id = "meta-llama/Llama-3.2-1B-Instruct"
    marin_id = "marin-community/marin-8b-instruct"
    llama_safe = llama_id.replace("/", "__")
    _ = marin_id.replace("/", "__")

    # PCA data (Llama 1B)
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    pca_path = Path(args.pca_dir) / f"{llama_safe}_pca.pt"
    pca_data = torch.load(pca_path, weights_only=False) if pca_path.exists() else None

    # RIASEC cosine matrices for both models
    llama_cosine, traits = load_riasec_cosine_matrix(llama_id, args.riasec_dir)
    marin_cosine, _ = load_riasec_cosine_matrix(marin_id, args.riasec_dir)

    # Comparison data (Llama 1B)
    comparison_path = Path(args.comparison_dir) / f"{llama_safe}_comparison.json"
    comparison = None
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)

    # ============================================================
    # Create comprehensive figure
    # ============================================================
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # Panel A: PCA Role Scatter
    if pca_data is not None:
        ax_a = fig.add_subplot(gs[0, 0])
        projections = pca_data["projections"].numpy()
        role_names = pca_data["role_names"]
        role_categories = pca_data["role_categories"]
        ev = pca_data["explained_variance_ratio"].numpy()

        categories = set(role_categories.values())
        for cat in sorted(categories):
            indices = [i for i, name in enumerate(role_names) if role_categories.get(name) == cat]
            if not indices:
                continue
            color = CATEGORY_COLORS.get(cat, "#333333")
            ax_a.scatter(
                projections[indices, 0],
                projections[indices, 1],
                c=color,
                label=cat,
                s=60,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
            )
            for idx in indices:
                name = role_names[idx].replace("_", " ")
                ax_a.annotate(
                    name,
                    (projections[idx, 0], projections[idx, 1]),
                    fontsize=5.5,
                    alpha=0.7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

        ax_a.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
        ax_a.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
        ax_a.set_title("A. Role Projections (Llama 1B)", fontsize=12, fontweight="bold")
        ax_a.legend(fontsize=7, loc="lower left")
        ax_a.axhline(y=0, color="gray", linestyle="-", alpha=0.2)
        ax_a.axvline(x=0, color="gray", linestyle="-", alpha=0.2)

    # Panel B: Cross-model RIASEC cosine comparison
    if llama_cosine is not None and marin_cosine is not None:
        ax_b = fig.add_subplot(gs[0, 1])

        # Extract upper triangular values
        n = len(traits)
        llama_upper = []
        marin_upper = []
        pair_labels = []
        for i in range(n):
            for j in range(i + 1, n):
                llama_upper.append(llama_cosine[i, j])
                marin_upper.append(marin_cosine[i, j])
                pair_labels.append(f"{traits[i][:3]}-{traits[j][:3]}")

        llama_upper = np.array(llama_upper)
        marin_upper = np.array(marin_upper)

        ax_b.scatter(llama_upper, marin_upper, c="#1f77b4", s=40, alpha=0.8, edgecolors="white")
        for k, label in enumerate(pair_labels):
            ax_b.annotate(
                label,
                (llama_upper[k], marin_upper[k]),
                fontsize=5,
                alpha=0.6,
                xytext=(2, 2),
                textcoords="offset points",
            )

        # Fit line and compute correlation
        r, p = stats.pearsonr(llama_upper, marin_upper)
        slope, intercept = np.polyfit(llama_upper, marin_upper, 1)
        x_line = np.linspace(min(llama_upper) - 0.05, max(llama_upper) + 0.05, 100)
        ax_b.plot(
            x_line, slope * x_line + intercept, "r--", alpha=0.5, label=f"r={r:.3f}, p={p:.2e}"
        )

        # Identity line
        lim = [
            min(min(llama_upper), min(marin_upper)) - 0.05,
            max(max(llama_upper), max(marin_upper)) + 0.05,
        ]
        ax_b.plot(lim, lim, "k:", alpha=0.3, label="identity")
        ax_b.set_xlim(lim)
        ax_b.set_ylim(lim)

        ax_b.set_xlabel("Llama 1B Cosine Similarity")
        ax_b.set_ylabel("Marin 8B Cosine Similarity")
        ax_b.set_title("B. Cross-Model RIASEC Geometry", fontsize=12, fontweight="bold")
        ax_b.legend(fontsize=7)

    # Panel C: RIASEC vs PCA components heatmap
    if comparison is not None:
        ax_c = fig.add_subplot(gs[0, 2])
        cosine_data = comparison["cosine_riasec_vs_pca"]
        trait_names = comparison["trait_names"]
        n_pca_show = min(6, len(comparison["pca_component_labels"]))
        pca_labels = comparison["pca_component_labels"][:n_pca_show]

        matrix = np.zeros((len(trait_names), n_pca_show))
        for i, trait in enumerate(trait_names):
            for j, pc in enumerate(pca_labels):
                matrix[i, j] = cosine_data[trait][pc]

        im = ax_c.imshow(matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
        ax_c.set_xticks(range(n_pca_show))
        ax_c.set_xticklabels(pca_labels, fontsize=8)
        ax_c.set_yticks(range(len(trait_names)))
        ax_c.set_yticklabels([t.capitalize() for t in trait_names], fontsize=8)
        for i in range(len(trait_names)):
            for j in range(n_pca_show):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.15 else "black"
                ax_c.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
        plt.colorbar(im, ax=ax_c, label="Cosine Sim", shrink=0.8)
        ax_c.set_title("C. RIASEC vs PCA Components", fontsize=12, fontweight="bold")

    # Panel D: PCA Scree Plot
    if pca_data is not None:
        ax_d = fig.add_subplot(gs[1, 0])
        ev = pca_data["explained_variance_ratio"].numpy()
        n_comp = len(ev)
        x = np.arange(1, n_comp + 1)

        ax_d.bar(x, ev * 100, alpha=0.7, color="#1f77b4")
        ax_d.set_xlabel("Principal Component")
        ax_d.set_ylabel("Explained Variance (%)")
        ax_d.set_title("D. PCA Scree Plot (Llama 1B)", fontsize=12, fontweight="bold")

        ax_d2 = ax_d.twinx()
        cumulative = np.cumsum(ev) * 100
        ax_d2.plot(x, cumulative, "r-o", markersize=3)
        ax_d2.set_ylabel("Cumulative (%)")
        ax_d2.set_ylim(0, 105)
        for threshold in [50, 70, 90]:
            ax_d2.axhline(y=threshold, color="gray", linestyle="--", alpha=0.3)

    # Panel E: PC1 Projections sorted bar chart
    if pca_data is not None:
        ax_e = fig.add_subplot(gs[1, 1])
        role_names = pca_data["role_names"]
        role_categories = pca_data["role_categories"]
        projections = pca_data["projections"].numpy()
        pc1 = projections[:, 0]

        # Sort by PC1 projection
        sorted_indices = np.argsort(pc1)
        sorted_names = [role_names[i].replace("_", " ") for i in sorted_indices]
        sorted_pc1 = pc1[sorted_indices]
        sorted_colors = [
            CATEGORY_COLORS.get(role_categories.get(role_names[i], ""), "#333")
            for i in sorted_indices
        ]

        ax_e.barh(range(len(sorted_names)), sorted_pc1, color=sorted_colors, alpha=0.8)
        ax_e.set_yticks(range(len(sorted_names)))
        ax_e.set_yticklabels(sorted_names, fontsize=7)
        ax_e.set_xlabel("PC1 Projection")
        ax_e.set_title("E. Role PC1 Rankings", fontsize=12, fontweight="bold")
        ax_e.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    # Panel F: RIASEC cosine heatmap (Marin 8B)
    if marin_cosine is not None:
        ax_f = fig.add_subplot(gs[1, 2])
        im = ax_f.imshow(marin_cosine, cmap="RdYlBu_r", vmin=0.2, vmax=1.0, aspect="equal")
        trait_labels = [t.capitalize() for t in traits]
        ax_f.set_xticks(range(len(traits)))
        ax_f.set_xticklabels(trait_labels, rotation=45, ha="right", fontsize=8)
        ax_f.set_yticks(range(len(traits)))
        ax_f.set_yticklabels(trait_labels, fontsize=8)
        for i in range(len(traits)):
            for j in range(len(traits)):
                val = marin_cosine[i, j]
                color = "white" if val > 0.7 else "black"
                ax_f.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax_f, label="Cosine Sim", shrink=0.8)
        ax_f.set_title("F. RIASEC Cosine (Marin 8B)", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Personality Vector Geometry: Cross-Model Analysis\n"
        "Llama-3.2-1B-Instruct vs Marin-8B-Instruct",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    output_path = output_dir / "comprehensive_analysis.png"
    fig.savefig(output_path, dpi=CLEAN_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved comprehensive figure to: {output_path}")

    # ============================================================
    # Also create RIASEC hexagon plot
    # ============================================================
    if llama_cosine is not None and marin_cosine is not None:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "polar"})

        angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False)

        for ax, cosine_mat, title in [
            (ax1, llama_cosine, "Llama 1B"),
            (ax2, marin_cosine, "Marin 8B"),
        ]:
            # Plot mean cosine similarity profile for each trait
            for i, trait in enumerate(traits):
                # Mean similarity to all other traits
                sims = [cosine_mat[i, j] for j in range(len(traits)) if i != j]
                mean_sim = np.mean(sims)
                color = TRAIT_COLORS.get(trait, "#333")
                ax.bar(
                    angles[i], mean_sim, width=0.8, alpha=0.7, color=color, label=trait.capitalize()
                )

            ax.set_xticks(angles)
            ax.set_xticklabels([t[:3].upper() for t in traits], fontsize=9)
            ax.set_ylim(0, 0.8)
            ax.set_title(title, pad=20, fontsize=12, fontweight="bold")

        ax1.legend(bbox_to_anchor=(-0.15, 0.5), loc="center right", fontsize=7)
        fig2.suptitle("RIASEC Mean Inter-Trait Cosine Similarity", fontsize=13, fontweight="bold")
        plt.tight_layout()

        hexagon_path = output_dir / "riasec_hexagon_comparison.png"
        fig2.savefig(hexagon_path, dpi=CLEAN_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"Saved RIASEC hexagon to: {hexagon_path}")


if __name__ == "__main__":
    main()
