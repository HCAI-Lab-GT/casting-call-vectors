#!/usr/bin/env python
"""
Generate clean, publication-quality figures.
300 DPI, ~5x5 inches, no bold, standard colors, clean labels.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open
from scipy import stats
from sklearn.decomposition import PCA

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
})

DPI = 300
riasec_dir = Path("persona_data/model_inits")
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Standard color palette (colorblind-friendly)
TRAIT_COLORS = {
    "realistic": "#D55E00",
    "investigative": "#0072B2",
    "artistic": "#009E73",
    "social": "#CC79A7",
    "enterprising": "#E69F00",
    "conventional": "#56B4E9",
}

CATEGORY_COLORS = {
    "scientist": "#0072B2",
    "artist": "#D55E00",
    "assistant": "#009E73",
}

hex_order = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
traits = sorted(hex_order)


def load_riasec_vectors(model_id):
    safe_model = model_id.replace("/", "__")
    vectors = {}
    for trait in traits:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            vectors[trait] = f.get_tensor("response_persona_vector").numpy().flatten()
    return vectors


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_cosine_matrix(vectors, trait_list):
    n = len(trait_list)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_sim(vectors[trait_list[i]], vectors[trait_list[j]])
    return matrix


# ========== Figure 1: Cross-model RIASEC cosine heatmaps ==========
def fig1_cross_model_heatmaps():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for ax, model_id, title in [
        (ax1, "meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
        (ax2, "marin-community/marin-8b-instruct", "Marin-8B"),
    ]:
        vectors = load_riasec_vectors(model_id)
        matrix = compute_cosine_matrix(vectors, traits)

        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0.3, vmax=1.0, aspect="equal")
        labels = [t.capitalize()[:4] for t in traits]
        ax.set_xticks(range(len(traits)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(traits)))
        ax.set_yticklabels(labels)
        for i in range(len(traits)):
            for j in range(len(traits)):
                val = matrix[i, j]
                color = "white" if val > 0.65 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color)
        ax.set_title(title)

    fig.colorbar(im, ax=[ax1, ax2], label="Cosine similarity", shrink=0.8, pad=0.02)
    fig.savefig(output_dir / "fig1_cross_model_heatmaps.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1_cross_model_heatmaps.png")


# ========== Figure 2: Cross-model correlation scatter ==========
def fig2_cross_model_scatter():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    llama_vecs = load_riasec_vectors("meta-llama/Llama-3.2-1B-Instruct")
    marin_vecs = load_riasec_vectors("marin-community/marin-8b-instruct")

    llama_upper, marin_upper, pair_labels = [], [], []
    for i in range(len(traits)):
        for j in range(i + 1, len(traits)):
            llama_upper.append(cosine_sim(llama_vecs[traits[i]], llama_vecs[traits[j]]))
            marin_upper.append(cosine_sim(marin_vecs[traits[i]], marin_vecs[traits[j]]))
            pair_labels.append(f"{traits[i][:3]}-{traits[j][:3]}")

    llama_upper = np.array(llama_upper)
    marin_upper = np.array(marin_upper)
    r, p = stats.pearsonr(llama_upper, marin_upper)

    ax.scatter(llama_upper, marin_upper, c="#0072B2", s=25, alpha=0.8, edgecolors="white", linewidths=0.3)
    for k, label in enumerate(pair_labels):
        ax.annotate(label, (llama_upper[k], marin_upper[k]),
                   fontsize=4.5, alpha=0.6, xytext=(2, 2), textcoords="offset points")

    slope, intercept = np.polyfit(llama_upper, marin_upper, 1)
    x_line = np.linspace(0.35, 0.85, 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="#D55E00", alpha=0.6,
            linewidth=0.8, label=f"r = {r:.3f}")

    lim = [0.35, 0.85]
    ax.plot(lim, lim, ":", color="gray", alpha=0.4, linewidth=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Llama-3.2-1B cosine similarity")
    ax.set_ylabel("Marin-8B cosine similarity")
    ax.set_title("Cross-model RIASEC geometry")
    ax.legend(frameon=False)
    ax.set_aspect("equal")

    fig.savefig(output_dir / "fig2_cross_model_scatter.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2_cross_model_scatter.png")


# ========== Figure 3: PCA role scatter ==========
def fig3_pca_role_scatter():
    pca_path = Path("data/assistant_axis/pca/meta-llama__Llama-3.2-1B-Instruct_pca.pt")
    if not pca_path.exists():
        print("Skipping fig3: no PCA data")
        return

    pca_data = torch.load(pca_path, weights_only=False)
    projections = pca_data["projections"].numpy()
    role_names = pca_data["role_names"]
    role_categories = pca_data["role_categories"]
    ev = pca_data["explained_variance_ratio"].numpy()

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    categories = set(role_categories.values())
    for cat in sorted(categories):
        indices = [i for i, name in enumerate(role_names) if role_categories.get(name) == cat]
        if not indices:
            continue
        color = CATEGORY_COLORS.get(cat, "#666666")
        ax.scatter(projections[indices, 0], projections[indices, 1],
                  c=color, label=cat.capitalize(), s=30, alpha=0.85,
                  edgecolors="white", linewidths=0.3)
        for idx in indices:
            name = role_names[idx].replace("_", " ")
            ax.annotate(name, (projections[idx, 0], projections[idx, 1]),
                       fontsize=5, alpha=0.65, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("Role projections (Llama-3.2-1B)")
    ax.legend(frameon=False, markerscale=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.15)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.15)

    fig.savefig(output_dir / "fig3_pca_role_scatter.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3_pca_role_scatter.png")


# ========== Figure 4: PCA scree plot ==========
def fig4_pca_scree():
    pca_path = Path("data/assistant_axis/pca/meta-llama__Llama-3.2-1B-Instruct_pca.pt")
    if not pca_path.exists():
        print("Skipping fig4: no PCA data")
        return

    pca_data = torch.load(pca_path, weights_only=False)
    ev = pca_data["explained_variance_ratio"].numpy()
    n_comp = len(ev)
    x = np.arange(1, n_comp + 1)

    fig, ax1 = plt.subplots(figsize=(3.5, 3))

    ax1.bar(x, ev * 100, alpha=0.7, color="#0072B2", width=0.7)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title("PCA scree plot (Llama-3.2-1B)")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    cumulative = np.cumsum(ev) * 100
    ax2.plot(x, cumulative, "o-", color="#D55E00", markersize=3, linewidth=0.8)
    ax2.set_ylabel("Cumulative (%)")
    ax2.set_ylim(0, 105)
    for threshold in [50, 70, 90]:
        ax2.axhline(y=threshold, color="gray", linestyle="--", alpha=0.2, linewidth=0.5)

    fig.savefig(output_dir / "fig4_pca_scree.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig4_pca_scree.png")


# ========== Figure 5: RIASEC vs PCA components heatmap ==========
def fig5_riasec_pca_heatmap():
    comparison_path = Path("outputs/analysis/meta-llama__Llama-3.2-1B-Instruct_comparison.json")
    if not comparison_path.exists():
        print("Skipping fig5: no comparison data")
        return

    with open(comparison_path) as f:
        comparison = json.load(f)

    cosine_data = comparison["cosine_riasec_vs_pca"]
    trait_names = comparison["trait_names"]
    n_pca_show = min(6, len(comparison["pca_component_labels"]))
    pca_labels = comparison["pca_component_labels"][:n_pca_show]

    matrix = np.zeros((len(trait_names), n_pca_show))
    for i, trait in enumerate(trait_names):
        for j, pc in enumerate(pca_labels):
            matrix[i, j] = cosine_data[trait][pc]

    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(n_pca_show))
    ax.set_xticklabels(pca_labels)
    ax.set_yticks(range(len(trait_names)))
    ax.set_yticklabels([t.capitalize() for t in trait_names])
    for i in range(len(trait_names)):
        for j in range(n_pca_show):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color)
    plt.colorbar(im, ax=ax, label="Cosine sim.", shrink=0.85)
    ax.set_title("RIASEC vs PCA components")

    fig.savefig(output_dir / "fig5_riasec_pca_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig5_riasec_pca_heatmap.png")


# ========== Figure 6: RIASEC vector PCA (both models) ==========
def fig6_riasec_pca():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3))

    for ax, model_id, title in [
        (ax1, "meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
        (ax2, "marin-community/marin-8b-instruct", "Marin-8B"),
    ]:
        vectors = load_riasec_vectors(model_id)
        matrix = np.stack([vectors[t] for t in hex_order])
        pca = PCA(n_components=min(6, matrix.shape[0]))
        proj = pca.fit_transform(matrix - matrix.mean(axis=0))
        ev = pca.explained_variance_ratio_

        for i, trait in enumerate(hex_order):
            ax.scatter(proj[i, 0], proj[i, 1], c=TRAIT_COLORS[trait], s=50,
                      edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(trait.capitalize(), (proj[i, 0], proj[i, 1]),
                       fontsize=6, xytext=(5, 4), textcoords="offset points")

        # Hexagonal edges
        for i in range(6):
            j = (i + 1) % 6
            ax.plot([proj[i, 0], proj[j, 0]], [proj[i, 1], proj[j, 1]],
                   "-", color="gray", alpha=0.3, linewidth=0.6)

        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title(title)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.15)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.15)

    fig.suptitle("RIASEC personality vectors in PCA space", fontsize=10)
    fig.savefig(output_dir / "fig6_riasec_pca.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig6_riasec_pca.png")


# ========== Figure 7: Hexagonal distance vs cosine ==========
def fig7_hexagonal_distance():
    fig, ax = plt.subplots(figsize=(3.5, 3))

    for model_id, marker, color, label in [
        ("meta-llama/Llama-3.2-1B-Instruct", "o", "#0072B2", "Llama 1B"),
        ("marin-community/marin-8b-instruct", "s", "#D55E00", "Marin 8B"),
    ]:
        vectors = load_riasec_vectors(model_id)
        for i in range(6):
            for j in range(i + 1, 6):
                dist = min(abs(i - j), 6 - abs(i - j))
                cos = cosine_sim(vectors[hex_order[i]], vectors[hex_order[j]])
                jitter = np.random.uniform(-0.08, 0.08)
                ax.scatter(dist + jitter, cos, marker=marker, c=color, s=20, alpha=0.7,
                          edgecolors="white", linewidths=0.2)

        # Mean bars
        for d in [1, 2, 3]:
            vals = []
            for i in range(6):
                for j in range(i + 1, 6):
                    if min(abs(i - j), 6 - abs(i - j)) == d:
                        vals.append(cosine_sim(vectors[hex_order[i]], vectors[hex_order[j]]))
            offset = -0.12 if "Llama" in label else 0.12
            ax.plot([d + offset - 0.08, d + offset + 0.08],
                   [np.mean(vals), np.mean(vals)], color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Hexagonal distance")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Holland hexagonal structure")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Adjacent", "Alternate", "Opposite"])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", markersize=5, label="Llama 1B"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#D55E00", markersize=5, label="Marin 8B"),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    fig.savefig(output_dir / "fig7_hexagonal_distance.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig7_hexagonal_distance.png")


# ========== Figure 8: PC1 projection bar chart ==========
def fig8_pc1_rankings():
    pca_path = Path("data/assistant_axis/pca/meta-llama__Llama-3.2-1B-Instruct_pca.pt")
    if not pca_path.exists():
        print("Skipping fig8: no PCA data")
        return

    pca_data = torch.load(pca_path, weights_only=False)
    role_names = pca_data["role_names"]
    role_categories = pca_data["role_categories"]
    projections = pca_data["projections"].numpy()
    pc1 = projections[:, 0]

    sorted_indices = np.argsort(pc1)
    sorted_names = [role_names[i].replace("_", " ") for i in sorted_indices]
    sorted_pc1 = pc1[sorted_indices]
    sorted_colors = [CATEGORY_COLORS.get(role_categories.get(role_names[i], ""), "#666")
                    for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(3.5, 4))
    ax.barh(range(len(sorted_names)), sorted_pc1, color=sorted_colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=6.5)
    ax.set_xlabel("PC1 projection")
    ax.set_title("Role rankings along assistant axis")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.25, linewidth=0.5)

    fig.savefig(output_dir / "fig8_pc1_rankings.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig8_pc1_rankings.png")


if __name__ == "__main__":
    fig1_cross_model_heatmaps()
    fig2_cross_model_scatter()
    fig3_pca_role_scatter()
    fig4_pca_scree()
    fig5_riasec_pca_heatmap()
    fig6_riasec_pca()
    fig7_hexagonal_distance()
    fig8_pc1_rankings()
    print(f"\nAll figures saved to: {output_dir}")
