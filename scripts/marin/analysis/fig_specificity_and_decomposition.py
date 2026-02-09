#!/usr/bin/env python
"""
Key diagnostic figures for the specificity finding.

Fig 6: Cross-trait specificity matrix (the negative result)
Fig 7: Vector decomposition and residual-assistant axis alignment

Usage:
  uv run python scripts/marin/analysis/fig_specificity_and_decomposition.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open
from sklearn.decomposition import PCA

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.3,
})

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
TRAIT_COLORS = {
    "realistic":     "#D55E00",
    "investigative": "#0072B2",
    "artistic":      "#009E73",
    "social":        "#CC79A7",
    "enterprising":  "#E69F00",
    "conventional":  "#56B4E9",
}

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def load_riasec_vectors(model_id):
    safe_model = model_id.replace("/", "__")
    vectors = {}
    for trait in TRAITS:
        path = Path(f"persona_data/model_inits/{trait}_persona_initialization/{safe_model}.safetensors")
        with safe_open(str(path), framework="pt") as f:
            vectors[trait] = f.get_tensor("response_persona_vector").numpy().flatten()
    return vectors


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Specificity Matrix + Random Baseline
# ══════════════════════════════════════════════════════════════════════════════

def fig6_specificity():
    spec_path = Path("outputs/specificity/meta-llama__Llama-3.2-1B-Instruct_cross_trait_specificity.json")
    if not spec_path.exists():
        print("Skipping fig6: no specificity data")
        return

    with open(spec_path) as f:
        data = json.load(f)

    fig, (ax_mat, ax_bar) = plt.subplots(1, 2, figsize=(6.5, 3.0),
                                          gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(wspace=0.4, left=0.1, right=0.95, top=0.88, bottom=0.15)

    # Panel (a): Specificity matrix at alpha=5
    alpha_str = "5"
    matrix = np.zeros((6, 6))
    for i, steer_t in enumerate(TRAITS):
        for j, eval_t in enumerate(TRAITS):
            matrix[i, j] = data["specificity_matrix"][steer_t][eval_t][alpha_str]["mean_gap"]

    im = ax_mat.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=7, aspect="equal")
    ax_mat.set_xticks(range(6))
    ax_mat.set_xticklabels([t.capitalize()[:4] + "." for t in TRAITS], rotation=45, ha="right", fontsize=6)
    ax_mat.set_yticks(range(6))
    ax_mat.set_yticklabels([t.capitalize()[:4] + "." for t in TRAITS], fontsize=6)
    ax_mat.set_xlabel("Evaluated on", fontsize=7)
    ax_mat.set_ylabel("Steered with", fontsize=7)
    ax_mat.tick_params(length=0)

    # Annotate values, highlight diagonal
    for i in range(6):
        for j in range(6):
            val = matrix[i, j]
            weight = "bold" if i == j else "normal"
            color = "white" if val > 5 else "black"
            ax_mat.text(j, i, f"{val:.1f}", ha="center", va="center",
                       fontsize=5.5, color=color, fontweight=weight)

    # Draw diagonal boxes
    for i in range(6):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                             edgecolor="black", linewidth=1.5)
        ax_mat.add_patch(rect)

    cb = fig.colorbar(im, ax=ax_mat, shrink=0.85, pad=0.02)
    cb.set_label("Logprob gap", fontsize=6)
    cb.ax.tick_params(labelsize=5, length=2, width=0.3)
    cb.outline.set_linewidth(0.3)
    ax_mat.set_title(r"Cross-trait specificity ($\alpha=5$)", fontsize=8, pad=4)
    panel_label(ax_mat, "a", x=-0.18, y=1.12)

    # Panel (b): Comparison bar chart - diagonal vs off-diagonal vs random
    spec_idx = data.get("specificity_index_alpha_5", {})
    diag = spec_idx.get("diagonal_mean", 0)
    off_diag = spec_idx.get("off_diagonal_mean", 0)
    random_mean = np.mean([data["random_baseline"]["5"][t]["mean_gap"] for t in TRAITS]) if "5" in data.get("random_baseline", {}) else 0

    bars = [diag, off_diag, random_mean]
    labels = ["On-diagonal\n(matching trait)", "Off-diagonal\n(other traits)", "Random\nbaseline"]
    colors = ["#0072B2", "#D55E00", "#888888"]

    x_pos = np.arange(3)
    ax_bar.bar(x_pos, bars, color=colors, alpha=0.8, width=0.6,
               edgecolor="white", linewidth=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=6)
    ax_bar.set_ylabel("Mean logprob gap")
    ax_bar.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.4)
    ax_bar.set_title("No trait specificity", fontsize=8, pad=4)

    # Add values on bars
    for i, v in enumerate(bars):
        ax_bar.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=7)

    panel_label(ax_bar, "b", x=-0.18, y=1.12)

    fig.savefig(OUTPUT_DIR / "fig6_specificity.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig6_specificity.png", dpi=300)
    plt.close(fig)
    print("Saved fig6_specificity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Decomposition + Residual-Assistant Axis Alignment
# ══════════════════════════════════════════════════════════════════════════════

def fig7_decomposition():
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2],
                          wspace=0.4, left=0.07, right=0.96, top=0.88, bottom=0.15)

    ax_pie = fig.add_subplot(gs[0, 0])
    ax_cos = fig.add_subplot(gs[0, 1])
    ax_align = fig.add_subplot(gs[0, 2])

    # ── Panel (a): Shared vs residual variance fraction ──
    models = {
        "Llama 1B": "meta-llama/Llama-3.2-1B-Instruct",
        "Marin 8B": "marin-community/marin-8b-instruct",
    }

    x = np.arange(len(TRAITS))
    width = 0.35
    for idx, (label, model_id) in enumerate(models.items()):
        vectors = load_riasec_vectors(model_id)
        V = np.stack([vectors[t] for t in TRAITS])
        mean_vec = V.mean(axis=0)
        mean_unit = mean_vec / np.linalg.norm(mean_vec)

        shared_fracs = []
        for trait in TRAITS:
            v = vectors[trait]
            proj_scalar = np.dot(v, mean_unit)
            shared_frac = proj_scalar**2 / (np.linalg.norm(v)**2 + 1e-10)
            shared_fracs.append(shared_frac)

        offset = -width/2 + idx * width
        color = "#0072B2" if idx == 0 else "#D55E00"
        ax_pie.bar(x + offset, shared_fracs, width=width, label=label,
                   color=color, alpha=0.8, edgecolor="white", linewidth=0.3)

    ax_pie.set_xticks(x)
    ax_pie.set_xticklabels([t.capitalize()[:4] + "." for t in TRAITS], rotation=45, ha="right", fontsize=6)
    ax_pie.set_ylabel("Shared direction fraction")
    ax_pie.set_ylim(0, 1)
    ax_pie.axhline(0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    ax_pie.legend(fontsize=6, loc="lower right")
    ax_pie.set_title("Shared direction dominance", fontsize=8, pad=4)
    panel_label(ax_pie, "a", x=-0.18, y=1.12)

    # ── Panel (b): Original vs residual cosine matrices side by side ──
    # Show as bar chart of off-diagonal means
    model_id = "marin-community/marin-8b-instruct"
    vectors = load_riasec_vectors(model_id)
    V = np.stack([vectors[t] for t in TRAITS])
    mean_vec = V.mean(axis=0)
    mean_unit = mean_vec / np.linalg.norm(mean_vec)

    residuals = {}
    for trait in TRAITS:
        proj = np.dot(vectors[trait], mean_unit) * mean_unit
        residuals[trait] = vectors[trait] - proj

    # Compute off-diagonal cosines for both
    orig_pairs = []
    resid_pairs = []
    labels_pairs = []
    for i in range(len(TRAITS)):
        for j in range(i+1, len(TRAITS)):
            orig_pairs.append(cosine_sim(vectors[TRAITS[i]], vectors[TRAITS[j]]))
            resid_pairs.append(cosine_sim(residuals[TRAITS[i]], residuals[TRAITS[j]]))
            labels_pairs.append(f"{TRAITS[i][:3]}-{TRAITS[j][:3]}")

    ax_cos.scatter(orig_pairs, resid_pairs, c="#0072B2", s=20, alpha=0.8,
                   edgecolors="white", linewidths=0.3, zorder=5)
    for k, lbl in enumerate(labels_pairs):
        ax_cos.annotate(lbl, (orig_pairs[k], resid_pairs[k]),
                       fontsize=4, alpha=0.5, xytext=(2, 2),
                       textcoords="offset points")

    ax_cos.axhline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.4)
    ax_cos.axvline(np.mean(orig_pairs), color="#D55E00", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_cos.axhline(np.mean(resid_pairs), color="#0072B2", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_cos.set_xlabel("Original cosine sim.")
    ax_cos.set_ylabel("Residual cosine sim.")
    ax_cos.set_title("Shared direction removal", fontsize=8, pad=4)
    panel_label(ax_cos, "b", x=-0.18, y=1.12)

    # ── Panel (c): Residual alignment with assistant axis ──
    for model_id, marker, color, label in [
        ("meta-llama/Llama-3.2-1B-Instruct", "o", "#0072B2", "Llama 1B"),
        ("marin-community/marin-8b-instruct", "s", "#D55E00", "Marin 8B"),
    ]:
        safe_model = model_id.replace("/", "__")
        vectors = load_riasec_vectors(model_id)
        V = np.stack([vectors[t] for t in TRAITS])
        mean_vec = V.mean(axis=0)
        mean_unit = mean_vec / np.linalg.norm(mean_vec)

        pca_path = Path(f"data/assistant_axis/pca/{safe_model}_pca.pt")
        if not pca_path.exists():
            continue
        pca_data = torch.load(pca_path, weights_only=False)
        pc1 = pca_data["components"].numpy()[0]

        # Check if PC1 needs flipping (default_assistant should have positive projection)
        role_names = pca_data["role_names"]
        projections = pca_data["projections"].numpy()
        da_idx = role_names.index("default_assistant") if "default_assistant" in role_names else None
        if da_idx is not None and projections[da_idx, 0] < 0:
            pc1 = -pc1  # flip so positive = assistant-like

        alignments = []
        for trait in TRAITS:
            proj = np.dot(vectors[trait], mean_unit) * mean_unit
            residual = vectors[trait] - proj
            cos = cosine_sim(residual, pc1)
            alignments.append(cos)

        y = np.arange(len(TRAITS))
        offset = -0.15 if "Llama" in label else 0.15
        ax_align.barh(y + offset, alignments, height=0.3, color=color, alpha=0.8,
                      edgecolor="white", linewidth=0.3, label=label)

    ax_align.set_yticks(range(len(TRAITS)))
    ax_align.set_yticklabels([t.capitalize() for t in TRAITS], fontsize=7)
    ax_align.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax_align.set_xlabel("cos(residual, assistant axis)")
    ax_align.set_title("Residual ↔ assistant axis", fontsize=8, pad=4)
    ax_align.legend(fontsize=6, loc="lower right")

    # Add interpretive arrows
    ax_align.annotate("← character-like", xy=(-0.3, -0.8), fontsize=5, alpha=0.5,
                      annotation_clip=False)
    ax_align.annotate("assistant-like →", xy=(0.15, -0.8), fontsize=5, alpha=0.5,
                      annotation_clip=False)

    ax_align.invert_yaxis()
    panel_label(ax_align, "c", x=-0.22, y=1.12)

    fig.savefig(OUTPUT_DIR / "fig7_decomposition.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig7_decomposition.png", dpi=300)
    plt.close(fig)
    print("Saved fig7_decomposition")


if __name__ == "__main__":
    fig6_specificity()
    fig7_decomposition()
