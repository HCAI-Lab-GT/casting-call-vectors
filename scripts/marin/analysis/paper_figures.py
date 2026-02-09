#!/usr/bin/env python
"""
Publication-quality figures for the personality vectors paper.

Generates 5 main figures:
  Fig 1: Cross-model RIASEC geometry (heatmaps + correlation)
  Fig 2: Steering effectiveness (logprob gap vs alpha)
  Fig 3: Persona space PCA structure (scatter, scree, rankings)
  Fig 4: RIASEC hexagonal structure in PCA space
  Fig 5: RIASEC-persona relationship (heatmap + hex distance)

Usage:
  uv run python scripts/marin/analysis/paper_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from safetensors import safe_open
from scipy import stats
from sklearn.decomposition import PCA

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
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
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.4,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
})

DPI = 300
RIASEC_DIR = Path("persona_data/model_inits")
PCA_DIR = Path("data/assistant_axis/pca")
EVAL_DIR = Path("outputs/riasec_eval")
ANALYSIS_DIR = Path("outputs/analysis")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette (Wong 2011)
TRAIT_COLORS = {
    "realistic":     "#D55E00",
    "investigative": "#0072B2",
    "artistic":      "#009E73",
    "social":        "#CC79A7",
    "enterprising":  "#E69F00",
    "conventional":  "#56B4E9",
}

CATEGORY_COLORS = {
    "scientist":      "#0072B2",
    "artist":         "#D55E00",
    "adventurer":     "#009E73",
    "craftsperson":   "#E69F00",
    "thinker":        "#56B4E9",
    "professional":   "#332288",
    "unconventional": "#CC79A7",
    "leader":         "#882255",
    "communicator":   "#44AA99",
    "assistant":      "#000000",
}

CATEGORY_MARKERS = {
    "scientist":      "o",
    "artist":         "s",
    "adventurer":     "^",
    "craftsperson":   "D",
    "thinker":        "v",
    "professional":   "P",
    "unconventional": "X",
    "leader":         "*",
    "communicator":   "h",
    "assistant":      "p",
}

HEX_ORDER = ["realistic", "investigative", "artistic",
             "social", "enterprising", "conventional"]
TRAITS = sorted(HEX_ORDER)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_riasec_vectors(model_id: str) -> dict[str, np.ndarray]:
    safe_model = model_id.replace("/", "__")
    vectors = {}
    for trait in TRAITS:
        path = RIASEC_DIR / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            vectors[trait] = f.get_tensor("response_persona_vector").numpy().flatten()
    return vectors


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_cosine_matrix(vectors, trait_list):
    n = len(trait_list)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine_sim(vectors[trait_list[i]], vectors[trait_list[j]])
    return mat


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold panel label (a), (b), etc."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Cross-Model RIASEC Geometry
# ══════════════════════════════════════════════════════════════════════════════

def fig1_cross_model_geometry():
    fig = plt.figure(figsize=(7.0, 2.4))

    # Custom grid: two heatmaps + one scatter with colorbar space
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 1],
                          wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.15)

    ax_h1 = fig.add_subplot(gs[0, 0])
    ax_h2 = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])
    ax_sc = fig.add_subplot(gs[0, 3])

    # Panel (a) and (b): heatmaps
    labels_short = [t.capitalize()[:4] + "." for t in TRAITS]
    ims = []
    for ax, model_id, title, label in [
        (ax_h1, "meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B", "a"),
        (ax_h2, "marin-community/marin-8b-instruct", "Marin-8B", "b"),
    ]:
        vectors = load_riasec_vectors(model_id)
        matrix = compute_cosine_matrix(vectors, TRAITS)

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.35, vmax=1.0, aspect="equal")
        ims.append(im)
        ax.set_xticks(range(len(TRAITS)))
        ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(TRAITS)))
        ax.set_yticklabels(labels_short if ax == ax_h1 else [], fontsize=6)

        for i in range(len(TRAITS)):
            for j in range(len(TRAITS)):
                val = matrix[i, j]
                color = "white" if val > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color=color)

        ax.set_title(title, fontsize=8, pad=4)
        panel_label(ax, label, x=-0.2, y=1.15)
        ax.tick_params(length=0)

    # Shared colorbar
    cb = fig.colorbar(ims[-1], cax=ax_cb)
    cb.ax.tick_params(labelsize=6, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)

    # Panel (c): cross-model correlation
    llama_vecs = load_riasec_vectors("meta-llama/Llama-3.2-1B-Instruct")
    marin_vecs = load_riasec_vectors("marin-community/marin-8b-instruct")

    llama_upper, marin_upper, pair_labels = [], [], []
    for i in range(len(TRAITS)):
        for j in range(i + 1, len(TRAITS)):
            llama_upper.append(cosine_sim(llama_vecs[TRAITS[i]], llama_vecs[TRAITS[j]]))
            marin_upper.append(cosine_sim(marin_vecs[TRAITS[i]], marin_vecs[TRAITS[j]]))
            pair_labels.append(f"{TRAITS[i][:3]}-{TRAITS[j][:3]}")

    llama_upper = np.array(llama_upper)
    marin_upper = np.array(marin_upper)
    r, p = stats.pearsonr(llama_upper, marin_upper)

    ax_sc.scatter(llama_upper, marin_upper, c="#0072B2", s=20, alpha=0.85,
                  edgecolors="white", linewidths=0.3, zorder=5)

    for k, label_text in enumerate(pair_labels):
        ax_sc.annotate(label_text, (llama_upper[k], marin_upper[k]),
                       fontsize=4, alpha=0.5, xytext=(2, 2),
                       textcoords="offset points")

    # Fit line
    slope, intercept = np.polyfit(llama_upper, marin_upper, 1)
    x_fit = np.linspace(llama_upper.min() - 0.02, llama_upper.max() + 0.02, 100)
    ax_sc.plot(x_fit, slope * x_fit + intercept, "--", color="#D55E00",
               alpha=0.7, linewidth=0.8)

    # Identity line
    lim = [min(llama_upper.min(), marin_upper.min()) - 0.03,
           max(llama_upper.max(), marin_upper.max()) + 0.03]
    ax_sc.plot(lim, lim, ":", color="gray", alpha=0.3, linewidth=0.5)
    ax_sc.set_xlim(lim)
    ax_sc.set_ylim(lim)
    ax_sc.set_xlabel("Llama-3.2-1B cosine sim.")
    ax_sc.set_ylabel("Marin-8B cosine sim.")
    ax_sc.set_title(f"Cross-model (r = {r:.2f})", fontsize=8, pad=4)
    ax_sc.set_aspect("equal")
    panel_label(ax_sc, "c", x=-0.18, y=1.15)

    fig.savefig(OUTPUT_DIR / "fig1_cross_model_geometry.pdf", dpi=DPI)
    fig.savefig(OUTPUT_DIR / "fig1_cross_model_geometry.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1_cross_model_geometry")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Steering Effectiveness
# ══════════════════════════════════════════════════════════════════════════════

def fig2_steering_effectiveness():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.09, right=0.97, top=0.88, bottom=0.15)

    for ax, model_tag, title, label in [
        (ax1, "meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B", "a"),
        (ax2, "marin-community__marin-8b-instruct", "Marin-8B", "b"),
    ]:
        eval_path = EVAL_DIR / f"{model_tag}_logprob_eval.json"
        if not eval_path.exists():
            print(f"  Skipping {title}: no logprob eval data")
            continue

        with open(eval_path) as f:
            data = json.load(f)

        for trait in HEX_ORDER:
            tdata = data["results"][trait]
            alphas = sorted(tdata["alphas"].keys(), key=float)
            alpha_vals = [float(a) for a in alphas]
            mean_gaps = [tdata["alphas"][a]["mean_gap"] for a in alphas]
            std_gaps = [tdata["alphas"][a]["std_gap"] for a in alphas]

            color = TRAIT_COLORS[trait]
            ax.plot(alpha_vals, mean_gaps, "-o", color=color, markersize=3,
                    label=trait.capitalize(), zorder=5)
            ax.fill_between(
                alpha_vals,
                [m - s for m, s in zip(mean_gaps, std_gaps)],
                [m + s for m, s in zip(mean_gaps, std_gaps)],
                alpha=0.08, color=color, linewidth=0,
            )

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
        ax.set_xlabel(r"Steering strength ($\alpha$)")
        if ax == ax1:
            ax.set_ylabel("Logprob gap (YES - NO)")
        ax.set_title(title, fontsize=8, pad=4)
        ax.grid(True, alpha=0.15, linewidth=0.3)
        panel_label(ax, label)

    # Single legend for both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.53, 1.02), fontsize=6.5,
               columnspacing=1.0, handletextpad=0.4)

    fig.savefig(OUTPUT_DIR / "fig2_steering_effectiveness.pdf", dpi=DPI)
    fig.savefig(OUTPUT_DIR / "fig2_steering_effectiveness.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2_steering_effectiveness")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Persona Space PCA (Marin 8B)
# ══════════════════════════════════════════════════════════════════════════════

def fig3_persona_space():
    model_tag = "marin-community__marin-8b-instruct"
    pca_path = PCA_DIR / f"{model_tag}_pca.pt"
    if not pca_path.exists():
        print("Skipping fig3: no Marin PCA data")
        return

    pca_data = torch.load(pca_path, weights_only=False)
    projections = pca_data["projections"].numpy()
    role_names = pca_data["role_names"]
    role_categories = pca_data["role_categories"]
    ev = pca_data["explained_variance_ratio"].numpy()

    fig = plt.figure(figsize=(7.0, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.8, 1.0],
                          wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.14)

    # ── Panel (a): PC1 vs PC2 scatter ──
    ax_sc = fig.add_subplot(gs[0, 0])

    for cat in sorted(set(role_categories.values())):
        indices = [i for i, n in enumerate(role_names) if role_categories.get(n) == cat]
        if not indices:
            continue
        color = CATEGORY_COLORS.get(cat, "#888888")
        marker = CATEGORY_MARKERS.get(cat, "o")
        size = 60 if cat == "assistant" else 18
        ax_sc.scatter(
            projections[indices, 0], projections[indices, 1],
            c=color, marker=marker, s=size, alpha=0.85,
            edgecolors="white", linewidths=0.3,
            label=cat.replace("_", " ").capitalize(), zorder=5,
        )

    # Annotate notable roles
    notable = {"default_assistant", "fortune_teller", "stand_up_comedian",
               "poet_laureate", "social_worker", "software_architect",
               "skeptic", "kindergarten_teacher"}
    for idx, name in enumerate(role_names):
        if name in notable:
            display = name.replace("_", " ")
            ax_sc.annotate(
                display,
                (projections[idx, 0], projections[idx, 1]),
                fontsize=4.5, alpha=0.7, xytext=(3, 3),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", lw=0.3, alpha=0.3) if abs(projections[idx, 0]) > 25 else None,
            )

    ax_sc.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)")
    ax_sc.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)")
    ax_sc.axhline(0, color="gray", linestyle="-", alpha=0.15, linewidth=0.3)
    ax_sc.axvline(0, color="gray", linestyle="-", alpha=0.15, linewidth=0.3)
    ax_sc.legend(fontsize=4.5, ncol=2, loc="lower left",
                 markerscale=0.7, handletextpad=0.2, columnspacing=0.5,
                 borderpad=0.3)
    panel_label(ax_sc, "a", x=-0.12, y=1.12)

    # ── Panel (b): Scree plot ──
    ax_sr = fig.add_subplot(gs[0, 1])

    n_show = min(15, len(ev))
    x = np.arange(1, n_show + 1)
    ax_sr.bar(x, ev[:n_show] * 100, color="#0072B2", alpha=0.7, width=0.65,
              edgecolor="white", linewidth=0.3)
    ax_sr.set_xlabel("Component")
    ax_sr.set_ylabel("Variance (%)")
    ax_sr.set_xticks([1, 5, 10, 15])

    ax_sr2 = ax_sr.twinx()
    cumulative = np.cumsum(ev[:n_show]) * 100
    ax_sr2.plot(x, cumulative, "o-", color="#D55E00", markersize=2, linewidth=0.8)
    ax_sr2.set_ylabel("Cumulative (%)", color="#D55E00")
    ax_sr2.tick_params(axis="y", colors="#D55E00")
    ax_sr2.set_ylim(0, 100)
    for thr in [50, 70, 90]:
        ax_sr2.axhline(thr, color="gray", linestyle="--", alpha=0.15, linewidth=0.3)
    ax_sr2.spines["top"].set_visible(False)
    panel_label(ax_sr, "b", x=-0.2, y=1.12)

    # ── Panel (c): PC1 rankings (top/bottom 12) ──
    ax_rk = fig.add_subplot(gs[0, 2])

    pc1 = projections[:, 0]
    sorted_idx = np.argsort(pc1)
    n_show_rk = 12

    # Bottom N and top N
    bottom_idx = sorted_idx[:n_show_rk]
    top_idx = sorted_idx[-n_show_rk:]
    show_idx = np.concatenate([bottom_idx, top_idx])

    show_names = [role_names[i].replace("_", " ") for i in show_idx]
    show_pc1 = pc1[show_idx]
    show_colors = []
    for i in show_idx:
        cat = role_categories.get(role_names[i], "")
        show_colors.append(CATEGORY_COLORS.get(cat, "#888888"))

    y_pos = np.arange(len(show_idx))
    ax_rk.barh(y_pos, show_pc1, color=show_colors, alpha=0.8, height=0.7,
               edgecolor="white", linewidth=0.2)
    ax_rk.set_yticks(y_pos)
    ax_rk.set_yticklabels(show_names, fontsize=5)
    ax_rk.set_xlabel("PC1 projection")
    ax_rk.axvline(0, color="gray", linestyle="-", alpha=0.3, linewidth=0.4)

    # Separator line between bottom and top groups
    ax_rk.axhline(n_show_rk - 0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.4)

    ax_rk.invert_yaxis()
    panel_label(ax_rk, "c", x=-0.35, y=1.12)

    fig.savefig(OUTPUT_DIR / "fig3_persona_space.pdf", dpi=DPI)
    fig.savefig(OUTPUT_DIR / "fig3_persona_space.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3_persona_space")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: RIASEC Hexagonal Structure
# ══════════════════════════════════════════════════════════════════════════════

def fig4_riasec_hexagonal():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    fig.subplots_adjust(wspace=0.3, left=0.08, right=0.96, top=0.88, bottom=0.12)

    for ax, model_id, title, label in [
        (axes[0], "meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B", "a"),
        (axes[1], "marin-community/marin-8b-instruct", "Marin-8B", "b"),
    ]:
        vectors = load_riasec_vectors(model_id)
        matrix = np.stack([vectors[t] for t in HEX_ORDER])
        pca = PCA(n_components=min(6, matrix.shape[0]))
        proj = pca.fit_transform(matrix - matrix.mean(axis=0))
        ev = pca.explained_variance_ratio_

        # Hexagonal edges (gray, behind)
        for i in range(6):
            j = (i + 1) % 6
            ax.plot([proj[i, 0], proj[j, 0]], [proj[i, 1], proj[j, 1]],
                    "-", color="#cccccc", alpha=0.5, linewidth=0.8, zorder=2)

        # Points
        for i, trait in enumerate(HEX_ORDER):
            ax.scatter(proj[i, 0], proj[i, 1], c=TRAIT_COLORS[trait], s=50,
                       edgecolors="black", linewidths=0.5, zorder=5)
            # Place labels to avoid overlaps
            offset_x, offset_y = 6, 4
            if proj[i, 0] < 0:
                offset_x = -6
            ax.annotate(
                trait.capitalize(), (proj[i, 0], proj[i, 1]),
                fontsize=6.5, fontweight="normal",
                xytext=(offset_x, offset_y), textcoords="offset points",
                ha="left" if offset_x > 0 else "right",
            )

        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title(title, fontsize=8, pad=4)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.1, linewidth=0.3)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.1, linewidth=0.3)
        panel_label(ax, label)

    fig.savefig(OUTPUT_DIR / "fig4_riasec_hexagonal.pdf", dpi=DPI)
    fig.savefig(OUTPUT_DIR / "fig4_riasec_hexagonal.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig4_riasec_hexagonal")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: RIASEC vs Persona Space + Hexagonal Distance
# ══════════════════════════════════════════════════════════════════════════════

def fig5_riasec_persona_relationship():
    fig, (ax_hm, ax_hex) = plt.subplots(1, 2, figsize=(6.0, 2.8))
    fig.subplots_adjust(wspace=0.45, left=0.08, right=0.94, top=0.88, bottom=0.15)

    # ── Panel (a): RIASEC vs PCA components heatmap ──
    comparison_path = ANALYSIS_DIR / "marin-community__marin-8b-instruct_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)

        cosine_data = comparison["cosine_riasec_vs_pca"]
        trait_names = comparison["trait_names"]
        n_pca = min(8, len(comparison["pca_component_labels"]))
        pca_labels = comparison["pca_component_labels"][:n_pca]

        matrix = np.zeros((len(trait_names), n_pca))
        for i, trait in enumerate(trait_names):
            for j, pc in enumerate(pca_labels):
                matrix[i, j] = cosine_data[trait][pc]

        im = ax_hm.imshow(matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
        ax_hm.set_xticks(range(n_pca))
        ax_hm.set_xticklabels(pca_labels, fontsize=6)
        ax_hm.set_yticks(range(len(trait_names)))
        ax_hm.set_yticklabels([t.capitalize() for t in trait_names], fontsize=6)
        ax_hm.tick_params(length=0)

        for i in range(len(trait_names)):
            for j in range(n_pca):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.2 else "black"
                ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=4.5, color=color)

        cb = fig.colorbar(im, ax=ax_hm, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=5, length=2, width=0.3)
        cb.set_label("Cosine sim.", fontsize=6)
        cb.outline.set_linewidth(0.3)
        ax_hm.set_title("RIASEC vs persona PCA", fontsize=8, pad=4)
    panel_label(ax_hm, "a", x=-0.18, y=1.12)

    # ── Panel (b): Hexagonal distance vs cosine similarity ──
    from matplotlib.lines import Line2D

    for model_id, marker, color, lbl in [
        ("meta-llama/Llama-3.2-1B-Instruct", "o", "#0072B2", "Llama 1B"),
        ("marin-community/marin-8b-instruct", "s", "#D55E00", "Marin 8B"),
    ]:
        vectors = load_riasec_vectors(model_id)
        # Group by hexagonal distance
        dist_vals = {1: [], 2: [], 3: []}
        for i in range(6):
            for j in range(i + 1, 6):
                dist = min(abs(i - j), 6 - abs(i - j))
                cos = cosine_sim(vectors[HEX_ORDER[i]], vectors[HEX_ORDER[j]])
                dist_vals[dist].append(cos)
                jitter = np.random.default_rng(42 + i * 6 + j).uniform(-0.08, 0.08)
                ax_hex.scatter(dist + jitter, cos, marker=marker, c=color,
                               s=15, alpha=0.5, edgecolors="white", linewidths=0.2)

        # Mean horizontal bars
        offset = -0.15 if "Llama" in lbl else 0.15
        for d in [1, 2, 3]:
            m = np.mean(dist_vals[d])
            ax_hex.plot([d + offset - 0.1, d + offset + 0.1], [m, m],
                        color=color, linewidth=2, alpha=0.9, solid_capstyle="round")

    ax_hex.set_xlabel("Hexagonal distance")
    ax_hex.set_ylabel("Cosine similarity")
    ax_hex.set_title("Holland hexagonal structure", fontsize=8, pad=4)
    ax_hex.set_xticks([1, 2, 3])
    ax_hex.set_xticklabels(["Adjacent\n(d=1)", "Alternate\n(d=2)", "Opposite\n(d=3)"],
                            fontsize=6)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
               markersize=5, label="Llama 1B"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#D55E00",
               markersize=5, label="Marin 8B"),
    ]
    ax_hex.legend(handles=legend_elements, loc="upper right", fontsize=6)
    panel_label(ax_hex, "b", x=-0.15, y=1.12)

    fig.savefig(OUTPUT_DIR / "fig5_riasec_persona.pdf", dpi=DPI)
    fig.savefig(OUTPUT_DIR / "fig5_riasec_persona.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig5_riasec_persona")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Vector Decomposition (Original vs Residual)
# ══════════════════════════════════════════════════════════════════════════════

def fig6_vector_decomposition():
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05],
                          wspace=0.15, left=0.06, right=0.97, top=0.85, bottom=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax_cb1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_cb2 = fig.add_subplot(gs[0, 3])

    labels_short = [t.capitalize() for t in TRAITS]

    for ax, ax_cb, matrix_key, cmap, vmin, vmax, title, label in [
        (ax1, ax_cb1, "original_cosine_matrix", "YlOrRd", 0.35, 1.0,
         "Original vectors", "a"),
        (ax2, ax_cb2, "residual_cosine_matrix", "RdBu_r", -0.7, 0.7,
         "Residual vectors", "b"),
    ]:
        # Use Marin 8B decomposition
        decomp_path = ANALYSIS_DIR / "marin-community__marin-8b-instruct_vector_decomposition.json"
        if not decomp_path.exists():
            print("Skipping fig6: no decomposition data")
            return

        with open(decomp_path) as f:
            decomp = json.load(f)

        matrix = np.zeros((6, 6))
        for i, ti in enumerate(TRAITS):
            for j, tj in enumerate(TRAITS):
                matrix[i, j] = decomp[matrix_key][ti][tj]

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(6))
        ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(6))
        ax.set_yticklabels(labels_short if ax == ax1 else [], fontsize=6)

        for i in range(6):
            for j in range(6):
                val = matrix[i, j]
                if i == j:
                    continue
                if matrix_key == "original_cosine_matrix":
                    color = "white" if val > 0.65 else "black"
                else:
                    color = "white" if abs(val) > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color=color)

        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(length=0)

        cb = fig.colorbar(im, cax=ax_cb)
        cb.ax.tick_params(labelsize=5, length=2, width=0.3)
        cb.outline.set_linewidth(0.3)

        panel_label(ax, label, x=-0.15, y=1.12)

    fig.savefig(OUTPUT_DIR / "fig6_vector_decomposition.pdf", dpi=DPI)
    fig.savefig(OUTPUT_DIR / "fig6_vector_decomposition.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig6_vector_decomposition")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating publication figures...")
    fig1_cross_model_geometry()
    fig2_steering_effectiveness()
    fig3_persona_space()
    fig4_riasec_hexagonal()
    fig5_riasec_persona_relationship()
    fig6_vector_decomposition()
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
