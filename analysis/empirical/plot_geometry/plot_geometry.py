"""
Plot role vectors (PCA, t-SNE) colored by empirical metrics.

Loads persona vectors from safetensors and empirical evaluation results
(both NLP metrics from per-role JSON caches and LLM judge scores from
gold prompt CSVs), then produces geometry plots where the color of each
point reflects an empirical metric instead of hand-coded ratings.

Output structure:
    analysis/empirical/plot_geometry/figures/
    ├── nlp_colored/
    │   ├── steered/
    │   └── assistant_axis/
    └── llm_colored/
        ├── steered/
        └── assistant_axis/

Usage:
    python analysis/empirical/plot_geometry/plot_geometry.py
    python analysis/empirical/plot_geometry/plot_geometry.py --method steered --metrics llm
    python analysis/empirical/plot_geometry/plot_geometry.py --method all --metrics all --assistant-axis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from safetensors.numpy import load_file
from scipy.spatial import cKDTree
from adjustText import adjust_text
import os
import re
import glob
import json
import argparse
import pandas as pd

# ── paths (relative to repo root) ──────────────────────────────────────
VECTOR_DIR = "persona_data/model_inits"
NLP_CACHE_DIR = "analysis/empirical/nlp_experiments/figures/per_role"
GOLD_CSV_DIR = "experiment_data/gold_prompt_experiments"
OUTPUT_BASE = "analysis/empirical/plot_geometry/figures"

# NLP metrics to visualize (metric_key, colorbar_label, colormap)
NLP_METRICS = [
    ("compression_ratio_mean", "Compression Ratio", "viridis"),
    ("bleu_mean", "BLEU Score", "viridis"),
    ("first_person_rate_mean", "First-Person Rate (%)", "magma"),
    ("unique_bigram_ratio_mean", "Unique Bigram Ratio", "viridis"),
    ("hedge_rate_mean", "Hedge Rate (%)", "cividis"),
    ("assertive_rate_mean", "Assertive Rate (%)", "inferno"),
    ("question_rate_mean", "Question Rate (%)", "plasma"),
    ("word_count_mean", "Word Count", "viridis"),
    ("epistemic_total_mean", "Epistemic Markers", "magma"),
    ("role_consistency_mean", "Role Consistency", "viridis"),
    ("ai_phrase_rate_mean", "AI Phrase Rate (%)", "inferno"),
]

# LLM judge metrics per method (column_key, colorbar_label, colormap)
LLM_METRICS_BY_METHOD = {
    "steered": [
        ("steered_score", "Steered Score (0–100)", "RdYlGn"),
        ("baseline_score", "Baseline Score (0–100)", "RdYlGn"),
        ("cmp_emotional_register", "Emotional Register (0–100)", "plasma"),
        ("cmp_vocab_choice", "Vocab Choice (0–100)", "plasma"),
        ("cmp_social_dynamic", "Social Dynamic (0–100)", "plasma"),
        ("cmp_motivation", "Motivation (0–100)", "magma"),
        ("cmp_worldview_alignment", "Worldview Alignment (0–100)", "magma"),
        ("steered_style", "Steered Style (avg)", "viridis"),
        ("steered_content", "Steered Content (avg)", "viridis"),
    ],
    "assistant_axis": [
        ("assistant_axis_score", "Assistant Axis Score (0–100)", "RdYlGn"),
        ("baseline_score", "Baseline Score (0–100)", "RdYlGn"),
        ("assistant_axis_cmp_emotional_register", "AA Emotional Register", "cividis"),
        ("assistant_axis_cmp_vocab_choice", "AA Vocab Choice", "cividis"),
        ("assistant_axis_cmp_social_dynamic", "AA Social Dynamic", "cividis"),
        ("assistant_axis_cmp_motivation", "AA Motivation", "cividis"),
        ("assistant_axis_cmp_worldview_alignment", "AA Worldview Alignment", "cividis"),
    ],
}
# Flat list for backwards compat (used by SCORE_COLS)
LLM_METRICS = [m for metrics in LLM_METRICS_BY_METHOD.values() for m in metrics]

STYLE_DIMS = ["cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic"]
CONTENT_DIMS = ["cmp_motivation", "cmp_worldview_alignment"]

SCORE_COLS = [
    "steered_score", "assistant_axis_score", "baseline_score",
    "cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic",
    "cmp_motivation", "cmp_worldview_alignment",
    "assistant_axis_cmp_emotional_register", "assistant_axis_cmp_vocab_choice",
    "assistant_axis_cmp_social_dynamic", "assistant_axis_cmp_motivation",
    "assistant_axis_cmp_worldview_alignment",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Geometry plots colored by empirical metrics")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "steered", "assistant_axis", "baseline"],
                        help="Which generation method's metrics to use (default: all)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="If set, filter by this alpha value")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity (auto-clamped to n-1)")
    parser.add_argument("--metrics", type=str, default="all",
                        choices=["all", "nlp", "llm"],
                        help="Which metric set to plot: nlp, llm, or all (default: all)")
    parser.add_argument("--assistant-axis", action="store_true",
                        help="Also color plots by assistant axis projection")
    return parser.parse_args()


# ── score parsing (same as plot_comparison_results.py) ──────────────────

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


# ── data loading ────────────────────────────────────────────────────────

def load_vectors(layer_number: int, num_questions: int):
    """Load role vectors from safetensors, returns dict[role_name, ndarray]."""
    vectors = {}
    for entry in os.listdir(VECTOR_DIR):
        full_path = os.path.join(VECTOR_DIR, entry)
        if not os.path.isdir(full_path):
            continue
        files = [f for f in os.listdir(full_path)
                 if f.endswith(".safetensors") and f"count{num_questions}" in f]
        if not files:
            continue
        data = load_file(os.path.join(full_path, files[0]))
        key = list(data.keys())[0]
        name = entry.replace("_persona_initialization", "")
        vectors[name] = data[key].astype(np.float32)[layer_number - 1].squeeze()
    return vectors


def load_nlp_metrics(method: str, alpha=None):
    """Load per-role NLP summary metrics from cached JSON files."""
    metrics = {}
    if not os.path.isdir(NLP_CACHE_DIR):
        return metrics
    for fname in os.listdir(NLP_CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        role = fname.replace(".json", "")
        with open(os.path.join(NLP_CACHE_DIR, fname)) as f:
            data = json.load(f)

        if alpha is not None:
            alpha_key = str(alpha)
            if "per_alpha" not in data or alpha_key not in data["per_alpha"]:
                continue
            summaries = data["per_alpha"][alpha_key].get("summaries", {})
        else:
            summaries = data.get("summaries", {})

        if method not in summaries:
            continue
        metrics[role] = summaries[method]
    return metrics


def load_llm_metrics(alpha=None, sample_count=50):
    """Load per-role LLM judge scores from gold prompt CSVs."""
    files = sorted(glob.glob(os.path.join(GOLD_CSV_DIR, "Comparison_GoldStandard_*.csv")))
    if not files:
        return {}

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return {}

    combined = pd.concat(dfs, ignore_index=True)

    for col in SCORE_COLS:
        if col in combined.columns:
            combined[col] = combined[col].apply(_parse_score)

    if "sample_count" in combined.columns:
        combined = combined[combined["sample_count"] == sample_count]
    if alpha is not None and "alpha" in combined.columns:
        combined = combined[combined["alpha"].apply(lambda x: abs(x - alpha) < 0.001)]

    for dim_cols, new_col in [(STYLE_DIMS, "steered_style"), (CONTENT_DIMS, "steered_content")]:
        valid_cols = [c for c in dim_cols if c in combined.columns]
        if valid_cols:
            combined[new_col] = combined[valid_cols].mean(axis=1)

    metrics = {}
    if "role" not in combined.columns:
        return metrics

    all_metric_cols = SCORE_COLS + ["steered_style", "steered_content"]
    available_cols = [c for c in all_metric_cols if c in combined.columns]

    grouped = combined.groupby("role")[available_cols].mean()
    for role, row in grouped.iterrows():
        metrics[role] = {col: row[col] for col in available_cols if not pd.isna(row[col])}

    return metrics


# ── plotting helpers ────────────────────────────────────────────────────

def _label_3d(ax, coords, names):
    """Add labels pushed away from nearest neighbors (same as full_analysis.py)."""
    tree = cKDTree(coords)
    k = min(6, len(names) - 1)
    _, ii = tree.query(coords, k=k + 1)
    scale = np.ptp(coords, axis=0) * 0.025
    for i, name in enumerate(names):
        neighbors = coords[ii[i, 1:]]
        direction = coords[i] - neighbors.mean(axis=0)
        norm_d = np.linalg.norm(direction)
        direction = direction / norm_d if norm_d > 0 else np.array([1, 0, 0])
        offset = direction * scale
        ax.text(coords[i, 0] + offset[0], coords[i, 1] + offset[1],
                coords[i, 2] + offset[2], name,
                fontsize=4.5, ha='center', va='center', alpha=0.85)


def _label_2d(ax, coords, names):
    """Add labels with adjust_text (same as full_analysis.py t-SNE 2D)."""
    texts = []
    for i, name in enumerate(names):
        texts.append(ax.text(coords[i, 0], coords[i, 1], name,
                             fontsize=5.5, alpha=0.85))
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5),
                expand=(1.5, 1.5), force_text=(0.8, 0.8), force_points=(0.3, 0.3))


def generate_plots(out_dir, metric_list, role_metrics, all_names, method,
                   layer_number, has_assistant, projected_aa, pca_resid,
                   pca_projected, pca, tsne2d_projected, tsne3d_projected):
    """Generate all geometry plots for a given metric list and output directory."""
    os.makedirs(out_dir, exist_ok=True)

    for metric_key, cbar_label, cmap in metric_list:
        color_vals = []
        for name in all_names:
            if name in role_metrics and metric_key in role_metrics[name]:
                color_vals.append(role_metrics[name][metric_key])
            else:
                color_vals.append(np.nan)
        color_vals = np.array(color_vals)

        valid_mask = ~np.isnan(color_vals)
        if valid_mask.sum() < 3:
            print(f"    Skipping {metric_key}: too few valid values")
            continue

        vmin = np.nanmin(color_vals)
        vmax = np.nanmax(color_vals)
        if vmin == vmax:
            vmax = vmin + 1e-6

        metric_slug = metric_key.replace("_mean", "")
        print(f"    Plotting {metric_slug} ...")

        # ── 1. PCA (assistant axis) 3D ──────────────────────────────────
        if has_assistant:
            fig = plt.figure(figsize=(16, 11))
            ax = fig.add_subplot(111, projection='3d')

            sc = ax.scatter(projected_aa[:, 0], projected_aa[:, 1], projected_aa[:, 2],
                            c=color_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                            s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

            _label_3d(ax, projected_aa, all_names)

            t_r = np.linspace(-1, 1, 2) * max(np.abs(projected_aa[:, 0])) * 1.2
            ax.plot(t_r, [0, 0], [0, 0], 'k--', linewidth=1.5, label="Assistant Axis")

            ax.set_xlabel("Assistant Axis")
            ax.set_ylabel(f"Residual PC1 ({pca_resid.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_zlabel(f"Residual PC2 ({pca_resid.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title(f"Role Vectors — colored by {cbar_label} [{method}] (Layer {layer_number})")
            ax.legend(loc='upper left')

            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label(cbar_label)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pca_aa_{metric_slug}.png"), dpi=150)
            plt.close(fig)

        # ── 2. Standard PCA 3D ──────────────────────────────────────────
        fig = plt.figure(figsize=(16, 11))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2],
                        c=color_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

        _label_3d(ax, pca_projected, all_names)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax.set_title(f"Standard PCA — colored by {cbar_label} [{method}] (Layer {layer_number})")

        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(cbar_label)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pca_{metric_slug}.png"), dpi=150)
        plt.close(fig)

        # ── 3. t-SNE 2D ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, 11))

        sc = ax.scatter(tsne2d_projected[:, 0], tsne2d_projected[:, 1],
                        c=color_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

        _label_2d(ax, tsne2d_projected, all_names)

        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE — colored by {cbar_label} [{method}] (Layer {layer_number})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tsne2d_{metric_slug}.png"), dpi=150)
        plt.close(fig)

        # ── 4. t-SNE 3D ────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 11))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(tsne3d_projected[:, 0], tsne3d_projected[:, 1], tsne3d_projected[:, 2],
                        c=color_vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

        _label_3d(ax, tsne3d_projected, all_names)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(cbar_label)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
        ax.set_title(f"3D t-SNE — colored by {cbar_label} [{method}] (Layer {layer_number})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tsne3d_{metric_slug}.png"), dpi=150)
        plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    layer_number = args.layer
    alpha = args.alpha

    # Resolve method list
    if args.method == "all":
        methods = ["steered", "assistant_axis"]
    else:
        methods = [args.method]

    # Resolve metric set list
    if args.metrics == "all":
        metric_sets = ["nlp", "llm"]
    else:
        metric_sets = [args.metrics]

    # Load vectors once (geometry is the same regardless of method/metrics)
    vectors = load_vectors(layer_number, args.num_questions)
    print(f"Loaded {len(vectors)} vectors")

    # We need at least one method's data to determine common roles for geometry.
    # Compute geometry from ALL vector roles, then filter per-method for coloring.
    all_role_names = sorted([r for r in vectors.keys() if r != "assistant"])
    has_assistant = "assistant" in vectors

    role_matrix = np.array([vectors[r] for r in all_role_names])
    if has_assistant:
        assistant_vector = vectors["assistant"]
        all_names = all_role_names + ["assistant"]
        all_vectors_raw = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
    else:
        all_names = all_role_names
        all_vectors_raw = role_matrix

    all_vectors_centered = all_vectors_raw - all_vectors_raw.mean(axis=0)

    # ── Dimensionality reductions (computed once) ──────────────────────

    projected_aa = None
    pca_resid = None
    aa_dim1 = None
    if has_assistant:
        assistant_axis = assistant_vector - role_matrix.mean(axis=0)
        assistant_axis = assistant_axis / np.linalg.norm(assistant_axis)
        aa_dim1 = all_vectors_centered @ assistant_axis
        residual = all_vectors_centered - np.outer(aa_dim1, assistant_axis)
        pca_resid = PCA(n_components=2)
        dims23 = pca_resid.fit_transform(residual)
        projected_aa = np.column_stack([aa_dim1, dims23])

    pca = PCA(n_components=3)
    pca_projected = pca.fit_transform(all_vectors_centered)

    perp = min(args.perplexity, len(all_names) - 1)
    tsne2d = TSNE(n_components=2, perplexity=perp, random_state=42)
    tsne2d_projected = tsne2d.fit_transform(all_vectors_centered)

    tsne3d = TSNE(n_components=3, perplexity=perp, random_state=42)
    tsne3d_projected = tsne3d.fit_transform(all_vectors_centered)

    # ── Loop over methods × metric sets ────────────────────────────────

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for metric_set in metric_sets:
            print(f"\n  Metric set: {metric_set}")

            # Load metrics for this method
            role_metrics = {}
            active_metrics = []

            if metric_set == "nlp":
                nlp_data = load_nlp_metrics(method, alpha)
                print(f"  Loaded NLP metrics for {len(nlp_data)} roles")
                for role in all_names:
                    if role in nlp_data:
                        role_metrics[role] = nlp_data[role]
                active_metrics = list(NLP_METRICS)

            elif metric_set == "llm":
                llm_data = load_llm_metrics(alpha, sample_count=args.num_questions)
                print(f"  Loaded LLM judge metrics for {len(llm_data)} roles")
                for role in all_names:
                    if role in llm_data:
                        role_metrics[role] = llm_data[role]
                active_metrics = list(LLM_METRICS_BY_METHOD.get(method, []))

            # Optionally add assistant axis projection
            if args.assistant_axis and has_assistant:
                for i, name in enumerate(all_names):
                    role_metrics.setdefault(name, {})["assistant_axis_proj"] = aa_dim1[i]
                active_metrics.append(("assistant_axis_proj", "Assistant Axis Projection", "coolwarm_r"))

            if not active_metrics:
                print("  No metrics to plot, skipping.")
                continue

            # Output: plot_geometry/{metric_set}_colored/{method}/
            out_dir = os.path.join(OUTPUT_BASE, f"{metric_set}_colored", method)

            generate_plots(
                out_dir=out_dir,
                metric_list=active_metrics,
                role_metrics=role_metrics,
                all_names=all_names,
                method=method,
                layer_number=layer_number,
                has_assistant=has_assistant,
                projected_aa=projected_aa,
                pca_resid=pca_resid,
                pca_projected=pca_projected,
                pca=pca,
                tsne2d_projected=tsne2d_projected,
                tsne3d_projected=tsne3d_projected,
            )

            print(f"  Saved to {out_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
