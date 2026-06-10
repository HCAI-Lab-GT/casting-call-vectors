"""
Scatter plots correlating geometric properties of role vectors with empirical metrics.

For each empirical metric (NLP + LLM judge), produces scatter plots where:
  x-axis = a geometric property (cosine distance to mean, assistant axis projection,
            PC1/PC2/PC3 loadings, vector norm)
  y-axis = the empirical metric value

Each point is a role. Pearson and Spearman correlations are annotated.

Usage:
    python analysis/empirical/plot_geometry/plot_correlation.py
    python analysis/empirical/plot_geometry/plot_correlation.py --method steered --alpha 2.0
    python analysis/empirical/plot_geometry/plot_correlation.py --metrics llm
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.numpy import load_file
from scipy import stats as sp_stats
import os
import re
import glob
import json
import argparse
import pandas as pd


VECTOR_DIR = "persona_data/model_inits"
NLP_CACHE_DIR = "analysis/empirical/nlp_experiments/figures/per_role"
GOLD_CSV_DIR = "experiment_data/gold_prompt_experiments"
OUTPUT_BASE = "analysis/empirical/plot_geometry/figures"

NLP_METRICS = [
    ("compression_ratio_mean", "Compression Ratio"),
    ("bleu_mean", "BLEU Score"),
    ("first_person_rate_mean", "First-Person Rate (%)"),
    ("unique_bigram_ratio_mean", "Unique Bigram Ratio"),
    ("hedge_rate_mean", "Hedge Rate (%)"),
    ("assertive_rate_mean", "Assertive Rate (%)"),
    ("question_rate_mean", "Question Rate (%)"),
    ("word_count_mean", "Word Count"),
    ("role_consistency_mean", "Role Consistency"),
    ("ai_phrase_rate_mean", "AI Phrase Rate (%)"),
]

LLM_METRICS_BY_METHOD = {
    "steered": [
        ("steered_score", "Steered Score (0–100)"),
        ("baseline_score", "Baseline Score (0–100)"),
        ("cmp_emotional_register", "Emotional Register (0–100)"),
        ("cmp_vocab_choice", "Vocab Choice (0–100)"),
        ("cmp_social_dynamic", "Social Dynamic (0–100)"),
        ("cmp_motivation", "Motivation (0–100)"),
        ("cmp_worldview_alignment", "Worldview Alignment (0–100)"),
        ("steered_style", "Steered Style (avg)"),
        ("steered_content", "Steered Content (avg)"),
    ],
    "assistant_axis": [
        ("assistant_axis_score", "Assistant Axis Score (0–100)"),
        ("baseline_score", "Baseline Score (0–100)"),
        ("assistant_axis_cmp_emotional_register", "AA Emotional Register"),
        ("assistant_axis_cmp_vocab_choice", "AA Vocab Choice"),
        ("assistant_axis_cmp_social_dynamic", "AA Social Dynamic"),
        ("assistant_axis_cmp_motivation", "AA Motivation"),
        ("assistant_axis_cmp_worldview_alignment", "AA Worldview Alignment"),
    ],
}

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

GEOMETRIC_FEATURES = [
    "cosine_dist_to_mean",
    "assistant_axis_proj",
    "vector_norm",
    "pc1",
    "pc2",
    "pc3",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "steered", "assistant_axis", "baseline"],
                        help="Which generation method (default: all)")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--metrics", type=str, default="all",
                        choices=["all", "nlp", "llm"],
                        help="Which metric set to use: nlp, llm, or all (default: all)")
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

def load_vectors(layer_number, num_questions):
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


def load_nlp_metrics(method, alpha=None):
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


def compute_geometric_features(names, vectors):
    """Compute per-role geometric features. Returns dict[feature_name, ndarray]."""
    from sklearn.decomposition import PCA

    role_names = [n for n in names if n != "assistant"]
    role_matrix = np.array([vectors[n] for n in role_names])

    has_assistant = "assistant" in vectors
    if has_assistant:
        assistant_vector = vectors["assistant"]
        all_matrix = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
        all_names = role_names + ["assistant"]
    else:
        all_matrix = role_matrix
        all_names = role_names

    mean_vec = role_matrix.mean(axis=0)
    centered = all_matrix - mean_vec

    # Cosine distance to mean
    norms = np.linalg.norm(all_matrix, axis=1, keepdims=True)
    mean_norm = np.linalg.norm(mean_vec)
    cos_sim = (all_matrix @ mean_vec) / (norms.squeeze() * mean_norm + 1e-12)
    cosine_dist = 1.0 - cos_sim

    # Assistant axis projection
    if has_assistant:
        assistant_axis = assistant_vector - mean_vec
        assistant_axis = assistant_axis / np.linalg.norm(assistant_axis)
        aa_proj = centered @ assistant_axis
    else:
        aa_proj = np.zeros(len(all_names))

    # PCA
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(centered)

    # Vector norms
    vec_norms = np.linalg.norm(all_matrix, axis=1)

    features = {}
    for i, name in enumerate(all_names):
        features[name] = {
            "cosine_dist_to_mean": cosine_dist[i],
            "assistant_axis_proj": aa_proj[i],
            "vector_norm": vec_norms[i],
            "pc1": pca_coords[i, 0],
            "pc2": pca_coords[i, 1],
            "pc3": pca_coords[i, 2],
        }
    return features


def generate_correlation_plots(out_dir, active_metrics, role_data, geo_features,
                               geo_keys, method, layer):
    """Generate correlation heatmap and scatter plots for one method × metric_set."""
    os.makedirs(out_dir, exist_ok=True)

    emp_keys = [k for k, _ in active_metrics]
    emp_labels = [l for _, l in active_metrics]

    roles_ordered = [r for r in sorted(role_data.keys()) if r in geo_features]

    geo_matrix = np.array([[geo_features[r][g] for g in geo_keys] for r in roles_ordered])
    emp_matrix = np.array([
        [role_data[r].get(k, np.nan) for k in emp_keys] for r in roles_ordered
    ])

    n_geo = len(geo_keys)
    n_emp = len(emp_keys)
    corr_r = np.zeros((n_geo, n_emp))
    corr_p = np.zeros((n_geo, n_emp))

    for gi in range(n_geo):
        for ei in range(n_emp):
            valid = ~np.isnan(emp_matrix[:, ei])
            if valid.sum() < 5:
                corr_r[gi, ei] = 0.0
                corr_p[gi, ei] = 1.0
            else:
                r, p = sp_stats.pearsonr(geo_matrix[valid, gi], emp_matrix[valid, ei])
                corr_r[gi, ei] = r
                corr_p[gi, ei] = p

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(10, n_emp * 1.2), max(5, n_geo * 0.8)))
    im = ax.imshow(corr_r, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n_emp))
    ax.set_xticklabels(emp_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_geo))
    ax.set_yticklabels([g.replace("_", " ").title() for g in geo_keys], fontsize=9)

    for gi in range(n_geo):
        for ei in range(n_emp):
            sig = "*" if corr_p[gi, ei] < 0.05 else ""
            sig += "*" if corr_p[gi, ei] < 0.01 else ""
            sig += "*" if corr_p[gi, ei] < 0.001 else ""
            ax.text(ei, gi, f"{corr_r[gi, ei]:.2f}{sig}",
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(corr_r[gi, ei]) > 0.5 else 'black')

    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title(f"Geometry × Empirical Metric Correlations [{method}] (Layer {layer})\n"
                 f"* p<0.05  ** p<0.01  *** p<0.001  |  n={len(roles_ordered)} roles")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("    Saved correlation_heatmap.png")

    # ── Individual scatter plots for significant correlations ───────────
    for gi, geo_key in enumerate(geo_keys):
        for ei, (emp_key, emp_label) in enumerate(active_metrics):
            if abs(corr_r[gi, ei]) < 0.15:
                continue

            valid = ~np.isnan(emp_matrix[:, ei])
            x = geo_matrix[valid, gi]
            y = emp_matrix[valid, ei]

            r_pearson, p_pearson = sp_stats.pearsonr(x, y)
            r_spearman, p_spearman = sp_stats.spearmanr(x, y)

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.scatter(x, y, s=30, alpha=0.6, edgecolors='k', linewidths=0.3)

            z = np.polyfit(x, y, 1)
            p_line = np.poly1d(z)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, p_line(x_sorted), 'r--', linewidth=1.2, alpha=0.7)

            geo_label = geo_key.replace("_", " ").title()
            ax.set_xlabel(geo_label, fontsize=11)
            ax.set_ylabel(emp_label, fontsize=11)
            ax.set_title(f"{geo_label} vs {emp_label} [{method}]")

            stats_text = (f"Pearson r={r_pearson:.3f} (p={p_pearson:.1e})\n"
                          f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.1e})\n"
                          f"n={len(x)} roles")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

            plt.tight_layout()
            geo_slug = geo_key
            emp_slug = emp_key.replace("_mean", "")
            plt.savefig(os.path.join(out_dir, f"scatter_{geo_slug}_vs_{emp_slug}.png"), dpi=150)
            plt.close(fig)


def main():
    args = parse_args()
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

    vectors = load_vectors(args.layer, args.num_questions)
    print(f"Loaded {len(vectors)} vectors")

    all_role_names = sorted(vectors.keys())
    geo_features = compute_geometric_features(all_role_names, vectors)

    geo_keys = GEOMETRIC_FEATURES

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for metric_set in metric_sets:
            print(f"\n  Metric set: {metric_set}")

            role_data = {}
            active_metrics = []

            if metric_set == "nlp":
                nlp_data = load_nlp_metrics(method, alpha)
                print(f"  Loaded NLP metrics for {len(nlp_data)} roles")
                for role in all_role_names:
                    if role in nlp_data:
                        role_data[role] = nlp_data[role]
                active_metrics = list(NLP_METRICS)

            elif metric_set == "llm":
                llm_data = load_llm_metrics(alpha, sample_count=args.num_questions)
                print(f"  Loaded LLM judge metrics for {len(llm_data)} roles")
                for role in all_role_names:
                    if role in llm_data:
                        role_data[role] = llm_data[role]
                active_metrics = list(LLM_METRICS_BY_METHOD.get(method, []))

            if not active_metrics or len(role_data) < 10:
                print("  Too few roles or no metrics, skipping.")
                continue

            # Output: plot_geometry/{metric_set}_colored/{method}/
            out_dir = os.path.join(OUTPUT_BASE, f"{metric_set}_colored", method)

            generate_correlation_plots(
                out_dir=out_dir,
                active_metrics=active_metrics,
                role_data=role_data,
                geo_features=geo_features,
                geo_keys=geo_keys,
                method=method,
                layer=args.layer,
            )

            print(f"  Saved to {out_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
