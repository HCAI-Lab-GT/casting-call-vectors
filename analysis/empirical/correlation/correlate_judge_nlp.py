#!/usr/bin/env python3
"""
Correlate LLM-as-a-judge scores with NLP markers.
===================================================
For each response method (assistant_axis / baseline / steered), computes
Pearson and Spearman correlations between the NLP markers produced by
`analysis/empirical/nlp_experiments/nlp_evaluator.py` and the judge scores
stored in `experiment_data/gold_prompt_experiments/*.csv`.

Judge score columns considered:
    {method}_score                           -- overall persona score
    cmp_{dim}                                -- per-dimension comparison (steered)
    assistant_axis_cmp_{dim}                 -- per-dimension comparison (assistant_axis)

By default this script reads the per-role NLP cache under
`analysis/empirical/nlp_experiments/figures/per_role/` so no spaCy recompute is needed.
If the cache is missing (or `--recompute` is passed) NLP markers are computed
on the fly via `PersonaEvaluator`.

Usage:
    python correlate_judge_nlp.py                               # folder mode (defaults)
    python correlate_judge_nlp.py --role accountant teacher
    python correlate_judge_nlp.py --input <csv_or_folder> --output ./out/
    python correlate_judge_nlp.py --recompute --no-spacy
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import PersonaEvaluator / Filters from the sibling nlp package.
NLP_DIR = Path(__file__).resolve().parent.parent / "nlp"
if str(NLP_DIR) not in sys.path:
    sys.path.insert(0, str(NLP_DIR))

from nlp_helpers import Filters, PersonaMetrics  # noqa: E402
import nlp_evaluator as nlp_mod  # noqa: E402
from nlp_evaluator import (  # noqa: E402
    PersonaEvaluator,
    extract_role_from_filename,
    init_spacy,
    _compute_bleu_against_reference,
)


# =============================================================================
# CONFIG
# =============================================================================

RESPONSE_COLUMNS = ["assistant_axis", "baseline", "steered"]

CMP_DIMS = [
    "emotional_register",
    "vocab_choice",
    "social_dynamic",
    "motivation",
    "worldview_alignment",
]

# Which judge-score columns each response method is paired against.
METHOD_SCORE_COLUMNS: Dict[str, List[str]] = {
    "assistant_axis": ["assistant_axis_score"]
    + [f"assistant_axis_cmp_{d}" for d in CMP_DIMS],
    "baseline": ["baseline_score"],
    "steered": ["steered_score"] + [f"cmp_{d}" for d in CMP_DIMS],
}

# Methods pooled for the combined "what NLP markers drive judge scores" view.
COMBINED_METHODS = ["assistant_axis", "steered"]

# Unified judge-column name → per-method CSV column name. Used to concat rows
# from assistant_axis and steered into a single (marker, score) series.
COMBINED_JUDGE_COLUMNS: Dict[str, Dict[str, str]] = {
    "persona_score": {
        "assistant_axis": "assistant_axis_score",
        "steered": "steered_score",
    },
    **{
        f"cmp_{d}": {
            "assistant_axis": f"assistant_axis_cmp_{d}",
            "steered": f"cmp_{d}",
        }
        for d in CMP_DIMS
    },
}

# Numeric fields from PersonaMetrics we correlate against.
NLP_MARKER_FIELDS = [
    "first_person_rate",
    "epistemic_total",
    "word_count",
    "unique_bigram_ratio",
    "hedge_rate",
    "assertive_rate",
    "question_rate",
    "transition_rate",
    "avg_sentence_length",
    "sentence_count",
    "modal_verb_rate",
    "passive_voice_rate",
    "ai_phrase_rate",
    "role_consistency_score",
    "bleu",
    "compression_ratio",
]

# Heatmap layout: coherence block on top, style block below, with a divider.
HEATMAP_COHERENCE_FIELDS = [
    "word_count",
    "sentence_count",
    "unique_bigram_ratio",
    "compression_ratio",
    "bleu",
]
HEATMAP_STYLE_FIELDS = [
    "first_person_rate",
    "modal_verb_rate",
    "question_rate",
    "ai_phrase_rate",
    "assertive_rate",
    "transition_rate",
    "passive_voice_rate",
]


# =============================================================================
# NLP MARKER LOADING
# =============================================================================


def _reorder_df_alpha_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder df so rows group by sorted alpha values — matches the row order
    produced when `analyze_single_file` aggregates `per_alpha` raw_metrics."""
    if "alpha" not in df.columns:
        return df.reset_index(drop=True)
    alpha_vals = sorted(df["alpha"].unique())
    frames = [
        df[df["alpha"].apply(lambda x, a=a: abs(x - a) < 1e-3)] for a in alpha_vals
    ]
    return pd.concat(frames, ignore_index=True)


def _load_markers_from_cache(
    cache_path: Path,
    columns: List[str],
) -> Optional[Dict[str, List[dict]]]:
    """Load NLP markers from nlp_experiments per_role cache, preserving
    alpha-sorted row order."""
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        cached = json.load(f)
    per_alpha = cached.get("per_alpha", {})
    if not per_alpha:
        # Fall back to top-level raw_metrics (order may or may not match CSV).
        rm = cached.get("raw_metrics", {})
        if not rm:
            return None
        return {col: rm.get(col, []) for col in columns}
    alpha_keys = sorted(per_alpha.keys(), key=float)
    out: Dict[str, List[dict]] = {col: [] for col in columns}
    for ak in alpha_keys:
        rm = per_alpha[ak].get("raw_metrics", {})
        for col in columns:
            out[col].extend(rm.get(col, []))
    return out


def _compute_markers_fresh(
    df: pd.DataFrame,
    role: str,
    columns: List[str],
    use_spacy: bool,
) -> Dict[str, List[dict]]:
    """Compute NLP markers row-by-row for each response column. df must already
    be in alpha-sorted order if ordering matters downstream."""
    evaluator = PersonaEvaluator(role=role, use_spacy=use_spacy)
    out: Dict[str, List[dict]] = {}

    # Precompute spaCy batches if available for speed.
    spacy_batches: Dict[str, List[Optional[dict]]] = {}
    if use_spacy and nlp_mod.SPACY_AVAILABLE and nlp_mod.nlp is not None:
        for col in columns:
            if col not in df.columns:
                continue
            texts = df[col].tolist()
            batch: List[Optional[dict]] = [None] * len(texts)
            valid = [
                (i, str(t)[:100_000])
                for i, t in enumerate(texts)
                if not pd.isna(t) and str(t).strip()
            ]
            if valid:
                idxs, trunc = zip(*valid)
                for j, doc in enumerate(
                    nlp_mod.nlp.pipe(trunc, batch_size=50)
                ):
                    sents = list(doc.sents)
                    batch[idxs[j]] = {
                        "sentence_count": len(sents),
                        "lemmas": [tk.lemma_.lower() for tk in doc if tk.is_alpha],
                        "modal_count": sum(1 for tk in doc if tk.tag_ == "MD"),
                        "passive_count": sum(
                            1 for tk in doc if tk.dep_ in ("auxpass", "nsubjpass")
                        ),
                    }
            spacy_batches[col] = batch

    # BLEU against baseline (same convention as nlp_evaluator).
    baseline_texts = None
    if "baseline" in df.columns:
        baseline_texts = [
            str(t) if not pd.isna(t) else "" for t in df["baseline"].tolist()
        ]

    for col in columns:
        if col not in df.columns:
            continue
        texts = df[col].tolist()
        clean = [str(t) if not pd.isna(t) else "" for t in texts]
        if col == "baseline":
            bleu_batch = [1.0] * len(clean)
        elif baseline_texts is not None:
            bleu_batch = _compute_bleu_against_reference(clean, baseline_texts)
        else:
            bleu_batch = [0.0] * len(clean)
        sp_batch = spacy_batches.get(col, [None] * len(texts))
        metrics = [
            evaluator.compute_metrics(t, spacy_result=sr, bleu=b)
            for t, sr, b in zip(texts, sp_batch, bleu_batch)
        ]
        out[col] = [asdict(m) for m in metrics]
    return out


def load_role_markers(
    csv_path: Path,
    cache_dir: Optional[Path],
    columns: List[str],
    filters: Filters,
    use_spacy: bool,
    recompute: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[dict]], str, bool]:
    """Returns (alpha-sorted filtered df, markers-by-column, role, from_cache)."""
    role = extract_role_from_filename(str(csv_path))
    df = pd.read_csv(csv_path)
    if not filters.is_empty():
        df = filters.apply(df)
    if len(df) == 0:
        return df, {c: [] for c in columns}, role, False
    df = _reorder_df_alpha_sorted(df)

    from_cache = False
    markers: Optional[Dict[str, List[dict]]] = None
    if not recompute and cache_dir is not None:
        markers = _load_markers_from_cache(cache_dir / f"{role}.json", columns)
        if markers is not None:
            lens = {c: len(v) for c, v in markers.items()}
            if all(l == len(df) for l in lens.values()):
                from_cache = True
            else:
                tqdm.write(
                    f"  [{role}] cache length mismatch (df={len(df)}, cache={lens}), "
                    f"recomputing"
                )
                markers = None

    if markers is None:
        markers = _compute_markers_fresh(df, role, columns, use_spacy)

    return df, markers, role, from_cache


# =============================================================================
# CORRELATION
# =============================================================================


def _corr_pair(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = ~(np.isnan(x) | np.isnan(y))
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan")}
    xm, ym = x[mask], y[mask]
    if np.std(xm) == 0 or np.std(ym) == 0:
        return {"n": n, "pearson_r": 0.0, "pearson_p": 1.0,
                "spearman_r": 0.0, "spearman_p": 1.0}
    pr, pp = stats.pearsonr(xm, ym)
    sr, sp = stats.spearmanr(xm, ym)
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def compute_correlations(
    df: pd.DataFrame,
    markers_by_col: Dict[str, List[dict]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Returns {method: {judge_col: {nlp_marker: corr_stats}}}."""
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    n_rows = len(df)
    for method, judge_cols in METHOD_SCORE_COLUMNS.items():
        metrics = markers_by_col.get(method, [])
        if not metrics or len(metrics) != n_rows:
            continue
        out[method] = {}
        marker_arrays = {
            f: np.array([m.get(f, np.nan) for m in metrics], dtype=float)
            for f in NLP_MARKER_FIELDS
        }
        for judge_col in judge_cols:
            if judge_col not in df.columns:
                continue
            y = pd.to_numeric(df[judge_col], errors="coerce").to_numpy(dtype=float)
            out[method][judge_col] = {
                field: _corr_pair(x, y) for field, x in marker_arrays.items()
            }
    return out


def compute_combined_correlations(
    df: pd.DataFrame,
    markers_by_col: Dict[str, List[dict]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Pool assistant_axis + steered rows and correlate NLP markers against
    judge scores. Each CSV row contributes two observations (one per method).
    Returns {judge_col: {nlp_marker: corr_stats}}.
    """
    n_rows = len(df)
    if n_rows == 0:
        return {}

    combined_markers: List[dict] = []
    for method in COMBINED_METHODS:
        metrics = markers_by_col.get(method, [])
        if len(metrics) != n_rows:
            return {}
        combined_markers.extend(metrics)

    marker_arrays = {
        f: np.array([m.get(f, np.nan) for m in combined_markers], dtype=float)
        for f in NLP_MARKER_FIELDS
    }

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for unified, col_map in COMBINED_JUDGE_COLUMNS.items():
        pieces: List[np.ndarray] = []
        missing = False
        for method in COMBINED_METHODS:
            col = col_map[method]
            if col not in df.columns:
                missing = True
                break
            pieces.append(
                pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            )
        if missing:
            continue
        y = np.concatenate(pieces)
        out[unified] = {
            field: _corr_pair(x, y) for field, x in marker_arrays.items()
        }
    return out


def compute_correlations_per_alpha(
    df: pd.DataFrame,
    markers_by_col: Dict[str, List[dict]],
) -> Dict[float, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]:
    """Returns {alpha: {method: {judge_col: {nlp_marker: corr_stats}}}}.

    Rows in `df` are assumed to be alpha-sorted (as produced by
    `_reorder_df_alpha_sorted`), and `markers_by_col[col]` is assumed to be
    aligned row-for-row with `df`.
    """
    if "alpha" not in df.columns or len(df) == 0:
        return {}

    alpha_vals = sorted(df["alpha"].unique())
    df_reset = df.reset_index(drop=True)
    out: Dict[float, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}

    for a in alpha_vals:
        mask = df_reset["alpha"].apply(lambda x, a=a: abs(x - a) < 1e-3).to_numpy()
        idxs = np.where(mask)[0].tolist()
        if not idxs:
            continue
        sub_df = df_reset.iloc[idxs].reset_index(drop=True)
        sub_markers: Dict[str, List[dict]] = {}
        for col, mlist in markers_by_col.items():
            if len(mlist) != len(df_reset):
                continue
            sub_markers[col] = [mlist[i] for i in idxs]
        out[float(a)] = compute_correlations(sub_df, sub_markers)
    return out


def aggregate_rows(
    per_role: Dict[str, Tuple[pd.DataFrame, Dict[str, List[dict]]]],
) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    """Concat every role's rows + markers into single global arrays."""
    dfs = []
    combined: Dict[str, List[dict]] = defaultdict(list)
    for role, (df, markers) in per_role.items():
        if len(df) == 0:
            continue
        df_copy = df.copy()
        df_copy["_role"] = role
        dfs.append(df_copy)
        for col, mlist in markers.items():
            combined[col].extend(mlist)
    if not dfs:
        return pd.DataFrame(), {}
    global_df = pd.concat(dfs, ignore_index=True)
    return global_df, dict(combined)


# =============================================================================
# REPORT
# =============================================================================


def _fmt_stat(stats_dict: Dict[str, float]) -> str:
    r = stats_dict.get("pearson_r", float("nan"))
    p = stats_dict.get("pearson_p", float("nan"))
    sr = stats_dict.get("spearman_r", float("nan"))
    if np.isnan(r):
        return "n/a"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{r:+.3f}{sig} (ρ={sr:+.3f})"


def generate_per_alpha_markdown(
    per_alpha: Dict[float, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]],
    metric: str = "pearson_r",
) -> str:
    """One table per (method, judge_col): rows=NLP markers, columns=alphas."""
    if not per_alpha:
        return ""

    alpha_keys = sorted(per_alpha.keys())
    # Build set of (method, judge_col) pairs present anywhere.
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for a in alpha_keys:
        for method, by_judge in per_alpha[a].items():
            for jc in by_judge.keys():
                key = (method, jc)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)

    nice_metric = {"pearson_r": "Pearson r", "spearman_r": "Spearman ρ"}.get(
        metric, metric
    )
    lines: List[str] = []
    lines.append(f"\n## Per-alpha correlation tables ({nice_metric})\n")
    lines.append(
        "Each cell is the correlation between the NLP marker (row) and the "
        "judge score column (table) at the given alpha (column). "
        "Significance stars: `*` p<0.05, `**` p<0.01, `***` p<0.001.\n"
    )

    for method, jc in pairs:
        lines.append(f"### `{method}` — judge `{jc}`\n")
        header = (
            "| NLP marker | "
            + " | ".join(f"α={a:.1f}" for a in alpha_keys)
            + " |"
        )
        sep = "|" + "|".join(["---"] * (len(alpha_keys) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for field in NLP_MARKER_FIELDS:
            cells = [field]
            for a in alpha_keys:
                st = (
                    per_alpha.get(a, {})
                    .get(method, {})
                    .get(jc, {})
                    .get(field, {})
                )
                r = st.get(metric, float("nan"))
                p = st.get(
                    "pearson_p" if metric == "pearson_r" else "spearman_p",
                    float("nan"),
                )
                if np.isnan(r):
                    cells.append("n/a")
                    continue
                sig = (
                    "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )
                cells.append(f"{r:+.3f}{sig}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def generate_markdown(
    correlations: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    meta: Dict,
) -> str:
    lines: List[str] = []
    lines.append("# Judge Score ↔ NLP Marker Correlations\n")
    lines.append(f"- **Roles**: {meta.get('n_roles', 1)}")
    lines.append(f"- **Total responses**: {meta.get('n_responses', 0)}")
    lines.append(f"- **Filters**: {meta.get('filters', 'None')}")
    cache_hits = meta.get("cache_hits")
    if cache_hits is not None:
        lines.append(
            f"- **Cache hits**: {cache_hits}/{meta.get('n_roles', 1)} "
            f"(recomputed: {meta.get('n_roles', 1) - cache_hits})"
        )
    lines.append("")
    lines.append("Entries show Pearson r (with significance stars) "
                 "and Spearman ρ in parentheses.  "
                 "`*` p<0.05, `**` p<0.01, `***` p<0.001.\n")

    for method, by_judge in correlations.items():
        lines.append(f"## Method: `{method}`\n")
        judge_cols = list(by_judge.keys())
        if not judge_cols:
            lines.append("_No judge columns available._\n")
            continue
        header = "| NLP marker | " + " | ".join(judge_cols) + " |"
        sep = "|" + "|".join(["---"] * (len(judge_cols) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for field in NLP_MARKER_FIELDS:
            cells = [field]
            for jc in judge_cols:
                st = by_judge[jc].get(field, {})
                cells.append(_fmt_stat(st))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # Top absolute correlations across all judge cols.
        flat: List[Tuple[float, float, str, str]] = []
        for jc, marker_map in by_judge.items():
            for field, st in marker_map.items():
                r = st.get("pearson_r", float("nan"))
                p = st.get("pearson_p", float("nan"))
                if not np.isnan(r):
                    flat.append((r, p, field, jc))
        flat.sort(key=lambda t: abs(t[0]), reverse=True)
        if flat:
            lines.append(f"### Top correlations (`{method}`)\n")
            lines.append("| Rank | Marker | Judge column | Pearson r | p |")
            lines.append("|---|---|---|---|---|")
            for i, (r, p, field, jc) in enumerate(flat[:10], start=1):
                sig = (
                    "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )
                lines.append(f"| {i} | {field} | {jc} | {r:+.3f}{sig} | {p:.3g} |")
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# VISUALIZATION (optional)
# =============================================================================


def generate_heatmaps(
    correlations: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    out_dir: Path,
    metric: str = "pearson_r",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping heatmaps")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = HEATMAP_COHERENCE_FIELDS + HEATMAP_STYLE_FIELDS
    divider = len(HEATMAP_COHERENCE_FIELDS)
    for method, by_judge in correlations.items():
        judge_cols = list(by_judge.keys())
        if not judge_cols:
            continue
        matrix = np.full((len(fields), len(judge_cols)), np.nan)
        for j, jc in enumerate(judge_cols):
            for i, field in enumerate(fields):
                matrix[i, j] = by_judge[jc].get(field, {}).get(metric, np.nan)
        fig, ax = plt.subplots(
            figsize=(max(6, 0.9 * len(judge_cols) + 4),
                     0.35 * len(fields) + 2)
        )
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(judge_cols)))
        ax.set_xticklabels(judge_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(fields)))
        ax.set_yticklabels(fields)
        ax.axhline(divider - 0.5, color="black", linewidth=1.2)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(v) > 0.5 else "black")
        ax.set_title(f"{method}: {metric}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{method}_{metric}.png", dpi=150)
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Correlate judge scores with NLP markers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=str,
        default="experiment_data/gold_prompt_experiments",
        help="CSV file or folder of CSVs (default: gold_prompt_experiments)",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="./analysis/empirical/correlation/figures/",
        help="Output directory",
    )
    parser.add_argument(
        "--nlp-cache", type=str,
        default="./analysis/empirical/nlp_experiments/figures/per_role",
        help="Path to nlp per-role cache (used when available)",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Ignore cached NLP markers and recompute from scratch",
    )
    parser.add_argument(
        "--no-spacy", action="store_true",
        help="Disable spaCy (only matters when recomputing)",
    )
    parser.add_argument(
        "--spacy-model", type=str, default="en_core_web_sm",
        help="spaCy model to use when recomputing",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Generate correlation heatmap PNGs",
    )
    parser.add_argument("--role", type=str, nargs="+", default=None)
    parser.add_argument("--layer", type=int, nargs="+", default=[16])
    parser.add_argument("--sample-count", type=int, nargs="+", default=[50])
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=None,
        help="Filter by alpha (default: all of 1.0 1.5 2.0 2.5)",
    )
    parser.add_argument("--temperature", type=float, nargs="+", default=None)
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else [1.0, 1.5, 2.0, 2.5]
    filters = Filters(
        role=args.role,
        layer=args.layer,
        sample_count=args.sample_count,
        alpha=alpha,
        temperature=args.temperature,
    )

    use_spacy = False
    if not args.no_spacy:
        init_spacy(args.spacy_model)
        use_spacy = nlp_mod.SPACY_AVAILABLE

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.nlp_cache) if args.nlp_cache else None
    if cache_dir is not None and not cache_dir.exists():
        print(f"NLP cache dir {cache_dir} not found; will recompute markers.")
        cache_dir = None

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
    else:
        csv_files = [input_path]
    print(f"Inputs: {len(csv_files)} CSV file(s) from {input_path}")
    print(f"Filters: {filters.describe()}")
    print(f"spaCy (for recompute): {'enabled' if use_spacy else 'disabled'}")
    print(f"NLP cache: {cache_dir or '(none)'}\n")

    per_role: Dict[str, Tuple[pd.DataFrame, Dict[str, List[dict]]]] = {}
    per_role_correlations: Dict[str, Dict] = {}
    cache_hits = 0
    total_rows = 0

    for fp in tqdm(csv_files, desc="Loading markers"):
        df, markers, role, from_cache = load_role_markers(
            fp, cache_dir, RESPONSE_COLUMNS, filters, use_spacy, args.recompute,
        )
        if len(df) == 0:
            continue
        per_role[role] = (df, markers)
        cache_hits += int(from_cache)
        total_rows += len(df)
        per_role_correlations[role] = compute_correlations(df, markers)

    if not per_role:
        print("No data after filtering; exiting.")
        return

    global_df, global_markers = aggregate_rows(per_role)
    aggregated_correlations = compute_correlations(global_df, global_markers)
    aggregated_per_alpha = compute_correlations_per_alpha(global_df, global_markers)

    combined = compute_combined_correlations(global_df, global_markers)
    if combined:
        aggregated_correlations["assistant_axis+steered"] = combined

    meta = {
        "input": str(input_path),
        "n_roles": len(per_role),
        "n_responses": total_rows,
        "filters": filters.describe(),
        "cache_hits": cache_hits,
        "recomputed": not args.recompute and cache_dir is not None
                      and cache_hits < len(per_role),
    }

    result = {
        "meta": meta,
        "aggregated": aggregated_correlations,
        "aggregated_per_alpha": {str(a): v for a, v in aggregated_per_alpha.items()},
        "per_role": per_role_correlations,
    }
    json_path = output_dir / "correlations.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    report = generate_markdown(aggregated_correlations, meta)
    report += generate_per_alpha_markdown(aggregated_per_alpha, metric="pearson_r")
    report += generate_per_alpha_markdown(aggregated_per_alpha, metric="spearman_r")
    md_path = output_dir / "report.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Saved: {md_path}")

    if args.visualize:
        heat_dir = output_dir / "heatmaps"
        generate_heatmaps(aggregated_correlations, heat_dir, metric="pearson_r")
        generate_heatmaps(aggregated_correlations, heat_dir, metric="spearman_r")
        print(f"Saved heatmaps: {heat_dir}")

    print("\n" + "=" * 60)
    print(report[:2000])
    if len(report) > 2000:
        print("\n... (see full report in file)")


if __name__ == "__main__":
    main()
