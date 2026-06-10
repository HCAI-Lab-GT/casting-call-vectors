"""
Re-plot the alpha-vs-score figures using *effective steering norm*
||delta|| = alpha * ||v|| on the x-axis instead of alpha.

Why: the proposed (steered) and assistant_axis methods both apply
delta = alpha * v, but their vectors have very different norms
(proposed median ||v|| ~3.78, assistant_axis median ~9.70). At fixed
alpha the assistant_axis is being driven ~2.55x harder, which makes
the standard alpha-axis comparison misleading. Plotting against
effective norm lets the existing data answer "how do they compare at
the same actual residual-stream perturbation magnitude?" without
re-running the experiments.

Inputs (no GPU, no model — just CSVs and stored vectors):
  experiment_data/gold_prompt_experiments/Comparison_GoldStandard_*.csv
  persona_data/pt_vectors/{role}.pt                      [proposed]
  persona_data/assistant-axis/olmo-3-7b-instruct/vectors/{role}.pt  [axis]

Outputs (analysis/empirical/judge/figures/effective_norm/):
  figure1_effnorm_vs_score.{pdf,png}
  figure1_effnorm_vs_score_with_alpha_ticks.{pdf,png}
  figure4_effnorm_dimension_comparison.{pdf,png}
  per_role_norms.csv
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch

PROPOSED_DIR = "persona_data/pt_vectors"
AA_DIR = "persona_data/assistant-axis/olmo-3-7b-instruct/vectors"
DEFAULT_INPUT = "experiment_data/gold_prompt_experiments"
DEFAULT_OUTPUT = "analysis/empirical/judge/figures/effective_norm"
DEFAULT_NLP_DIR = "analysis/empirical/nlp_experiments/figures/per_role"

ALPHAS = [1.0, 1.5, 2.0, 2.5]

# NLP metrics worth plotting (mean fields ending in _mean from per_role JSONs)
NLP_METRICS = [
    ("compression_ratio_mean", "Compression Ratio"),
    ("bleu_mean", "BLEU vs baseline"),
    ("first_person_rate_mean", "First-Person Rate (%)"),
    ("unique_bigram_ratio_mean", "Unique Bigram Ratio"),
    ("hedge_rate_mean", "Hedge Rate (%)"),
    ("assertive_rate_mean", "Assertive Rate (%)"),
    ("question_rate_mean", "Question Rate (%)"),
    ("word_count_mean", "Word Count"),
    ("epistemic_total_mean", "Epistemic Markers"),
    ("role_consistency_mean", "Role Consistency"),
    ("ai_phrase_rate_mean", "AI Phrase Rate (%)"),
]

LLM_DIM_PANEL = [
    ("cmp_emotional_register", "assistant_axis_cmp_emotional_register",
     "Emotional Register"),
    ("cmp_vocab_choice", "assistant_axis_cmp_vocab_choice",
     "Vocab Choice"),
    ("cmp_social_dynamic", "assistant_axis_cmp_social_dynamic",
     "Social Dynamic"),
    ("cmp_motivation", "assistant_axis_cmp_motivation",
     "Motivation"),
    ("cmp_worldview_alignment", "assistant_axis_cmp_worldview_alignment",
     "Worldview Alignment"),
]

STEERED_COLOR = "#2ecc71"
AXIS_COLOR = "#3498db"
BASELINE_COLOR = "#e74c3c"

STYLE_DIMS = ["cmp_emotional_register", "cmp_vocab_choice", "cmp_social_dynamic"]
CONTENT_DIMS = ["cmp_motivation", "cmp_worldview_alignment"]
AXIS_STYLE_DIMS = ["assistant_axis_cmp_emotional_register",
                   "assistant_axis_cmp_vocab_choice",
                   "assistant_axis_cmp_social_dynamic"]
AXIS_CONTENT_DIMS = ["assistant_axis_cmp_motivation",
                     "assistant_axis_cmp_worldview_alignment"]
DIM_LABELS = ["Emotional\nRegister", "Vocab\nChoice", "Social\nDynamic",
              "Motivation", "Worldview\nAlignment"]

SCORE_COLS = [
    "steered_score", "assistant_axis_score", "baseline_score",
    *STYLE_DIMS, *CONTENT_DIMS, *AXIS_STYLE_DIMS, *AXIS_CONTENT_DIMS,
]


# ── data helpers ──────────────────────────────────────────────────────────────
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


USE_COLS = ["role", "alpha", "sample_count"] + SCORE_COLS


def load_results(input_dir: str) -> pd.DataFrame:
    """Stream CSVs reading only the score / metadata columns, drop the multi-MB
    text columns, filter to sample_count=50 per file, then concat."""
    files = sorted(glob.glob(f"{input_dir}/Comparison_GoldStandard_*.csv"))
    dfs = []
    for f in files:
        try:
            # peek header to keep only intersecting columns (older CSVs may
            # be missing some assistant_axis_cmp_* fields)
            header = pd.read_csv(f, nrows=0).columns
            cols = [c for c in USE_COLS if c in header]
            df = pd.read_csv(f, usecols=cols)
            if df.empty:
                continue
            df = df[df["sample_count"] == 50]
            if df.empty:
                continue
            for col in SCORE_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(_parse_score).astype("float32")
            dfs.append(df)
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
    combined = pd.concat(dfs, ignore_index=True)
    combined["steered_style"] = combined[STYLE_DIMS].mean(axis=1)
    combined["steered_content"] = combined[CONTENT_DIMS].mean(axis=1)
    combined["axis_style"] = combined[AXIS_STYLE_DIMS].mean(axis=1)
    combined["axis_content"] = combined[AXIS_CONTENT_DIMS].mean(axis=1)
    return combined


def _vec_norm(t: torch.Tensor) -> float:
    """Normalize torch payload shape and return the layer-16 vector's L2 norm."""
    t = t.detach().float()
    if t.ndim == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 2:
        idx = 15 if t.shape[0] > 15 else t.shape[0] - 1
        t = t[idx]
    return float(torch.linalg.vector_norm(t).item())


def load_norms(directory: str) -> dict[str, float]:
    out = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".pt"):
            continue
        role = fname[:-3]
        try:
            payload = torch.load(os.path.join(directory, fname),
                                 map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                tensor = None
                for k in ("vector", "persona_vector", "delta", "axis"):
                    if k in payload:
                        tensor = payload[k]; break
                if tensor is None:
                    for v in payload.values():
                        if torch.is_tensor(v):
                            tensor = v; break
                if tensor is None:
                    continue
            else:
                tensor = payload
            out[role] = _vec_norm(tensor)
        except Exception as e:
            print(f"  skip {directory}/{fname}: {e}")
    return out


def attach_effective_norms(df: pd.DataFrame,
                           steered_norms: dict[str, float],
                           axis_norms: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["steered_vec_norm"] = df["role"].map(steered_norms)
    df["axis_vec_norm"] = df["role"].map(axis_norms)
    df["steered_eff_norm"] = df["alpha"] * df["steered_vec_norm"]
    df["axis_eff_norm"] = df["alpha"] * df["axis_vec_norm"]
    return df


# ── plots ─────────────────────────────────────────────────────────────────────
def figure1_effnorm_vs_score(df: pd.DataFrame, out_dir: Path,
                             show_alpha_ticks: bool = False) -> None:
    """Both methods on a shared effective-norm x-axis."""
    rows = []
    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha]
        rows.append({
            "alpha": alpha,
            "steered_eff_norm": sub["steered_eff_norm"].mean(),
            "axis_eff_norm": sub["axis_eff_norm"].mean(),
            "steered_score": sub["steered_score"].mean(),
            "axis_score": sub["assistant_axis_score"].mean(),
            "baseline_score": sub["baseline_score"].mean(),
            "steered_score_std": sub["steered_score"].std(),
            "axis_score_std": sub["assistant_axis_score"].std(),
        })
    g = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # steered band
    ax.fill_between(
        g["steered_eff_norm"],
        g["steered_score"] - g["steered_score_std"],
        g["steered_score"] + g["steered_score_std"],
        color=STEERED_COLOR, alpha=0.08,
    )
    ax.fill_between(
        g["axis_eff_norm"],
        g["axis_score"] - g["axis_score_std"],
        g["axis_score"] + g["axis_score_std"],
        color=AXIS_COLOR, alpha=0.08,
    )
    ax.plot(g["steered_eff_norm"], g["steered_score"], marker="o",
            color=STEERED_COLOR, linewidth=2, label="Steered (proposed)")
    ax.plot(g["axis_eff_norm"], g["axis_score"], marker="s",
            color=AXIS_COLOR, linewidth=2, linestyle="--",
            label="Assistant axis")

    # baseline is alpha-independent; draw a flat line at its mean across alphas
    ax.axhline(g["baseline_score"].mean(), color=BASELINE_COLOR,
               linewidth=1.5, linestyle=":", label="Baseline")

    if show_alpha_ticks:
        for _, row in g.iterrows():
            ax.annotate(rf"$\alpha={row['alpha']}$",
                        xy=(row["steered_eff_norm"], row["steered_score"]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=STEERED_COLOR)
            ax.annotate(rf"$\alpha={row['alpha']}$",
                        xy=(row["axis_eff_norm"], row["axis_score"]),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color=AXIS_COLOR)

    ax.set_xlabel(r"Effective steering norm  $\|\Delta\|_2 = \alpha \cdot \|v\|_2$")
    ax.set_ylabel("Mean role alignment score (0–100)")
    ax.set_title("Role Alignment vs. Effective Steering Norm")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    suffix = "_with_alpha_ticks" if show_alpha_ticks else ""
    path = out_dir / f"figure1_effnorm_vs_score{suffix}.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=200)
    plt.close()
    print(f"Saved: {path}")


def figure4_effnorm_dimension_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """5-dim grouped bar at each alpha, with eff-norm annotated below the title."""
    cmp_df = df.dropna(subset=["assistant_axis_cmp_emotional_register"])
    if cmp_df.empty:
        print("Figure 4 skipped: no assistant_axis_cmp data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(DIM_LABELS))
    width = 0.35

    for i, alpha in enumerate(ALPHAS):
        ax = axes[i]
        sub = cmp_df[cmp_df["alpha"] == alpha]

        steered_means = [sub[c].mean() for c in STYLE_DIMS + CONTENT_DIMS]
        axis_means = [sub[c].mean() for c in AXIS_STYLE_DIMS + AXIS_CONTENT_DIMS]

        ax.bar(x - width / 2, steered_means, width,
               color=STEERED_COLOR, label="Steered")
        ax.bar(x + width / 2, axis_means, width,
               color=AXIS_COLOR, label="Assistant axis")

        steered_eff = sub["steered_eff_norm"].mean()
        axis_eff = sub["axis_eff_norm"].mean()
        ax.set_title(
            rf"$\alpha = {alpha}$"
            "\n"
            rf"$\|\Delta\|$:  steered $\approx$ {steered_eff:.1f}, "
            rf"axis $\approx$ {axis_eff:.1f}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS, fontsize=9)
        ax.set_ylim(0, 100)
        ax.axhline(50, color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Per-dimension Alignment by Method (effective norm in subtitle)",
                 y=1.01)
    plt.tight_layout()
    path = out_dir / "figure4_effnorm_dimension_comparison.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=200)
    plt.close()
    print(f"Saved: {path}")


def figure5_per_role_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-row scatter: x = effective norm, y = score, colored by method.
    Shows the actual distribution rather than just the means."""
    fig, ax = plt.subplots(figsize=(8, 5))
    s = df.dropna(subset=["steered_eff_norm", "steered_score"])
    a = df.dropna(subset=["axis_eff_norm", "assistant_axis_score"])

    ax.scatter(s["steered_eff_norm"], s["steered_score"], s=8, alpha=0.25,
               color=STEERED_COLOR, label="Steered (per-role)")
    ax.scatter(a["axis_eff_norm"], a["assistant_axis_score"], s=8, alpha=0.25,
               color=AXIS_COLOR, label="Assistant axis (per-role)")

    # overlay alpha-grouped means
    means = []
    for alpha in ALPHAS:
        sub_s = s[s["alpha"] == alpha]
        sub_a = a[a["alpha"] == alpha]
        means.append((alpha,
                      sub_s["steered_eff_norm"].mean(), sub_s["steered_score"].mean(),
                      sub_a["axis_eff_norm"].mean(), sub_a["assistant_axis_score"].mean()))
    means = pd.DataFrame(means,
        columns=["alpha", "s_x", "s_y", "a_x", "a_y"])
    ax.plot(means["s_x"], means["s_y"], "-o", color="darkgreen", linewidth=2,
            markersize=7, label=r"Steered, mean per $\alpha$")
    ax.plot(means["a_x"], means["a_y"], "-s", color="darkblue", linewidth=2,
            markersize=7, label=r"Axis, mean per $\alpha$")
    for _, row in means.iterrows():
        ax.annotate(rf"$\alpha={row['alpha']}$", (row["s_x"], row["s_y"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="darkgreen")
        ax.annotate(rf"$\alpha={row['alpha']}$", (row["a_x"], row["a_y"]),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8, color="darkblue")

    ax.set_xlabel(r"Effective steering norm  $\|\Delta\|_2 = \alpha \cdot \|v\|_2$")
    ax.set_ylabel("Role alignment score (0–100)")
    ax.set_title("Score vs. Effective Steering Norm (per-role + means)")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    path = out_dir / "figure5_effnorm_per_role_scatter.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=200)
    plt.close()
    print(f"Saved: {path}")


def load_nlp_per_role(nlp_dir: str) -> pd.DataFrame:
    """Read every analysis/empirical/nlp_experiments/figures/per_role/{role}.json
    into a long table with columns: role, alpha, method, <metric>, ..."""
    import json
    rows = []
    for fname in sorted(os.listdir(nlp_dir)):
        if not fname.endswith(".json"):
            continue
        role = fname[:-5]
        with open(os.path.join(nlp_dir, fname)) as f:
            data = json.load(f)
        per_alpha = data.get("per_alpha", {})
        for alpha_str, payload in per_alpha.items():
            try:
                alpha = float(alpha_str)
            except ValueError:
                continue
            for method, summary in (payload.get("summaries") or {}).items():
                if not isinstance(summary, dict):
                    continue
                row = {"role": role, "alpha": alpha, "method": method}
                for metric, _label in NLP_METRICS:
                    v = summary.get(metric)
                    row[metric] = float(v) if isinstance(v, (int, float)) else float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def attach_eff_norm_to_nlp(df: pd.DataFrame,
                           steered_norms: dict[str, float],
                           axis_norms: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    norm_lookup = {
        "steered": steered_norms,
        "assistant_axis": axis_norms,
    }
    eff = []
    for _, r in df.iterrows():
        nm = norm_lookup.get(r["method"], {}).get(r["role"])
        eff.append(r["alpha"] * nm if nm is not None else float("nan"))
    df["eff_norm"] = eff
    return df


def figure_nlp_panel(df: pd.DataFrame, out_dir: Path) -> None:
    """One panel per NLP metric. Both methods plotted at their actual eff_norm,
    plus a baseline horizontal line (alpha-independent)."""
    n_metrics = len(NLP_METRICS)
    cols = 3
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, NLP_METRICS):
        for method, color, marker, ls, lbl in [
            ("steered", STEERED_COLOR, "o", "-", "Steered (proposed)"),
            ("assistant_axis", AXIS_COLOR, "s", "--", "Assistant axis"),
        ]:
            sub = df[df["method"] == method].dropna(subset=[metric, "eff_norm"])
            if sub.empty:
                continue
            agg = sub.groupby("alpha").agg(
                eff_norm=("eff_norm", "mean"),
                metric_mean=(metric, "mean"),
                metric_std=(metric, "std"),
            ).reset_index().sort_values("eff_norm")
            ax.fill_between(agg["eff_norm"],
                            agg["metric_mean"] - agg["metric_std"],
                            agg["metric_mean"] + agg["metric_std"],
                            color=color, alpha=0.07)
            ax.plot(agg["eff_norm"], agg["metric_mean"], marker=marker,
                    color=color, linewidth=1.7, linestyle=ls, label=lbl)
        baseline = df[df["method"] == "baseline"][metric].mean()
        if pd.notna(baseline):
            ax.axhline(baseline, color=BASELINE_COLOR, linewidth=1.2,
                       linestyle=":", label=f"Baseline ({baseline:.2f})")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r"$\|\Delta\|_2$", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    for j in range(len(NLP_METRICS), len(axes)):
        axes[j].axis("off")
    fig.legend(handles, labels, loc="lower right",
               bbox_to_anchor=(0.99, 0.02), fontsize=10, frameon=True)
    fig.suptitle(
        r"NLP metrics vs. Effective Steering Norm  $\|\Delta\|_2 = \alpha \cdot \|v\|_2$",
        y=1.0, fontsize=13,
    )
    plt.tight_layout()
    path = out_dir / "nlp_metrics_effnorm_panel.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=200)
    plt.close()
    print(f"Saved: {path}")


def figure_llm_dim_panel(df: pd.DataFrame, out_dir: Path) -> None:
    """5-cell grid: each LLM-judge cmp_* dimension vs effective norm, both
    methods. Uses the per-row gold_prompt CSV data attached with eff norms."""
    cmp_df = df.dropna(subset=["assistant_axis_cmp_emotional_register"])
    if cmp_df.empty:
        print("LLM dim panel skipped: no assistant_axis_cmp data")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (st_col, ax_col, label) in zip(axes, LLM_DIM_PANEL):
        rows = []
        for alpha in ALPHAS:
            sub = cmp_df[cmp_df["alpha"] == alpha]
            rows.append({
                "alpha": alpha,
                "s_eff": sub["steered_eff_norm"].mean(),
                "s_mean": sub[st_col].mean(),
                "s_std": sub[st_col].std(),
                "a_eff": sub["axis_eff_norm"].mean(),
                "a_mean": sub[ax_col].mean(),
                "a_std": sub[ax_col].std(),
            })
        g = pd.DataFrame(rows).sort_values("s_eff")
        ax.fill_between(g["s_eff"], g["s_mean"] - g["s_std"],
                        g["s_mean"] + g["s_std"], color=STEERED_COLOR, alpha=0.08)
        ax.fill_between(g["a_eff"], g["a_mean"] - g["a_std"],
                        g["a_mean"] + g["a_std"], color=AXIS_COLOR, alpha=0.08)
        ax.plot(g["s_eff"], g["s_mean"], "-o", color=STEERED_COLOR,
                linewidth=2, label="Steered (proposed)")
        ax.plot(g["a_eff"], g["a_mean"], "--s", color=AXIS_COLOR,
                linewidth=2, label="Assistant axis")
        ax.axhline(50, color="lightgray", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(r"$\|\Delta\|_2$")
        ax.set_ylabel("Score (0–100)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right",
               bbox_to_anchor=(0.99, 0.05), fontsize=10, frameon=True)
    fig.suptitle(
        r"LLM-judge dimensions vs. Effective Steering Norm  $\|\Delta\|_2 = \alpha \cdot \|v\|_2$",
        y=1.01, fontsize=13,
    )
    plt.tight_layout()
    path = out_dir / "llm_dimensions_effnorm_panel.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=200)
    plt.close()
    print(f"Saved: {path}")


def figure_llm_aggregates(df: pd.DataFrame, out_dir: Path) -> None:
    """Three panels: overall score, style aggregate, content aggregate. All
    three plotted vs effective norm for both methods."""
    cmp_df = df.dropna(subset=["assistant_axis_cmp_emotional_register"])
    if cmp_df.empty:
        print("LLM aggregates skipped: no assistant_axis_cmp data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panels = [
        (axes[0], "Overall Role Alignment",
         "steered_score", "assistant_axis_score"),
        (axes[1], "Style aggregate (3 dims)",
         "steered_style", "axis_style"),
        (axes[2], "Content aggregate (2 dims)",
         "steered_content", "axis_content"),
    ]

    for ax, title, st_col, ax_col in panels:
        rows = []
        for alpha in ALPHAS:
            sub = cmp_df[cmp_df["alpha"] == alpha]
            rows.append({
                "alpha": alpha,
                "s_eff": sub["steered_eff_norm"].mean(),
                "s_mean": sub[st_col].mean(),
                "s_std": sub[st_col].std(),
                "a_eff": sub["axis_eff_norm"].mean(),
                "a_mean": sub[ax_col].mean(),
                "a_std": sub[ax_col].std(),
            })
        g = pd.DataFrame(rows).sort_values("s_eff")
        ax.fill_between(g["s_eff"], g["s_mean"] - g["s_std"],
                        g["s_mean"] + g["s_std"], color=STEERED_COLOR, alpha=0.08)
        ax.fill_between(g["a_eff"], g["a_mean"] - g["a_std"],
                        g["a_mean"] + g["a_std"], color=AXIS_COLOR, alpha=0.08)
        ax.plot(g["s_eff"], g["s_mean"], "-o", color=STEERED_COLOR,
                linewidth=2, label="Steered (proposed)")
        ax.plot(g["a_eff"], g["a_mean"], "--s", color=AXIS_COLOR,
                linewidth=2, label="Assistant axis")
        baseline_col = {
            "steered_score": "baseline_score",
        }.get(st_col)
        if baseline_col and baseline_col in cmp_df.columns:
            ax.axhline(cmp_df[baseline_col].mean(), color=BASELINE_COLOR,
                       linewidth=1.4, linestyle=":", label="Baseline")
        ax.axhline(50, color="lightgray", linewidth=0.5, linestyle="--", zorder=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$\|\Delta\|_2$")
        ax.set_ylabel("Score (0–100)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        r"LLM-judge aggregates vs. Effective Steering Norm  $\|\Delta\|_2 = \alpha \cdot \|v\|_2$",
        y=1.02, fontsize=13,
    )
    plt.tight_layout()
    path = out_dir / "llm_aggregates_effnorm.pdf"
    plt.savefig(path)
    plt.savefig(str(path).replace(".pdf", ".png"), dpi=200)
    plt.close()
    print(f"Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--proposed_dir", default=PROPOSED_DIR)
    parser.add_argument("--axis_dir", default=AA_DIR)
    parser.add_argument("--nlp_dir", default=DEFAULT_NLP_DIR)
    parser.add_argument("--skip_nlp", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading norms ...")
    steered_norms = load_norms(args.proposed_dir)
    axis_norms = load_norms(args.axis_dir)
    print(f"  proposed: n={len(steered_norms)}, "
          f"median ||v|| = {np.median(list(steered_norms.values())):.3f}")
    print(f"  axis:     n={len(axis_norms)}, "
          f"median ||v|| = {np.median(list(axis_norms.values())):.3f}")

    norm_df = pd.DataFrame({
        "role": sorted(set(steered_norms) | set(axis_norms)),
    })
    norm_df["steered_vec_norm"] = norm_df["role"].map(steered_norms)
    norm_df["axis_vec_norm"] = norm_df["role"].map(axis_norms)
    norm_df.to_csv(out_dir / "per_role_norms.csv", index=False)
    print(f"Wrote {out_dir / 'per_role_norms.csv'}")

    print("Loading experiment results ...")
    df = load_results(args.input_dir)
    df = attach_effective_norms(df, steered_norms, axis_norms)
    missing_steered = df["steered_vec_norm"].isna().sum()
    missing_axis = df["axis_vec_norm"].isna().sum()
    if missing_steered or missing_axis:
        print(f"  warning: {missing_steered} rows missing proposed norm, "
              f"{missing_axis} missing axis norm (these rows are dropped from plots)")

    figure1_effnorm_vs_score(df, out_dir, show_alpha_ticks=False)
    figure1_effnorm_vs_score(df, out_dir, show_alpha_ticks=True)
    figure4_effnorm_dimension_comparison(df, out_dir)
    figure5_per_role_scatter(df, out_dir)
    figure_llm_dim_panel(df, out_dir)
    figure_llm_aggregates(df, out_dir)

    if not args.skip_nlp and os.path.isdir(args.nlp_dir):
        print("Loading NLP per-role JSONs ...")
        nlp_df = load_nlp_per_role(args.nlp_dir)
        nlp_df = attach_eff_norm_to_nlp(nlp_df, steered_norms, axis_norms)
        print(f"  NLP rows: {len(nlp_df)} (roles={nlp_df['role'].nunique()})")
        figure_nlp_panel(nlp_df, out_dir)


if __name__ == "__main__":
    main()
