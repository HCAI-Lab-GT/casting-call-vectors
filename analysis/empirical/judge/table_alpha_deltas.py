#!/usr/bin/env python3
"""
Per-dimension score + delta table across increasing alphas.

Row groups: the overall role alignment score and each of the five judging
dimensions (emotional_register, vocab_choice, social_dynamic, motivation,
worldview_alignment). Within each group, there is one row for the steered
vector (ours) and one for the assistant_axis baseline vector.

The horizontal axis is grouped by alpha; within each alpha group the first
column is the raw mean score and the second is the successive delta from
the previous alpha (blank for the lowest alpha).

Usage:
    python analysis/empirical/judge/table_alpha_deltas.py \
        --input_dir ./experiment_data/gold_prompt_experiments/ \
        --output_dir ./analysis/empirical/judge/figures/
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

DIMENSIONS = [
    # (label, steered_col, assistant_axis_col)
    ("Overall Role Alignment", "steered_score", "assistant_axis_score"),
    ("Emotional Register", "cmp_emotional_register", "assistant_axis_cmp_emotional_register"),
    ("Vocab Choice", "cmp_vocab_choice", "assistant_axis_cmp_vocab_choice"),
    ("Social Dynamic", "cmp_social_dynamic", "assistant_axis_cmp_social_dynamic"),
    ("Motivation", "cmp_motivation", "assistant_axis_cmp_motivation"),
    ("Worldview Alignment", "cmp_worldview_alignment", "assistant_axis_cmp_worldview_alignment"),
]

VECTOR_ROWS = [
    # (label, which column index in DIMENSIONS: 1 = steered, 2 = assistant_axis)
    ("steered (ours)", 1),
    ("assistant_axis", 2),
]


def _parse_score(val):
    """Coerce a judge-score cell to float; handles serialized Series strings."""
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
        except Exception:
            continue
        if not df.empty:
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No Comparison_GoldStandard_*.csv files in {input_dir}")
    combined = pd.concat(dfs, ignore_index=True)

    score_cols = {c for dim in DIMENSIONS for c in dim[1:]}
    for col in score_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(_parse_score)

    if "alpha" in combined.columns:
        combined["alpha"] = pd.to_numeric(combined["alpha"], errors="coerce")
    return combined


def _fmt_alpha(alpha: float) -> str:
    return f"{alpha:g}"


def score_col(alpha: float) -> str:
    return f"α={_fmt_alpha(alpha)} score"


def delta_col(alpha: float) -> str:
    return f"α={_fmt_alpha(alpha)} Δ"


def build_delta_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    alphas = sorted(a for a in df["alpha"].dropna().unique())
    if len(alphas) < 2:
        raise ValueError(f"Need at least two alphas to compute deltas (found {alphas})")

    rows = []
    for dim in DIMENSIONS:
        dim_label = dim[0]
        for vector_label, col_idx in VECTOR_ROWS:
            col = dim[col_idx]
            if col not in df.columns:
                continue
            mean_by_alpha = df.groupby("alpha")[col].mean()
            row = {"dimension": dim_label, "vector": vector_label}
            prev_score = np.nan
            for i, alpha in enumerate(alphas):
                score = mean_by_alpha.get(alpha, np.nan)
                row[score_col(alpha)] = score
                row[delta_col(alpha)] = np.nan if i == 0 else score - prev_score
                prev_score = score
            rows.append(row)
    return pd.DataFrame(rows), alphas


def to_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    last_dim = None
    for _, row in table.iterrows():
        if last_dim is not None and row["dimension"] != last_dim:
            lines.append("| " + " | ".join([""] * len(cols)) + " |")
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if pd.isna(v):
                    cells.append("—")
                elif c.endswith(" Δ"):
                    cells.append(f"{v:+.2f}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
        last_dim = row["dimension"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments/")
    ap.add_argument("--output_dir", default="./analysis/empirical/judge/figures/")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input_dir)
    alphas = sorted(df["alpha"].dropna().unique())
    print(f"Loaded {len(df):,} rows | roles: {df['role'].nunique()} | alphas: {alphas}")

    table, _ = build_delta_table(df)
    csv_path = output_dir / "alpha_deltas_per_dimension.csv"
    md_path = output_dir / "alpha_deltas_per_dimension.md"
    table.to_csv(csv_path, index=False)
    md = to_markdown(table)
    md_path.write_text(md + "\n")
    print(md)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
