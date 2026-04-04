"""
Backfill missing data in gold_prompt_experiments CSVs using recovered_csvs.

For each file in gold_prompt_experiments, if a parallel file exists in recovered_csvs,
match rows by key (role, layer, sample_count, alpha, temperature, question) and fill in
any field that is null or -1 with the corresponding value from the recovered file
(if that value is itself not null or -1).
"""

import os
import pandas as pd
import numpy as np

GOLD_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_data", "gold_prompt_experiments")
RECOVERED_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_data", "recovered_csvs")

KEY_COLS = ["role", "layer", "sample_count", "alpha", "temperature", "question"]


def is_missing(series: pd.Series) -> pd.Series:
    """Return mask where values are null or -1."""
    return series.isna() | (series == -1)


def backfill_file(gold_path: str, recovered_path: str) -> int:
    gold = pd.read_csv(gold_path)
    recovered = pd.read_csv(recovered_path)

    value_cols = [c for c in gold.columns if c not in KEY_COLS]

    # Merge on key columns
    merged = gold.merge(recovered, on=KEY_COLS, how="left", suffixes=("", "_rec"))

    total_filled = 0
    for col in value_cols:
        rec_col = f"{col}_rec"
        if rec_col not in merged.columns:
            continue

        missing_mask = is_missing(merged[col])
        has_recovery = ~is_missing(merged[rec_col])
        fill_mask = missing_mask & has_recovery

        count = fill_mask.sum()
        if count > 0:
            merged.loc[fill_mask, col] = merged.loc[fill_mask, rec_col]
            total_filled += count

    # Drop recovery columns and save back
    rec_cols = [c for c in merged.columns if c.endswith("_rec")]
    merged.drop(columns=rec_cols, inplace=True)

    if total_filled > 0:
        merged.to_csv(gold_path, index=False)

    return total_filled


def main():
    gold_files = sorted(os.listdir(GOLD_DIR))
    total_updated_files = 0
    total_filled_values = 0
    
    print(len(gold_files))

    for fname in gold_files:
        if not fname.endswith(".csv"):
            continue

        gold_path = os.path.join(GOLD_DIR, fname)
        recovered_path = os.path.join(RECOVERED_DIR, fname)

        if not os.path.exists(recovered_path):
            continue

        filled = backfill_file(gold_path, recovered_path)
        if filled > 0:
            total_updated_files += 1
            total_filled_values += filled
            print(f"  {fname}: filled {filled} values")

    print(f"\nDone. Updated {total_updated_files} files, filled {total_filled_values} total values.")


if __name__ == "__main__":
    main()
