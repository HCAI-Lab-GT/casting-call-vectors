import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

def has_value(value):
    return not pd.isna(value) and str(value).strip() != ""

def is_malformed(value):
    if not has_value(value):
        return True
    text = str(value)
    if "dtype:" in text or "Name:" in text:
        return True
    return False

def is_duplicate_across_alphas(df, idx, col):
    # Check if the same response appears for the same question but different alpha
    row = df.loc[idx]
    question = row["question"]
    alpha = row["alpha"]
    value = str(row[col]).strip()
    if not has_value(value):
        return False
    # Find all rows with the same question but different alpha
    mask = (df["question"] == question) & (df["alpha"] != alpha)
    return any(str(v).strip() == value for v in df.loc[mask, col] if has_value(v))

def main():
    parser = argparse.ArgumentParser(description="Report malformed/missing steered and baseline responses by alpha.")
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--role", type=str, default=None, help="Role name (matches CSV suffix, e.g., 'aberration')")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.role:
        role_csv = f"Comparison_GoldStandard_{args.role.replace('/', '_')}.csv"
        csv_paths = [input_dir / role_csv] if (input_dir / role_csv).exists() else []
        if not csv_paths:
            print(f"No CSV found for role '{args.role}' in {input_dir}")
            return
    else:
        csv_paths = sorted(input_dir.glob("Comparison_GoldStandard_*.csv"))

    # Ensure output directory exists
    output_dir = Path("experiment_data/experiment_missing_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # Check for required columns
        required_cols = ["question", "alpha", "steered", "baseline", "assistant_axis"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Skipping {csv_path}: missing required columns: {missing_cols}")
            continue
        report = defaultdict(lambda: {
            "steered_missing": 0, "steered_malformed": 0, "steered_duplicate": 0,
            "baseline_missing": 0, "baseline_malformed": 0,
            "assistant_axis_missing": 0, "assistant_axis_malformed": 0, "assistant_axis_duplicate": 0,
            "total": 0
        })
        for idx, row in df.iterrows():
            alpha = row["alpha"]
            report[alpha]["total"] += 1
            # Steered
            steered = row["steered"]
            if not has_value(steered):
                report[alpha]["steered_missing"] += 1
            elif is_malformed(steered):
                report[alpha]["steered_malformed"] += 1
            elif is_duplicate_across_alphas(df, idx, "steered"):
                report[alpha]["steered_duplicate"] += 1
            # Baseline
            baseline = row["baseline"]
            if not has_value(baseline):
                report[alpha]["baseline_missing"] += 1
            elif is_malformed(baseline):
                report[alpha]["baseline_malformed"] += 1
            # Assistant Axis
            assistant_axis = row["assistant_axis"]
            if not has_value(assistant_axis):
                report[alpha]["assistant_axis_missing"] += 1
            elif is_malformed(assistant_axis):
                report[alpha]["assistant_axis_malformed"] += 1
            elif is_duplicate_across_alphas(df, idx, "assistant_axis"):
                report[alpha]["assistant_axis_duplicate"] += 1

        # Prepare table for CSV output
        import csv
        table_rows = []
        for alpha in sorted(report.keys()):
            stats = report[alpha]
            table_rows.append({
                "alpha": alpha,
                "total": stats["total"],
                "steered_missing": stats["steered_missing"],
                "steered_malformed": stats["steered_malformed"],
                "steered_duplicate": stats["steered_duplicate"],
                "baseline_missing": stats["baseline_missing"],
                "baseline_malformed": stats["baseline_malformed"],
                "assistant_axis_missing": stats["assistant_axis_missing"],
                "assistant_axis_malformed": stats["assistant_axis_malformed"],
                "assistant_axis_duplicate": stats["assistant_axis_duplicate"]
            })

        # Write CSV summary to experiment_data/experiment_missing_data
        role_name = csv_path.stem.replace("Comparison_GoldStandard_", "")
        out_csv = output_dir / f"{role_name}_missing_data.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "alpha", "total",
                "steered_missing", "steered_malformed", "steered_duplicate",
                "baseline_missing", "baseline_malformed",
                "assistant_axis_missing", "assistant_axis_malformed", "assistant_axis_duplicate"
            ])
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"Report for {csv_path.name} written to {out_csv}")

if __name__ == "__main__":
    main()
