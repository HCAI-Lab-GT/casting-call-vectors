import argparse
from pathlib import Path
import pandas as pd
from collections import Counter

def has_value(value):
    # Treat NaN, empty string, and string 'nan' as missing
    text = str(value).strip().lower()
    return not pd.isna(value) and text != "" and text != "nan"

def is_malformed_baseline(value):
    if not has_value(value):
        return True
    text = str(value)
    return "dtype:" in text or "Name: baseline" in text


def choose_best_baseline(df, question):
    # Prioritize alpha=2.5, no dtype
    candidates = df[(df["question"] == question) & df["baseline"].apply(has_value) & ~df["baseline"].apply(is_malformed_baseline)]
    if not candidates.empty:
        alpha_25 = candidates[candidates["alpha"] == 2.5]
        if not alpha_25.empty:
            # If multiple, pick the most common
            return Counter(alpha_25["baseline"]).most_common(1)[0][0]
        # Otherwise, pick the most common among all candidates
        return Counter(candidates["baseline"]).most_common(1)[0][0]
    return None

def main():
    parser = argparse.ArgumentParser(description="Backfill/repair baseline responses in gold standard CSVs.")
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--role", type=str, default=None, help="Role name (matches CSV suffix, e.g., 'aberration')")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.role:
        # Normalize role name to match CSV naming
        role_csv = f"Comparison_GoldStandard_{args.role.replace('/', '_')}.csv"
        csv_paths = [input_dir / role_csv] if (input_dir / role_csv).exists() else []
        if not csv_paths:
            print(f"No CSV found for role '{args.role}' in {input_dir}")
            return
    else:
        csv_paths = sorted(input_dir.glob("Comparison_GoldStandard_*.csv"))
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "baseline" not in df.columns or "question" not in df.columns or "alpha" not in df.columns:
            print(f"Skipping {csv_path}: missing required columns.")
            continue
        changed = False
        dry_run_updates = []  # Collect (question, alpha, reason) for dry run
        no_baseline_count = 0
        per_alpha_fill_counts = {}
        for question, q_indices in df.groupby("question", dropna=False).groups.items():
            question_mask = df.index.isin(q_indices)
            best_baseline = choose_best_baseline(df, question)
            if best_baseline is None:
                no_baseline_count += 1
                continue
            for idx in df.index[question_mask]:
                current = str(df.at[idx, "baseline"]).strip()
                alpha = df.at[idx, "alpha"]
                # If slot is empty or malformed, fill it
                if not has_value(current) or is_malformed_baseline(current):
                    if args.dry_run:
                        dry_run_updates.append((question, alpha, "fill/repair"))
                        per_alpha_fill_counts[alpha] = per_alpha_fill_counts.get(alpha, 0) + 1
                    else:
                        df.at[idx, "baseline"] = best_baseline
                        per_alpha_fill_counts[alpha] = per_alpha_fill_counts.get(alpha, 0) + 1
                    changed = True
                # If alpha==2.5, ensure it matches best_baseline
                elif alpha == 2.5 and current != best_baseline:
                    if args.dry_run:
                        dry_run_updates.append((question, alpha, "force match alpha=2.5"))
                        per_alpha_fill_counts[alpha] = per_alpha_fill_counts.get(alpha, 0) + 1
                    else:
                        df.at[idx, "baseline"] = best_baseline
                        per_alpha_fill_counts[alpha] = per_alpha_fill_counts.get(alpha, 0) + 1
                    changed = True
        if changed:
            if args.dry_run:
                if dry_run_updates:
                    print(f"{csv_path.name}: would update baseline responses for:")
                    for question, alpha, reason in dry_run_updates:
                        print(f"  - Question: {repr(question)[:60]}... | alpha: {alpha} | reason: {reason}")
                    print("Per-alpha fill/repair counts:")
                    for alpha, count in sorted(per_alpha_fill_counts.items()):
                        print(f"  alpha={alpha}: {count}")
                else:
                    print(f"{csv_path.name}: would update baseline responses (details unavailable)")
            else:
                df.to_csv(csv_path, index=False)
                print(f"{csv_path.name}: updated baseline responses.")
                print("Per-alpha fill/repair counts:")
                for alpha, count in sorted(per_alpha_fill_counts.items()):
                    print(f"  alpha={alpha}: {count}")
        else:
            print(f"{csv_path.name}: no changes needed.")
            print(f"Total no baseline counts: {no_baseline_count}")

if __name__ == "__main__":
    main()
