"""
Check per-role coverage of validation questions across all 4 alphas (1.0, 1.5, 2.0, 2.5)
filtered to sample_count=50 in gold_prompt_experiments CSVs.
"""

import csv
import json
from pathlib import Path

GOLD_DIR = Path(__file__).parent.parent / "experiment_data" / "gold_prompt_experiments"
VALIDATION_QUESTIONS_PATH = Path(__file__).parent.parent / "configs" / "validation_questions.jsonl"
ALPHAS = {1.0, 1.5, 2.0, 2.5}
SAMPLE_COUNT = 50


def load_validation_questions(path: Path) -> list[str]:
    seen: set[str] = set()
    questions: list[str] = []
    with path.open() as f:
        for line in f:
            q = json.loads(line)["question"].strip()
            if q not in seen:
                seen.add(q)
                questions.append(q)
    return questions


def check_role(csv_path: Path, validation_questions: list[str]) -> dict:
    question_alphas: dict[str, set] = {q: set() for q in validation_questions}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sc = int(row["sample_count"])
                alpha = float(row["alpha"])
            except (ValueError, KeyError):
                continue

            if sc != SAMPLE_COUNT:
                continue

            question = row["question"].strip()
            if question in question_alphas:
                question_alphas[question].add(alpha)

    expected = len(validation_questions)
    covered = sum(1 for alphas in question_alphas.values() if ALPHAS.issubset(alphas))
    alpha_counts = {a: sum(1 for alphas in question_alphas.values() if a in alphas) for a in ALPHAS}

    return {
        "expected": expected,
        "all_4_alphas": covered,
        "alpha_counts": alpha_counts,
        "question_alphas": question_alphas,
    }


def main():
    validation_questions = load_validation_questions(VALIDATION_QUESTIONS_PATH)
    expected = len(validation_questions)

    csv_files = sorted(GOLD_DIR.glob("Comparison_GoldStandard_*.csv"))
    if not csv_files:
        print(f"No CSVs found in {GOLD_DIR}")
        return

    results = []
    for csv_path in csv_files:
        role = csv_path.stem.replace("Comparison_GoldStandard_", "")
        stats = check_role(csv_path, validation_questions)
        results.append((role, stats["all_4_alphas"], stats["alpha_counts"], stats["question_alphas"]))

    col_w = max(len(r[0]) for r in results) + 2
    header = f"{'role':<{col_w}} {'all_4_alphas':>13}"
    print(f"Expected questions per alpha: {expected}")
    print(header)
    print("-" * len(header))

    for role, covered, _, __ in results:
        print(f"{role:<{col_w}} {covered:>13}")

    print("-" * len(header))
    roles_with_full = sum(1 for _, c, _, __ in results if c == expected)
    print(f"\nRoles with full {expected}q coverage across all 4 alphas: {roles_with_full}/{len(results)}")

    incomplete = [
        (role, covered, sum(expected - alpha_counts[a] for a in ALPHAS), alpha_counts, question_alphas)
        for role, covered, alpha_counts, question_alphas in results
        if any(alpha_counts[a] < expected for a in ALPHAS)
    ]
    if not incomplete:
        print("No roles are missing questions.")
        return

    sorted_alphas = sorted(ALPHAS)
    alpha_header = "  ".join(f"a{a}" for a in sorted_alphas)
    print(f"\nIncomplete roles ({len(incomplete)}):")
    inc_col_w = max(len(r[0]) for r in incomplete) + 2
    inc_header = f"  {'role':<{inc_col_w}} {'have':>6} {'missing':>8}  {alpha_header}"
    print(inc_header)
    print("  " + "-" * (len(inc_header) - 2))
    for role, covered, missing, alpha_counts, question_alphas in sorted(incomplete, key=lambda x: x[2], reverse=True):
        alpha_cells = "  ".join(f"{alpha_counts[a]:>4}" for a in sorted_alphas)
        missing_nums = [
            str(i + 1)
            for i, q in enumerate(validation_questions)
            if not ALPHAS.issubset(question_alphas[q])
        ]
        print(f"  {role:<{inc_col_w}} {covered:>6} {missing:>8}  {alpha_cells}")
        print(f"  {'':>{inc_col_w}} missing q#s: {', '.join(missing_nums)}")


if __name__ == "__main__":
    main()
