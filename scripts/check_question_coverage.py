"""
Check that each validation question appears exactly once per alpha (1.0, 1.5, 2.0, 2.5)
across all role CSVs in gold_prompt_experiments/.
"""

import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = REPO_ROOT / "configs" / "validation_questions.jsonl"
CSV_DIR = REPO_ROOT / "experiment_data" / "regenerated_modal" / "gold_prompt_experiments"
EXPECTED_ALPHAS = {1.0, 1.5, 2.0, 2.5}
GENERATIONS_PER_PAIR = 2  # steered + assistant_axis


def load_validation_questions() -> dict[int, str]:
    questions = {}
    with QUESTIONS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                questions[obj["id"]] = obj["question"]
    return questions


def check_csv(csv_path: Path, expected_questions: dict[int, str]) -> tuple[int, list[str]]:
    """Return (missing_pairs_count, verbose_error_lines)."""
    details = []
    question_alphas: dict[str, set[float]] = defaultdict(set)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["sample_count"]) != 50:
                    continue
            except (ValueError, KeyError):
                continue
            q = row["question"].strip()
            try:
                alpha = float(row["alpha"])
            except (ValueError, KeyError):
                details.append(f"  invalid alpha value: {row.get('alpha')!r}")
                continue
            question_alphas[q].add(alpha)

    expected_texts = set(expected_questions.values())
    missing_pairs = 0

    missing = expected_texts - set(question_alphas.keys())
    for q in sorted(missing):
        missing_pairs += len(EXPECTED_ALPHAS)
        details.append(f"  MISSING question entirely: {q[:80]!r}")

    for q_text, alphas_seen in question_alphas.items():
        if q_text not in expected_texts:
            continue
        missing_alphas = EXPECTED_ALPHAS - alphas_seen
        extra_alphas = alphas_seen - EXPECTED_ALPHAS
        if missing_alphas:
            missing_pairs += len(missing_alphas)
            details.append(f"  missing alphas {sorted(missing_alphas)} for: {q_text[:60]!r}")
        if extra_alphas:
            details.append(f"  unexpected alphas {sorted(extra_alphas)} for: {q_text[:60]!r}")

    return missing_pairs, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Check validation question coverage across role CSVs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-question detail for each role with issues")
    args = parser.parse_args()

    expected_questions = load_validation_questions()
    csv_files = sorted(CSV_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSVs found in {CSV_DIR}")
        return

    total_roles = len(csv_files)
    roles_with_errors = 0
    all_results: dict[str, tuple[int, list[str]]] = {}

    for csv_path in csv_files:
        role = csv_path.stem.replace("Comparison_GoldStandard_", "")
        missing_pairs, details = check_csv(csv_path, expected_questions)
        if missing_pairs > 0:
            roles_with_errors += 1
            all_results[role] = (missing_pairs, details)

    expected_texts = set(expected_questions.values())
    print(f"Checked {total_roles} role CSVs")
    print(f"Expected {len(expected_texts)} unique questions x 4 alphas = {len(expected_texts) * 4} rows per role\n")

    if not all_results:
        print("All roles pass: every question present for all 4 alphas.")
        return

    print(f"{roles_with_errors}/{total_roles} roles have issues:\n")
    for role, (missing_pairs, details) in sorted(all_results.items()):
        generations = missing_pairs * GENERATIONS_PER_PAIR
        print(f"[{role}]  {missing_pairs} missing pair(s) -> {generations} generation(s) needed")
        if args.verbose:
            for line in details:
                print(line)
            print()


if __name__ == "__main__":
    main()
