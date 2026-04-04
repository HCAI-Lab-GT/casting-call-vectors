"""
For each role CSV, report two types of score issues per question:
  1. Scores that are -1 (judge placeholder / skipped)
  2. Score columns where all values across different alphas are identical
     (may indicate duplicated responses or non-discriminating judgement)

Usage:
    # All roles
    python scripts/patch_scripts/report_score_issues.py

    # Specific roles
    python scripts/patch_scripts/report_score_issues.py --roles accountant critic

    # Restrict to specific score columns
    python scripts/patch_scripts/report_score_issues.py --columns steered_score baseline_score
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from pvx import setup_logging

logger = setup_logging(name="report-score-issues")

EXPECTED_ALPHAS = {1.0, 1.5, 2.0, 2.5}

SCORE_COLUMNS = [
    "assistant_axis_score",
    "baseline_score",
    "steered_score",
    "cmp_emotional_register",
    "cmp_vocab_choice",
    "cmp_social_dynamic",
    "cmp_motivation",
    "cmp_worldview_alignment",
    "assistant_axis_cmp_emotional_register",
    "assistant_axis_cmp_vocab_choice",
    "assistant_axis_cmp_social_dynamic",
    "assistant_axis_cmp_motivation",
    "assistant_axis_cmp_worldview_alignment",
]


def is_neg_one(value: object) -> bool:
    if pd.isna(value):
        return False
    try:
        return float(value) == -1.0
    except (ValueError, TypeError):
        return False


def is_malformed(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if text == "":
        return False
    try:
        float(text)
        return False
    except (ValueError, TypeError):
        return True


def load_roles(role_list_path: Path) -> list[str]:
    with open(role_list_path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys()) if isinstance(data, dict) else list(data)


def report_role(
    df: pd.DataFrame,
    role: str,
    columns: list[str],
    validation_questions: set[str] | None = None,
) -> tuple[int, int, int, int, int, int]:
    present_cols = [c for c in columns if c in df.columns]
    if not present_cols:
        logger.warning("%s: none of the requested score columns found", role)
        return 0, 0, 0, 0, 0, 0

    # Only consider sample_count==50 rows
    if "sample_count" in df.columns:
        df = df[df["sample_count"] == 50].copy()

    if df.empty:
        return 0, 0, 0, 0, 0, 0

    neg_one_issues: list[str] = []
    malformed_issues: list[str] = []
    uniform_issues: list[str] = []
    baseline_inconsistent_issues: list[str] = []

    for question, group in df.groupby("question", sort=False):
        q_label = str(question)[:60]

        for col in present_cols:
            neg_one_count = group[col].apply(is_neg_one).sum()
            if neg_one_count > 0:
                neg_one_issues.append(
                    f"  q={q_label!r}  {col}: {neg_one_count}/{len(group)} values are -1"
                )
            malformed_count = group[col].apply(is_malformed).sum()
            if malformed_count > 0:
                examples = group[col][group[col].apply(is_malformed)].unique().tolist()[:3]
                malformed_issues.append(
                    f"  q={q_label!r}  {col}: {malformed_count}/{len(group)} malformed {examples}"
                )

        # Rows with identical score vectors across different alphas
        if group["alpha"].nunique() > 1:
            score_rows = group[present_cols].apply(
                lambda row: tuple(
                    None if pd.isna(v) or is_neg_one(v) else v
                    for v in row
                ),
                axis=1,
            )
            duplicated_alphas = group["alpha"][score_rows.duplicated(keep=False)]
            if not duplicated_alphas.empty:
                alphas_str = ", ".join(f"{a:.2f}" for a in sorted(duplicated_alphas))
                uniform_issues.append(
                    f"  q={q_label!r}: score vector duplicated across alpha(s) [{alphas_str}]"
                )

        # Baseline score must be consistent across alphas for the same question
        if "baseline_score" in present_cols and group["alpha"].nunique() > 1:
            valid_mask = group["baseline_score"].apply(
                lambda v: not pd.isna(v) and not is_neg_one(v) and not is_malformed(v)
            )
            valid_scores = group.loc[valid_mask, ["alpha", "baseline_score"]]
            if valid_scores["baseline_score"].nunique() > 1:
                detail = ", ".join(
                    f"α{float(row['alpha']):.1f}→{row['baseline_score']}"
                    for _, row in valid_scores.sort_values("alpha").iterrows()
                )
                baseline_inconsistent_issues.append(
                    f"  q={q_label!r}: baseline_score differs across alphas [{detail}]"
                )

    # Each validation question must have valid scores for all expected alphas
    missing_coverage_issues: list[str] = []
    if validation_questions:
        for question in sorted(validation_questions):
            q_label = str(question)[:60]
            q_rows = df[df["question"].str.strip() == question.strip()]
            present_alphas = set(q_rows["alpha"].apply(lambda a: float(round(float(a), 9))))
            missing_alphas = {a for a in EXPECTED_ALPHAS if not any(abs(a - p) < 1e-9 for p in present_alphas)}
            if missing_alphas:
                missing_coverage_issues.append(
                    f"  q={q_label!r}: no row for alpha(s) {sorted(missing_alphas)}"
                )
                continue
            for col in present_cols:
                bad_alphas = []
                for alpha in sorted(EXPECTED_ALPHAS):
                    alpha_rows = q_rows[q_rows["alpha"].apply(lambda a: abs(float(a) - alpha) < 1e-9)]
                    if alpha_rows.empty:
                        bad_alphas.append(f"α{alpha:.1f}(missing)")
                    elif alpha_rows[col].apply(lambda v: pd.isna(v) or is_neg_one(v) or is_malformed(v)).all():
                        bad_alphas.append(f"α{alpha:.1f}(invalid)")
                if bad_alphas:
                    missing_coverage_issues.append(
                        f"  q={q_label!r}  {col}: {', '.join(bad_alphas)}"
                    )

    total_rows = len(df)
    neg_one_total = sum(df[col].apply(is_neg_one).sum() for col in present_cols)
    malformed_total = sum(df[col].apply(is_malformed).sum() for col in present_cols)
    uniform_total = len(uniform_issues)
    baseline_inconsistent_total = len(baseline_inconsistent_issues)
    missing_coverage_total = len(missing_coverage_issues)
    affected_questions_neg_one = len({
        line.split("q=")[1].split("'")[1] for line in neg_one_issues
    }) if neg_one_issues else 0

    if not any([neg_one_issues, malformed_issues, uniform_issues,
                baseline_inconsistent_issues, missing_coverage_issues]):
        logger.info("%s: no issues found (%s rows, %s columns checked)",
                    role, total_rows, len(present_cols))
        return 0, 0, 0, 0, 0, total_rows

    logger.info("=== %s ===", role)
    if neg_one_issues:
        logger.info("  [-1 scores]")
        for line in neg_one_issues:
            logger.info(line)
    if malformed_issues:
        logger.info("  [malformed scores]")
        for line in malformed_issues:
            logger.info(line)
    if uniform_issues:
        logger.info("  [uniform across alphas]")
        for line in uniform_issues:
            logger.info(line)
    if baseline_inconsistent_issues:
        logger.info("  [baseline score inconsistent across alphas]")
        for line in baseline_inconsistent_issues:
            logger.info(line)
    if missing_coverage_issues:
        logger.info("  [missing score coverage]")
        for line in missing_coverage_issues:
            logger.info(line)

    logger.info(
        "  [summary] %s total rows | %s -1 score(s) across %s question(s) | "
        "%s malformed score(s) | %s question(s) with duplicate score vectors | "
        "%s baseline inconsistencies | %s missing coverage entries",
        total_rows,
        neg_one_total, affected_questions_neg_one,
        malformed_total,
        uniform_total,
        baseline_inconsistent_total,
        missing_coverage_total,
    )
    return neg_one_total, malformed_total, uniform_total, baseline_inconsistent_total, missing_coverage_total, total_rows


def load_validation_questions(path: Path) -> set[str]:
    questions: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.add(json.loads(line)["question"].strip())
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report -1 scores and uniform-across-alpha scores per question per role."
    )
    parser.add_argument(
        "--input_dir",
        default="./experiment_data/gold_prompt_experiments/",
        help="Directory containing Comparison_GoldStandard_<role>.csv files",
    )
    parser.add_argument(
        "--role_list",
        default="./configs/role_list.json",
        help="Path to role_list.json (ignored when --roles is provided)",
    )
    parser.add_argument("--roles", nargs="+", default=None,
                        help="Roles to check (default: all in role_list.json)")
    parser.add_argument("--columns", nargs="+", default=SCORE_COLUMNS,
                        help=f"Score columns to check (default: all). Available: {SCORE_COLUMNS}")
    parser.add_argument(
        "--questions_file",
        default="./configs/validation_questions.jsonl",
        help="Path to validation_questions.jsonl for coverage checks",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    columns = args.columns or SCORE_COLUMNS
    roles = args.roles if args.roles else load_roles(Path(args.role_list))
    validation_questions = load_validation_questions(Path(args.questions_file))

    missing: list[str] = []
    grand_neg_one = 0
    grand_malformed = 0
    grand_uniform = 0
    grand_baseline_inconsistent = 0
    grand_missing_coverage = 0
    grand_rows = 0

    for role in roles:
        csv_path = input_dir / f"Comparison_GoldStandard_{role}.csv"
        if not csv_path.exists():
            missing.append(role)
            continue
        df = pd.read_csv(csv_path)
        neg_one, malformed, uniform, baseline_inconsistent, missing_coverage, rows = report_role(
            df, role, columns, validation_questions
        )
        grand_neg_one += neg_one
        grand_malformed += malformed
        grand_uniform += uniform
        grand_baseline_inconsistent += baseline_inconsistent
        grand_missing_coverage += missing_coverage
        grand_rows += rows

    if missing:
        logger.warning("No CSV found for %s role(s): %s", len(missing), ", ".join(missing))

    logger.info(
        "=== Grand total: %s rows across %s role(s) | %s -1 score(s) | "
        "%s malformed score(s) | %s question(s) with duplicate score vectors | "
        "%s baseline inconsistencies | %s missing coverage entries ===",
        grand_rows, len(roles) - len(missing), grand_neg_one, grand_malformed, grand_uniform,
        grand_baseline_inconsistent, grand_missing_coverage,
    )


if __name__ == "__main__":
    main()
