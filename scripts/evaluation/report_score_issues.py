"""
For each role CSV, report two types of score issues per question:
  1. Scores that are -1 (judge placeholder / skipped)
  2. Score columns where all values across different alphas are identical
     (may indicate duplicated responses or non-discriminating judgement)

Usage:
    # All roles
    python scripts/evaluation/report_score_issues.py

    # Specific roles
    python scripts/evaluation/report_score_issues.py --roles accountant critic

    # Restrict to specific score columns
    python scripts/evaluation/report_score_issues.py --columns steered_score baseline_score
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from pvx import setup_logging

logger = setup_logging(name="report-score-issues")

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


def report_role(df: pd.DataFrame, role: str, columns: list[str]) -> tuple[int, int, int, int]:
    present_cols = [c for c in columns if c in df.columns]
    if not present_cols:
        logger.warning("%s: none of the requested score columns found", role)
        return 0, 0, 0, 0

    # Only consider sample_count==50 rows
    if "sample_count" in df.columns:
        df = df[df["sample_count"] == 50].copy()

    if df.empty:
        return 0, 0, 0, 0

    neg_one_issues: list[str] = []
    malformed_issues: list[str] = []
    uniform_issues: list[str] = []

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

    total_rows = len(df)
    neg_one_total = sum(df[col].apply(is_neg_one).sum() for col in present_cols)
    malformed_total = sum(df[col].apply(is_malformed).sum() for col in present_cols)
    uniform_total = len(uniform_issues)
    affected_questions_neg_one = len({
        line.split("q=")[1].split("'")[1] for line in neg_one_issues
    }) if neg_one_issues else 0

    if not neg_one_issues and not malformed_issues and not uniform_issues:
        logger.info("%s: no issues found (%s rows, %s columns checked)",
                    role, total_rows, len(present_cols))
        return 0, 0, 0, total_rows

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

    logger.info(
        "  [summary] %s total rows | %s -1 score(s) across %s question(s) | "
        "%s malformed score(s) | %s question(s) with duplicate score vectors across alphas",
        total_rows,
        neg_one_total, affected_questions_neg_one,
        malformed_total,
        uniform_total,
    )
    return neg_one_total, malformed_total, uniform_total, total_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report -1 scores and uniform-across-alpha scores per question per role."
    )
    parser.add_argument(
        "--input_dir",
        default="./experiment_data/regenerated_modal/gold_prompt_experiments/",
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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    columns = args.columns or SCORE_COLUMNS
    roles = args.roles if args.roles else load_roles(Path(args.role_list))

    missing: list[str] = []
    grand_neg_one = 0
    grand_malformed = 0
    grand_uniform = 0
    grand_rows = 0

    for role in roles:
        csv_path = input_dir / f"Comparison_GoldStandard_{role}.csv"
        if not csv_path.exists():
            missing.append(role)
            continue
        df = pd.read_csv(csv_path)
        neg_one, malformed, uniform, rows = report_role(df, role, columns)
        grand_neg_one += neg_one
        grand_malformed += malformed
        grand_uniform += uniform
        grand_rows += rows

    if missing:
        logger.warning("No CSV found for %s role(s): %s", len(missing), ", ".join(missing))

    logger.info(
        "=== Grand total: %s rows across %s role(s) | %s -1 score(s) | "
        "%s malformed score(s) | %s question(s) with duplicate score vectors across alphas ===",
        grand_rows, len(roles) - len(missing), grand_neg_one, grand_malformed, grand_uniform,
    )


if __name__ == "__main__":
    main()
