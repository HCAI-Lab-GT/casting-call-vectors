import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from pvx import setup_logging

logger = setup_logging(name="fix-assistant-axis-alpha-response")


def has_value(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def is_malformed_assistant_axis(value: object) -> bool:
    if not has_value(value):
        return True

    text = str(value)
    # Malformed entries in this dataset come from pandas Series stringification.
    return "Name: assistant_axis" in text or "dtype:" in text


def target_row_needs_repair(value: object) -> bool:
    if not has_value(value):
        return False
    # For target alpha rows, only repair when the dtype artifact is present.
    return "dtype:" in str(value)


def choose_clean_response(values: pd.Series) -> str | None:
    candidates = [
        str(value).strip()
        for value in values
        if has_value(value) and not is_malformed_assistant_axis(value)
    ]
    if not candidates:
        return None

    counts = Counter(candidates)
    # Prefer the most common clean candidate; break ties by longer response.
    return max(counts.items(), key=lambda item: (item[1], len(item[0])))[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repair assistant_axis responses for a target alpha by copying a clean "
            "per-question response from the same CSV"
        )
    )
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_paths = sorted(input_dir.glob("Comparison_GoldStandard_*.csv"))
    if args.roles:
        allowed = {role.replace("/", "_") for role in args.roles}
        csv_paths = [path for path in csv_paths if path.stem.removeprefix("Comparison_GoldStandard_") in allowed]

    if not csv_paths:
        logger.info("No matching CSV files found in %s", input_dir)
        return

    total_rows_updated = 0
    roles_with_updates: list[str] = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "assistant_axis" not in df.columns or "question" not in df.columns or "alpha" not in df.columns:
            logger.warning("Skipping %s: missing one of required columns [assistant_axis, question, alpha]", csv_path)
            continue

        alpha_values = pd.to_numeric(df["alpha"], errors="coerce")
        target_alpha_mask = (alpha_values - args.alpha).abs() < 1e-9
        if not target_alpha_mask.any():
            logger.info("%s: no rows with alpha=%s", csv_path.name, args.alpha)
            continue

        question_order = [str(question) for question in df["question"].dropna().drop_duplicates().tolist()]
        question_number_map = {question: idx + 1 for idx, question in enumerate(question_order)}

        changed_rows = 0
        unresolved_questions = 0
        already_clean_target_rows = 0
        questions_with_updates: set[str] = set()

        for question, q_indices in df.groupby("question", dropna=False).groups.items():
            question_mask = df.index.isin(q_indices)
            target_rows_mask = question_mask & target_alpha_mask
            if not target_rows_mask.any():
                continue

            clean_response = choose_clean_response(df.loc[question_mask, "assistant_axis"])
            if clean_response is None:
                unresolved_questions += 1
                logger.warning(
                    "%s: no clean assistant_axis response found for question=%r; skipping alpha=%s rows",
                    csv_path.name,
                    question,
                    args.alpha,
                )
                continue

            for index in df.index[target_rows_mask]:
                current_value = df.at[index, "assistant_axis"]
                if not target_row_needs_repair(current_value):
                    already_clean_target_rows += 1
                    continue

                current = str(current_value).strip()
                if current != clean_response:
                    changed_rows += 1
                    questions_with_updates.add(str(question))
                    if not args.dry_run:
                        df.at[index, "assistant_axis"] = clean_response

        if args.dry_run:
            logger.info(
                "%s: %s alpha=%s row(s) would be updated; unresolved questions=%s",
                csv_path.name,
                changed_rows,
                args.alpha,
                unresolved_questions,
            )
            logger.info(
                "%s: %s question(s) would be updated for alpha=%s",
                csv_path.name,
                len(questions_with_updates),
                args.alpha,
            )
            updated_question_numbers = sorted(
                question_number_map.get(question)
                for question in questions_with_updates
                if question_number_map.get(question) is not None
            )
            if updated_question_numbers:
                logger.info(
                    "%s: updated question numbers for alpha=%s -> %s",
                    csv_path.name,
                    args.alpha,
                    updated_question_numbers,
                )
            logger.info(
                "%s: %s alpha=%s row(s) already clean (no dtype artifact)",
                csv_path.name,
                already_clean_target_rows,
                args.alpha,
            )
            if changed_rows > 0:
                roles_with_updates.append(csv_path.stem.removeprefix("Comparison_GoldStandard_"))
            total_rows_updated += changed_rows
            continue

        if changed_rows > 0:
            df.to_csv(csv_path, index=False)
            logger.info(
                "%s: updated %s alpha=%s row(s); unresolved questions=%s",
                csv_path.name,
                changed_rows,
                args.alpha,
                unresolved_questions,
            )
            logger.info(
                "%s: updated question count for alpha=%s is %s",
                csv_path.name,
                args.alpha,
                len(questions_with_updates),
            )
            updated_question_numbers = sorted(
                question_number_map.get(question)
                for question in questions_with_updates
                if question_number_map.get(question) is not None
            )
            if updated_question_numbers:
                logger.info(
                    "%s: updated question numbers for alpha=%s -> %s",
                    csv_path.name,
                    args.alpha,
                    updated_question_numbers,
                )
            logger.info(
                "%s: %s alpha=%s row(s) already clean (no dtype artifact)",
                csv_path.name,
                already_clean_target_rows,
                args.alpha,
            )
        else:
            logger.info("%s: no assistant_axis updates needed for alpha=%s", csv_path.name, args.alpha)

        total_rows_updated += changed_rows

    if args.dry_run:
        if roles_with_updates:
            logger.info(
                "Roles needing updates (%s): %s",
                len(roles_with_updates),
                ", ".join(sorted(roles_with_updates)),
            )
        else:
            logger.info("No roles need updates for alpha=%s", args.alpha)
        logger.info("Dry run complete: %s total row(s) would be updated", total_rows_updated)
    else:
        logger.info("Completed repair: %s total row(s) updated", total_rows_updated)


if __name__ == "__main__":
    main()
