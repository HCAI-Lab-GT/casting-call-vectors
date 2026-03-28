import argparse
import json
from pathlib import Path

import pandas as pd

from pvx import setup_logging

logger = setup_logging(name="add-missing-questions-to-gold-standard")


def load_validation_questions(questions_file: Path) -> list[str]:
    """Load questions from a jsonl file."""
    questions = []
    with open(questions_file, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                questions.append(obj.get("question", ""))
    return [q for q in questions if q]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add missing questions from validation_questions.jsonl to Comparison_GoldStandard CSVs. "
            "Creates new rows with specified alphas, temperature=0.2, sample_count=50. "
            "Response columns are left empty for later generation."
        )
    )
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--roles", nargs="+", required=True, help="Roles to process")
    parser.add_argument("--alphas", nargs="+", type=float, required=True, help="Alpha values for new rows")
    parser.add_argument("--layer", type=int, required=True, help="Layer value for new rows")
    parser.add_argument("--questions_file", default="./configs/validation_questions.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    questions_file = Path(args.questions_file)

    if not questions_file.exists():
        logger.error("Questions file not found: %s", questions_file)
        return

    all_questions = load_validation_questions(questions_file)
    logger.info("Loaded %s questions from %s", len(all_questions), questions_file)

    total_rows_added = 0
    temperature = 0.2
    sample_count = 50



    for role in args.roles:
        csv_path = input_dir / f"Comparison_GoldStandard_{role}.csv"

        if not csv_path.exists():
            logger.warning("CSV not found for role %s: %s", role, csv_path)
            continue

        df = pd.read_csv(csv_path)

        # Use layer from CLI argument
        layer = args.layer
        logger.info("%s: using layer=%s", csv_path.name, layer)

        all_new_rows = []
        for alpha_value in args.alphas:
            # Find existing questions for this alpha
            df_alpha = df[df["alpha"] == alpha_value]
            existing_questions = set(df_alpha["question"].dropna().astype(str).unique())
            missing_questions = [q for q in all_questions if q not in existing_questions]

            logger.info(f"Alpha={alpha_value}: Existing Questions Count: {len(existing_questions)}")
            logger.info(f"Alpha={alpha_value}: All Questions Count: {len(all_questions)}")

            if not missing_questions:
                logger.info("%s: no missing questions for alpha=%.4f", csv_path.name, alpha_value)
                continue

            logger.info(
                "%s: %s missing question(s) out of %s total for alpha=%.4f",
                csv_path.name,
                len(missing_questions),
                len(all_questions),
                alpha_value,
            )

            if args.dry_run:
                logger.info("Dry run: would add %s rows for alpha=%.4f", len(missing_questions), alpha_value)
                continue

            for question in missing_questions:
                logger.info("Adding row: role=%s alpha=%.4f question=%s", role, alpha_value, question[:80])

                new_row = {
                    "role": role,
                    "layer": layer,
                    "sample_count": sample_count,
                    "alpha": alpha_value,
                    "temperature": temperature,
                    "question": question,
                    "assistant_axis": "",
                    "baseline": "",
                    "steered": "",
                    "assistant_axis_score": -1,
                    "baseline_score": -1,
                    "steered_score": -1,
                    "cmp_emotional_register": -1,
                    "cmp_vocab_choice": -1,
                    "cmp_social_dynamic": -1,
                    "cmp_motivation": -1,
                    "cmp_worldview_alignment": -1,
                    "assistant_axis_cmp_emotional_register": -1,
                    "assistant_axis_cmp_vocab_choice": -1,
                    "assistant_axis_cmp_social_dynamic": -1,
                    "assistant_axis_cmp_motivation": -1,
                    "assistant_axis_cmp_worldview_alignment": -1,
                }
                all_new_rows.append(new_row)

        if all_new_rows and not args.dry_run:
            df_new = pd.DataFrame(all_new_rows)
            # Ensure all columns from original CSV are present in new rows
            for col in df.columns:
                if col not in df_new.columns:
                    df_new[col] = ""
            # Reorder to match original CSV columns
            df_new = df_new[df.columns]

            df_combined = pd.concat([df, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            logger.info("%s: appended %s new row(s) across all alphas", csv_path.name, len(all_new_rows))
            total_rows_added += len(all_new_rows)

    if args.dry_run:
        logger.info("Dry run complete.")
    else:
        logger.info("Completed: added %s total row(s) across all roles", total_rows_added)


if __name__ == "__main__":
    main()
