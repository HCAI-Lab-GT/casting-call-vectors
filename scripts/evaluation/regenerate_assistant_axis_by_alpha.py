import argparse
from pathlib import Path

import pandas as pd

from pvx import setup_logging
from pvx.implementations.roles_layers.assistant_axis_persona_model import AssistantAxisPersonaModel

logger = setup_logging(name="regenerate-assistant-axis-by-alpha")


def parse_alphas(alpha_values: list[float] | None) -> set[float] | None:
    if not alpha_values:
        return None
    return {float(value) for value in alpha_values}


def alpha_matches(alpha: float, allowed: set[float] | None, tol: float = 1e-9) -> bool:
    if allowed is None:
        return True
    return any(abs(alpha - target) < tol for target in allowed)


def build_question_number_map(df: pd.DataFrame) -> dict[str, int]:
    ordered_questions = [str(question) for question in df["question"].dropna().drop_duplicates().tolist()]
    return {question: idx + 1 for idx, question in enumerate(ordered_questions)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate assistant_axis responses in Comparison_GoldStandard CSVs "
            "using each row's alpha value (matching steered alpha)."
        )
    )
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of alphas to regenerate (defaults to all alphas present in CSV rows)",
    )
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--json_filepath", default="./persona_data/persona_models.json")
    parser.add_argument("--pt_dir", default="persona_data/assistant-axis/olmo-3-7b-instruct/vectors/")
    parser.add_argument(
        "--assistant_axis_layer",
        type=int,
        default=None,
        help="Optional fixed layer for Assistant Axis. If omitted, uses each row's layer.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument(
        "--temperature_mode",
        choices=["row", "fixed"],
        default="row",
        help="Use per-row temperature or a fixed temperature for all generations.",
    )
    parser.add_argument("--fixed_temperature", type=float, default=0.2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    allowed_alphas = parse_alphas(args.alphas)

    csv_paths = sorted(input_dir.glob("Comparison_GoldStandard_*.csv"))
    if args.roles:
        allowed_roles = {role.replace("/", "_") for role in args.roles}
        csv_paths = [path for path in csv_paths if path.stem.removeprefix("Comparison_GoldStandard_") in allowed_roles]

    if not csv_paths:
        logger.info("No matching CSV files found in %s", input_dir)
        return

    total_rows_targeted = 0
    total_rows_updated = 0

    for csv_path in csv_paths:
        role = csv_path.stem.removeprefix("Comparison_GoldStandard_")
        df = pd.read_csv(csv_path)

        required_columns = {"question", "alpha", "temperature", "assistant_axis", "sample_count", "layer"}
        if not required_columns.issubset(df.columns):
            missing = sorted(required_columns - set(df.columns))
            logger.warning("Skipping %s: missing required columns %s", csv_path.name, missing)
            continue

        question_number_map = build_question_number_map(df)
        row_alphas = pd.to_numeric(df["alpha"], errors="coerce")
        target_indices: list[int] = []

        for index, alpha_value in row_alphas.items():
            if pd.isna(alpha_value):
                continue
            if alpha_matches(float(alpha_value), allowed_alphas):
                target_indices.append(index)

        if not target_indices:
            logger.info("%s: no rows matched requested alphas", csv_path.name)
            continue

        targeted_questions = {
            question_number_map.get(str(df.at[index, "question"]))
            for index in target_indices
            if question_number_map.get(str(df.at[index, "question"])) is not None
        }
        targeted_questions_sorted = sorted(targeted_questions)

        total_rows_targeted += len(target_indices)
        logger.info(
            "%s: %s row(s) targeted across %s question(s). Question numbers: %s",
            csv_path.name,
            len(target_indices),
            len(targeted_questions_sorted),
            targeted_questions_sorted,
        )

        if args.dry_run:
            continue

        model_cache: dict[tuple[int, int], AssistantAxisPersonaModel] = {}
        response_cache: dict[tuple[str, float, float, int], str] = {}
        changed_rows = 0

        # Group rows by alpha so we can save incrementally after each alpha completes.
        alpha_groups: dict[float, list[int]] = {}
        for index in target_indices:
            alpha_value = float(row_alphas.loc[index])
            alpha_groups.setdefault(alpha_value, []).append(index)

        progress_index = 0
        total_targets = len(target_indices)

        for alpha_value in sorted(alpha_groups.keys()):
            alpha_indices = alpha_groups[alpha_value]
            alpha_changed_rows = 0
            logger.info("Starting alpha %.4f (%s row(s))", alpha_value, len(alpha_indices))

            for index in alpha_indices:
                progress_index += 1
                question = str(df.at[index, "question"]).strip()
                layer_value = args.assistant_axis_layer if args.assistant_axis_layer is not None else int(df.at[index, "layer"])
                sample_count = int(df.at[index, "sample_count"])
                q_num = question_number_map.get(question)

                if args.temperature_mode == "row":
                    temperature = float(df.at[index, "temperature"])
                else:
                    temperature = float(args.fixed_temperature)

                logger.info(
                    "[%s/%s] row=%s q=%s alpha=%.4f layer=%s temp=%.4f",
                    progress_index, total_targets, index, q_num, alpha_value, layer_value, temperature,
                )

                model_key = (layer_value, sample_count)
                if model_key not in model_cache:
                    logger.info("Loading model: layer=%s sample_count=%s", layer_value, sample_count)
                    model_cache[model_key] = AssistantAxisPersonaModel.load_or_create(
                        target_model_id=args.model,
                        concept=role,
                        layer=layer_value,
                        target_pairs=sample_count,
                        json_filepath=args.json_filepath,
                        safetensors_dir=args.pt_dir,
                    )

                response_key = (question, alpha_value, temperature, layer_value)
                if response_key not in response_cache:
                    logger.info("Generating response (cache miss): q=%s alpha=%.4f", q_num, alpha_value)
                    response_cache[response_key] = model_cache[model_key].generate(
                        prompt=question,
                        alpha=alpha_value,
                        max_new_tokens=args.max_new_tokens,
                        temperature=temperature,
                    )
                else:
                    logger.info("Using cached response: q=%s alpha=%.4f", q_num, alpha_value)

                new_response = response_cache[response_key]
                current_response = str(df.at[index, "assistant_axis"]) if not pd.isna(df.at[index, "assistant_axis"]) else ""
                if current_response != new_response:
                    logger.info(
                        "  row=%s q=%s: updating response\n  OLD: %s\n  NEW: %s",
                        index, q_num,
                        current_response[:120].replace("\n", " "),
                        new_response[:120].replace("\n", " "),
                    )
                    df.at[index, "assistant_axis"] = new_response
                    changed_rows += 1
                    alpha_changed_rows += 1
                else:
                    logger.info("  row=%s q=%s: response unchanged", index, q_num)

            if alpha_changed_rows > 0:
                df.to_csv(csv_path, index=False)
                logger.info("Saved %s after alpha %.4f (%s row(s) updated)", csv_path.name, alpha_value, alpha_changed_rows)
            else:
                logger.info("No changes for alpha %.4f; skipping save", alpha_value)

        if changed_rows > 0:
            logger.info("%s: updated %s row(s)", csv_path.name, changed_rows)
        else:
            logger.info("%s: no assistant_axis changes needed", csv_path.name)

        total_rows_updated += changed_rows

    if args.dry_run:
        logger.info("Dry run complete: %s row(s) would be targeted", total_rows_targeted)
    else:
        logger.info(
            "Completed regeneration: %s targeted row(s), %s row(s) updated",
            total_rows_targeted,
            total_rows_updated,
        )


if __name__ == "__main__":
    main()
