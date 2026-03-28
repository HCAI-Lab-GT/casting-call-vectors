import argparse
import json
from pathlib import Path
import pandas as pd

from pvx import setup_logging
from pvx.implementations.judges.llm_as_judge import GOLD_COMPARATOR_PROMPT_TEMPLATE, LLMJudge

logger = setup_logging(name="backfill-assistant-axis-gold-comparisons")

COMPARE_FIELDS = (
    ("emotional_register", "style", "emotional_register"),
    ("vocab_choice", "style", "vocab_choice"),
    ("social_dynamic", "style", "social_dynamic"),
    ("motivation", "content", "motivation"),
    ("worldview_alignment", "content", "worldview_alignment"),
)
STEERED_COMPARE_COLUMNS = {f"cmp_{n}": (s, k) for n, s, k in COMPARE_FIELDS}
AA_COMPARE_COLUMNS = {f"assistant_axis_cmp_{n}": (s, k) for n, s, k in COMPARE_FIELDS}


def has_value(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def load_role_description(role: str, gold_prompts_dir: Path, cache: dict[str, str]) -> str:
    if role in cache:
        return cache[role]
    gold_path = gold_prompts_dir / f"{role.replace('/', '_')}_gold_label.json"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold prompt file not found for role '{role}': {gold_path}")
    with open(gold_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    role_description = str(payload.get("role_description", "")).strip()
    cache[role] = role_description
    return role_description


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill assistant-axis gold comparison scores into Comparison_GoldStandard_*.csv files"
    )
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--gold_prompts_dir", default="./persona_data/gold_labels_prompts_dataset")
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_steered_refresh", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--backend", default="openai", choices=["openai", "vllm", "hf_local"])
    parser.add_argument("--judge_model", default="openai/gpt-4.1-mini")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gold_prompts_dir = Path(args.gold_prompts_dir)
    csv_paths = sorted(input_dir.glob("Comparison_GoldStandard_*.csv"))
    if args.roles:
        allowed = {role.replace("/", "_") for role in args.roles}
        csv_paths = [path for path in csv_paths if path.stem.removeprefix("Comparison_GoldStandard_") in allowed]
    if not csv_paths:
        logger.info("No matching CSV files found in %s", input_dir)
        return

    judge = LLMJudge(
        backend=args.backend,
        model=args.judge_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        prompt_template=GOLD_COMPARATOR_PROMPT_TEMPLATE,
    )
    judge.judge_func = judge._aggregate_gold_comparator_score
    role_description_cache: dict[str, str] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        changed = False
        for column in (*STEERED_COMPARE_COLUMNS, *AA_COMPARE_COLUMNS):
            if column not in df.columns:
                df[column] = pd.NA
                changed = True
        pending_assistant_axis_rows, pending_steered_rows = 0, 0

        for index, row in df.iterrows():
            role = str(row.get("role", "")).strip()
            question = str(row.get("question", "")).strip()
            baseline = str(row.get("baseline", "")).strip()
            assistant_axis = str(row.get("assistant_axis", "")).strip()
            steered = str(row.get("steered", "")).strip()

            refresh_assistant_axis = has_value(assistant_axis) and has_value(baseline) and (
                args.overwrite
                or any(not has_value(row.get(column)) for column in AA_COMPARE_COLUMNS)
            )
            refresh_steered = not args.skip_steered_refresh and has_value(steered) and has_value(baseline)
            if not refresh_assistant_axis and not refresh_steered:
                continue

            role_description = load_role_description(role, gold_prompts_dir, role_description_cache)
            pending_assistant_axis_rows += int(refresh_assistant_axis)
            pending_steered_rows += int(refresh_steered)

            if args.dry_run:
                continue

            if refresh_assistant_axis:
                assistant_axis_scores = judge(
                    role=role,
                    role_description=role_description,
                    question=question,
                    baseline=baseline,
                    answer=assistant_axis,
                )
                for column, (section, key) in AA_COMPARE_COLUMNS.items():
                    df.at[index, column] = assistant_axis_scores[section][key]
                changed = True

            if refresh_steered:
                steered_scores = judge(
                    role=role,
                    role_description=role_description,
                    question=question,
                    baseline=baseline,
                    answer=steered,
                )
                for column, (section, key) in STEERED_COMPARE_COLUMNS.items():
                    df.at[index, column] = steered_scores[section][key]
                changed = True

        if args.dry_run:
            logger.info(
                "%s: %s assistant-axis rows and %s steered rows would be updated",
                csv_path.name,
                pending_assistant_axis_rows,
                pending_steered_rows,
            )
            continue
        if changed:
            df.to_csv(csv_path, index=False)
            logger.info(
                "Updated %s with %s assistant-axis rows and %s steered rows",
                csv_path,
                pending_assistant_axis_rows,
                pending_steered_rows,
            )
        else:
            logger.info("No changes needed for %s", csv_path)


if __name__ == "__main__":
    main()