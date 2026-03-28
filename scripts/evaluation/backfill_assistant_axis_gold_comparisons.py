import argparse
import asyncio
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pvx import setup_logging
from pvx.implementations.judges.llm_as_judge import (
    GOLD_COMPARATOR_PROMPT_TEMPLATE,
    LLMJudge,
    PROMPT_TEMPLATE,
)

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
ROLE_SCORE_COLUMNS = ("assistant_axis_score", "baseline_score", "steered_score")


def has_value(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def needs_backfill(value: object) -> bool:
    if not has_value(value):
        return True
    text = str(value).strip()
    if text == "-1":
        return True
    try:
        return float(text) == -1.0
    except ValueError:
        return False


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


# ── Async task descriptors ───────────────────────────────────────────────

@dataclass
class RoleScoreTask:
    csv_path: Path
    index: Any
    column: str  # "assistant_axis_score" | "baseline_score" | "steered_score"
    kwargs: dict = field(default_factory=dict)
    result: float | None = None


@dataclass
class ComparatorTask:
    csv_path: Path
    index: Any
    column_map: dict  # e.g. AA_COMPARE_COLUMNS or STEERED_COMPARE_COLUMNS
    kwargs: dict = field(default_factory=dict)
    result: dict | None = None


# ── Async workers ────────────────────────────────────────────────────────

async def run_role_score(task: RoleScoreTask, judge: LLMJudge, sem: asyncio.Semaphore) -> None:
    async with sem:
        task.result = await judge.async_judge(**task.kwargs)


async def run_comparator(task: ComparatorTask, judge: LLMJudge, sem: asyncio.Semaphore) -> None:
    async with sem:
        task.result = await judge.async_gold_comparator_judge(**task.kwargs)


# ── Main ─────────────────────────────────────────────────────────────────

async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill role and comparative judge scores into Comparison_GoldStandard_*.csv files"
    )
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--gold_prompts_dir", default="./persona_data/gold_labels_prompts_dataset")
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate all scores unconditionally")
    parser.add_argument("--overwrite_assistant_axis", action="store_true", help="Regenerate all assistant_axis scores and comparisons unconditionally")
    parser.add_argument("--skip_steered_refresh", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--backend", default="openai", choices=["openai", "vllm", "hf_local"])
    parser.add_argument("--judge_model", default="openai/gpt-4.1-mini")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max_concurrency", type=int, default=50, help="Max parallel API requests")
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

    role_judge = LLMJudge(
        backend=args.backend,
        model=args.judge_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        prompt_template=PROMPT_TEMPLATE,
    )

    comparator_judge = LLMJudge(
        backend=args.backend,
        model=args.judge_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        prompt_template=GOLD_COMPARATOR_PROMPT_TEMPLATE,
    )

    sem = asyncio.Semaphore(args.max_concurrency)
    role_description_cache: dict[str, str] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        changed = False
        for column in (*STEERED_COMPARE_COLUMNS, *AA_COMPARE_COLUMNS, *ROLE_SCORE_COLUMNS):
            if column not in df.columns:
                df[column] = pd.NA
                changed = True

        pending_counts = {
            "assistant_axis_role": 0,
            "baseline_role": 0,
            "steered_role": 0,
            "assistant_axis_compare": 0,
            "steered_compare": 0,
        }

        role_score_tasks: list[RoleScoreTask] = []
        comparator_tasks: list[ComparatorTask] = []

        for index, row in df.iterrows():
            role = str(row.get("role", "")).strip()
            question = str(row.get("question", "")).strip()
            baseline = str(row.get("baseline", "")).strip()
            assistant_axis = str(row.get("assistant_axis", "")).strip()
            steered = str(row.get("steered", "")).strip()

            refresh_assistant_axis_role = has_value(assistant_axis) and (
                args.overwrite or args.overwrite_assistant_axis or needs_backfill(row.get("assistant_axis_score"))
            )
            refresh_baseline_role = has_value(baseline) and (
                args.overwrite or needs_backfill(row.get("baseline_score"))
            )
            refresh_steered_role = has_value(steered) and (
                args.overwrite or needs_backfill(row.get("steered_score"))
            )
            refresh_assistant_axis_compare = has_value(assistant_axis) and has_value(baseline) and (
                args.overwrite
                or args.overwrite_assistant_axis
                or any(needs_backfill(row.get(column)) for column in AA_COMPARE_COLUMNS)
            )
            refresh_steered_compare = not args.skip_steered_refresh and has_value(steered) and has_value(baseline) and (
                args.overwrite
                or any(needs_backfill(row.get(column)) for column in STEERED_COMPARE_COLUMNS)
            )
            if not any(
                (
                    refresh_assistant_axis_role,
                    refresh_baseline_role,
                    refresh_steered_role,
                    refresh_assistant_axis_compare,
                    refresh_steered_compare,
                )
            ):
                continue

            role_description = load_role_description(role, gold_prompts_dir, role_description_cache)

            if refresh_assistant_axis_role:
                pending_counts["assistant_axis_role"] += 1
                if not args.dry_run:
                    role_score_tasks.append(RoleScoreTask(
                        csv_path=csv_path, index=index, column="assistant_axis_score",
                        kwargs=dict(role=role, role_description=role_description, question=question, answer=assistant_axis),
                    ))

            if refresh_baseline_role:
                pending_counts["baseline_role"] += 1
                if not args.dry_run:
                    role_score_tasks.append(RoleScoreTask(
                        csv_path=csv_path, index=index, column="baseline_score",
                        kwargs=dict(role=role, role_description=role_description, question=question, answer=baseline),
                    ))

            if refresh_steered_role:
                pending_counts["steered_role"] += 1
                if not args.dry_run:
                    role_score_tasks.append(RoleScoreTask(
                        csv_path=csv_path, index=index, column="steered_score",
                        kwargs=dict(role=role, role_description=role_description, question=question, answer=steered),
                    ))

            if refresh_assistant_axis_compare:
                pending_counts["assistant_axis_compare"] += 1
                if not args.dry_run:
                    comparator_tasks.append(ComparatorTask(
                        csv_path=csv_path, index=index, column_map=AA_COMPARE_COLUMNS,
                        kwargs=dict(role=role, role_description=role_description, question=question, baseline=baseline, answer=assistant_axis),
                    ))

            if refresh_steered_compare:
                pending_counts["steered_compare"] += 1
                if not args.dry_run:
                    comparator_tasks.append(ComparatorTask(
                        csv_path=csv_path, index=index, column_map=STEERED_COMPARE_COLUMNS,
                        kwargs=dict(role=role, role_description=role_description, question=question, baseline=baseline, answer=steered),
                    ))

        total_tasks = len(role_score_tasks) + len(comparator_tasks)
        total_pending = sum(pending_counts.values())

        if args.dry_run:
            logger.info(
                "%s: role scores (assistant_axis=%s baseline=%s steered=%s), comparisons (assistant_axis=%s steered=%s) would be updated"
                " — %s total API requests, max_concurrency=%s",
                csv_path.name,
                pending_counts["assistant_axis_role"],
                pending_counts["baseline_role"],
                pending_counts["steered_role"],
                pending_counts["assistant_axis_compare"],
                pending_counts["steered_compare"],
                total_pending,
                args.max_concurrency,
            )
            continue

        if total_tasks == 0:
            logger.info("No changes needed for %s", csv_path)
            continue

        logger.info(
            "%s: launching %s async tasks (role_score=%s, comparator=%s, concurrency=%s)",
            csv_path.name, total_tasks, len(role_score_tasks), len(comparator_tasks), args.max_concurrency,
        )

        # Fire all requests concurrently
        await asyncio.gather(
            *(run_role_score(t, role_judge, sem) for t in role_score_tasks),
            *(run_comparator(t, comparator_judge, sem) for t in comparator_tasks),
        )

        # Write results back into the DataFrame
        for task in role_score_tasks:
            df.at[task.index, task.column] = task.result
            changed = True

        for task in comparator_tasks:
            if task.result is not None:
                for column, (section, key) in task.column_map.items():
                    df.at[task.index, column] = task.result[section][key]
                changed = True

        if changed:
            df.to_csv(csv_path, index=False)
            logger.info(
                "Updated %s: role scores (assistant_axis=%s baseline=%s steered=%s), comparisons (assistant_axis=%s steered=%s)",
                csv_path,
                pending_counts["assistant_axis_role"],
                pending_counts["baseline_role"],
                pending_counts["steered_role"],
                pending_counts["assistant_axis_compare"],
                pending_counts["steered_compare"],
            )
        else:
            logger.info("No changes needed for %s", csv_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
