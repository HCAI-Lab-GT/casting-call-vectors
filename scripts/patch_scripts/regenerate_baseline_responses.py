"""
Scan every row in a role's gold-standard CSV and regenerate any baseline
response that is missing or malformed, using the same API-based generation
as GoldPromptExperiments (AsyncOpenAI + gold label prompt).

Generates once per unique (role, question) and populates all alpha rows with
the same response. Requests are sent concurrently up to --max_concurrency.

Usage:
    # Dry run for all roles
    python scripts/patch_scripts/regenerate_baseline_responses.py --dry_run

    # Fix all roles (default concurrency=50)
    python scripts/patch_scripts/regenerate_baseline_responses.py

    # Fix specific roles with custom concurrency
    python scripts/patch_scripts/regenerate_baseline_responses.py --roles accountant lawyer --max_concurrency 20

    # Fix with a specific alpha filter
    python scripts/patch_scripts/regenerate_baseline_responses.py --roles accountant --alphas 1.0 2.5
"""

import argparse
import asyncio
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from pvx import setup_logging

load_dotenv()

logger = setup_logging(name="regenerate-baseline-responses")


# ---------------------------------------------------------------------------
# Malformedness detection (mirrors regenerate_malformed_responses.py)
# ---------------------------------------------------------------------------

def has_value(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def is_malformed(value: object) -> bool:
    if not has_value(value):
        return True
    return "dtype:" in str(value)


# ---------------------------------------------------------------------------
# Gold label prompt loading (mirrors GoldPromptExperiments._load_gold_baseline_messages)
# ---------------------------------------------------------------------------

def load_gold_baseline_messages(role: str, gold_prompts_dir: Path) -> list[dict[str, str]]:
    gold_path = gold_prompts_dir / f"{role.replace('/', '_')}_gold_label.json"
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold prompt file not found for role '{role}' at {gold_path}. "
            "Generate it with scripts/patch_scripts/create_gold_standard_prompts.py first."
        )

    with open(gold_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    messages: list[dict[str, str]] = []
    if isinstance(payload.get("gold_label_prompt"), dict):
        messages = payload["gold_label_prompt"].get("messages", [])
    elif isinstance(payload.get("gold_label_prompts"), list) and payload["gold_label_prompts"]:
        messages = payload["gold_label_prompts"][0].get("messages", [])

    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"Gold prompt file for role '{role}' does not contain baseline messages: {gold_path}"
        )

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role_name = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role_name not in {"system", "user", "assistant"} or not content:
            continue
        normalized.append({"role": role_name, "content": content})

    # Drop trailing user turn (the question slot) — we append the real question ourselves
    if normalized and normalized[-1]["role"] == "user":
        normalized = normalized[:-1]

    if not normalized:
        raise ValueError(
            f"Gold prompt file for role '{role}' has no valid baseline messages: {gold_path}"
        )

    return normalized


# ---------------------------------------------------------------------------
# Role list loading
# ---------------------------------------------------------------------------

def load_roles(role_list_path: Path) -> list[str]:
    with open(role_list_path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys()) if isinstance(data, dict) else list(data)


# ---------------------------------------------------------------------------
# Async task descriptor and worker
# ---------------------------------------------------------------------------

@dataclass
class GenerationTask:
    question: str
    messages: list[dict[str, str]]
    indices: list[Any]  # df row indices to populate with the result
    result: str | None = field(default=None, repr=False)


async def run_generation(
    task: GenerationTask,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    temperature: float,
    max_new_tokens: int,
) -> None:
    async with sem:
        response = await client.chat.completions.create(
            model=model,
            messages=task.messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        task.result = response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate missing/malformed baseline responses in role CSVs via async API."
    )
    parser.add_argument(
        "--input_dir",
        default="./experiment_data/gold_prompt_experiments/",
        help="Directory containing Comparison_GoldStandard_<role>.csv files",
    )
    parser.add_argument(
        "--gold_prompts_dir",
        default="./persona_data/gold_labels_prompts_dataset",
        help="Directory containing <role>_gold_label.json files",
    )
    parser.add_argument(
        "--role_list",
        default="./configs/role_list.json",
        help="Path to role_list.json (ignored when --roles is provided)",
    )
    parser.add_argument("--roles", nargs="+", default=None,
                        help="Roles to process directly (skips role_list.json)")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="Only process rows with these alpha values")
    parser.add_argument("--backend", default="openai", choices=["openai", "vllm"])
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Baseline generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--max_concurrency", type=int, default=50,
                        help="Max parallel API requests")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be regenerated without changing files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gold_prompts_dir = Path(args.gold_prompts_dir)
    allowed_alphas = set(args.alphas) if args.alphas else None

    roles = args.roles if args.roles else load_roles(Path(args.role_list))

    # --- Pre-scan: find unique questions that need regeneration ---
    # We generate once per question and populate all alpha rows with the same response.
    scan_results: list[tuple[Path, str, pd.DataFrame, dict[str, list[int]]]] = []
    global_question_count = 0
    global_row_count = 0
    missing_files: list[str] = []

    for role in roles:
        csv_path = input_dir / f"Comparison_GoldStandard_{role}.csv"
        if not csv_path.exists():
            missing_files.append(role)
            continue

        df = pd.read_csv(csv_path)
        if "baseline" not in df.columns:
            logger.warning("Skipping %s: no 'baseline' column", csv_path.name)
            continue
        if "question" not in df.columns or "alpha" not in df.columns:
            logger.warning("Skipping %s: missing 'question' or 'alpha' column", csv_path.name)
            continue

        df["baseline"] = df["baseline"].astype(object)

        question_indices: dict[str, list[int]] = {}
        for idx in df.index:
            if "sample_count" in df.columns and int(df.at[idx, "sample_count"]) != 50:
                continue
            alpha = df.at[idx, "alpha"]
            if allowed_alphas is not None and not any(abs(alpha - a) < 1e-9 for a in allowed_alphas):
                continue
            if is_malformed(df.at[idx, "baseline"]):
                question = str(df.at[idx, "question"]).strip()
                question_indices.setdefault(question, []).append(idx)

        if not question_indices:
            continue

        global_question_count += len(question_indices)
        global_row_count += sum(len(v) for v in question_indices.values())
        scan_results.append((csv_path, role, df, question_indices))

    if missing_files:
        logger.warning("No CSV found for %s role(s): %s", len(missing_files), ", ".join(missing_files))

    if global_question_count == 0:
        logger.info("No missing/malformed baseline responses found.")
        return

    logger.info(
        "=== Baseline regeneration summary: %s unique question(s) → %s row(s) across %s role(s) ===",
        global_question_count, global_row_count, len(scan_results),
    )

    for csv_path, role, df, question_indices in scan_results:
        logger.info("  %s: %s unique question(s), %s row(s) total",
                    role, len(question_indices), sum(len(v) for v in question_indices.values()))
        for question, indices in question_indices.items():
            alphas = sorted({float(df.at[i, "alpha"]) for i in indices})
            logger.info("    q=%r → %s alpha(s): %s", question[:50], len(alphas),
                        ", ".join(f"{a:.2f}" for a in alphas))

    if args.dry_run:
        logger.info(
            "Dry run complete: %s unique question(s) would be generated, "
            "populating %s row(s) — max_concurrency=%s",
            global_question_count, global_row_count, args.max_concurrency,
        )
        return

    # --- Async regeneration: all roles/questions fired concurrently ---
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"{args.api_key_env} or OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    sem = asyncio.Semaphore(args.max_concurrency)

    # Build all tasks across all roles upfront, keyed back to their role entry
    role_tasks: list[tuple[Path, str, pd.DataFrame, list[GenerationTask]]] = []
    for csv_path, role, df, question_indices in scan_results:
        try:
            seed_messages = load_gold_baseline_messages(role, gold_prompts_dir)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Skipping %s: %s", role, e)
            continue

        tasks = [
            GenerationTask(
                question=question,
                messages=deepcopy(seed_messages) + [{"role": "user", "content": question}],
                indices=indices,
            )
            for question, indices in question_indices.items()
        ]
        role_tasks.append((csv_path, role, df, tasks))

    all_tasks = [t for _, _, _, tasks in role_tasks for t in tasks]
    logger.info("Launching %s async generation tasks across %s role(s), concurrency=%s",
                len(all_tasks), len(role_tasks), args.max_concurrency)

    await asyncio.gather(
        *(run_generation(t, client, sem, args.model, args.temperature, args.max_new_tokens)
          for t in all_tasks),
        return_exceptions=True,
    )

    # Write results back per role
    total_regenerated = 0
    for csv_path, role, df, tasks in role_tasks:
        changed = 0
        for task in tasks:
            if task.result is None:
                logger.error("Generation failed for role=%s q=%r", role, task.question[:50])
                continue
            for idx in task.indices:
                df.at[idx, "baseline"] = task.result
                changed += 1
            logger.info("Populated %s row(s) for role=%s q=%r", len(task.indices), role, task.question[:50])

        if changed:
            df.to_csv(csv_path, index=False)
            logger.info("%s: saved %s row(s)", csv_path.name, changed)

        total_regenerated += changed

    logger.info("Completed: %s total baseline row(s) populated from %s unique generation(s)",
                total_regenerated, global_question_count)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
