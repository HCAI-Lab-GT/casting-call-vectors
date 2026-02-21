"""
Async batch scorer for AbstractJudge subclasses.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as async_tqdm

from pvx import setup_logging
from pvx.abstraction.judges.abstract_judge import AbstractJudge
from pvx.implementations.judges.role_judge import PROMPT_TEMPLATE, RoleJudge

load_dotenv()

logger = setup_logging(name="batch-scorer")


class BatchScorer:
    """
    Async fan-out scorer that wraps any AbstractJudge subclass.

    Uses asyncio.Semaphore for back-pressure to avoid flooding the API.
    Scores are returned in the same order as the input items.
    Failed / unparseable scores are returned as -1 (with a warning logged).
    """

    def __init__(self, judge: AbstractJudge, concurrency: int = 50):
        """
        Args:
            judge: An AbstractJudge subclass configured for openai or vllm backend.
            concurrency: Maximum number of in-flight API calls at once.
        """
        if judge.backend == "hf_local":
            raise ValueError("BatchScorer only supports openai/vllm backends")
        self.judge = judge
        self.concurrency = concurrency

    async def _score_one(
        self, semaphore: asyncio.Semaphore, item: dict[str, Any], index: int
    ) -> tuple[int, int]:
        """Score a single item, returning (index, score)."""
        async with semaphore:
            try:
                score = await self.judge.async_judge(**item)
            except Exception as exc:
                logger.warning(
                    "item %d failed with %s: %s — defaulting to -1", index, type(exc).__name__, exc
                )
                score = -1
            return index, score

    async def score_batch(self, items: list[dict[str, Any]]) -> list[int]:
        """
        Score a batch of items concurrently.

        Args:
            items: List of kwarg dicts matching the judge's prompt template
                   (e.g. [{"question": "...", "answer": "..."}, ...]).

        Returns:
            List of int scores in the same order as items.
            Unparseable / failed items get score -1.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [
            asyncio.create_task(self._score_one(semaphore, item, i))
            for i, item in enumerate(items)
        ]

        results: list[tuple[int, int]] = []
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Scoring"):
            result = await coro
            results.append(result)

        # Re-order by original index
        results.sort(key=lambda t: t[0])
        return [score for _, score in results]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at line %d: %s", lineno, exc)
    return items


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


async def _run_cli(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    items = _load_jsonl(input_path)
    if not items:
        logger.error("No valid items found in %s", input_path)
        sys.exit(1)

    logger.info("Loaded %d items from %s", len(items), input_path)

    judge = RoleJudge(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        prompt_template=PROMPT_TEMPLATE,
        dtype=torch.float16,
    )

    scorer = BatchScorer(judge=judge, concurrency=args.concurrency)
    scores = await scorer.score_batch(
        [{"question": item.get("question", ""), "answer": item.get("answer", "")} for item in items]
    )

    output_records = []
    for item, score in zip(items, scores):
        record = dict(item)
        record["score"] = score
        output_records.append(record)

    _write_jsonl(output_path, output_records)
    logger.info("Wrote %d scored records to %s", len(output_records), output_path)

    failed = sum(1 for s in scores if s == -1)
    if failed:
        logger.warning("%d items received score -1 (parse failure or API error)", failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Async batch scorer for RoleJudge")
    ap.add_argument("--input", required=True, help="Path to input JSONL file")
    ap.add_argument("--output", required=True, help="Path to output JSONL file")
    ap.add_argument(
        "--backend", default="openai", choices=("openai", "vllm"), help="Inference backend"
    )
    ap.add_argument("--model", default="openai/gpt-oss-20b", help="Model identifier")
    ap.add_argument(
        "--base_url",
        default="https://api.together.xyz/v1",
        help="Base URL for OpenAI/vLLM endpoint",
    )
    ap.add_argument(
        "--api_key_env", default="TOGETHER_API_KEY", help="Env var holding the API key"
    )
    ap.add_argument(
        "--concurrency", type=int, default=50, help="Max concurrent API calls (default: 50)"
    )
    args = ap.parse_args()

    asyncio.run(_run_cli(args))
