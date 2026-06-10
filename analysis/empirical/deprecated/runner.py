"""Async runner functions for gold-standard conversation generation.

Handles semaphore-bounded parallel execution of persona conversations
with tqdm progress tracking. Extracted from generate.py to keep both
files under the 150-line limit.
"""

import asyncio
import logging
from pathlib import Path

from conversation import run_conversation
from interviewee import get_system_prompt, list_available_personas, load_persona, slug_from_path
from openai import AsyncOpenAI
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


async def generate_for_persona(
    client: AsyncOpenAI,
    persona_path: Path,
    interviewer_model: str,
    interviewee_model: str,
    prompt_index: int,
    max_questions: int,
) -> dict:
    """Generate a gold-standard conversation for a single persona JSON file."""
    persona = load_persona(persona_path)
    system_prompt = get_system_prompt(persona, prompt_index)
    slug = slug_from_path(persona_path)
    concept = persona.get("concept", slug)

    conversation = await run_conversation(
        interviewer_client=client,
        interviewee_client=client,
        interviewer_model=interviewer_model,
        interviewee_model=interviewee_model,
        system_prompt=system_prompt,
        persona_slug=slug,
        persona_concept=concept,
        prompt_index=prompt_index,
        max_questions=max_questions,
    )
    logger.info("Generated %d msgs for %s", len(conversation.messages), slug)
    return conversation.model_dump()


def save_single(result: dict, output_dir: Path) -> Path:
    """Save a single conversation result as <slug>.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = result["persona_slug"]
    filepath = output_dir / f"{slug}.json"
    import json

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", filepath)
    return filepath


async def _run_one(sem, client, path, args, output_dir, pbar, errors):
    """Semaphore-guarded worker for a single persona."""
    slug = slug_from_path(path)
    async with sem:
        try:
            result = await generate_for_persona(
                client,
                path,
                args.interviewer_model,
                args.interviewee_model,
                args.prompt_index,
                args.max_questions,
            )
            save_single(result, output_dir)
        except Exception as e:
            logger.error("Failed for %s: %s", slug, e)
            errors.append(slug)
        finally:
            pbar.update(1)
            pbar.set_postfix(fail=len(errors), refresh=True)


async def _run_batch(paths, args, client, output_dir) -> int:
    """Shared parallel runner for a list of persona paths."""
    total = len(paths)
    sem = asyncio.Semaphore(args.concurrency)
    errors: list[str] = []

    pbar = tqdm(total=total, desc="Gold-standard", unit="persona", dynamic_ncols=True)
    tasks = [_run_one(sem, client, p, args, output_dir, pbar, errors) for p in paths]
    await asyncio.gather(*tasks)
    pbar.close()

    if errors:
        logger.warning("Failed personas (%d): %s", len(errors), ", ".join(errors))
    return total - len(errors)


async def run_all(args, client: AsyncOpenAI, output_dir: Path) -> int:
    """Generate for every persona with semaphore-bounded parallelism."""
    paths = list_available_personas(args.instructions_dir)
    return await _run_batch(paths, args, client, output_dir)


async def run_list(paths: list[Path], args, client: AsyncOpenAI, output_dir: Path) -> int:
    """Generate for an explicit list of persona paths in parallel."""
    return await _run_batch(paths, args, client, output_dir)


async def run_single(args, client: AsyncOpenAI, output_dir: Path) -> int:
    """Generate for a single persona JSON path."""
    result = await generate_for_persona(
        client,
        args.persona,
        args.interviewer_model,
        args.interviewee_model,
        args.prompt_index,
        args.max_questions,
    )
    save_single(result, output_dir)
    return 1
