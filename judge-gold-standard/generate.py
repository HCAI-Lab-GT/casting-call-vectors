"""CLI entrypoint for gold-standard conversation generation.

Orchestrates the interviewer ↔ interviewee pipeline: loads persona
instruction files, runs multi-turn conversations in parallel, and saves
each result as a per-persona JSON in persona_data/vocational_personas/gold_standard/.

Logs go to a file (gold_standard_gen.log), not the terminal — the
terminal shows only a tqdm progress bar.

Usage:
    uv run judge-gold-standard/generate.py --persona persona_data/.../architect.json
    uv run judge-gold-standard/generate.py --all --max-questions 4 --concurrency 5
"""

import asyncio
import logging
import os
from pathlib import Path

from cli import parse_args
from dotenv import load_dotenv
from openai import AsyncOpenAI
from runner import run_all, run_single

load_dotenv()

GOLD_STANDARD_DIR = (
    Path(__file__).parent.parent / "persona_data" / "vocational_personas" / "gold_standard"
)
LOG_FILE = Path(__file__).parent / "gold_standard_gen.log"


def _setup_logging():
    """Route all logs to a file. Terminal stays clean for tqdm."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)


def _build_client() -> AsyncOpenAI:
    """Build an AsyncOpenAI client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return AsyncOpenAI(base_url=base_url, api_key=api_key)
    return AsyncOpenAI(api_key=api_key)


def main():
    args = parse_args()
    _setup_logging()
    client = _build_client()

    if args.all:
        count = asyncio.run(run_all(args, client, GOLD_STANDARD_DIR))
    else:
        if not args.persona.exists():
            print(f"Persona file not found: {args.persona}")
            raise SystemExit(1)
        count = asyncio.run(run_single(args, client, GOLD_STANDARD_DIR))

    print(f"Done. {count} conversation(s) saved to {GOLD_STANDARD_DIR}")
    print(f"Logs: {LOG_FILE}")


if __name__ == "__main__":
    main()
