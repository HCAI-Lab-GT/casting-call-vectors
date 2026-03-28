"""CLI argument parsing for gold-standard conversation generation.

Accepts persona selection in three ways:
  --persona <path>   Path to a single instruction JSON file.
  --all              Generate for every JSON in the instructions directory.
  --list <file>      Text file with JSON filenames to regenerate.
"""

import argparse
import re
from pathlib import Path

DEFAULT_MAX_QUESTIONS = 5
DEFAULT_CONCURRENCY = 4

INSTRUCTIONS_DIR = (
    Path(__file__).parent.parent / "persona_data" / "vocational_personas" / "instructions"
)


def parse_list_file(filepath: Path, instructions_dir: Path) -> list[Path]:
    """Parse a text file of JSON filenames into resolved instruction paths.

    Handles formats like: 'ambassador.json', "detective.json", ambassador.json
    with optional trailing commas and whitespace.
    """
    text = filepath.read_text(encoding="utf-8")
    filenames = re.findall(r"['\"]?(\w+\.json)['\"]?", text)
    paths = []
    for name in filenames:
        resolved = instructions_dir / name
        if resolved.exists():
            paths.append(resolved)
        else:
            print(f"Warning: skipping {name} — not found in {instructions_dir}")
    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate gold-standard persona conversations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--persona",
        type=Path,
        help="Path to a single persona instruction JSON file",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Generate for every persona in the instructions directory",
    )
    group.add_argument(
        "--list",
        type=Path,
        dest="list_file",
        help="Text file with JSON filenames to regenerate (one per line or comma-separated)",
    )
    parser.add_argument(
        "--interviewer-model",
        default="gpt-4o-mini",
        help="Model for the interviewer LLM (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--interviewee-model",
        default="gpt-4o-mini",
        help="Model for the interviewee LLM (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Which positive_prompt variant to use, 0-4 (default: 0)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help="Maximum interviewer questions per conversation (default: 5)",
    )
    parser.add_argument(
        "--instructions-dir",
        type=Path,
        default=INSTRUCTIONS_DIR,
        help="Directory containing persona instruction JSONs (used with --all/--list)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of personas to generate in parallel (default: 4)",
    )
    return parser.parse_args()
