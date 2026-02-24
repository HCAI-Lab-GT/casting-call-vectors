#!/usr/bin/env python
"""Generic persona vector extraction script.

This script provides a command-line interface for extracting persona vectors
from any PersonaSource implementation. It is persona-type agnostic - specific
experiments (RIASEC, Big Five, etc.) should use wrapper scripts or configs.

Usage:
    # Extract from a specific persona JSON file
    uv run python scripts/run_extraction.py --persona path/to/persona.json

    # Extract from a directory of persona files
    uv run python scripts/run_extraction.py --persona-dir path/to/personas/

    # With custom model and layer
    uv run python scripts/run_extraction.py \\
        --persona-dir personas/ \\
        --model allenai/OLMo-7B-Instruct \\
        --layer 14

    # With run tracking (recommended for experiments)
    uv run python scripts/run_extraction.py \\
        --persona-dir personas/ \\
        --model allenai/OLMo-7B-Instruct \\
        --track

For RIASEC-specific experiments, see experiments/phase1/.
"""

import logging
import sys
from pathlib import Path
from typing import Iterator

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pvx.config import RunConfig
from pvx.extraction import ExtractionPipeline, QuestionBank
from pvx.sources.base import BaselineSource, PersonaSource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_personas_from_path(path: Path) -> Iterator[PersonaSource]:
    """Load personas from file or directory.

    Attempts to detect persona type from file structure and load
    appropriate PersonaSource implementation.
    """
    # Import here to avoid circular deps
    from pvx.sources.vocational import VocationalPersonaSource

    if path.is_file():
        # Single file - try vocational format
        yield VocationalPersonaSource.from_json(path)
    elif path.is_dir():
        # Directory - load all JSON files
        for json_file in sorted(path.glob("*.json")):
            try:
                yield VocationalPersonaSource.from_json(json_file)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")


@click.command()
@click.option(
    "--persona",
    "persona_path",
    type=click.Path(exists=True),
    help="Path to single persona JSON file",
)
@click.option(
    "--persona-dir",
    type=click.Path(exists=True),
    help="Directory containing persona JSON files",
)
@click.option(
    "--model",
    default="allenai/OLMo-7B-Instruct",
    help="HuggingFace model identifier",
)
@click.option(
    "--layer",
    default=14,
    type=int,
    help="Layer for activation extraction",
)
@click.option(
    "--output-dir",
    default="outputs/vectors",
    help="Directory for saving extracted vectors",
)
@click.option(
    "--num-questions",
    default=50,
    type=int,
    help="Number of questions per persona",
)
@click.option(
    "--questions-path",
    default="_vendor/assistant-axis/data/extraction_questions.jsonl",
    help="Path to extraction questions JSONL",
)
@click.option(
    "--wandb-project",
    default=None,
    help="W&B project name (None to disable logging)",
)
@click.option(
    "--wandb-run-name",
    default=None,
    help="W&B run name (auto-generated if None)",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume from checkpoint directory",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of personas to extract",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be extracted without running",
)
@click.option(
    "--track",
    is_flag=True,
    help="Enable run tracking with RunConfig (outputs to outputs/runs/{model}/{run_id}/)",
)
def main(
    persona_path: str | None,
    persona_dir: str | None,
    model: str,
    layer: int,
    output_dir: str,
    num_questions: int,
    questions_path: str,
    wandb_project: str | None,
    wandb_run_name: str | None,
    resume: str | None,
    limit: int | None,
    dry_run: bool,
    track: bool,
):
    """Extract persona vectors using contrastive activation differences.

    Provide either --persona (single file) or --persona-dir (directory).
    The script automatically detects persona format and loads appropriately.
    """
    if not persona_path and not persona_dir:
        raise click.UsageError("Must provide either --persona or --persona-dir")

    logger.info("Starting persona vector extraction")
    logger.info(f"Model: {model}, Layer: {layer}")

    # Load questions
    questions_file = Path(questions_path)
    if not questions_file.exists():
        # Fall back to the local copy in persona_data
        questions_file = Path("persona_data/vocational_personas/questions/extraction_questions.jsonl")
    if questions_file.exists():
        questions = QuestionBank.from_jsonl(questions_file)
        logger.info(f"Loaded {len(questions)} questions from {questions_file}")
    else:
        logger.warning(f"Questions file not found: {questions_path}")
        logger.info("Using fallback questions")
        questions = QuestionBank.from_fallback()

    # Load personas
    # One of these is guaranteed to be set by the earlier check
    assert persona_path is not None or persona_dir is not None
    source_path = Path(persona_path) if persona_path else Path(persona_dir)  # type: ignore[arg-type]
    personas = list(load_personas_from_path(source_path))

    if limit:
        personas = personas[:limit]

    if not personas:
        logger.error("No personas found")
        sys.exit(1)

    logger.info(f"Loaded {len(personas)} personas")

    if dry_run:
        logger.info("Dry run - would extract:")
        for p in personas[:10]:
            meta = p.get_metadata()
            logger.info(f"  - {p.persona_id}: {meta.get('title', 'Unknown')}")
        if len(personas) > 10:
            logger.info(f"  ... and {len(personas) - 10} more")
        return

    # Create baseline
    baseline = BaselineSource.from_vocational_default()

    # Initialize pipeline with optional run tracking
    if track:
        # Create RunConfig for tracked experiment
        run_config = RunConfig.create(
            model_id=model,
            layer=layer,
            num_questions=num_questions,
            personas_dir=str(source_path),
        )
        pipeline = ExtractionPipeline.from_run_config(
            run_config=run_config,
            questions=questions,
            wandb_project=wandb_project,
        )
        logger.info(f"Run tracking enabled: {run_config.run_id}")
    else:
        # Legacy mode without run tracking
        pipeline = ExtractionPipeline(
            model_id=model,
            layer=layer,
            questions=questions,
            output_dir=output_dir,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )

    logger.info(f"Pipeline initialized: {pipeline}")

    # Run extraction
    try:
        vectors = pipeline.extract_batch(
            sources=personas,
            baseline=baseline,
            num_questions=num_questions,
            checkpoint_every=1,
            resume_from=resume,
        )

        logger.info(f"Extraction complete: {len(vectors)} vectors")

    except KeyboardInterrupt:
        logger.warning("Extraction interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
