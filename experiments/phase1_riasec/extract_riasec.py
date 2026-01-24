#!/usr/bin/env python
"""Phase 1 RIASEC persona extraction experiment.

This script extracts persona vectors for vocational personas grouped by
RIASEC type (Holland codes). It is the main extraction entry point for
Phase 1 of the LM-VECTOR research.

Usage:
    # Extract all RIASEC types (150 personas total, 25 per type)
    uv run python experiments/phase1_riasec/extract_riasec.py

    # Extract specific RIASEC type
    uv run python experiments/phase1_riasec/extract_riasec.py --riasec S

    # Dry run to see what would be extracted
    uv run python experiments/phase1_riasec/extract_riasec.py --dry-run

SLURM submission:
    # Generate and submit SLURM jobs (one per RIASEC type)
    uv run python experiments/phase1_riasec/extract_riasec.py --slurm
"""

import logging
import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pvx.extraction import ExtractionPipeline, QuestionBank
from pvx.sources.vocational import load_vocational_personas
from pvx.sources.base import BaselineSource
from pvx.infra.slurm import write_job_scripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Phase 1 configuration
RIASEC_TYPES = ["R", "I", "A", "S", "E", "C"]
PERSONAS_PER_TYPE = 25
TOTAL_PERSONAS = 150

DEFAULT_CONFIG = {
    "model": "allenai/OLMo-7B-Instruct",
    "layer": 14,
    "num_questions": 50,
    "persona_dir": "persona_data/vocational_personas/instructions",
    "output_dir": "outputs/phase1_riasec/vectors",
    "wandb_project": "pvx-phase1",
}


@click.command()
@click.option(
    "--riasec",
    type=click.Choice(RIASEC_TYPES),
    default=None,
    help="Extract only this RIASEC type (default: all types)",
)
@click.option(
    "--limit",
    default=PERSONAS_PER_TYPE,
    type=int,
    help=f"Personas per RIASEC type (default: {PERSONAS_PER_TYPE})",
)
@click.option(
    "--model",
    default=DEFAULT_CONFIG["model"],
    help="HuggingFace model identifier",
)
@click.option(
    "--output-dir",
    default=DEFAULT_CONFIG["output_dir"],
    help="Base output directory",
)
@click.option(
    "--wandb-project",
    default=DEFAULT_CONFIG["wandb_project"],
    help="W&B project name",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    help="Resume from checkpoint directory",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be extracted without running",
)
@click.option(
    "--slurm",
    is_flag=True,
    help="Generate SLURM job scripts instead of running locally",
)
@click.option(
    "--slurm-dir",
    default="jobs/phase1_riasec",
    help="Directory for SLURM scripts",
)
def main(
    riasec: str | None,
    limit: int,
    model: str,
    output_dir: str,
    wandb_project: str,
    resume: str | None,
    dry_run: bool,
    slurm: bool,
    slurm_dir: str,
):
    """Extract RIASEC vocational persona vectors.

    Phase 1 target: 150 personas (25 per RIASEC type)
    """
    if slurm:
        # Generate SLURM scripts
        logger.info(f"Generating SLURM scripts in {slurm_dir}")
        scripts = write_job_scripts(
            output_dir=slurm_dir,
            persona_dir=DEFAULT_CONFIG["persona_dir"],
            model_id=model,
            output_base=output_dir,
            personas_per_job=limit,
            wandb_project=wandb_project,
        )
        logger.info(f"Created {len(scripts)} job scripts")
        logger.info(f"Submit all jobs: bash {slurm_dir}/submit_all.sh")
        return

    # Determine which RIASEC types to extract
    types_to_extract = [riasec] if riasec else RIASEC_TYPES

    logger.info(f"Phase 1 RIASEC Extraction")
    logger.info(f"Types: {types_to_extract}")
    logger.info(f"Personas per type: {limit}")
    logger.info(f"Model: {model}")

    # Load questions
    questions_path = Path("_vendor/assistant-axis/data/extraction_questions.jsonl")
    if questions_path.exists():
        questions = QuestionBank.from_assistant_axis(str(questions_path))
    else:
        questions = QuestionBank.from_fallback()
    logger.info(f"Loaded {len(questions)} questions")

    # Load personas for each RIASEC type
    all_personas = []
    for r_type in types_to_extract:
        personas = list(load_vocational_personas(
            directory=Path(DEFAULT_CONFIG["persona_dir"]),
            riasec_filter=r_type,
            limit=limit,
        ))
        logger.info(f"  {r_type}: {len(personas)} personas")
        all_personas.extend(personas)

    if not all_personas:
        logger.error("No personas found. Run generate_vocational_personas.py first.")
        sys.exit(1)

    logger.info(f"Total: {len(all_personas)} personas")

    if dry_run:
        logger.info("Dry run complete - would extract above personas")
        return

    # Create baseline
    baseline = BaselineSource.from_vocational_default()

    # Initialize pipeline with RIASEC-specific run name
    run_name = f"riasec-{riasec}" if riasec else "riasec-all"

    pipeline = ExtractionPipeline(
        model_id=model,
        layer=DEFAULT_CONFIG["layer"],
        questions=questions,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_run_name=run_name,
    )

    # Extract
    try:
        vectors = pipeline.extract_batch(
            sources=all_personas,
            baseline=baseline,
            num_questions=DEFAULT_CONFIG["num_questions"],
            checkpoint_every=1,
            resume_from=resume,
        )

        logger.info(f"Extraction complete: {len(vectors)} vectors")

        # RIASEC-specific summary
        riasec_counts = {}
        for v in vectors:
            r_type = v.metadata.get("riasec_primary", "?")
            riasec_counts[r_type] = riasec_counts.get(r_type, 0) + 1

        logger.info("RIASEC distribution of extracted vectors:")
        for r_type in RIASEC_TYPES:
            count = riasec_counts.get(r_type, 0)
            status = "✓" if count >= limit else "⚠"
            logger.info(f"  {r_type}: {count}/{limit} {status}")

    except KeyboardInterrupt:
        logger.warning("Extraction interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
