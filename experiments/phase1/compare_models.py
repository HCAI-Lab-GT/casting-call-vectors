#!/usr/bin/env python3
"""Compare results across multiple model runs.

Usage:
    # Compare latest runs from all models
    uv run python experiments/phase1/compare_models.py

    # Compare specific models
    uv run python experiments/phase1/compare_models.py --models olmo-7b-instruct smollm3-3b

    # Compare specific runs
    uv run python experiments/phase1/compare_models.py \
        --run olmo-7b-instruct/2026-01-25_abc123 \
        --run smollm3-3b/2026-01-25_def456

    # Save comparison to file
    uv run python experiments/phase1/compare_models.py --output outputs/comparisons/
"""

import logging
import sys
from pathlib import Path

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pvx.analysis import ModelComparison, save_comparison_plots
from pvx.outputs import OutputManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--models",
    "-m",
    multiple=True,
    help="Model slugs to compare (uses latest run for each)",
)
@click.option(
    "--run",
    "-r",
    "runs",
    multiple=True,
    help="Specific runs to compare (format: model_slug/run_id)",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Directory to save comparison results",
)
@click.option(
    "--all-models",
    is_flag=True,
    help="Compare latest runs from all available models",
)
def main(models: tuple, runs: tuple, output: str | None, all_models: bool):
    """Compare results across multiple model runs."""
    mgr = OutputManager()

    # Collect runs to compare
    run_configs = []

    if all_models:
        # Get latest run from each model
        for model_slug in mgr.list_models():
            run = mgr.get_latest(model_slug)
            if run:
                run_configs.append(run)
                logger.info(f"Added: {model_slug}/{run.run_id}")

    if models:
        # Get latest run for each specified model
        for model_slug in models:
            run = mgr.get_latest(model_slug)
            if run:
                run_configs.append(run)
                logger.info(f"Added: {model_slug}/{run.run_id}")
            else:
                logger.warning(f"No runs found for model: {model_slug}")

    if runs:
        # Parse specific run paths
        for run_path in runs:
            parts = run_path.split("/")
            if len(parts) != 2:
                logger.error(f"Invalid run path: {run_path} (expected model_slug/run_id)")
                continue
            model_slug, run_id = parts
            run = mgr.get_run(model_slug, run_id)
            if run:
                run_configs.append(run)
                logger.info(f"Added: {model_slug}/{run_id}")
            else:
                logger.warning(f"Run not found: {run_path}")

    if not run_configs:
        logger.error("No runs to compare. Use --all-models, --models, or --run")
        sys.exit(1)

    if len(run_configs) < 2:
        logger.warning("Only one run found - showing metrics anyway")

    # Run comparison
    comparison = ModelComparison(run_configs)

    # Print summary
    print(comparison.summary())

    # Save if output specified
    if output:
        output_path = comparison.save(output)
        logger.info(f"Saved comparison to: {output_path}")

        # Also save plots
        try:
            save_comparison_plots(comparison, output)
            logger.info(f"Saved comparison plots to: {output}")
        except Exception as e:
            logger.warning(f"Could not save plots: {e}")

    # Return non-zero if no GO results
    go_count = sum(1 for m in comparison.metrics if m.go_no_go)
    if go_count == 0 and len(run_configs) > 0:
        logger.warning("No models achieved GO status")


if __name__ == "__main__":
    main()
