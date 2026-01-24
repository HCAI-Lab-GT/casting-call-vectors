#!/usr/bin/env python
"""Generic persona vector analysis script.

This script provides a command-line interface for analyzing extracted
persona vectors. It is analysis-type agnostic - specific analysis frameworks
(RIASEC, Big Five, etc.) should use wrapper scripts or the Python API.

Usage:
    # Run basic PCA analysis
    uv run python scripts/run_analysis.py --vectors-dir outputs/vectors

    # With specific number of components
    uv run python scripts/run_analysis.py \\
        --vectors-dir outputs/vectors \\
        --n-pcs 10 \\
        --output-dir outputs/analysis

For RIASEC-specific analysis, see experiments/phase1/.
"""

import json
import logging
import sys
from pathlib import Path

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_vectors(vectors_dir: Path) -> tuple[dict, dict]:
    """Load extracted vectors and metadata from directory.

    Returns:
        Tuple of (vectors dict, metadata dict)
    """
    import torch

    vectors = {}
    metadata = {}

    for pt_file in vectors_dir.glob("*.pt"):
        persona_id = pt_file.stem
        json_file = pt_file.with_suffix(".json")

        # Load tensors
        tensors = torch.load(pt_file, weights_only=True)
        vectors[persona_id] = tensors["response_mean_diff"]

        # Load metadata
        if json_file.exists():
            with open(json_file) as f:
                meta = json.load(f)
                metadata[persona_id] = meta.get("metadata", {})

    return vectors, metadata


@click.command()
@click.option(
    "--vectors-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing extracted vectors (.pt and .json files)",
)
@click.option(
    "--output-dir",
    default="outputs/analysis",
    help="Directory for analysis outputs",
)
@click.option(
    "--n-pcs",
    default=10,
    type=int,
    help="Number of principal components to compute",
)
@click.option(
    "--wandb-project",
    default=None,
    help="W&B project for logging visualizations",
)
@click.option(
    "--wandb-run-name",
    default=None,
    help="W&B run name",
)
@click.option(
    "--save-plots",
    is_flag=True,
    help="Save plots to output directory",
)
def main(
    vectors_dir: str,
    output_dir: str,
    n_pcs: int,
    wandb_project: str | None,
    wandb_run_name: str | None,
    save_plots: bool,
):
    """Run PCA and clustering analysis on extracted persona vectors.

    This script performs basic geometric analysis. For framework-specific
    analysis (e.g., RIASEC correlations), use the Python API or
    experiment-specific scripts.
    """
    from pvx.analysis import PersonaGeometry, PersonaVisualizer

    logger.info(f"Loading vectors from {vectors_dir}")

    vectors_path = Path(vectors_dir)
    vectors, metadata = load_vectors(vectors_path)

    if not vectors:
        logger.error("No vectors found")
        sys.exit(1)

    logger.info(f"Loaded {len(vectors)} vectors")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize geometry analyzer
    geometry = PersonaGeometry(vectors=vectors, metadata=metadata)
    logger.info(
        f"Geometry analyzer: {len(geometry.persona_ids)} personas, {geometry.hidden_dim} dimensions"
    )

    # Run PCA
    logger.info(f"Computing PCA with {n_pcs} components")
    pca_result = geometry.compute_pca(n_components=n_pcs)

    # Log variance explained
    logger.info("Variance explained by top PCs:")
    cumulative = 0.0
    for i, var in enumerate(pca_result.explained_variance_ratio[:5]):
        cumulative += var
        logger.info(f"  PC{i + 1}: {var:.3f} (cumulative: {cumulative:.3f})")

    # Save PCA results
    pca_output = {
        "n_personas": len(vectors),
        "n_components": n_pcs,
        "explained_variance_ratio": pca_result.explained_variance_ratio.tolist(),
        "cumulative_variance": [
            sum(pca_result.explained_variance_ratio[: i + 1])
            for i in range(len(pca_result.explained_variance_ratio))
        ],
    }
    with open(output_path / "pca_results.json", "w") as f:
        json.dump(pca_output, f, indent=2)
    logger.info(f"Saved PCA results to {output_path / 'pca_results.json'}")

    # Run clustering
    logger.info("Computing clusters")
    cluster_result = geometry.compute_clusters(n_clusters=6)
    logger.info(f"Silhouette score: {cluster_result.silhouette_score:.3f}")

    # Create visualizer
    visualizer = PersonaVisualizer(geometry)

    if save_plots:
        # Save static plots
        import matplotlib.pyplot as plt

        fig = visualizer.plot_pca_2d(pca_result)
        fig.savefig(output_path / "pca_2d.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved 2D PCA plot")

        fig = visualizer.plot_variance_explained(pca_result)
        fig.savefig(output_path / "variance_explained.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved variance explained plot")

    # W&B logging
    if wandb_project:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or "analysis",
                config={
                    "n_vectors": len(vectors),
                    "n_pcs": n_pcs,
                    "vectors_dir": vectors_dir,
                },
            )

            visualizer.log_to_wandb(run, pca_result)
            run.finish()
            logger.info("Logged to W&B")

        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
