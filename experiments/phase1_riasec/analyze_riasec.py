#!/usr/bin/env python
"""Phase 1 RIASEC geometry analysis experiment.

This script runs RIASEC-specific analysis on extracted persona vectors:
1. PCA with RIASEC correlation analysis
2. RIASEC contrast vectors (R-S, I-E, A-C)
3. Cluster purity by RIASEC type
4. Go/No-Go evaluation for H1 hypothesis

Usage:
    # Run full analysis
    uv run python experiments/phase1_riasec/analyze_riasec.py

    # With custom vectors directory
    uv run python experiments/phase1_riasec/analyze_riasec.py \\
        --vectors-dir outputs/phase1_riasec/vectors

    # Generate interactive plots
    uv run python experiments/phase1_riasec/analyze_riasec.py --interactive
"""

import json
import logging
import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Phase 1 configuration
DEFAULT_CONFIG = {
    "vectors_dir": "outputs/phase1_riasec/vectors",
    "output_dir": "outputs/phase1_riasec/analysis",
    "wandb_project": "pvx-phase1",
    "n_pcs": 10,
    "alignment_threshold": 0.6,  # Go/no-go threshold for H1
}


def load_vectors(vectors_dir: Path):
    """Load extracted vectors and metadata."""
    import torch

    vectors = {}
    metadata = {}

    for pt_file in vectors_dir.glob("*.pt"):
        persona_id = pt_file.stem
        json_file = pt_file.with_suffix(".json")

        tensors = torch.load(pt_file, weights_only=True)
        vectors[persona_id] = tensors["response_mean_diff"]

        if json_file.exists():
            with open(json_file) as f:
                meta = json.load(f)
                metadata[persona_id] = meta.get("metadata", {})

    return vectors, metadata


@click.command()
@click.option(
    "--vectors-dir",
    default=DEFAULT_CONFIG["vectors_dir"],
    type=click.Path(exists=True),
    help="Directory containing extracted vectors",
)
@click.option(
    "--output-dir",
    default=DEFAULT_CONFIG["output_dir"],
    help="Directory for analysis outputs",
)
@click.option(
    "--n-pcs",
    default=DEFAULT_CONFIG["n_pcs"],
    type=int,
    help="Number of principal components",
)
@click.option(
    "--threshold",
    default=DEFAULT_CONFIG["alignment_threshold"],
    type=float,
    help="Alignment threshold for H1 go/no-go (0-1)",
)
@click.option(
    "--wandb-project",
    default=DEFAULT_CONFIG["wandb_project"],
    help="W&B project for logging",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Generate interactive Plotly visualizations",
)
@click.option(
    "--no-wandb",
    is_flag=True,
    help="Disable W&B logging",
)
def main(
    vectors_dir: str,
    output_dir: str,
    n_pcs: int,
    threshold: float,
    wandb_project: str,
    interactive: bool,
    no_wandb: bool,
):
    """Run RIASEC-specific geometry analysis.

    Tests H1: Do RIASEC dimensions align with principal components?
    """
    from pvx.analysis import PersonaGeometry, PersonaVisualizer
    from pvx.analysis import riasec as riasec_analysis

    logger.info("Phase 1 RIASEC Analysis")
    logger.info(f"Vectors: {vectors_dir}")
    logger.info(f"Output: {output_dir}")

    # Load vectors
    vectors_path = Path(vectors_dir)
    vectors, metadata = load_vectors(vectors_path)

    if not vectors:
        logger.error("No vectors found")
        sys.exit(1)

    # Count RIASEC distribution
    riasec_counts = {}
    for pid, meta in metadata.items():
        r_type = meta.get("riasec_primary", "?")
        riasec_counts[r_type] = riasec_counts.get(r_type, 0) + 1

    logger.info(f"Loaded {len(vectors)} vectors")
    logger.info(f"RIASEC distribution: {riasec_counts}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize geometry analyzer
    geometry = PersonaGeometry(vectors=vectors, metadata=metadata)

    # Run PCA
    logger.info(f"Computing PCA ({n_pcs} components)")
    pca_result = geometry.compute_pca(n_components=n_pcs)

    logger.info("Variance explained:")
    cumulative = 0.0
    for i, var in enumerate(pca_result.explained_variance_ratio[:5]):
        cumulative += var
        logger.info(f"  PC{i+1}: {var:.3f} (cumulative: {cumulative:.3f})")

    # Run RIASEC-specific analysis
    logger.info("Running RIASEC analysis")
    riasec_result = riasec_analysis.analyze_riasec(
        geometry=geometry,
        pca_result=pca_result,
        n_pcs=n_pcs,
        alignment_threshold=threshold,
    )

    # H1 Go/No-Go decision
    logger.info("=" * 60)
    logger.info("H1 HYPOTHESIS TEST: RIASEC-PC Alignment")
    logger.info("=" * 60)
    logger.info(f"Alignment score: {riasec_result.axis_alignment_score:.3f}")
    logger.info(f"Threshold: {threshold}")

    if riasec_result.go_no_go:
        logger.info("RESULT: ✓ GO - RIASEC dimensions align with PCs")
        logger.info("Proceed with vocational persona steering experiments")
    else:
        logger.info("RESULT: ✗ NO-GO - Weak RIASEC-PC alignment")
        logger.info("Consider alternative frameworks (Big Five, direct traits)")

    logger.info("=" * 60)

    # Log contrast results
    logger.info("RIASEC Contrast Vectors:")
    for contrast in riasec_result.contrast_results:
        cosines_str = ", ".join(f"{c:.2f}" for c in (contrast.pc_cosines or [])[:3])
        logger.info(f"  {contrast.name}: PC cosines [{cosines_str}]")

    # Log PC correlations
    logger.info("PC-RIASEC Correlations:")
    for pc_idx, corrs in riasec_result.pc_correlations.items():
        top_dim = max(corrs, key=lambda d: abs(corrs[d]))
        top_corr = corrs[top_dim]
        logger.info(f"  PC{pc_idx+1}: {riasec_analysis.RIASEC_FULL_NAMES[top_dim]} (r={top_corr:.3f})")

    # Save results
    results = {
        "n_personas": len(vectors),
        "riasec_distribution": riasec_counts,
        "pca": {
            "n_components": n_pcs,
            "explained_variance_ratio": pca_result.explained_variance_ratio.tolist(),
        },
        "riasec_analysis": {
            "alignment_score": riasec_result.axis_alignment_score,
            "threshold": threshold,
            "go_no_go": riasec_result.go_no_go,
            "pc_correlations": {
                str(k): v for k, v in riasec_result.pc_correlations.items()
            },
            "contrasts": [
                {
                    "name": c.name,
                    "group_a_size": c.group_a_size,
                    "group_b_size": c.group_b_size,
                    "pc_cosines": c.pc_cosines,
                }
                for c in riasec_result.contrast_results
            ],
        },
    }

    with open(output_path / "riasec_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved analysis to {output_path / 'riasec_analysis.json'}")

    # Create visualizations
    visualizer = PersonaVisualizer(geometry)

    # Get RIASEC color function and map
    color_fn = riasec_analysis.get_riasec_color_fn()
    color_map = riasec_analysis.get_riasec_color_map()

    # Static plots
    import matplotlib.pyplot as plt

    fig = visualizer.plot_pca_2d(
        pca_result,
        color_by=color_fn,
        color_map=color_map,
    )
    fig.suptitle("RIASEC Personas in PC Space")
    fig.savefig(output_path / "riasec_pca_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 2D PCA plot")

    fig = visualizer.plot_variance_explained(pca_result)
    fig.savefig(output_path / "variance_explained.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved variance explained plot")

    # Interactive plots
    if interactive:
        try:
            import plotly.io as pio

            fig_3d = visualizer.plot_interactive_pca(
                pca_result,
                color_by=color_fn,
                color_map=color_map,
                hover_keys=["title", "riasec_primary"],
            )
            fig_3d.update_layout(title="RIASEC Personas in 3D PC Space")
            pio.write_html(fig_3d, output_path / "riasec_pca_3d.html")
            logger.info("Saved interactive 3D plot")

        except ImportError:
            logger.warning("plotly not installed, skipping interactive plots")

    # W&B logging
    if not no_wandb and wandb_project:
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                name="riasec-analysis",
                config={
                    "n_personas": len(vectors),
                    "n_pcs": n_pcs,
                    "alignment_threshold": threshold,
                },
            )

            # Log key metrics
            run.summary["alignment_score"] = riasec_result.axis_alignment_score
            run.summary["go_no_go"] = riasec_result.go_no_go
            run.summary["variance_explained_3pc"] = sum(
                pca_result.explained_variance_ratio[:3]
            )

            # Log visualizations
            visualizer.log_to_wandb(run, pca_result, color_by=color_fn)

            run.finish()
            logger.info("Logged to W&B")

        except ImportError:
            logger.warning("wandb not installed")

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
