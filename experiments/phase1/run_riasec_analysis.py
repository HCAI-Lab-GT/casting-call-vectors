#!/usr/bin/env python3
"""Phase 1: RIASEC Analysis and W&B Dashboard.

Runs comprehensive RIASEC analysis on extracted persona vectors and
generates visualizations for the W&B dashboard.

Usage:
    # Analyze extracted vectors
    uv run python experiments/phase1/run_riasec_analysis.py \
        --vectors-dir outputs/phase1_vectors

    # With W&B logging
    uv run python experiments/phase1/run_riasec_analysis.py \
        --vectors-dir outputs/phase1_vectors \
        --wandb-project pvx-phase1

    # Save plots locally
    uv run python experiments/phase1/run_riasec_analysis.py \
        --vectors-dir outputs/phase1_vectors \
        --save-plots
"""

import json
import logging
import sys
from pathlib import Path

import click

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pvx.analysis import PersonaGeometry
from pvx.analysis.riasec import (
    RIASEC_COLORS,
    RIASEC_DIMS,
    RIASEC_FULL_NAMES,
    analyze_riasec,
    compute_cluster_riasec_purity,
    group_by_riasec,
)
from pvx.extraction.pipeline import PersonaVector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_vectors(vectors_dir: Path) -> tuple[dict, dict]:
    """Load persona vectors and metadata from directory."""
    import torch

    vectors = {}
    metadata = {}

    for pt_file in vectors_dir.glob("*.pt"):
        vec = PersonaVector.load(pt_file.with_suffix(""))
        vectors[vec.persona_id] = vec.prompt_last_diff.squeeze()
        metadata[vec.persona_id] = dict(vec.metadata)

    return vectors, metadata


@click.command()
@click.option(
    "--vectors-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing extracted vectors",
)
@click.option(
    "--output-dir",
    default="outputs/phase1_analysis",
    help="Directory for analysis outputs",
)
@click.option(
    "--n-pcs",
    default=10,
    type=int,
    help="Number of principal components to analyze",
)
@click.option(
    "--alignment-threshold",
    default=0.6,
    type=float,
    help="Threshold for GO/NO-GO decision",
)
@click.option(
    "--wandb-project",
    default=None,
    help="W&B project for logging",
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
    alignment_threshold: float,
    wandb_project: str | None,
    save_plots: bool,
):
    """Run comprehensive RIASEC analysis on extracted persona vectors."""
    logger.info("=" * 60)
    logger.info("PHASE 1: RIASEC Analysis")
    logger.info("=" * 60)

    vectors_path = Path(vectors_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load vectors
    logger.info("-" * 60)
    logger.info("Loading vectors")
    logger.info("-" * 60)

    vectors, metadata = load_vectors(vectors_path)
    logger.info(f"Loaded {len(vectors)} vectors")

    # Check RIASEC distribution
    groups_raw = {}
    for pid, meta in metadata.items():
        primary = meta.get("riasec_primary")
        if primary:
            groups_raw.setdefault(primary, []).append(pid)

    logger.info("RIASEC distribution:")
    for dim in RIASEC_DIMS:
        count = len(groups_raw.get(dim, []))
        logger.info(f"  {dim} ({RIASEC_FULL_NAMES[dim]}): {count}")

    # Initialize geometry
    logger.info("-" * 60)
    logger.info("Running geometry analysis")
    logger.info("-" * 60)

    geometry = PersonaGeometry(vectors=vectors, metadata=metadata)
    logger.info(f"Geometry: {geometry.n_personas} personas, {geometry.hidden_dim} dims")

    # PCA analysis
    n_components = min(n_pcs, geometry.n_personas - 1)
    pca_result = geometry.compute_pca(n_components=n_components)

    logger.info(f"\nVariance explained (top {min(10, n_components)} PCs):")
    cumulative = 0.0
    for i, var in enumerate(pca_result.explained_variance_ratio[: min(10, n_components)]):
        cumulative += var
        logger.info(f"  PC{i + 1}: {var:.4f} (cumulative: {cumulative:.4f})")

    # RIASEC analysis
    logger.info("-" * 60)
    logger.info("Running RIASEC analysis")
    logger.info("-" * 60)

    riasec_result = analyze_riasec(
        geometry=geometry,
        pca_result=pca_result,
        n_pcs=min(5, n_components),
        alignment_threshold=alignment_threshold,
    )

    # Clustering analysis
    logger.info("-" * 60)
    logger.info("Running clustering analysis")
    logger.info("-" * 60)

    cluster_result = geometry.cluster(n_clusters=6)
    logger.info(f"K-means (k=6) silhouette score: {cluster_result.silhouette_score:.3f}")

    # Compute RIASEC purity of clusters
    purity = compute_cluster_riasec_purity(geometry, cluster_result.labels)
    logger.info("\nCluster RIASEC purity:")
    for cluster_id, info in sorted(purity.items()):
        logger.info(
            f"  Cluster {cluster_id}: {info['dominant']} ({info['purity']:.2f})"
        )

    mean_purity = sum(p["purity"] for p in purity.values()) / len(purity) if purity else 0
    logger.info(f"Mean purity: {mean_purity:.3f}")

    # Results summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\n1. PCA Structure")
    logger.info(f"   - {pca_result.components_for_variance(0.5)} PCs for 50% variance")
    logger.info(f"   - {pca_result.components_for_variance(0.8)} PCs for 80% variance")
    logger.info(f"   - {pca_result.components_for_variance(0.95)} PCs for 95% variance")

    logger.info(f"\n2. RIASEC Alignment")
    logger.info(f"   - Alignment score: {riasec_result.axis_alignment_score:.3f}")
    logger.info(f"   - Threshold: {alignment_threshold}")

    if riasec_result.contrast_results:
        logger.info("\n3. Contrast Vectors (opposite pairs)")
        for contrast in riasec_result.contrast_results:
            if contrast.pc_cosines:
                top_pc = max(range(len(contrast.pc_cosines[:5])),
                            key=lambda i: abs(contrast.pc_cosines[i]))
                max_cos = abs(contrast.pc_cosines[top_pc])
                logger.info(f"   - {contrast.name}: |cos| = {max_cos:.3f} (PC{top_pc + 1})")

    logger.info(f"\n4. Clustering")
    logger.info(f"   - Silhouette score: {cluster_result.silhouette_score:.3f}")
    logger.info(f"   - Mean RIASEC purity: {mean_purity:.3f}")

    # GO/NO-GO decision
    logger.info("\n" + "=" * 60)
    if riasec_result.go_no_go:
        logger.info("H1 HYPOTHESIS: GO")
        logger.info("RIASEC vocational dimensions are reflected in model activations!")
        logger.info("")
        logger.info("Interpretation:")
        logger.info("  - Persona vectors cluster by RIASEC type")
        logger.info("  - RIASEC contrast vectors align with principal components")
        logger.info("  - This suggests vocational personas induce systematic")
        logger.info("    activation patterns that mirror Holland's hexagon")
    else:
        logger.info("H1 HYPOTHESIS: NO-GO")
        logger.info(f"RIASEC alignment ({riasec_result.axis_alignment_score:.3f}) below threshold ({alignment_threshold})")
        logger.info("")
        logger.info("Possible reasons:")
        logger.info("  - Sample size may be too small")
        logger.info("  - Model layer may not be optimal")
        logger.info("  - RIASEC may not be the right framework for this model")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  - Try different layers (earlier vs later)")
        logger.info("  - Increase questions per persona")
        logger.info("  - Consider alternative personality frameworks (Big Five)")
    logger.info("=" * 60)

    # Save results
    results = {
        "n_personas": len(vectors),
        "hidden_dim": geometry.hidden_dim,
        "n_pcs_computed": n_components,
        "pca": {
            "explained_variance_ratio": pca_result.explained_variance_ratio.tolist(),
            "pcs_for_50_variance": pca_result.components_for_variance(0.5),
            "pcs_for_80_variance": pca_result.components_for_variance(0.8),
            "pcs_for_95_variance": pca_result.components_for_variance(0.95),
        },
        "riasec": {
            "alignment_score": riasec_result.axis_alignment_score,
            "alignment_threshold": alignment_threshold,
            "go_no_go": riasec_result.go_no_go,
            "distribution": {dim: len(groups_raw.get(dim, [])) for dim in RIASEC_DIMS},
            "contrasts": [
                {
                    "name": c.name,
                    "pc_cosines": c.pc_cosines,
                }
                for c in riasec_result.contrast_results
            ],
            "pc_correlations": {
                str(k): v for k, v in riasec_result.pc_correlations.items()
            },
        },
        "clustering": {
            "n_clusters": 6,
            "silhouette_score": cluster_result.silhouette_score,
            "mean_riasec_purity": mean_purity,
            "cluster_purity": {
                str(k): v for k, v in purity.items()
            },
        },
    }

    results_file = output_path / "riasec_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to: {results_file}")

    # Visualizations
    if save_plots or wandb_project:
        logger.info("-" * 60)
        logger.info("Generating visualizations")
        logger.info("-" * 60)

        from pvx.analysis.viz import PersonaVisualizer

        visualizer = PersonaVisualizer(geometry)

        if save_plots:
            import matplotlib.pyplot as plt

            # PCA 2D plot
            fig = visualizer.plot_pca_2d(
                pca_result,
                color_by=lambda pid, meta: meta.get("riasec_primary"),
                color_map=RIASEC_COLORS,
            )
            fig.savefig(output_path / "pca_2d.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved: {output_path / 'pca_2d.png'}")

            # Variance explained
            fig = visualizer.plot_variance_explained(pca_result)
            fig.savefig(output_path / "variance_explained.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved: {output_path / 'variance_explained.png'}")

        if wandb_project:
            try:
                import wandb

                run = wandb.init(
                    project=wandb_project,
                    name="riasec-analysis",
                    config={
                        "n_personas": len(vectors),
                        "n_pcs": n_components,
                        "alignment_threshold": alignment_threshold,
                    },
                )

                # Log metrics
                run.log({
                    "alignment_score": riasec_result.axis_alignment_score,
                    "silhouette_score": cluster_result.silhouette_score,
                    "mean_riasec_purity": mean_purity,
                    "go_no_go": riasec_result.go_no_go,
                })

                # Log visualizations
                visualizer.log_to_wandb(run, pca_result)

                # Log results file as artifact
                artifact = wandb.Artifact("riasec-analysis", type="analysis")
                artifact.add_file(str(results_file))
                run.log_artifact(artifact)

                run.finish()
                logger.info("  Logged to W&B")
            except ImportError:
                logger.warning("  wandb not installed, skipping W&B logging")

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
