#!/usr/bin/env python3
"""Phase 1 End-to-End Pipeline Validation.

This script validates the complete extraction and analysis pipeline
using existing vocational personas with a small model (SmolLM2-135M).

The validation confirms:
1. Persona loading and baseline contrast
2. Activation extraction (even on CPU/MPS)
3. Vector saving and loading
4. PCA and geometry analysis
5. RIASEC correlation analysis

Usage:
    # Quick validation (2-3 minutes)
    uv run python experiments/phase1/validate_pipeline.py

    # With larger model for proper validation
    uv run python experiments/phase1/validate_pipeline.py --model smol3-3b

    # With full OLMo-7B (requires GPU)
    uv run python experiments/phase1/validate_pipeline.py --model olmo-7b
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import click
import torch

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pvx.analysis import PersonaGeometry
from pvx.analysis.riasec import RIASEC_COLORS, analyze_riasec, group_by_riasec
from pvx.config import MODEL_PRESETS, SMOLLM_MODELS
from pvx.extraction import ExtractionPipeline, QuestionBank
from pvx.extraction.pipeline import PersonaVector
from pvx.sources.base import BaselineSource
from pvx.sources.vocational import load_vocational_personas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_model_config(model_key: str) -> tuple[str, int]:
    """Get model ID and layer from key."""
    # Common aliases
    ALIASES = {
        "olmo-7b": "production",
        "olmo": "production",
    }
    model_key = ALIASES.get(model_key, model_key)

    if model_key in SMOLLM_MODELS:
        model_id = SMOLLM_MODELS[model_key]
        # Use middle layer for small models
        layer = 4 if "135" in model_id else 16
        return model_id, layer
    elif model_key in MODEL_PRESETS:
        config = MODEL_PRESETS[model_key]
        return config["model_id"], config["layer"]
    else:
        # Assume it's a HuggingFace model ID
        return model_key, 14


@click.command()
@click.option(
    "--model",
    default="smol2-135m",
    help="Model to use: smol2-135m, smol3-3b, olmo-7b, or HF model ID",
)
@click.option(
    "--num-questions",
    default=5,
    type=int,
    help="Number of questions per persona (more = better validation)",
)
@click.option(
    "--max-personas",
    default=4,
    type=int,
    help="Maximum personas to process",
)
@click.option(
    "--output-dir",
    default=None,
    help="Output directory (uses temp dir if not specified)",
)
@click.option(
    "--keep-outputs",
    is_flag=True,
    help="Keep output files after validation",
)
def main(
    model: str,
    num_questions: int,
    max_personas: int,
    output_dir: str | None,
    keep_outputs: bool,
):
    """Run end-to-end pipeline validation for Phase 1."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Pipeline Validation")
    logger.info("=" * 60)

    # Get model config
    model_id, layer = get_model_config(model)
    logger.info(f"Model: {model_id} (layer {layer})")

    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(tempfile.mkdtemp(prefix="pvx_validate_"))
        logger.info(f"Using temp directory: {output_path}")

    # Step 1: Load personas
    logger.info("-" * 60)
    logger.info("Step 1: Loading vocational personas")
    logger.info("-" * 60)

    personas = load_vocational_personas(limit=max_personas)
    if not personas:
        logger.error("No personas found! Generate with:")
        logger.error("  uv run python scripts/generate_vocational_personas.py --limit 10")
        sys.exit(1)

    logger.info(f"Loaded {len(personas)} personas:")
    for p in personas:
        meta = p.get_metadata()
        logger.info(f"  - {meta.get('title', p.persona_id)} ({meta.get('riasec_primary', '?')})")

    # Step 2: Load questions
    logger.info("-" * 60)
    logger.info("Step 2: Loading extraction questions")
    logger.info("-" * 60)

    questions_path = Path("_vendor/assistant-axis/data/extraction_questions.jsonl")
    if questions_path.exists():
        questions = QuestionBank.from_assistant_axis()
        logger.info(f"Loaded {len(questions)} questions from assistant-axis")
    else:
        questions = QuestionBank.from_fallback()
        logger.info(f"Using {len(questions)} fallback questions")

    # Step 3: Run extraction
    logger.info("-" * 60)
    logger.info("Step 3: Running extraction pipeline")
    logger.info("-" * 60)

    vectors_dir = output_path / "vectors"
    pipeline = ExtractionPipeline(
        model_id=model_id,
        layer=layer,
        questions=questions,
        output_dir=vectors_dir,
        wandb_project=None,  # Disable W&B for validation
    )

    logger.info(f"Pipeline: {pipeline}")

    baseline = BaselineSource.from_vocational_default()
    vectors = pipeline.extract_batch(
        sources=personas,
        baseline=baseline,
        num_questions=num_questions,
        checkpoint_every=1,
    )

    logger.info(f"Extracted {len(vectors)} persona vectors")

    # Verify extraction stats
    for vec in vectors:
        stats = vec.extraction_stats
        logger.info(
            f"  {vec.persona_id}: {stats.get('valid_pairs', 0)}/{stats.get('total_pairs', 0)} valid pairs, "
            f"norm={torch.norm(vec.prompt_last_diff).item():.3f}"
        )

    # Step 4: Save and reload vectors
    logger.info("-" * 60)
    logger.info("Step 4: Verifying vector persistence")
    logger.info("-" * 60)

    # Try loading back
    loaded_vectors = []
    for pt_file in vectors_dir.glob("*.pt"):
        vec = PersonaVector.load(pt_file.with_suffix(""))
        loaded_vectors.append(vec)
        logger.info(f"  Loaded: {vec.persona_id} (shape: {vec.prompt_last_diff.shape})")

    assert len(loaded_vectors) == len(vectors), "Vector count mismatch after reload"
    logger.info("Vector persistence: OK")

    # Step 5: Geometry analysis
    logger.info("-" * 60)
    logger.info("Step 5: Running geometry analysis")
    logger.info("-" * 60)

    # Build vectors and metadata dicts
    vectors_dict = {v.persona_id: v.prompt_last_diff.squeeze() for v in loaded_vectors}
    metadata_dict = {v.persona_id: dict(v.metadata) for v in loaded_vectors}

    geometry = PersonaGeometry(vectors=vectors_dict, metadata=metadata_dict)
    logger.info(f"Geometry: {geometry.n_personas} personas, {geometry.hidden_dim} dimensions")

    # PCA
    n_components = min(5, len(vectors_dict) - 1)
    pca_result = geometry.compute_pca(n_components=n_components)
    logger.info("PCA variance explained:")
    cumulative = 0.0
    for i, var in enumerate(pca_result.explained_variance_ratio[:5]):
        cumulative += var
        logger.info(f"  PC{i + 1}: {var:.3f} (cumulative: {cumulative:.3f})")

    # Clustering
    n_clusters = min(3, len(vectors_dict) - 1)
    if n_clusters >= 2:
        cluster_result = geometry.cluster(n_clusters=n_clusters)
        logger.info(f"Clustering: {cluster_result.n_clusters} clusters, silhouette={cluster_result.silhouette_score:.3f}")

    # Step 6: RIASEC analysis
    logger.info("-" * 60)
    logger.info("Step 6: Running RIASEC analysis")
    logger.info("-" * 60)

    groups = group_by_riasec(geometry)
    logger.info("RIASEC distribution:")
    for dim in "RIASEC":
        count = len(groups.get(dim, []))
        if count > 0:
            logger.info(f"  {dim}: {count} personas")

    # Only run full RIASEC analysis if we have enough personas
    min_per_group = 2
    valid_groups = sum(1 for g in groups.values() if len(g) >= min_per_group)
    if valid_groups >= 2:
        riasec_result = analyze_riasec(
            geometry=geometry,
            pca_result=pca_result,
            n_pcs=min(3, n_components),
        )
        logger.info(f"RIASEC alignment score: {riasec_result.axis_alignment_score:.3f}")
        logger.info(f"GO/NO-GO: {'GO' if riasec_result.go_no_go else 'NO-GO (insufficient data)'}")
    else:
        logger.warning(f"Insufficient data for RIASEC analysis (need at least 2 groups with 2+ personas)")

    # Step 7: Save analysis results
    logger.info("-" * 60)
    logger.info("Step 7: Saving analysis results")
    logger.info("-" * 60)

    analysis_dir = output_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Save PCA results
    pca_output = {
        "n_personas": len(vectors_dict),
        "n_components": n_components,
        "explained_variance_ratio": pca_result.explained_variance_ratio.tolist(),
        "model": model_id,
        "layer": layer,
        "num_questions": num_questions,
    }
    with open(analysis_dir / "pca_results.json", "w") as f:
        json.dump(pca_output, f, indent=2)
    logger.info(f"Saved: {analysis_dir / 'pca_results.json'}")

    # Summary
    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_path}")
    logger.info("")
    logger.info("Pipeline validated successfully!")
    logger.info("")
    logger.info("Next steps for Phase 1:")
    logger.info("  1. Generate 150 personas (25 per RIASEC type):")
    logger.info("     for r in R I A S E C; do")
    logger.info("       uv run python scripts/generate_vocational_personas.py --riasec $r --limit 25")
    logger.info("     done")
    logger.info("")
    logger.info("  2. Run full extraction (requires GPU):")
    logger.info("     uv run python scripts/run_extraction.py \\")
    logger.info("       --persona-dir persona_data/vocational_personas/instructions \\")
    logger.info("       --model allenai/OLMo-7B-Instruct \\")
    logger.info("       --wandb-project pvx-phase1")
    logger.info("")
    logger.info("  3. Run RIASEC analysis:")
    logger.info("     uv run python scripts/run_analysis.py \\")
    logger.info("       --vectors-dir outputs/vectors \\")
    logger.info("       --wandb-project pvx-phase1")

    if not keep_outputs and not output_dir:
        import shutil
        shutil.rmtree(output_path)
        logger.info(f"\nCleaned up temp directory: {output_path}")


if __name__ == "__main__":
    main()
