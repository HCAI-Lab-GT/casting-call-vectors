#!/usr/bin/env python3
"""Phase 1 RIASEC Balanced Validation.

This script validates that RIASEC analysis works correctly with
balanced persona data (2 personas per RIASEC type = 12 total).

This is the go/no-go test before running the full 150 persona extraction.

Usage:
    # Quick validation (SmolLM2-135M, ~6-8 minutes)
    uv run python experiments/phase1/validate_riasec.py

    # With better model (SmolLM3-3B)
    uv run python experiments/phase1/validate_riasec.py --model smol3-3b

    # More questions per persona for better vectors
    uv run python experiments/phase1/validate_riasec.py --num-questions 10
"""

import json
import logging
import sys
from pathlib import Path

import click
import torch

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pvx.analysis import PersonaGeometry
from pvx.analysis.riasec import (
    RIASEC_COLORS,
    RIASEC_DIMS,
    RIASEC_FULL_NAMES,
    analyze_riasec,
    group_by_riasec,
)
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
    ALIASES = {"olmo-7b": "production", "olmo": "production"}
    model_key = ALIASES.get(model_key, model_key)

    if model_key in SMOLLM_MODELS:
        model_id = SMOLLM_MODELS[model_key]
        layer = 4 if "135" in model_id else 16
        return model_id, layer
    elif model_key in MODEL_PRESETS:
        config = MODEL_PRESETS[model_key]
        return config["model_id"], config["layer"]
    else:
        return model_key, 14


def load_balanced_personas(per_type: int = 2) -> list:
    """Load balanced personas (N per RIASEC type)."""
    all_personas = []
    for dim in RIASEC_DIMS:
        personas = load_vocational_personas(riasec_filter=dim, limit=per_type)
        logger.info(f"  {dim} ({RIASEC_FULL_NAMES[dim]}): {len(personas)} personas")
        all_personas.extend(personas)
    return all_personas


@click.command()
@click.option(
    "--model",
    default="smol2-135m",
    help="Model: smol2-135m, smol3-3b, olmo-7b, or HF model ID",
)
@click.option(
    "--num-questions",
    default=5,
    type=int,
    help="Number of questions per persona",
)
@click.option(
    "--per-type",
    default=2,
    type=int,
    help="Number of personas per RIASEC type",
)
@click.option(
    "--output-dir",
    default="outputs/riasec_validation",
    help="Output directory for vectors and analysis",
)
@click.option(
    "--skip-extraction",
    is_flag=True,
    help="Skip extraction, use existing vectors in output-dir",
)
def main(
    model: str,
    num_questions: int,
    per_type: int,
    output_dir: str,
    skip_extraction: bool,
):
    """Validate RIASEC analysis with balanced persona data."""
    logger.info("=" * 60)
    logger.info("PHASE 1: RIASEC Balanced Validation")
    logger.info("=" * 60)

    model_id, layer = get_model_config(model)
    output_path = Path(output_dir)
    vectors_dir = output_path / "vectors"

    logger.info(f"Model: {model_id} (layer {layer})")
    logger.info(f"Questions per persona: {num_questions}")
    logger.info(f"Personas per RIASEC type: {per_type}")
    logger.info(f"Total personas: {per_type * 6}")

    if not skip_extraction:
        # Step 1: Load balanced personas
        logger.info("-" * 60)
        logger.info("Step 1: Loading balanced personas")
        logger.info("-" * 60)

        personas = load_balanced_personas(per_type)
        logger.info(f"Total: {len(personas)} personas")

        # Step 2: Load questions
        logger.info("-" * 60)
        logger.info("Step 2: Loading extraction questions")
        logger.info("-" * 60)

        questions_path = Path("_vendor/assistant-axis/data/extraction_questions.jsonl")
        if questions_path.exists():
            questions = QuestionBank.from_assistant_axis()
            logger.info(f"Loaded {len(questions)} questions")
        else:
            questions = QuestionBank.from_fallback()
            logger.info(f"Using {len(questions)} fallback questions")

        # Step 3: Run extraction
        logger.info("-" * 60)
        logger.info("Step 3: Running extraction pipeline")
        logger.info("-" * 60)

        pipeline = ExtractionPipeline(
            model_id=model_id,
            layer=layer,
            questions=questions,
            output_dir=vectors_dir,
            wandb_project=None,
        )

        baseline = BaselineSource.from_vocational_default()
        vectors = pipeline.extract_batch(
            sources=personas,
            baseline=baseline,
            num_questions=num_questions,
            checkpoint_every=1,
        )
        logger.info(f"Extracted {len(vectors)} vectors")

    # Step 4: Load vectors for analysis
    logger.info("-" * 60)
    logger.info("Step 4: Loading vectors for analysis")
    logger.info("-" * 60)

    vectors_dict = {}
    metadata_dict = {}
    for pt_file in vectors_dir.glob("*.pt"):
        vec = PersonaVector.load(pt_file.with_suffix(""))
        vectors_dict[vec.persona_id] = vec.prompt_last_diff.squeeze()
        metadata_dict[vec.persona_id] = dict(vec.metadata)

    logger.info(f"Loaded {len(vectors_dict)} vectors")

    # Step 5: Geometry analysis
    logger.info("-" * 60)
    logger.info("Step 5: Running geometry analysis")
    logger.info("-" * 60)

    geometry = PersonaGeometry(vectors=vectors_dict, metadata=metadata_dict)
    logger.info(f"Geometry: {geometry.n_personas} personas, {geometry.hidden_dim} dims")

    # PCA
    n_components = min(10, geometry.n_personas - 1)
    pca_result = geometry.compute_pca(n_components=n_components)

    logger.info("Variance explained:")
    cumulative = 0.0
    for i, var in enumerate(pca_result.explained_variance_ratio[:5]):
        cumulative += var
        logger.info(f"  PC{i + 1}: {var:.3f} (cumulative: {cumulative:.3f})")

    # Step 6: RIASEC analysis
    logger.info("-" * 60)
    logger.info("Step 6: Running RIASEC analysis")
    logger.info("-" * 60)

    groups = group_by_riasec(geometry)
    logger.info("RIASEC distribution:")
    for dim in RIASEC_DIMS:
        count = len(groups.get(dim, []))
        logger.info(f"  {dim} ({RIASEC_FULL_NAMES[dim]}): {count}")

    # Run full RIASEC analysis
    riasec_result = analyze_riasec(
        geometry=geometry,
        pca_result=pca_result,
        n_pcs=min(5, n_components),
        alignment_threshold=0.3,  # Lower threshold for small sample
    )

    # Step 7: Results summary
    logger.info("-" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("-" * 60)

    logger.info(f"Alignment score: {riasec_result.axis_alignment_score:.3f}")

    if riasec_result.contrast_results:
        logger.info("\nContrast vectors (opposite RIASEC pairs):")
        for contrast in riasec_result.contrast_results:
            if contrast.pc_cosines:
                max_cos = max(abs(c) for c in contrast.pc_cosines[:3])
                max_pc = contrast.pc_cosines[:3].index(
                    max(contrast.pc_cosines[:3], key=abs)
                )
                logger.info(
                    f"  {contrast.name}: max|cos| = {max_cos:.3f} (PC{max_pc + 1})"
                )

    logger.info("\nPC correlations with RIASEC dimensions:")
    for pc_idx, corrs in riasec_result.pc_correlations.items():
        if corrs:
            top_dim = max(corrs, key=lambda d: abs(corrs[d]))
            top_corr = corrs[top_dim]
            logger.info(
                f"  PC{pc_idx + 1}: r={top_corr:.3f} with {RIASEC_FULL_NAMES[top_dim]}"
            )

    # GO/NO-GO decision
    logger.info("=" * 60)
    if riasec_result.go_no_go:
        logger.info("GO: RIASEC dimensions align with principal components!")
        logger.info("Proceed with full 150-persona extraction.")
    else:
        logger.info("NO-GO: RIASEC alignment below threshold")
        logger.info("This is expected with small sample sizes.")
        logger.info("Consider:")
        logger.info("  - Using more personas per type (--per-type 5)")
        logger.info("  - Using more questions (--num-questions 20)")
        logger.info("  - Using a larger model (--model smol3-3b)")
    logger.info("=" * 60)

    # Save results
    analysis_dir = output_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    results = {
        "model": model_id,
        "layer": layer,
        "num_questions": num_questions,
        "per_type": per_type,
        "total_personas": len(vectors_dict),
        "alignment_score": riasec_result.axis_alignment_score,
        "go_no_go": riasec_result.go_no_go,
        "pca_variance": pca_result.explained_variance_ratio.tolist(),
        "riasec_distribution": {dim: len(groups.get(dim, [])) for dim in RIASEC_DIMS},
    }
    with open(analysis_dir / "riasec_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {analysis_dir / 'riasec_validation.json'}")


if __name__ == "__main__":
    main()
