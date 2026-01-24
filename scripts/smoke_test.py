#!/usr/bin/env python3
"""Smoke test script for persona-vectors.

Validates that all components work correctly on the local machine.
Runs without API calls using SmolLM models for Apple Silicon.

Usage:
    # Fast check (no model loading)
    uv run python scripts/smoke_test.py

    # Unit-level model check (SmolLM2-135M - fast)
    uv run python scripts/smoke_test.py --model smol2-135m

    # Full smoke test (SmolLM3-3B - proper validation)
    uv run python scripts/smoke_test.py --model smol3-3b
"""

import logging
import sys
from pathlib import Path

import click

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_imports() -> bool:
    """Verify all core imports work."""
    logger.info("Checking imports...")
    try:
        # Core modules - imports are intentionally unused (import verification)
        from pvx.analysis.geometry import ClusterResult, PCAResult, PersonaGeometry  # noqa: F401
        from pvx.config import MODEL_PRESETS, SMOLLM_MODELS  # noqa: F401
        from pvx.extraction.activations import ActivationExtractor, ActivationResult  # noqa: F401
        from pvx.extraction.pipeline import ExtractionPipeline, PersonaVector  # noqa: F401
        from pvx.extraction.questions import QuestionBank  # noqa: F401
        from pvx.sources.base import BaselineSource, PersonaSource  # noqa: F401

        logger.info("  All core imports OK")
        return True
    except ImportError as e:
        logger.error(f"  Import failed: {e}")
        return False


def check_onet_data() -> bool:
    """Verify O*NET data loads correctly."""
    logger.info("Checking O*NET data...")
    try:
        from pvx.data.onet_loader import ONETLoader

        data_path = Path("data/onet_raw")
        if not data_path.exists():
            logger.warning("  O*NET data not found (optional)")
            logger.warning("  Run: ./scripts/download_onet.sh")
            return True  # Not a failure

        loader = ONETLoader(data_path)
        occupations = loader.load_occupations()
        logger.info(f"  Loaded {len(occupations)} occupations")

        # Check RIASEC scores
        riasec = loader.get_riasec_scores()
        logger.info(f"  RIASEC scores for {len(riasec)} occupations")

        # Sample profile
        profile = loader.get_occupation_profile("29-1141.00")  # Registered Nurses
        logger.info(f"  Sample: {profile['title']} ({profile['riasec_primary']})")

        return True
    except Exception as e:
        logger.error(f"  O*NET loading failed: {e}")
        return False


def check_vocational_personas() -> bool:
    """Check if vocational persona files exist."""
    logger.info("Checking vocational personas...")
    try:
        persona_dir = Path("persona_data/vocational_personas/instructions")
        if not persona_dir.exists():
            logger.warning("  Vocational personas not generated (optional)")
            logger.warning("  Run: uv run python scripts/generate_vocational_personas.py")
            return True  # Not a failure

        json_files = list(persona_dir.glob("*.json"))
        logger.info(f"  Found {len(json_files)} persona files")

        if json_files:
            # Load one to verify format
            import json

            with open(json_files[0]) as f:
                _ = json.load(f)  # Verify valid JSON
            logger.info(f"  Sample persona: {json_files[0].stem}")

        return True
    except Exception as e:
        logger.error(f"  Persona check failed: {e}")
        return False


def check_question_bank() -> bool:
    """Verify QuestionBank works."""
    logger.info("Checking QuestionBank...")
    try:
        from pvx.extraction.questions import QuestionBank

        # Try loading from assistant-axis
        aa_path = Path("_vendor/assistant-axis/data/extraction_questions.jsonl")
        if aa_path.exists():
            bank = QuestionBank.from_assistant_axis()
            logger.info(f"  Loaded {len(bank)} questions from assistant-axis")
        else:
            # Create from list for testing
            test_questions = [
                "What is the most important thing in your work?",
                "How do you handle stress?",
                "What skills are essential?",
            ]
            bank = QuestionBank.from_list(test_questions)
            logger.info(f"  Created test bank with {len(bank)} questions")

        # Test sampling
        sample = bank.sample(min(2, len(bank)))
        logger.info(f"  Sampling works: got {len(sample)} questions")

        return True
    except Exception as e:
        logger.error(f"  QuestionBank check failed: {e}")
        return False


def check_geometry_analysis() -> bool:
    """Verify geometry analysis works on synthetic data."""
    logger.info("Checking geometry analysis...")
    try:
        import torch

        from pvx.analysis.geometry import PersonaGeometry

        # Create synthetic vectors
        torch.manual_seed(42)
        vectors = {f"persona_{i}": torch.randn(768) for i in range(10)}

        geometry = PersonaGeometry(vectors)
        logger.info(f"  Created geometry with {geometry.n_personas} vectors")

        # PCA
        pca = geometry.compute_pca(n_components=5)
        logger.info(f"  PCA: {pca.components_for_variance(0.9)} components for 90% variance")

        # Clustering
        clusters = geometry.cluster(n_clusters=3)
        logger.info(f"  Clustering: {clusters.n_clusters} clusters")

        # Distances
        distances = geometry.compute_distances()
        logger.info(f"  Distance matrix: {distances.shape}")

        return True
    except Exception as e:
        logger.error(f"  Geometry check failed: {e}")
        return False


def check_device() -> str:
    """Detect available compute device."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon MPS"
    else:
        device = "cpu"
        device_name = "CPU"

    logger.info(f"Compute device: {device_name} ({device})")
    return device


def check_model_loading(model_key: str) -> bool:
    """Load and test a SmolLM model."""
    from pvx.config import MODEL_PRESETS, SMOLLM_MODELS
    from pvx.extraction.activations import ActivationExtractor

    if model_key in SMOLLM_MODELS:
        model_id = SMOLLM_MODELS[model_key]
        layer = 4 if "135" in model_id else 16
    elif model_key in MODEL_PRESETS:
        config = MODEL_PRESETS[model_key]
        model_id = config["model_id"]
        layer = config["layer"]
    else:
        logger.error(f"Unknown model: {model_key}")
        return False

    logger.info(f"Loading model: {model_id}...")
    try:
        extractor = ActivationExtractor(
            model_id=model_id,
            layer=layer,
            device="auto",
        )
        logger.info(f"  Model loaded on {extractor.device} with dtype {extractor.dtype}")

        # Test extraction
        logger.info("Testing activation extraction...")
        result = extractor.extract(
            system_prompt="You are a helpful assistant.",
            question="What is 2+2?",
            max_new_tokens=32,
        )

        logger.info(f"  Prompt last shape: {result.prompt_last.shape}")
        logger.info(f"  Response mean shape: {result.response_mean.shape}")
        logger.info(f"  Generated {result.num_response_tokens} tokens")
        logger.info(f"  Response: {result.response_text[:100]}...")

        return True
    except Exception as e:
        logger.error(f"  Model loading failed: {e}")
        return False


@click.command()
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model to test: smol2-135m, smol3-3b, unit_test, smoke_test",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
def main(model: str | None, verbose: bool):
    """Run smoke tests for persona-vectors."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 50)
    logger.info("Persona-Vectors Smoke Test")
    logger.info("=" * 50)

    results = {}

    # Basic checks (no model loading)
    results["imports"] = check_imports()
    results["device"] = check_device()
    results["onet"] = check_onet_data()
    results["personas"] = check_vocational_personas()
    results["questions"] = check_question_bank()
    results["geometry"] = check_geometry_analysis()

    # Model loading (optional)
    if model:
        logger.info("-" * 50)
        results["model"] = check_model_loading(model)

    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)

    all_passed = True
    for name, passed in results.items():
        if isinstance(passed, str):
            status = passed
        else:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
        logger.info(f"  {name}: {status}")

    if all_passed:
        logger.info("\nAll checks passed!")
        sys.exit(0)
    else:
        logger.error("\nSome checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
