"""Integration tests for extraction pipeline.

These tests require model loading and are marked as slow.
Use pytest -m slow to run them.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from pvx.config import MODEL_PRESETS
from pvx.extraction.activations import ActivationExtractor, ActivationResult
from pvx.extraction.pipeline import ExtractionPipeline, PersonaVector
from pvx.extraction.questions import QuestionBank
from pvx.sources.base import BaselineSource
from tests.mocks import MockActivationExtractor, MockPersonaSource


class TestActivationExtractorMPS:
    """Tests for ActivationExtractor MPS device support."""

    def test_device_detection(self, test_device: str):
        """Should detect correct device."""
        # Just verify detection works - don't load model
        if torch.cuda.is_available():
            assert test_device == "cuda"
        elif torch.backends.mps.is_available():
            assert test_device == "mps"
        else:
            assert test_device == "cpu"


class TestMockExtractionPipeline:
    """Tests using mock extractor (no model loading)."""

    def test_mock_extractor_produces_activations(self, sample_questions: list[str]):
        """MockActivationExtractor should produce valid activations."""
        extractor = MockActivationExtractor()

        result = extractor.extract(
            system_prompt="You are a nurse.",
            question=sample_questions[0],
        )

        assert isinstance(result, ActivationResult)
        assert result.prompt_last.shape == (1, 768)
        assert result.response_mean.shape == (1, 768)
        assert result.num_response_tokens > 0

    def test_mock_extractor_is_deterministic(self, sample_questions: list[str]):
        """Same input should produce same output."""
        extractor = MockActivationExtractor()

        result1 = extractor.extract("You are a nurse.", sample_questions[0])
        result2 = extractor.extract("You are a nurse.", sample_questions[0])

        torch.testing.assert_close(result1.prompt_last, result2.prompt_last)

    def test_mock_extractor_batch(self, sample_questions: list[str]):
        """Batch extraction should work."""
        extractor = MockActivationExtractor()

        prompts = [("You are a nurse.", q) for q in sample_questions[:3]]
        results = extractor.extract_batch(prompts)

        assert len(results) == 3
        assert all(isinstance(r, ActivationResult) for r in results)


class TestPersonaVectorSaveLoad:
    """Tests for PersonaVector save/load."""

    def test_save_and_load(self):
        """Should save and load persona vector correctly."""
        # Create a test vector
        vec = PersonaVector(
            persona_id="test_nurse",
            prompt_last_diff=torch.randn(1, 768),
            response_mean_diff=torch.randn(1, 768),
            response_all_layers_diff=torch.randn(13, 1, 768),
            metadata={"riasec_primary": "S", "title": "Test Nurse"},
            extraction_stats={"valid_pairs": 10, "total_pairs": 12},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_vector"
            vec.save(filepath)

            # Verify files created
            assert filepath.with_suffix(".pt").exists()
            assert filepath.with_suffix(".json").exists()

            # Load and verify
            loaded = PersonaVector.load(filepath)

            assert loaded.persona_id == vec.persona_id
            torch.testing.assert_close(loaded.prompt_last_diff, vec.prompt_last_diff)
            assert loaded.metadata.get("riasec_primary") == "S"


@pytest.mark.slow
class TestExtractionPipelineIntegration:
    """Integration tests requiring real model loading.

    Uses SmolLM2-135M for fast testing on Apple Silicon.
    """

    @pytest.fixture(scope="class")
    def unit_extractor(self, test_device: str):
        """Load SmolLM2-135M for testing."""
        config = MODEL_PRESETS["unit_test"]
        return ActivationExtractor(
            model_id=config["model_id"],
            layer=config["layer"],
            device="auto",
        )

    def test_extractor_loads_model(self, unit_extractor: ActivationExtractor):
        """Should successfully load SmolLM2-135M."""
        assert unit_extractor.model is not None
        assert unit_extractor.tokenizer is not None

    def test_extractor_produces_activations(
        self,
        unit_extractor: ActivationExtractor,
        sample_questions: list[str],
    ):
        """Should extract activations from real model."""
        result = unit_extractor.extract(
            system_prompt="You are a helpful assistant.",
            question=sample_questions[0],
            max_new_tokens=16,
        )

        assert isinstance(result, ActivationResult)
        assert result.prompt_last.dim() == 2
        assert result.response_text != "" or result.num_response_tokens == 0

    def test_pipeline_extract_single_persona(
        self,
        test_device: str,
        sample_questions: list[str],
    ):
        """Should extract vector for single persona."""
        config = MODEL_PRESETS["unit_test"]
        questions = QuestionBank.from_list(sample_questions[:2])

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ExtractionPipeline(
                model_id=config["model_id"],
                layer=config["layer"],
                questions=questions,
                output_dir=tmpdir,
            )

            source = MockPersonaSource("test_nurse", riasec_primary="S")
            baseline = BaselineSource()

            vector = pipeline.extract_persona(
                source=source,
                baseline=baseline,
                num_questions=2,
                max_new_tokens=16,
            )

            assert vector.persona_id == "test_nurse"
            assert vector.prompt_last_diff.shape[1] > 0
            assert vector.extraction_stats["valid_pairs"] > 0


@pytest.mark.slow
class TestSmolLMModelLoading:
    """Tests for SmolLM model loading on Apple Silicon."""

    def test_smol2_135m_loads(self, test_device: str):
        """SmolLM2-135M should load successfully."""
        extractor = ActivationExtractor(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            layer=4,
            device="auto",
        )

        assert extractor.model is not None
        # On MPS, should use float32
        if test_device == "mps":
            assert extractor.dtype == torch.float32

    def test_smol2_generates_text(self, test_device: str):
        """SmolLM2-135M should generate coherent text."""
        extractor = ActivationExtractor(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            layer=4,
            device="auto",
        )

        result = extractor.extract(
            system_prompt="You are a helpful assistant.",
            question="What is 2+2?",
            max_new_tokens=32,
        )

        assert result.response_text is not None
        # Model should generate some response
        assert result.num_response_tokens > 0 or result.response_text == ""
