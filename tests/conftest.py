"""Shared pytest fixtures for persona-vectors tests."""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_device() -> str:
    """Detect available compute device.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def unit_model_id() -> str:
    """Return SmolLM2-135M model ID for fast unit tests."""
    return "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture
def smoke_model_id() -> str:
    """Return SmolLM3-3B model ID for proper smoke tests."""
    return "HuggingFaceTB/SmolLM3-3B"


@pytest.fixture
def sample_questions() -> list[str]:
    """Return sample questions for testing."""
    return [
        "What is the most important thing in your work?",
        "How do you handle difficult situations?",
        "What skills are essential for your role?",
        "How do you approach problem-solving?",
        "What motivates you in your profession?",
    ]


@pytest.fixture
def sample_persona_data() -> dict:
    """Return mock vocational persona data."""
    return {
        "soc_code": "29-1141.00",
        "title": "Registered Nurses",
        "riasec": {"R": 2.3, "I": 3.5, "A": 2.1, "S": 6.2, "E": 3.8, "C": 3.0},
        "riasec_primary": "S",
        "highpoint_codes": ["S", "I", "E"],
        "instructions": [
            "You are a Registered Nurse working in a hospital.",
            "You are a caring and professional nurse.",
            "You are an experienced healthcare professional.",
        ],
        "baseline_instructions": [
            "",
            "You are an AI assistant.",
            "You are a helpful assistant.",
        ],
    }


@pytest.fixture
def sample_vectors() -> dict[str, torch.Tensor]:
    """Return sample persona vectors for geometry testing."""
    torch.manual_seed(42)
    hidden_dim = 768
    return {
        "nurse_1": torch.randn(hidden_dim),
        "nurse_2": torch.randn(hidden_dim),
        "engineer_1": torch.randn(hidden_dim),
        "engineer_2": torch.randn(hidden_dim),
        "artist_1": torch.randn(hidden_dim),
        "artist_2": torch.randn(hidden_dim),
    }


@pytest.fixture
def sample_metadata() -> dict[str, dict]:
    """Return sample metadata for geometry testing."""
    return {
        "nurse_1": {"riasec_primary": "S", "title": "Registered Nurse"},
        "nurse_2": {"riasec_primary": "S", "title": "Nurse Practitioner"},
        "engineer_1": {"riasec_primary": "R", "title": "Software Engineer"},
        "engineer_2": {"riasec_primary": "R", "title": "Mechanical Engineer"},
        "artist_1": {"riasec_primary": "A", "title": "Graphic Designer"},
        "artist_2": {"riasec_primary": "A", "title": "Illustrator"},
    }
