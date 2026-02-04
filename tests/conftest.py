import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def test_device() -> str:
    """Determine the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def sample_questions() -> list[str]:
    """Sample questions for testing question bank and extraction."""
    return [
        "What motivates you in your work?",
        "How do you handle difficult situations?",
        "Describe your approach to problem-solving.",
        "What skills are most important in your field?",
        "How do you collaborate with others?",
        "What challenges do you face regularly?",
        "How do you stay current in your profession?",
        "What advice would you give to someone new?",
    ]


@pytest.fixture
def sample_vectors() -> dict[str, torch.Tensor]:
    """Sample persona vectors for geometry tests (768-dim like SmolLM)."""
    torch.manual_seed(42)
    return {
        "nurse_1": torch.randn(768),
        "nurse_2": torch.randn(768),
        "engineer_1": torch.randn(768),
        "engineer_2": torch.randn(768),
        "artist_1": torch.randn(768),
        "artist_2": torch.randn(768),
    }


@pytest.fixture
def sample_metadata() -> dict[str, dict]:
    """Sample metadata for geometry tests."""
    return {
        "nurse_1": {"riasec_primary": "S", "title": "Registered Nurse"},
        "nurse_2": {"riasec_primary": "S", "title": "Nurse Practitioner"},
        "engineer_1": {"riasec_primary": "R", "title": "Civil Engineer"},
        "engineer_2": {"riasec_primary": "R", "title": "Mechanical Engineer"},
        "artist_1": {"riasec_primary": "A", "title": "Graphic Designer"},
        "artist_2": {"riasec_primary": "A", "title": "Illustrator"},
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for test files, cleaned up after test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def mock_persona_vectors():
    """Mock persona vectors with realistic shapes (Qwen2.5-7B has 4096 hidden dim, 29 layers)."""
    torch.manual_seed(42)
    return {
        "prompt": torch.randn(1, 4096),
        "response": torch.randn(1, 4096),
        "all_layers": torch.randn(29, 1, 4096),
    }


@pytest.fixture
def mock_metadata():
    """Mock metadata for safetensors files."""
    return {
        "target_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "trait": "artistic",
        "layer_steering": "14",
    }


@pytest.fixture
def legacy_json_file(temp_dir, mock_persona_vectors):
    """Create a legacy JSON initialization file for backward compatibility testing."""
    json_dir = temp_dir / "artistic_persona_initialization" / "Qwen"
    json_dir.mkdir(parents=True)
    json_path = json_dir / "Qwen2.5-7B-Instruct.json"

    data = {
        "target_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "trait": "artistic",
        "layer_steering": 14,
        "device": "cpu",
        "prompt_persona_vector": mock_persona_vectors["prompt"].tolist(),
        "response_persona_vector": mock_persona_vectors["response"].tolist(),
        "all_layers_response_persona_vector": mock_persona_vectors["all_layers"].tolist(),
        "prompt_persona_vector_shape": list(mock_persona_vectors["prompt"].shape),
        "response_persona_vector_shape": list(mock_persona_vectors["response"].shape),
        "created_at": "2026-01-15T10:00:00",
        "dataset_info": {
            "trait": "artistic",
            "num_questions": 100,
            "num_pos_neg_pairs": 5,
        },
    }

    with open(json_path, "w") as f:
        json.dump(data, f)

    return json_path
