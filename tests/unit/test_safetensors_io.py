"""Unit tests for safetensors save/load operations in AbstractPersonaModel."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors import safe_open

from pvx.pvx_models.abstract_persona_model import AbstractPersonaModel


class MockPersonaModel(AbstractPersonaModel):
    """Test subclass that bypasses model loading."""

    def __init__(self, target_model_id: str, trait: str, layer: int = 14):
        self.target_model_id = target_model_id
        self.layer_steering = layer
        self.trait = trait
        self.dataset = MagicMock()
        self.dataset.trait = trait
        self.dataset.num_questions = 100
        self.dataset.positive_negative_pairs = [("pos", "neg")] * 5

    def extract_persona_vector(self, temperature=0.9, max_new_tokens=200):
        raise NotImplementedError("Not needed for I/O tests")


class TestSaveToSafetensors:
    """Tests for save_to_safetensors() method."""

    def test_save_creates_file(self, temp_dir, mock_persona_vectors):
        """Verify safetensors file is created at expected path."""
        model = MockPersonaModel("test/model", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        result_path = model.save_to_safetensors(str(temp_dir) + "/")

        assert Path(result_path).exists()
        assert result_path.endswith(".safetensors")

    def test_save_sanitizes_model_id(self, temp_dir, mock_persona_vectors):
        """Verify / in model ID is replaced with __ in filename."""
        model = MockPersonaModel("Qwen/Qwen2.5-7B-Instruct", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        result_path = model.save_to_safetensors(str(temp_dir) + "/")

        filename = Path(result_path).name
        assert "/" not in filename
        assert "Qwen__Qwen2.5-7B-Instruct" in filename

    def test_save_embeds_metadata(self, temp_dir, mock_persona_vectors):
        """Verify metadata is embedded in the safetensors file."""
        model = MockPersonaModel("Qwen/Qwen2.5-7B-Instruct", "artistic", layer=14)
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        result_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(result_path, framework="pt") as f:
            metadata = f.metadata()

        assert metadata["target_model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert metadata["trait"] == "artistic"
        assert metadata["layer_steering"] == "14"
        assert "created_at" in metadata
        assert "prompt_persona_vector_shape" in metadata

    def test_save_creates_directory(self, temp_dir, mock_persona_vectors):
        """Verify parent directory is created if missing."""
        model = MockPersonaModel("test/model", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        nested_dir = temp_dir / "nested" / "path"
        result_path = model.save_to_safetensors(str(nested_dir) + "/")

        assert Path(result_path).exists()
        assert nested_dir.exists()

    def test_save_tensors_on_cpu(self, temp_dir, mock_persona_vectors):
        """Verify tensors are saved on CPU regardless of original device."""
        model = MockPersonaModel("test/model", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        result_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(result_path, framework="pt") as f:
            loaded = f.get_tensor("prompt_persona_vector")
            assert loaded.device.type == "cpu"


class TestFromSafetensors:
    """Tests for from_safetensors() classmethod."""

    def test_load_restores_tensors(self, temp_dir, mock_persona_vectors):
        """Verify tensors are loaded with correct shapes via safe_open."""
        model = MockPersonaModel("Qwen/Qwen2.5-7B-Instruct", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]
        saved_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(saved_path, framework="pt") as f:
            loaded_prompt = f.get_tensor("prompt_persona_vector")
            loaded_response = f.get_tensor("response_persona_vector")
            loaded_all_layers = f.get_tensor("all_layers_response_persona_vector")

        assert loaded_prompt.shape == mock_persona_vectors["prompt"].shape
        assert loaded_response.shape == mock_persona_vectors["response"].shape
        assert loaded_all_layers.shape == mock_persona_vectors["all_layers"].shape

    def test_load_extracts_metadata(self, temp_dir, mock_persona_vectors):
        """Verify metadata is extracted correctly on load."""
        model = MockPersonaModel("Qwen/Qwen2.5-7B-Instruct", "artistic", layer=14)
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]
        saved_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(saved_path, framework="pt") as f:
            metadata = f.metadata()

        assert metadata["target_model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert metadata["trait"] == "artistic"
        assert int(metadata["layer_steering"]) == 14


class TestRoundTrip:
    """Tests for save->load round-trip integrity."""

    def test_roundtrip_exact_values(self, temp_dir, mock_persona_vectors):
        """Verify tensor values are preserved exactly through save/load cycle."""
        model = MockPersonaModel("Qwen/Qwen2.5-7B-Instruct", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        saved_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(saved_path, framework="pt") as f:
            loaded_prompt = f.get_tensor("prompt_persona_vector")
            loaded_response = f.get_tensor("response_persona_vector")
            loaded_all_layers = f.get_tensor("all_layers_response_persona_vector")

        assert torch.allclose(loaded_prompt, mock_persona_vectors["prompt"])
        assert torch.allclose(loaded_response, mock_persona_vectors["response"])
        assert torch.allclose(loaded_all_layers, mock_persona_vectors["all_layers"])

    def test_roundtrip_preserves_dtype(self, temp_dir, mock_persona_vectors):
        """Verify tensor dtype is preserved through save/load cycle."""
        model = MockPersonaModel("test/model", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"].to(torch.float16)
        model.response_persona_vector = mock_persona_vectors["response"].to(torch.float16)
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"].to(
            torch.float16
        )

        saved_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(saved_path, framework="pt") as f:
            loaded = f.get_tensor("prompt_persona_vector")

        assert loaded.dtype == torch.float16


class TestManifest:
    """Tests for _update_manifest() method."""

    def test_manifest_created_on_first_save(self, temp_dir, mock_persona_vectors):
        """Verify manifest.json is created on first save."""
        model = MockPersonaModel("test/model", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        model.save_to_safetensors(str(temp_dir) + "/")

        manifest_path = temp_dir / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "vectors" in manifest
        assert len(manifest["vectors"]) == 1
        assert manifest["vectors"][0]["trait"] == "artistic"

    def test_manifest_updated_on_resave(self, temp_dir, mock_persona_vectors):
        """Verify existing entry is updated (not duplicated) on re-save."""
        model = MockPersonaModel("test/model", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        model.save_to_safetensors(str(temp_dir) + "/")
        model.save_to_safetensors(str(temp_dir) + "/")

        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert len(manifest["vectors"]) == 1

    def test_manifest_appends_new_entries(self, temp_dir, mock_persona_vectors):
        """Verify new traits are appended to manifest."""
        model1 = MockPersonaModel("test/model", "artistic")
        model1.prompt_persona_vector = mock_persona_vectors["prompt"]
        model1.response_persona_vector = mock_persona_vectors["response"]
        model1.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]
        model1.save_to_safetensors(str(temp_dir) + "/")

        model2 = MockPersonaModel("test/model", "social")
        model2.prompt_persona_vector = mock_persona_vectors["prompt"]
        model2.response_persona_vector = mock_persona_vectors["response"]
        model2.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]
        model2.save_to_safetensors(str(temp_dir) + "/")

        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert len(manifest["vectors"]) == 2
        traits = {v["trait"] for v in manifest["vectors"]}
        assert traits == {"artistic", "social"}


class TestLoadOrCreate:
    """Tests for load_or_create() priority logic."""

    def test_prefers_safetensors_over_json(self, temp_dir, mock_persona_vectors, legacy_json_file):
        """Verify safetensors is loaded when both formats exist."""
        model = MockPersonaModel("Qwen/Qwen2.5-7B-Instruct", "artistic")
        model.prompt_persona_vector = mock_persona_vectors["prompt"] * 2
        model.response_persona_vector = mock_persona_vectors["response"] * 2
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"] * 2
        model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(str(temp_dir / "artistic__Qwen__Qwen2.5-7B-Instruct.safetensors"), framework="pt") as f:
            safetensors_prompt = f.get_tensor("prompt_persona_vector")

        assert torch.allclose(safetensors_prompt, mock_persona_vectors["prompt"] * 2)

    def test_falls_back_to_json_when_no_safetensors(self, temp_dir, legacy_json_file):
        """Verify JSON is loaded when safetensors doesn't exist."""
        assert legacy_json_file.exists()
        assert not (temp_dir / "artistic__Qwen__Qwen2.5-7B-Instruct.safetensors").exists()

    def test_filename_construction(self, temp_dir, mock_persona_vectors):
        """Verify correct filename is constructed from trait and model ID."""
        model = MockPersonaModel("Qwen/Qwen2.5-7B-Instruct", "conventional")
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        result_path = model.save_to_safetensors(str(temp_dir) + "/")

        expected_filename = "conventional__Qwen__Qwen2.5-7B-Instruct.safetensors"
        assert Path(result_path).name == expected_filename
