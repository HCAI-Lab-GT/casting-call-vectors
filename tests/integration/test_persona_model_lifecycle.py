"""Integration tests for persona model save/load lifecycle."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors import safe_open

from pvx.pvx_models.abstract_persona_model import AbstractPersonaModel


class StubPersonaModel(AbstractPersonaModel):
    """Stub implementation for lifecycle testing without real model loading."""

    def __init__(
        self,
        target_model_id: str = "test/model",
        trait: str = "artistic",
        layer: int = 14,
        from_json: bool = False,
        **kwargs,
    ):
        self.target_model_id = target_model_id
        self.layer_steering = layer
        self.trait = trait

        if not from_json:
            self.dataset = MagicMock()
            self.dataset.trait = trait
            self.dataset.num_questions = 100
            self.dataset.positive_negative_pairs = [("pos", "neg")] * 5

    def extract_persona_vector(self, temperature=0.9, max_new_tokens=200):
        raise NotImplementedError("Stub does not extract vectors")


class TestPersonaModelLifecycle:
    """End-to-end tests for persona model serialization lifecycle."""

    def test_full_save_load_cycle(self, temp_dir, mock_persona_vectors):
        """Test complete lifecycle: create -> save -> load -> verify."""
        original = StubPersonaModel(
            target_model_id="Qwen/Qwen2.5-7B-Instruct",
            trait="artistic",
            layer=14,
        )
        original.prompt_persona_vector = mock_persona_vectors["prompt"]
        original.response_persona_vector = mock_persona_vectors["response"]
        original.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        saved_path = original.save_to_safetensors(str(temp_dir) + "/")
        assert Path(saved_path).exists()

        loaded = StubPersonaModel.from_safetensors(saved_path, trait="artistic")

        assert torch.allclose(loaded.prompt_persona_vector, original.prompt_persona_vector)
        assert torch.allclose(loaded.response_persona_vector, original.response_persona_vector)
        assert torch.allclose(
            loaded.all_layers_response_persona_vector,
            original.all_layers_response_persona_vector,
        )

    def test_multiple_traits_coexist(self, temp_dir, mock_persona_vectors):
        """Test that multiple trait vectors can be saved and loaded independently."""
        traits = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]

        for trait in traits:
            model = StubPersonaModel(
                target_model_id="Qwen/Qwen2.5-7B-Instruct",
                trait=trait,
            )
            model.prompt_persona_vector = mock_persona_vectors["prompt"] * (traits.index(trait) + 1)
            model.response_persona_vector = mock_persona_vectors["response"]
            model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]
            model.save_to_safetensors(str(temp_dir) + "/")

        safetensors_files = list(temp_dir.glob("*.safetensors"))
        assert len(safetensors_files) == 6

        manifest_path = temp_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest["vectors"]) == 6

        for i, trait in enumerate(traits):
            path = temp_dir / f"{trait}__Qwen__Qwen2.5-7B-Instruct.safetensors"
            with safe_open(str(path), framework="pt") as f:
                loaded_prompt = f.get_tensor("prompt_persona_vector")
            expected = mock_persona_vectors["prompt"] * (i + 1)
            assert torch.allclose(loaded_prompt, expected)

    def test_overwrite_existing_file(self, temp_dir, mock_persona_vectors):
        """Test that re-saving overwrites the existing file with new values."""
        model = StubPersonaModel(
            target_model_id="Qwen/Qwen2.5-7B-Instruct",
            trait="artistic",
        )

        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]
        first_path = model.save_to_safetensors(str(temp_dir) + "/")

        model.prompt_persona_vector = mock_persona_vectors["prompt"] * 10
        second_path = model.save_to_safetensors(str(temp_dir) + "/")

        assert first_path == second_path

        with safe_open(second_path, framework="pt") as f:
            loaded = f.get_tensor("prompt_persona_vector")
        assert torch.allclose(loaded, mock_persona_vectors["prompt"] * 10)

    def test_backward_compatibility_json_to_safetensors(
        self, temp_dir, mock_persona_vectors, legacy_json_file
    ):
        """Test that JSON files can be loaded and then saved as safetensors."""
        with open(legacy_json_file) as f:
            json_data = json.load(f)

        model = StubPersonaModel(
            target_model_id=json_data["target_model_id"],
            trait=json_data["dataset_info"]["trait"],
            layer=json_data["layer_steering"],
        )
        model.prompt_persona_vector = torch.tensor(json_data["prompt_persona_vector"])
        model.response_persona_vector = torch.tensor(json_data["response_persona_vector"])
        model.all_layers_response_persona_vector = torch.tensor(
            json_data["all_layers_response_persona_vector"]
        )

        safetensors_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(safetensors_path, framework="pt") as f:
            loaded_prompt = f.get_tensor("prompt_persona_vector")

        original_prompt = torch.tensor(json_data["prompt_persona_vector"])
        assert torch.allclose(loaded_prompt, original_prompt)

    def test_metadata_integrity(self, temp_dir, mock_persona_vectors):
        """Test that all metadata fields survive the save/load cycle."""
        model = StubPersonaModel(
            target_model_id="Qwen/Qwen2.5-7B-Instruct",
            trait="social",
            layer=16,
        )
        model.prompt_persona_vector = mock_persona_vectors["prompt"]
        model.response_persona_vector = mock_persona_vectors["response"]
        model.all_layers_response_persona_vector = mock_persona_vectors["all_layers"]

        saved_path = model.save_to_safetensors(str(temp_dir) + "/")

        with safe_open(saved_path, framework="pt") as f:
            metadata = f.metadata()

        assert metadata["target_model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert metadata["trait"] == "social"
        assert metadata["layer_steering"] == "16"
        assert "created_at" in metadata
        assert metadata["num_questions"] == "100"
        assert metadata["num_pos_neg_pairs"] == "5"
