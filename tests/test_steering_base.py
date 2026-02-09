"""Unit test validating that response persona vectors dominate prompt persona vectors in known inits."""

from pathlib import Path

import torch
from safetensors.torch import load_file


class TestSteeringBase:
    """Sanity checks for prompt vs response persona vector magnitudes."""

    def test_known_llama_init_prompt_is_near_zero(self):
        repo_root = Path(__file__).resolve().parents[1]
        safetensors_path = (
            repo_root
            / "persona_data"
            / "model_inits"
            / "artistic_persona_initialization"
            / "meta-llama__Llama-3.2-1B-Instruct.safetensors"
        )

        assert safetensors_path.exists()

        tensors = load_file(str(safetensors_path))
        prompt_vec = tensors["prompt_persona_vector"]
        response_vec = tensors["response_persona_vector"]

        prompt_norm = torch.norm(prompt_vec).item()
        response_norm = torch.norm(response_vec).item()

        assert response_norm > 1.0, f"response_norm={response_norm} expected > 1.0"
        assert prompt_norm < 0.01, f"prompt_norm={prompt_norm} expected < 0.01"
        assert response_norm > 100 * prompt_norm, (
            f"response_norm={response_norm} prompt_norm={prompt_norm} expected response > 100 * prompt"
        )

