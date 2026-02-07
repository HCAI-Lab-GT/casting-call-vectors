"""Model presets for testing and production.

SmolLM models are used for local testing on Apple Silicon without API calls.
Production uses larger models like OLMo-7B for actual persona extraction.
"""

from typing import TypedDict


class ModelConfig(TypedDict):
    """Configuration for a model preset."""

    model_id: str
    layer: int
    max_new_tokens: int


# SmolLM model family - https://huggingface.co/collections/HuggingFaceTB/smollm2
SMOLLM_MODELS: dict[str, str] = {
    # SmolLM2 - for fast unit tests (smaller, basic validation)
    "smol2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smol2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smol2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # SmolLM3 - for proper smoke/integration tests
    "smol3-3b": "HuggingFaceTB/SmolLM3-3B",
}


MODEL_PRESETS: dict[str, ModelConfig] = {
    "production": {
        "model_id": "allenai/OLMo-7B-Instruct",
        "layer": 14,
        "max_new_tokens": 256,
    },
    "unit_test": {
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "layer": 4,  # SmolLM2-135M has 9 layers
        "max_new_tokens": 32,
    },
    "smoke_test": {
        "model_id": "HuggingFaceTB/SmolLM3-3B",
        "layer": 16,  # SmolLM3-3B has 32 layers
        "max_new_tokens": 64,
    },
}


def get_model_config(preset: str) -> ModelConfig:
    """Get model configuration by preset name.

    Args:
        preset: Preset name (production, unit_test, smoke_test)

    Returns:
        ModelConfig with model_id, layer, and max_new_tokens

    Raises:
        KeyError: If preset name is not found
    """
    if preset not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise KeyError(f"Unknown preset '{preset}'. Available: {available}")
    return MODEL_PRESETS[preset]
