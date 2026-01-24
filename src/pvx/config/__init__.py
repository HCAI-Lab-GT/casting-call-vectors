"""Configuration module for persona-vectors.

Provides model presets and configuration utilities.
"""

from .models import MODEL_PRESETS, SMOLLM_MODELS, get_model_config

__all__ = ["MODEL_PRESETS", "SMOLLM_MODELS", "get_model_config"]
