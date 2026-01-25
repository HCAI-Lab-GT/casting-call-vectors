"""Configuration module for persona-vectors.

Provides model presets, run configuration, and utilities.
"""

from .models import MODEL_PRESETS, SMOLLM_MODELS, get_model_config
from .run_config import RunConfig

__all__ = ["MODEL_PRESETS", "SMOLLM_MODELS", "get_model_config", "RunConfig"]
