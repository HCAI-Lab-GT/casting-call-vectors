"""Vector extraction pipeline for persona vectors.

This module provides the core extraction functionality:

- QuestionBank: Manages extraction questions from various sources
- ActivationExtractor: Extracts activations from transformer models
- ExtractionPipeline: End-to-end pipeline with W&B logging and checkpointing
"""

from .questions import QuestionBank
from .activations import ActivationExtractor
from .pipeline import ExtractionPipeline, PersonaVector

__all__ = [
    "QuestionBank",
    "ActivationExtractor",
    "ExtractionPipeline",
    "PersonaVector",
]
