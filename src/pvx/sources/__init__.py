"""Persona source definitions for vector extraction.

This module provides extensible persona source abstractions that can be used
with the extraction pipeline. Currently implemented:

- VocationalPersonaSource: O*NET occupation-based personas (nurses, engineers, etc.)
- BaselineSource: Default AI assistant baseline for contrastive extraction

Future extensions could include:
- TraitPersonaSource: Trait-based personas (e.g., "angry", "sarcastic", "analytical")
- RolePersonaSource: Non-vocational role-playing personas (e.g., "ghost", "demon", "mystic")
"""

from .base import BaselineSource, PersonaMetadata, PersonaSource
from .vocational import VocationalPersonaSource

__all__ = [
    "PersonaMetadata",
    "PersonaSource",
    "BaselineSource",
    "VocationalPersonaSource",
]
