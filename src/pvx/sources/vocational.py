"""Vocational persona source for O*NET occupation-based personas.

This module wraps the output of VocationalPersonaGenerator into the
PersonaSource interface for use with the extraction pipeline.
"""

import json
import logging
import re
from pathlib import Path
from typing import Iterator

from .base import BaselineSource, PersonaMetadata

logger = logging.getLogger(__name__)


class VocationalPersonaSource:
    """Load and manage vocational personas from JSON files.

    Vocational personas are generated from O*NET occupation data and include
    RIASEC (Holland code) metadata for personality dimension analysis.

    Example:
        >>> source = VocationalPersonaSource.from_json("persona_data/.../registered_nurses.json")
        >>> source.persona_id
        'registered_nurses'
        >>> source.get_metadata()["riasec_primary"]
        'S'
    """

    # Shared baseline instance for all vocational personas
    _baseline: BaselineSource | None = None

    def __init__(
        self,
        data: dict,
        persona_id: str,
        baseline: BaselineSource | None = None,
    ):
        """Initialize from parsed persona data.

        Args:
            data: Parsed JSON data from vocational persona file
            persona_id: Unique identifier (filesystem-safe slug)
            baseline: Baseline source for contrastive extraction
        """
        self._data = data
        self._persona_id = persona_id
        self._baseline = baseline or self._get_shared_baseline()

    @classmethod
    def _get_shared_baseline(cls) -> BaselineSource:
        """Get or create the shared baseline instance."""
        if cls._baseline is None:
            cls._baseline = BaselineSource.from_vocational_default()
        return cls._baseline

    @classmethod
    def from_json(cls, filepath: Path | str) -> "VocationalPersonaSource":
        """Load vocational persona from JSON file.

        Args:
            filepath: Path to the persona JSON file

        Returns:
            VocationalPersonaSource instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file isn't valid JSON
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)

        # Extract persona_id from filename (e.g., "registered_nurses.json" -> "registered_nurses")
        persona_id = filepath.stem

        return cls(data=data, persona_id=persona_id)

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        riasec_filter: str | None = None,
        limit: int | None = None,
    ) -> Iterator["VocationalPersonaSource"]:
        """Load all vocational personas from a directory.

        Args:
            directory: Path to directory containing persona JSON files
            riasec_filter: Only return personas with this primary RIASEC type
            limit: Maximum number of personas to return

        Yields:
            VocationalPersonaSource instances
        """
        directory = Path(directory)
        count = 0

        for filepath in sorted(directory.glob("*.json")):
            # Skip the default baseline file
            if filepath.stem == "default":
                continue

            try:
                source = cls.from_json(filepath)

                # Apply RIASEC filter if specified
                if riasec_filter is not None:
                    metadata = source.get_metadata()
                    if metadata.get("riasec_primary") != riasec_filter:
                        continue

                yield source
                count += 1

                if limit is not None and count >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

    @property
    def persona_id(self) -> str:
        """Unique identifier for this persona."""
        return self._persona_id

    def get_system_prompts(self) -> list[str]:
        """Return list of system prompt variants.

        Returns:
            List of 5 system prompts from the persona file.
        """
        instructions = self._data.get("instruction", [])
        return [inst["pos"] for inst in instructions]

    def get_baseline_prompts(self) -> list[str]:
        """Return baseline prompts for contrastive extraction.

        Returns:
            List of baseline prompts from the shared baseline source.
        """
        return self._baseline.get_system_prompts()

    def get_eval_prompt(self) -> str:
        """Return the evaluation prompt template.

        Returns:
            Evaluation prompt with {question} and {answer} placeholders.
        """
        return self._data.get("eval_prompt", "")

    def get_metadata(self) -> PersonaMetadata:
        """Return metadata including RIASEC scores.

        Returns:
            PersonaMetadata dict with vocational-specific fields.
        """
        meta = self._data.get("_metadata", {})
        return PersonaMetadata(
            soc_code=meta.get("soc_code", ""),
            title=meta.get("title", self._persona_id),
            riasec=meta.get("riasec", {}),
            riasec_primary=meta.get("riasec_primary", ""),
            highpoint_codes=meta.get("highpoint_codes", []),
            source="onet",
        )

    def __repr__(self) -> str:
        meta = self.get_metadata()
        return f"VocationalPersonaSource(id={self.persona_id!r}, title={meta.get('title')!r}, riasec={meta.get('riasec_primary')!r})"


def load_vocational_personas(
    directory: Path | str = "persona_data/vocational_personas/instructions",
    riasec_filter: str | None = None,
    limit: int | None = None,
) -> list[VocationalPersonaSource]:
    """Convenience function to load vocational personas as a list.

    Args:
        directory: Path to directory containing persona JSON files
        riasec_filter: Only return personas with this primary RIASEC type (R, I, A, S, E, or C)
        limit: Maximum number of personas to return

    Returns:
        List of VocationalPersonaSource instances
    """
    return list(
        VocationalPersonaSource.from_directory(
            directory=directory,
            riasec_filter=riasec_filter,
            limit=limit,
        )
    )


def to_slug(title: str) -> str:
    """Convert title to filesystem-safe slug.

    Args:
        title: Human-readable title (e.g., "Chief Executives")

    Returns:
        Lowercase slug with underscores (e.g., "chief_executives")
    """
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")
