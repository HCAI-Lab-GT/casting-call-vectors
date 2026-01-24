"""Base abstractions for persona sources.

This module defines the PersonaSource protocol that all persona implementations
must follow, enabling a unified extraction pipeline regardless of persona type.
"""

import json
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable


class PersonaMetadata(TypedDict, total=False):
    """Metadata associated with a persona.

    Attributes:
        soc_code: O*NET-SOC occupation code (for vocational personas)
        title: Human-readable persona title
        riasec: RIASEC dimension scores {R, I, A, S, E, C}
        riasec_primary: Primary RIASEC type (single letter)
        highpoint_codes: Top 2-3 RIASEC codes in order
        source: Origin of the persona (e.g., "onet", "trait", "role")
    """

    soc_code: str
    title: str
    riasec: dict[str, float]
    riasec_primary: str
    highpoint_codes: list[str]
    source: str


@runtime_checkable
class PersonaSource(Protocol):
    """Protocol defining the interface for persona sources.

    All persona implementations must provide these methods to be usable
    with the ExtractionPipeline. The protocol uses structural subtyping,
    so any class implementing these methods is automatically compatible.

    Example:
        >>> class MyPersonaSource:
        ...     @property
        ...     def persona_id(self) -> str:
        ...         return "my_persona"
        ...     def get_system_prompts(self) -> list[str]:
        ...         return ["You are a helpful assistant."]
        ...     # ... other methods
        >>> isinstance(MyPersonaSource(), PersonaSource)
        True
    """

    @property
    def persona_id(self) -> str:
        """Unique identifier for this persona.

        Should be filesystem-safe (lowercase, underscores, no spaces).
        Examples: "registered_nurses", "chief_executives", "angry"
        """
        ...

    def get_system_prompts(self) -> list[str]:
        """Return list of system prompt variants for this persona.

        Multiple variants enable robust vector extraction by averaging
        across different phrasings of the same persona.

        Returns:
            List of 3-5 system prompts instructing the model to embody the persona.
        """
        ...

    def get_baseline_prompts(self) -> list[str]:
        """Return list of baseline/contrast prompts.

        For contrastive extraction, we need prompts that represent the
        "opposite" or "neutral" state. For vocational personas, this is
        typically the default AI assistant baseline.

        Returns:
            List of baseline prompts for contrastive vector computation.
        """
        ...

    def get_eval_prompt(self) -> str:
        """Return the evaluation prompt template for LLM judge scoring.

        The template should contain {question} and {answer} placeholders
        that will be filled in during evaluation.

        Returns:
            Evaluation prompt string with placeholders.
        """
        ...

    def get_metadata(self) -> PersonaMetadata:
        """Return metadata associated with this persona.

        Returns:
            PersonaMetadata dict with available fields populated.
        """
        ...


class BaselineSource:
    """Default AI assistant baseline for contrastive extraction.

    This source represents the "default" or "neutral" model behavior,
    used as the negative/baseline in contrastive vector computation:
    persona_vector = mean(persona_activations) - mean(baseline_activations)

    The prompts are sourced from assistant-axis default.json format.
    """

    # Default baseline prompts (from assistant-axis)
    DEFAULT_PROMPTS = [
        "",
        "You are an AI assistant.",
        "You are a large language model.",
        "You are a helpful assistant.",
        "Respond as yourself.",
    ]

    def __init__(
        self,
        prompts: list[str] | None = None,
        baseline_path: Path | str | None = None,
    ):
        """Initialize baseline source.

        Args:
            prompts: Custom baseline prompts (overrides defaults)
            baseline_path: Path to baseline JSON file (assistant-axis format)
        """
        if prompts is not None:
            self._prompts = prompts
        elif baseline_path is not None:
            self._prompts = self._load_from_json(Path(baseline_path))
        else:
            self._prompts = self.DEFAULT_PROMPTS

    def _load_from_json(self, path: Path) -> list[str]:
        """Load prompts from assistant-axis format JSON."""
        with open(path) as f:
            data = json.load(f)
        return [inst["pos"] for inst in data.get("instruction", [])]

    @property
    def persona_id(self) -> str:
        return "baseline"

    def get_system_prompts(self) -> list[str]:
        return self._prompts

    def get_baseline_prompts(self) -> list[str]:
        # Baseline is its own baseline (used for consistency in pipeline)
        return self._prompts

    def get_eval_prompt(self) -> str:
        # Baseline doesn't need evaluation
        return ""

    def get_metadata(self) -> PersonaMetadata:
        return PersonaMetadata(
            title="Default Assistant",
            source="baseline",
        )

    @classmethod
    def from_vocational_default(cls) -> "BaselineSource":
        """Load from the vocational personas default.json file."""
        default_path = Path("persona_data/vocational_personas/instructions/default.json")
        if default_path.exists():
            return cls(baseline_path=default_path)
        return cls()  # Fall back to hardcoded defaults
