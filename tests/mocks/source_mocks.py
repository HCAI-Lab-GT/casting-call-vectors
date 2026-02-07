"""Mock persona source implementations for testing.

These mocks provide minimal persona source implementations for testing
pipeline logic without requiring actual persona data files.
"""

from pvx.sources.base import PersonaMetadata


class MockPersonaSource:
    """Mock persona source for pipeline testing.

    Provides minimal implementation of PersonaSource protocol
    for testing extraction pipelines.

    Example:
        >>> source = MockPersonaSource("test_nurse", riasec_primary="S")
        >>> source.get_system_prompts()
        ['You are a test_nurse.']
    """

    def __init__(
        self,
        persona_id: str = "mock_persona",
        riasec_primary: str = "S",
        title: str | None = None,
    ):
        """Initialize mock persona source.

        Args:
            persona_id: Unique identifier for the persona
            riasec_primary: Primary RIASEC type
            title: Human-readable title (defaults to persona_id)
        """
        self._id = persona_id
        self._riasec = riasec_primary
        self._title = title or persona_id.replace("_", " ").title()

    @property
    def persona_id(self) -> str:
        """Return unique identifier."""
        return self._id

    def get_system_prompts(self) -> list[str]:
        """Return mock system prompts."""
        return [
            f"You are a {self._title}.",
            f"You are an experienced {self._title} professional.",
            f"Act as a {self._title} in your responses.",
        ]

    def get_baseline_prompts(self) -> list[str]:
        """Return mock baseline prompts."""
        return [
            "",
            "You are an AI assistant.",
            "You are a helpful assistant.",
        ]

    def get_eval_prompt(self) -> str:
        """Return mock evaluation prompt."""
        return (
            f"Rate how well this response reflects a {self._title} perspective.\n"
            "Question: {question}\n"
            "Answer: {answer}\n"
            "Score (0-3):"
        )

    def get_metadata(self) -> PersonaMetadata:
        """Return mock metadata."""
        return PersonaMetadata(
            title=self._title,
            riasec_primary=self._riasec,
            riasec={
                "R": 3.0 if self._riasec == "R" else 2.0,
                "I": 3.0 if self._riasec == "I" else 2.0,
                "A": 3.0 if self._riasec == "A" else 2.0,
                "S": 3.0 if self._riasec == "S" else 2.0,
                "E": 3.0 if self._riasec == "E" else 2.0,
                "C": 3.0 if self._riasec == "C" else 2.0,
            },
            source="mock",
        )

    def __repr__(self) -> str:
        return f"MockPersonaSource(id={self._id!r}, riasec={self._riasec!r})"
