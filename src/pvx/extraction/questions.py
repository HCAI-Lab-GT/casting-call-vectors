"""Question bank management for persona vector extraction.

This module provides the QuestionBank class for loading and managing
extraction questions from various sources (assistant-axis, custom, etc.).
"""

import json
import random
from pathlib import Path
from typing import Iterator


class QuestionBank:
    """Manage extraction questions for persona vector extraction.

    The question bank loads questions from JSONL files and provides
    methods for sampling and iterating over questions.

    Example:
        >>> bank = QuestionBank.from_assistant_axis()
        >>> len(bank)
        240
        >>> questions = bank.sample(50)
        >>> len(questions)
        50
    """

    def __init__(self, questions: list[dict], source: str = "unknown"):
        """Initialize with a list of question dictionaries.

        Args:
            questions: List of dicts with at least a "question" key
            source: Identifier for the question source
        """
        self._questions = questions
        self._source = source

    @classmethod
    def from_jsonl(cls, filepath: Path | str, source: str | None = None) -> "QuestionBank":
        """Load questions from a JSONL file.

        Expected format: One JSON object per line with a "question" key.
        Example: {"question": "What is the meaning of life?", "id": 0}

        Args:
            filepath: Path to the JSONL file
            source: Optional source identifier (defaults to filename)

        Returns:
            QuestionBank instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        filepath = Path(filepath)
        questions = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

        return cls(
            questions=questions,
            source=source or filepath.stem,
        )

    @classmethod
    def from_assistant_axis(
        cls,
        path: str | Path = "_vendor/assistant-axis/data/extraction_questions.jsonl",
    ) -> "QuestionBank":
        """Load the 240 questions from assistant-axis.

        This is the primary question source for vocational persona extraction,
        providing a standardized set of questions for cross-persona comparison.

        Args:
            path: Path to the assistant-axis extraction questions JSONL

        Returns:
            QuestionBank with 240 questions
        """
        return cls.from_jsonl(path, source="assistant-axis")

    @classmethod
    def from_list(cls, questions: list[str], source: str = "custom") -> "QuestionBank":
        """Create a QuestionBank from a simple list of question strings.

        Args:
            questions: List of question strings
            source: Source identifier

        Returns:
            QuestionBank instance
        """
        question_dicts = [{"question": q, "id": i} for i, q in enumerate(questions)]
        return cls(questions=question_dicts, source=source)

    def __len__(self) -> int:
        """Return the number of questions."""
        return len(self._questions)

    def __iter__(self) -> Iterator[str]:
        """Iterate over question strings."""
        for q in self._questions:
            yield q["question"]

    def __getitem__(self, index: int) -> str:
        """Get a question by index."""
        return self._questions[index]["question"]

    @property
    def source(self) -> str:
        """Return the source identifier."""
        return self._source

    def get_all(self) -> list[str]:
        """Return all questions as a list of strings.

        Returns:
            List of all 240 (or however many) question strings.
        """
        return [q["question"] for q in self._questions]

    def get_all_with_ids(self) -> list[dict]:
        """Return all questions with their IDs and metadata.

        Returns:
            List of question dictionaries with all original fields.
        """
        return self._questions.copy()

    def sample(self, n: int, seed: int | None = None) -> list[str]:
        """Return a random sample of n questions.

        Args:
            n: Number of questions to sample
            seed: Optional random seed for reproducibility

        Returns:
            List of n randomly sampled question strings

        Raises:
            ValueError: If n > len(self)
        """
        if n > len(self):
            raise ValueError(f"Cannot sample {n} questions from bank of size {len(self)}")

        rng = random.Random(seed)
        sampled = rng.sample(self._questions, n)
        return [q["question"] for q in sampled]

    def sample_with_ids(self, n: int, seed: int | None = None) -> list[dict]:
        """Return a random sample with question IDs preserved.

        Args:
            n: Number of questions to sample
            seed: Optional random seed for reproducibility

        Returns:
            List of n question dictionaries with IDs
        """
        if n > len(self):
            raise ValueError(f"Cannot sample {n} questions from bank of size {len(self)}")

        rng = random.Random(seed)
        return rng.sample(self._questions, n)

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int | None = None,
    ) -> tuple["QuestionBank", "QuestionBank"]:
        """Split into train and test question banks.

        Useful for evaluating extraction quality on held-out questions.

        Args:
            train_ratio: Fraction of questions for training (default 0.8)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_bank, test_bank)
        """
        rng = random.Random(seed)
        questions = self._questions.copy()
        rng.shuffle(questions)

        split_idx = int(len(questions) * train_ratio)
        train_qs = questions[:split_idx]
        test_qs = questions[split_idx:]

        return (
            QuestionBank(train_qs, source=f"{self._source}_train"),
            QuestionBank(test_qs, source=f"{self._source}_test"),
        )

    def __repr__(self) -> str:
        return f"QuestionBank(n={len(self)}, source={self._source!r})"
