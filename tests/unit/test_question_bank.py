"""Unit tests for QuestionBank."""

import tempfile
from pathlib import Path

import pytest

from pvx.extraction.questions import QuestionBank


class TestQuestionBankFromList:
    """Tests for QuestionBank.from_list()."""

    def test_creates_from_list(self, sample_questions: list[str]):
        """Should create QuestionBank from list of strings."""
        bank = QuestionBank.from_list(sample_questions)

        assert len(bank) == len(sample_questions)
        assert bank.source == "custom"

    def test_preserves_question_order(self, sample_questions: list[str]):
        """Should preserve original question order."""
        bank = QuestionBank.from_list(sample_questions)

        for i, question in enumerate(sample_questions):
            assert bank[i] == question

    def test_iteration(self, sample_questions: list[str]):
        """Should iterate over questions."""
        bank = QuestionBank.from_list(sample_questions)

        iterated = list(bank)
        assert iterated == sample_questions


class TestQuestionBankSample:
    """Tests for QuestionBank.sample()."""

    def test_sample_returns_correct_count(self, sample_questions: list[str]):
        """Should return requested number of questions."""
        bank = QuestionBank.from_list(sample_questions)

        sampled = bank.sample(3)
        assert len(sampled) == 3

    def test_sample_with_seed_is_reproducible(self, sample_questions: list[str]):
        """Should return same sample with same seed."""
        bank = QuestionBank.from_list(sample_questions)

        sample1 = bank.sample(3, seed=42)
        sample2 = bank.sample(3, seed=42)

        assert sample1 == sample2

    def test_sample_with_different_seeds_differs(self, sample_questions: list[str]):
        """Should return different samples with different seeds."""
        bank = QuestionBank.from_list(sample_questions)

        sample1 = bank.sample(3, seed=42)
        sample2 = bank.sample(3, seed=123)

        assert sample1 != sample2

    def test_sample_too_many_raises_error(self, sample_questions: list[str]):
        """Should raise ValueError when sampling more than available."""
        bank = QuestionBank.from_list(sample_questions)

        with pytest.raises(ValueError, match="Cannot sample"):
            bank.sample(100)


class TestQuestionBankSplit:
    """Tests for QuestionBank.split()."""

    def test_split_default_ratio(self, sample_questions: list[str]):
        """Should split with default 80/20 ratio."""
        bank = QuestionBank.from_list(sample_questions)

        train, test = bank.split()

        assert len(train) + len(test) == len(bank)
        expected_train = int(len(bank) * 0.8)
        assert len(train) == expected_train
        assert len(test) == len(bank) - expected_train

    def test_split_custom_ratio(self, sample_questions: list[str]):
        """Should split with custom ratio."""
        bank = QuestionBank.from_list(sample_questions)

        train, test = bank.split(train_ratio=0.6)

        expected_train = int(len(bank) * 0.6)
        assert len(train) == expected_train
        assert len(test) == len(bank) - expected_train

    def test_split_is_reproducible(self, sample_questions: list[str]):
        """Should produce same split with same seed."""
        bank = QuestionBank.from_list(sample_questions)

        train1, test1 = bank.split(seed=42)
        train2, test2 = bank.split(seed=42)

        assert list(train1) == list(train2)
        assert list(test1) == list(test2)

    def test_split_sources_are_named(self, sample_questions: list[str]):
        """Should name train/test banks appropriately."""
        bank = QuestionBank.from_list(sample_questions, source="my_source")

        train, test = bank.split()

        assert train.source == "my_source_train"
        assert test.source == "my_source_test"


class TestQuestionBankJSONL:
    """Tests for JSONL loading."""

    def test_from_jsonl(self, sample_questions: list[str]):
        """Should load questions from JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i, q in enumerate(sample_questions):
                f.write(f'{{"question": "{q}", "id": {i}}}\n')
            filepath = f.name

        try:
            bank = QuestionBank.from_jsonl(filepath)
            assert len(bank) == len(sample_questions)
            assert bank[0] == sample_questions[0]
        finally:
            Path(filepath).unlink()

    def test_from_jsonl_nonexistent_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            QuestionBank.from_jsonl("/nonexistent/path.jsonl")


class TestQuestionBankGetMethods:
    """Tests for get_all and get_all_with_ids."""

    def test_get_all(self, sample_questions: list[str]):
        """Should return all questions as list."""
        bank = QuestionBank.from_list(sample_questions)

        all_questions = bank.get_all()
        assert all_questions == sample_questions

    def test_get_all_with_ids(self, sample_questions: list[str]):
        """Should return questions with IDs."""
        bank = QuestionBank.from_list(sample_questions)

        with_ids = bank.get_all_with_ids()

        assert len(with_ids) == len(sample_questions)
        assert all("question" in q for q in with_ids)
        assert all("id" in q for q in with_ids)
