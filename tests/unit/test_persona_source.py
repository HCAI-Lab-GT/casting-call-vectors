"""Unit tests for persona source abstractions."""

from pvx.sources.base import BaselineSource, PersonaSource
from tests.mocks import MockPersonaSource


class TestBaselineSource:
    """Tests for BaselineSource."""

    def test_default_prompts(self):
        """Should have default prompts."""
        source = BaselineSource()

        prompts = source.get_system_prompts()
        assert len(prompts) > 0
        assert "" in prompts  # Empty prompt is a valid baseline

    def test_persona_id_is_baseline(self):
        """Should have 'baseline' as persona_id."""
        source = BaselineSource()

        assert source.persona_id == "baseline"

    def test_custom_prompts(self):
        """Should accept custom prompts."""
        custom = ["Custom baseline 1", "Custom baseline 2"]
        source = BaselineSource(prompts=custom)

        assert source.get_system_prompts() == custom

    def test_baseline_prompts_same_as_system(self):
        """Baseline should return same prompts for both methods."""
        source = BaselineSource()

        assert source.get_system_prompts() == source.get_baseline_prompts()

    def test_metadata_source_is_baseline(self):
        """Metadata should indicate baseline source."""
        source = BaselineSource()

        meta = source.get_metadata()
        assert meta.get("source") == "baseline"


class TestMockPersonaSource:
    """Tests for MockPersonaSource."""

    def test_protocol_compliance(self):
        """MockPersonaSource should satisfy PersonaSource protocol."""
        source = MockPersonaSource()

        # Should be recognized as PersonaSource
        assert isinstance(source, PersonaSource)

    def test_persona_id(self):
        """Should return provided persona_id."""
        source = MockPersonaSource("test_persona")

        assert source.persona_id == "test_persona"

    def test_system_prompts(self):
        """Should return list of system prompts."""
        source = MockPersonaSource("nurse", riasec_primary="S")

        prompts = source.get_system_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    def test_baseline_prompts(self):
        """Should return list of baseline prompts."""
        source = MockPersonaSource()

        prompts = source.get_baseline_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_eval_prompt_has_placeholders(self):
        """Eval prompt should have {question} and {answer} placeholders."""
        source = MockPersonaSource()

        eval_prompt = source.get_eval_prompt()
        assert "{question}" in eval_prompt
        assert "{answer}" in eval_prompt

    def test_metadata_has_riasec(self):
        """Metadata should include RIASEC primary type."""
        source = MockPersonaSource("test", riasec_primary="R")

        meta = source.get_metadata()
        assert meta.get("riasec_primary") == "R"

    def test_metadata_has_title(self):
        """Metadata should include title."""
        source = MockPersonaSource("software_engineer")

        meta = source.get_metadata()
        assert "title" in meta

    def test_custom_title(self):
        """Should use custom title when provided."""
        source = MockPersonaSource("test", title="Custom Title")

        meta = source.get_metadata()
        assert meta.get("title") == "Custom Title"


class TestPersonaSourceProtocol:
    """Tests for PersonaSource protocol compliance."""

    def test_baseline_satisfies_protocol(self):
        """BaselineSource should satisfy PersonaSource protocol."""
        source = BaselineSource()

        # Protocol checks
        assert hasattr(source, "persona_id")
        assert hasattr(source, "get_system_prompts")
        assert hasattr(source, "get_baseline_prompts")
        assert hasattr(source, "get_eval_prompt")
        assert hasattr(source, "get_metadata")

    def test_required_methods_return_correct_types(self):
        """Protocol methods should return expected types."""
        source = MockPersonaSource()

        assert isinstance(source.persona_id, str)
        assert isinstance(source.get_system_prompts(), list)
        assert isinstance(source.get_baseline_prompts(), list)
        assert isinstance(source.get_eval_prompt(), str)
        assert isinstance(source.get_metadata(), dict)
