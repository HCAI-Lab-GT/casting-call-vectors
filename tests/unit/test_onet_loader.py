"""Unit tests for O*NET data loader.

These tests skip if O*NET data is not available, since it requires
downloading the database first.
"""

from pathlib import Path

import pytest

from pvx.data.onet_loader import (
    BIG_FIVE_MAPPING,
    RIASEC_ELEMENTS,
    RIASEC_FULL_NAMES,
    WORK_STYLE_ELEMENTS,
    WORK_VALUE_ELEMENTS,
    ONETLoader,
)

# Check if O*NET data exists
ONET_DATA_PATH = Path("data/onet_raw")
HAS_ONET_DATA = ONET_DATA_PATH.exists()

pytestmark = pytest.mark.skipif(
    not HAS_ONET_DATA,
    reason="O*NET data not downloaded. Run: ./scripts/download_onet.sh",
)


class TestONETLoaderInit:
    """Tests for ONETLoader initialization."""

    def test_init_with_valid_path(self):
        """Should initialize with valid data directory."""
        loader = ONETLoader(ONET_DATA_PATH)

        assert loader.data_dir == ONET_DATA_PATH

    def test_init_with_invalid_path_raises(self):
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            ONETLoader("/nonexistent/path")


class TestLoadOccupations:
    """Tests for loading occupation data."""

    def test_load_occupations(self):
        """Should load occupation data."""
        loader = ONETLoader(ONET_DATA_PATH)

        occupations = loader.load_occupations()

        assert len(occupations) > 0
        assert "soc_code" in occupations.columns
        assert "title" in occupations.columns
        assert "description" in occupations.columns

    def test_occupations_cached(self):
        """Should cache loaded occupations."""
        loader = ONETLoader(ONET_DATA_PATH)

        occ1 = loader.load_occupations()
        occ2 = loader.load_occupations()

        assert occ1 is occ2  # Same object (cached)


class TestRIASECScores:
    """Tests for RIASEC score loading."""

    def test_get_riasec_scores(self):
        """Should return RIASEC scores for occupations."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_riasec_scores()

        assert len(scores) > 0
        # Check structure for first occupation
        first_soc = next(iter(scores))
        first_scores = scores[first_soc]
        assert set(first_scores.keys()) == {"R", "I", "A", "S", "E", "C"}

    def test_riasec_scores_in_range(self):
        """RIASEC scores should be in valid range."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_riasec_scores()

        for soc_code, riasec in scores.items():
            for dim, score in riasec.items():
                assert 0 <= score <= 7, f"Score out of range for {soc_code} {dim}: {score}"


class TestHighpointCodes:
    """Tests for RIASEC high-point code loading."""

    def test_get_highpoint_codes(self):
        """Should return high-point codes for occupations."""
        loader = ONETLoader(ONET_DATA_PATH)

        highpoints = loader.get_highpoint_codes()

        assert len(highpoints) > 0
        # Check structure
        first_soc = next(iter(highpoints))
        codes = highpoints[first_soc]
        assert isinstance(codes, list)
        assert all(c in RIASEC_FULL_NAMES for c in codes)


class TestOccupationProfile:
    """Tests for full occupation profile loading."""

    def test_get_occupation_profile(self):
        """Should return full profile for occupation."""
        loader = ONETLoader(ONET_DATA_PATH)

        # Registered Nurses (common test occupation)
        profile = loader.get_occupation_profile("29-1141.00")

        assert profile["soc_code"] == "29-1141.00"
        assert "title" in profile
        assert "riasec" in profile
        assert "tasks" in profile

    def test_profile_has_tasks(self):
        """Profile should include task statements."""
        loader = ONETLoader(ONET_DATA_PATH)

        profile = loader.get_occupation_profile("29-1141.00")

        assert len(profile["tasks"]) > 0
        assert all(isinstance(t, str) for t in profile["tasks"])

    def test_profile_nonexistent_raises(self):
        """Should raise for nonexistent occupation."""
        loader = ONETLoader(ONET_DATA_PATH)

        with pytest.raises(ValueError, match="not found"):
            loader.get_occupation_profile("99-9999.99")


class TestFilterByRIASEC:
    """Tests for RIASEC-based filtering."""

    def test_filter_by_riasec(self):
        """Should filter occupations by RIASEC dimension."""
        loader = ONETLoader(ONET_DATA_PATH)

        # Social occupations with high scores
        social_occs = loader.filter_by_riasec("S", min_score=5.0)

        assert len(social_occs) > 0
        # Verify all returned have high S scores
        scores = loader.get_riasec_scores()
        for soc_code in social_occs:
            assert scores[soc_code]["S"] >= 5.0

    def test_filter_invalid_dimension_raises(self):
        """Should raise for invalid RIASEC dimension."""
        loader = ONETLoader(ONET_DATA_PATH)

        with pytest.raises(ValueError, match="Invalid RIASEC"):
            loader.filter_by_riasec("X")


class TestRIASECDistribution:
    """Tests for RIASEC distribution analysis."""

    def test_get_riasec_distribution(self):
        """Should return count of occupations by primary type."""
        loader = ONETLoader(ONET_DATA_PATH)

        dist = loader.get_riasec_distribution()

        assert set(dist.keys()) == {"R", "I", "A", "S", "E", "C"}
        assert all(count >= 0 for count in dist.values())
        assert sum(dist.values()) > 0


class TestSlug:
    """Tests for title-to-slug conversion."""

    def test_to_slug_basic(self):
        """Should convert title to lowercase slug."""
        loader = ONETLoader(ONET_DATA_PATH)

        slug = loader.to_slug("Chief Executives")

        assert slug == "chief_executives"

    def test_to_slug_special_chars(self):
        """Should handle special characters."""
        loader = ONETLoader(ONET_DATA_PATH)

        slug = loader.to_slug("First-Line Supervisors/Managers")

        assert slug == "first_line_supervisors_managers"


class TestRIASECConstants:
    """Tests for RIASEC constant definitions."""

    def test_riasec_elements_complete(self):
        """RIASEC_ELEMENTS should map all 6 dimensions."""
        assert len(RIASEC_ELEMENTS) == 6

    def test_riasec_full_names_complete(self):
        """RIASEC_FULL_NAMES should have all 6 dimensions."""
        assert set(RIASEC_FULL_NAMES.keys()) == {"R", "I", "A", "S", "E", "C"}


# ============================================================================
# Work Styles Tests
# ============================================================================


class TestWorkStyles:
    """Tests for Work Styles loading."""

    def test_load_work_styles(self):
        """Should load Work Styles data."""
        loader = ONETLoader(ONET_DATA_PATH)

        ws = loader.load_work_styles()

        assert len(ws) > 0
        assert "soc_code" in ws.columns
        assert "element_id" in ws.columns
        assert "data_value" in ws.columns

    def test_work_styles_cached(self):
        """Should cache loaded Work Styles."""
        loader = ONETLoader(ONET_DATA_PATH)

        ws1 = loader.load_work_styles()
        ws2 = loader.load_work_styles()

        assert ws1 is ws2  # Same object (cached)

    def test_get_work_style_scores(self):
        """Should return Work Style scores for occupations."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_work_style_scores()

        assert len(scores) > 0
        # Check a sample occupation has expected traits
        sample_soc = next(iter(scores))
        assert "Achievement/Effort" in scores[sample_soc]
        assert "Analytical Thinking" in scores[sample_soc]

    def test_work_style_scores_in_range(self):
        """Work Style scores should be in valid range (1-5)."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_work_style_scores()

        for soc, traits in scores.items():
            for trait, value in traits.items():
                assert 1.0 <= value <= 5.0, (
                    f"Work Style score out of range for {soc}: {trait}={value}"
                )

    def test_work_style_has_16_traits(self):
        """Each occupation should have 16 Work Style traits."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_work_style_scores()

        # Check a sample occupation
        sample_soc = next(iter(scores))
        assert len(scores[sample_soc]) == 16


class TestWorkStyleConstants:
    """Tests for Work Style constant definitions."""

    def test_work_style_elements_complete(self):
        """WORK_STYLE_ELEMENTS should have 16 traits."""
        assert len(WORK_STYLE_ELEMENTS) == 16

    def test_big_five_mapping_complete(self):
        """BIG_FIVE_MAPPING should have 5 dimensions."""
        assert len(BIG_FIVE_MAPPING) == 5
        assert set(BIG_FIVE_MAPPING.keys()) == {"O", "C", "E", "A", "N_inv"}


# ============================================================================
# Big Five Tests
# ============================================================================


class TestBigFive:
    """Tests for Big Five derivation from Work Styles."""

    def test_get_big_five_scores(self):
        """Should compute Big Five scores for occupations."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_big_five_scores()

        assert len(scores) > 0
        sample_soc = next(iter(scores))
        # Should have all 5 dimensions
        assert set(scores[sample_soc].keys()) == {"O", "C", "E", "A", "N_inv"}

    def test_big_five_scores_in_range(self):
        """Big Five scores should be in valid range (1-5)."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_big_five_scores()

        for soc, dims in scores.items():
            for dim, value in dims.items():
                assert 1.0 <= value <= 5.0, f"Big Five score out of range for {soc}: {dim}={value}"

    def test_big_five_cached(self):
        """Should cache Big Five scores."""
        loader = ONETLoader(ONET_DATA_PATH)

        b5_1 = loader.get_big_five_scores()
        b5_2 = loader.get_big_five_scores()

        assert b5_1 is b5_2  # Same object (cached)


# ============================================================================
# Work Values Tests
# ============================================================================


class TestWorkValues:
    """Tests for Work Values loading."""

    def test_load_work_values(self):
        """Should load Work Values data."""
        loader = ONETLoader(ONET_DATA_PATH)

        wv = loader.load_work_values()

        assert len(wv) > 0
        assert "soc_code" in wv.columns
        assert "element_id" in wv.columns

    def test_work_values_cached(self):
        """Should cache loaded Work Values."""
        loader = ONETLoader(ONET_DATA_PATH)

        wv1 = loader.load_work_values()
        wv2 = loader.load_work_values()

        assert wv1 is wv2  # Same object (cached)

    def test_get_work_value_scores(self):
        """Should return Work Value scores for occupations."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_work_value_scores()

        assert len(scores) > 0
        sample_soc = next(iter(scores))
        assert "Achievement" in scores[sample_soc]
        assert "Independence" in scores[sample_soc]

    def test_work_value_scores_in_range(self):
        """Work Value scores should be in valid range (1-7)."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_work_value_scores()

        for soc, values in scores.items():
            for value_name, score in values.items():
                assert 1.0 <= score <= 7.0, (
                    f"Work Value score out of range for {soc}: {value_name}={score}"
                )

    def test_work_value_has_6_values(self):
        """Each occupation should have 6 Work Values."""
        loader = ONETLoader(ONET_DATA_PATH)

        scores = loader.get_work_value_scores()

        # Check a sample occupation
        sample_soc = next(iter(scores))
        assert len(scores[sample_soc]) == 6

    def test_get_work_value_highpoints(self):
        """Should return Work Value high-points for occupations."""
        loader = ONETLoader(ONET_DATA_PATH)

        hp = loader.get_work_value_highpoints()

        assert len(hp) > 0
        sample_soc = next(iter(hp))
        assert len(hp[sample_soc]) <= 3  # At most 3 high-points
        # Should be valid value names
        valid_values = set(WORK_VALUE_ELEMENTS.values())
        for code in hp[sample_soc]:
            assert code in valid_values


class TestWorkValueConstants:
    """Tests for Work Value constant definitions."""

    def test_work_value_elements_complete(self):
        """WORK_VALUE_ELEMENTS should have 6 values."""
        assert len(WORK_VALUE_ELEMENTS) == 6


# ============================================================================
# Extended Profile Tests
# ============================================================================


class TestExtendedProfile:
    """Tests for extended occupation profile with all psychometrics."""

    def test_profile_has_work_styles(self):
        """Profile should include Work Styles data."""
        loader = ONETLoader(ONET_DATA_PATH)

        # Use Registered Nurses as test occupation
        profile = loader.get_occupation_profile("29-1141.00")

        assert "work_styles" in profile
        assert len(profile["work_styles"]) > 0

    def test_profile_has_big_five(self):
        """Profile should include Big Five derived scores."""
        loader = ONETLoader(ONET_DATA_PATH)

        profile = loader.get_occupation_profile("29-1141.00")

        assert "big_five" in profile
        assert len(profile["big_five"]) > 0
        assert set(profile["big_five"].keys()) == {"O", "C", "E", "A", "N_inv"}

    def test_profile_has_work_values(self):
        """Profile should include Work Values data."""
        loader = ONETLoader(ONET_DATA_PATH)

        profile = loader.get_occupation_profile("29-1141.00")

        assert "work_values" in profile
        assert len(profile["work_values"]) > 0

    def test_profile_has_work_value_highpoints(self):
        """Profile should include Work Value high-points."""
        loader = ONETLoader(ONET_DATA_PATH)

        profile = loader.get_occupation_profile("29-1141.00")

        assert "work_value_highpoints" in profile
        assert isinstance(profile["work_value_highpoints"], list)

    def test_profile_all_fields_present(self):
        """Profile should have all expected fields."""
        loader = ONETLoader(ONET_DATA_PATH)

        profile = loader.get_occupation_profile("29-1141.00")

        expected_fields = [
            "soc_code",
            "title",
            "description",
            "riasec",
            "riasec_primary",
            "highpoint_codes",
            "work_styles",
            "big_five",
            "work_values",
            "work_value_highpoints",
            "tasks",
        ]
        for field in expected_fields:
            assert field in profile, f"Missing field: {field}"
