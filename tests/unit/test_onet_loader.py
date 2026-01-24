"""Unit tests for O*NET data loader.

These tests skip if O*NET data is not available, since it requires
downloading the database first.
"""

from pathlib import Path

import pytest

from pvx.data.onet_loader import RIASEC_ELEMENTS, RIASEC_FULL_NAMES, ONETLoader

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
