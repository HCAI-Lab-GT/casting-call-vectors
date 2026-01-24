"""Unit tests for PersonaGeometry analysis."""

import numpy as np
import pytest
import torch

from pvx.analysis.geometry import PersonaGeometry


class TestPersonaGeometryInit:
    """Tests for PersonaGeometry initialization."""

    def test_creates_from_vectors(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should initialize from dict of vectors."""
        geometry = PersonaGeometry(sample_vectors)

        assert geometry.n_personas == len(sample_vectors)
        assert geometry.hidden_dim == sample_vectors["nurse_1"].shape[0]

    def test_stores_persona_ids(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should store persona IDs."""
        geometry = PersonaGeometry(sample_vectors)

        assert set(geometry.persona_ids) == set(sample_vectors.keys())

    def test_with_metadata(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, dict],
    ):
        """Should store optional metadata."""
        geometry = PersonaGeometry(sample_vectors, metadata=sample_metadata)

        assert geometry.metadata == sample_metadata


class TestPCA:
    """Tests for PersonaGeometry.compute_pca()."""

    def test_compute_pca(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should compute PCA on vectors."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.compute_pca(n_components=3)

        assert result.components.shape[0] == 3
        assert result.projections.shape == (len(sample_vectors), 3)

    def test_explained_variance_sums_to_one(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Explained variance ratios should sum to <= 1."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.compute_pca(n_components=3)

        assert result.explained_variance_ratio.sum() <= 1.0

    def test_cumulative_variance_increases(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Cumulative variance should be monotonically increasing."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.compute_pca(n_components=3)

        for i in range(1, len(result.cumulative_variance_ratio)):
            assert result.cumulative_variance_ratio[i] >= result.cumulative_variance_ratio[i - 1]

    def test_components_for_variance(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should report components needed for variance threshold."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.compute_pca(n_components=5)
        n_for_90 = result.components_for_variance(0.9)

        assert 1 <= n_for_90 <= 5

    def test_get_projection(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should retrieve projection for specific persona."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.compute_pca(n_components=3)
        proj = result.get_projection("nurse_1")

        assert proj is not None
        assert proj.shape == (3,)

    def test_get_projection_nonexistent(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should return None for nonexistent persona."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.compute_pca(n_components=3)
        proj = result.get_projection("nonexistent")

        assert proj is None


class TestClustering:
    """Tests for PersonaGeometry.cluster()."""

    def test_kmeans_clustering(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should cluster with k-means."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.cluster(n_clusters=2, method="kmeans")

        assert result.n_clusters == 2
        assert len(result.labels) == len(sample_vectors)
        assert set(result.labels) == {0, 1}

    def test_hierarchical_clustering(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should cluster with hierarchical clustering."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.cluster(n_clusters=2, method="hierarchical")

        assert result.n_clusters == 2
        assert len(result.labels) == len(sample_vectors)

    def test_get_cluster_members(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should get members of a cluster."""
        geometry = PersonaGeometry(sample_vectors)

        result = geometry.cluster(n_clusters=2, method="kmeans")
        members_0 = result.get_cluster_members(0)
        members_1 = result.get_cluster_members(1)

        # All personas should be in exactly one cluster
        assert len(members_0) + len(members_1) == len(sample_vectors)
        assert set(members_0 + members_1) == set(sample_vectors.keys())


class TestDistances:
    """Tests for PersonaGeometry distance computation."""

    def test_compute_distances_cosine(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should compute cosine distances."""
        geometry = PersonaGeometry(sample_vectors)

        distances = geometry.compute_distances(metric="cosine")

        assert distances.shape == (len(sample_vectors), len(sample_vectors))
        # Diagonal should be zero (distance to self)
        np.testing.assert_array_almost_equal(np.diag(distances), 0.0)

    def test_compute_distances_euclidean(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should compute Euclidean distances."""
        geometry = PersonaGeometry(sample_vectors)

        distances = geometry.compute_distances(metric="euclidean")

        assert distances.shape == (len(sample_vectors), len(sample_vectors))
        np.testing.assert_array_almost_equal(np.diag(distances), 0.0)

    def test_compute_similarities(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should compute cosine similarities."""
        geometry = PersonaGeometry(sample_vectors)

        similarities = geometry.compute_similarities()

        assert similarities.shape == (len(sample_vectors), len(sample_vectors))
        # Diagonal should be 1 (similarity to self)
        np.testing.assert_array_almost_equal(np.diag(similarities), 1.0)


class TestContrast:
    """Tests for PersonaGeometry.compute_contrast()."""

    def test_compute_contrast(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should compute contrast between groups."""
        geometry = PersonaGeometry(sample_vectors)

        group_a = ["nurse_1", "nurse_2"]
        group_b = ["engineer_1", "engineer_2"]

        result = geometry.compute_contrast(group_a, group_b, name="nurse_vs_engineer")

        assert result.name == "nurse_vs_engineer"
        assert result.vector.shape == (768,)
        assert result.group_a_ids == group_a
        assert result.group_b_ids == group_b

    def test_contrast_with_pca(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should compute PC cosines when PCA result provided."""
        geometry = PersonaGeometry(sample_vectors)
        pca = geometry.compute_pca(n_components=3)

        group_a = ["nurse_1", "nurse_2"]
        group_b = ["engineer_1", "engineer_2"]

        result = geometry.compute_contrast(group_a, group_b, pca_result=pca)

        assert len(result.pc_cosines) == 3

    def test_contrast_insufficient_group_raises(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should raise error if groups too small."""
        geometry = PersonaGeometry(sample_vectors)

        with pytest.raises(ValueError, match="at least 2"):
            geometry.compute_contrast(["nurse_1"], ["engineer_1", "engineer_2"])


class TestGroupBy:
    """Tests for PersonaGeometry.group_by()."""

    def test_group_by_metadata(
        self,
        sample_vectors: dict[str, torch.Tensor],
        sample_metadata: dict[str, dict],
    ):
        """Should group personas by metadata field."""
        geometry = PersonaGeometry(sample_vectors, metadata=sample_metadata)

        groups = geometry.group_by(lambda pid, meta: meta.get("riasec_primary"))

        assert "S" in groups  # nurses
        assert "R" in groups  # engineers
        assert "A" in groups  # artists
        assert len(groups["S"]) == 2


class TestGetVector:
    """Tests for PersonaGeometry.get_vector()."""

    def test_get_vector(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should retrieve vector for specific persona."""
        geometry = PersonaGeometry(sample_vectors)

        vec = geometry.get_vector("nurse_1")

        assert vec is not None
        assert vec.shape == (768,)

    def test_get_vector_nonexistent(
        self,
        sample_vectors: dict[str, torch.Tensor],
    ):
        """Should return None for nonexistent persona."""
        geometry = PersonaGeometry(sample_vectors)

        vec = geometry.get_vector("nonexistent")

        assert vec is None
