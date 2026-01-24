"""Geometric analysis of persona vectors.

This module provides generic PCA, clustering, and distance analysis
for understanding the structure of persona vectors in activation space.
Axis-specific analysis (e.g., RIASEC) is handled by separate modules.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler

from ..sources.base import PersonaMetadata

logger = logging.getLogger(__name__)


@dataclass
class PCAResult:
    """Results from PCA analysis.

    Attributes:
        components: Principal component vectors (n_components, hidden_dim)
        explained_variance: Variance explained by each component
        explained_variance_ratio: Ratio of variance explained
        cumulative_variance_ratio: Cumulative variance explained
        projections: Persona projections onto PCs (n_personas, n_components)
        persona_ids: Mapping of indices to persona IDs
    """

    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    projections: np.ndarray
    persona_ids: list[str]

    def components_for_variance(self, threshold: float = 0.9) -> int:
        """Number of components needed to explain given variance threshold."""
        return int(np.searchsorted(self.cumulative_variance_ratio, threshold) + 1)

    def get_projection(self, persona_id: str) -> np.ndarray | None:
        """Get projection for a specific persona."""
        try:
            idx = self.persona_ids.index(persona_id)
            return self.projections[idx]
        except ValueError:
            return None


@dataclass
class ClusterResult:
    """Results from clustering analysis.

    Attributes:
        labels: Cluster assignments for each persona
        n_clusters: Number of clusters
        method: Clustering method used
        persona_ids: Mapping of indices to persona IDs
        cluster_centers: Cluster centers if available
    """

    labels: np.ndarray
    n_clusters: int
    method: str
    persona_ids: list[str]
    cluster_centers: np.ndarray | None = None

    def get_cluster_members(self, cluster_id: int) -> list[str]:
        """Get persona IDs belonging to a cluster."""
        indices = np.where(self.labels == cluster_id)[0]
        return [self.persona_ids[i] for i in indices]


@dataclass
class ContrastResult:
    """Results from contrast vector analysis.

    Attributes:
        name: Contrast name (e.g., "group_a_vs_group_b")
        vector: Contrast vector (mean_a - mean_b)
        group_a_ids: Persona IDs in group A
        group_b_ids: Persona IDs in group B
        pc_cosines: Cosine similarities with top PCs (if computed)
    """

    name: str
    vector: np.ndarray
    group_a_ids: list[str]
    group_b_ids: list[str]
    pc_cosines: list[float] = field(default_factory=list)


class PersonaGeometry:
    """Generic geometric analysis of persona vectors in activation space.

    Provides PCA, clustering, distance computation, and contrast analysis
    without assumptions about specific axis systems (RIASEC, HEXACO, etc.).

    Example:
        >>> vectors = {p.persona_id: p.prompt_last_diff for p in extracted}
        >>> metadata = {p.persona_id: p.metadata for p in extracted}
        >>> geometry = PersonaGeometry(vectors, metadata)
        >>> pca = geometry.compute_pca(n_components=10)
        >>> print(f"Components for 90% variance: {pca.components_for_variance(0.9)}")
    """

    def __init__(
        self,
        vectors: dict[str, torch.Tensor],
        metadata: dict[str, PersonaMetadata] | None = None,
    ):
        """Initialize geometry analyzer.

        Args:
            vectors: Dict mapping persona_id -> vector tensor
            metadata: Optional dict mapping persona_id -> PersonaMetadata
        """
        self.persona_ids = list(vectors.keys())
        self.metadata = metadata or {}

        # Stack vectors into matrix (n_personas, hidden_dim)
        self.vectors = torch.stack([vectors[pid].squeeze() for pid in self.persona_ids])
        if self.vectors.dim() == 1:
            self.vectors = self.vectors.unsqueeze(0)

        self.vectors_np = self.vectors.numpy()
        self.n_personas, self.hidden_dim = self.vectors_np.shape

        logger.info(f"Loaded {self.n_personas} persona vectors of dim {self.hidden_dim}")

    def compute_pca(
        self,
        n_components: int = 10,
        standardize: bool = True,
    ) -> PCAResult:
        """Compute PCA on persona vectors.

        Args:
            n_components: Number of components to compute
            standardize: Whether to standardize vectors before PCA

        Returns:
            PCAResult with components, variance, and projections
        """
        # Optionally standardize
        if standardize:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.vectors_np)
        else:
            data = self.vectors_np

        # Compute PCA
        n_components = min(n_components, self.n_personas, self.hidden_dim)
        pca = PCA(n_components=n_components)
        projections = pca.fit_transform(data)

        cumulative = np.cumsum(pca.explained_variance_ratio_)

        result = PCAResult(
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            cumulative_variance_ratio=cumulative,
            projections=projections,
            persona_ids=self.persona_ids,
        )

        logger.info(f"PCA: {result.components_for_variance(0.9)} components for 90% variance")
        logger.info(
            f"Top 3 components explain {cumulative[min(2, len(cumulative) - 1)]:.1%} variance"
        )

        return result

    def compute_contrast(
        self,
        group_a_ids: list[str],
        group_b_ids: list[str],
        name: str | None = None,
        pca_result: PCAResult | None = None,
        n_pcs: int = 5,
    ) -> ContrastResult:
        """Compute contrast vector between two groups of personas.

        Args:
            group_a_ids: Persona IDs for group A
            group_b_ids: Persona IDs for group B
            name: Optional name for the contrast
            pca_result: Optional PCA result for computing PC cosines
            n_pcs: Number of PCs to compute cosines with

        Returns:
            ContrastResult with contrast vector and PC cosines
        """
        # Get indices
        indices_a = [self.persona_ids.index(pid) for pid in group_a_ids if pid in self.persona_ids]
        indices_b = [self.persona_ids.index(pid) for pid in group_b_ids if pid in self.persona_ids]

        if len(indices_a) < 2 or len(indices_b) < 2:
            raise ValueError("Each group needs at least 2 personas")

        # Compute contrast
        mean_a = self.vectors_np[indices_a].mean(axis=0)
        mean_b = self.vectors_np[indices_b].mean(axis=0)
        contrast_vec = mean_a - mean_b

        # Compute cosines with PCs if available
        pc_cosines = []
        if pca_result is not None:
            for pc_idx in range(min(n_pcs, pca_result.components.shape[0])):
                pc_vec = pca_result.components[pc_idx]
                cosine = np.dot(contrast_vec, pc_vec) / (
                    np.linalg.norm(contrast_vec) * np.linalg.norm(pc_vec) + 1e-8
                )
                pc_cosines.append(float(cosine))

        return ContrastResult(
            name=name or "group_a_vs_group_b",
            vector=contrast_vec,
            group_a_ids=[self.persona_ids[i] for i in indices_a],
            group_b_ids=[self.persona_ids[i] for i in indices_b],
            pc_cosines=pc_cosines,
        )

    def group_by(
        self,
        key_fn: Callable[[str, PersonaMetadata], str | None],
    ) -> dict[str, list[str]]:
        """Group personas by a metadata key.

        Args:
            key_fn: Function taking (persona_id, metadata) and returning group key

        Returns:
            Dict mapping group key -> list of persona IDs
        """
        groups: dict[str, list[str]] = {}
        for pid in self.persona_ids:
            meta = self.metadata.get(pid, {})
            key = key_fn(pid, meta)
            if key is not None:
                groups.setdefault(key, []).append(pid)
        return groups

    def cluster(
        self,
        n_clusters: int = 6,
        method: Literal["kmeans", "hierarchical"] = "kmeans",
        use_pca: bool = True,
        n_components: int = 10,
    ) -> ClusterResult:
        """Cluster personas in vector space.

        Args:
            n_clusters: Number of clusters
            method: Clustering method
            use_pca: Whether to cluster in PCA space
            n_components: PCA components if use_pca=True

        Returns:
            ClusterResult with labels and cluster info
        """
        # Prepare data
        if use_pca:
            pca = self.compute_pca(n_components=n_components)
            data = pca.projections
        else:
            data = StandardScaler().fit_transform(self.vectors_np)

        # Cluster
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(data)
            centers = clusterer.cluster_centers_
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(data)
            centers = None

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            method=method,
            persona_ids=self.persona_ids,
            cluster_centers=centers,
        )

    def compute_distances(self, metric: str = "cosine") -> np.ndarray:
        """Compute pairwise distances between persona vectors.

        Args:
            metric: Distance metric ("cosine" or "euclidean")

        Returns:
            Distance matrix (n_personas, n_personas)
        """
        if metric == "cosine":
            return cosine_distances(self.vectors_np)
        from sklearn.metrics.pairwise import euclidean_distances

        return euclidean_distances(self.vectors_np)

    def compute_similarities(self) -> np.ndarray:
        """Compute pairwise cosine similarities between persona vectors.

        Returns:
            Similarity matrix (n_personas, n_personas)
        """
        return cosine_similarity(self.vectors_np)

    def correlate_with_metadata(
        self,
        pca_result: PCAResult,
        metadata_key: str,
        n_pcs: int = 5,
    ) -> dict[int, float]:
        """Correlate PC projections with a numeric metadata field.

        Args:
            pca_result: PCA result to use
            metadata_key: Key to extract from metadata (must be numeric)
            n_pcs: Number of PCs to analyze

        Returns:
            Dict mapping PC index -> correlation coefficient
        """
        correlations = {}

        for pc_idx in range(min(n_pcs, pca_result.projections.shape[1])):
            pc_proj = pca_result.projections[:, pc_idx]

            # Extract metadata values
            values = []
            proj_values = []
            for i, pid in enumerate(self.persona_ids):
                meta = self.metadata.get(pid, {})
                if metadata_key in meta and meta[metadata_key] is not None:
                    try:
                        values.append(float(meta[metadata_key]))
                        proj_values.append(pc_proj[i])
                    except (ValueError, TypeError):
                        continue

            if len(values) >= 3:
                corr = np.corrcoef(values, proj_values)[0, 1]
                correlations[pc_idx] = float(corr) if not np.isnan(corr) else 0.0
            else:
                correlations[pc_idx] = 0.0

        return correlations

    def get_vector(self, persona_id: str) -> np.ndarray | None:
        """Get vector for a specific persona."""
        try:
            idx = self.persona_ids.index(persona_id)
            return self.vectors_np[idx]
        except ValueError:
            return None

    def save(self, output_dir: Path | str) -> None:
        """Save vectors and metadata to disk.

        Args:
            output_dir: Directory for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save vectors
        torch.save(
            {"vectors": self.vectors, "persona_ids": self.persona_ids},
            output_dir / "vectors.pt",
        )

        # Save metadata
        if self.metadata:
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        pid: dict(self.metadata[pid])
                        for pid in self.persona_ids
                        if pid in self.metadata
                    },
                    f,
                    indent=2,
                )

        logger.info(f"Saved geometry data to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path | str) -> "PersonaGeometry":
        """Load geometry from disk.

        Args:
            input_dir: Directory containing saved files

        Returns:
            PersonaGeometry instance
        """
        input_dir = Path(input_dir)

        # Load vectors
        data = torch.load(input_dir / "vectors.pt", weights_only=True)
        vectors = {pid: data["vectors"][i] for i, pid in enumerate(data["persona_ids"])}

        # Load metadata if available
        metadata = {}
        meta_path = input_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return cls(vectors=vectors, metadata=metadata)

    def __repr__(self) -> str:
        return f"PersonaGeometry(n_personas={self.n_personas}, hidden_dim={self.hidden_dim})"
