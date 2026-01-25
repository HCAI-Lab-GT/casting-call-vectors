"""RIASEC-specific analysis for vocational persona vectors.

This module provides Holland code (RIASEC) specific analysis functions
that build on top of the generic PersonaGeometry class.

RIASEC Dimensions:
- R (Realistic): Practical, hands-on work
- I (Investigative): Research, analysis
- A (Artistic): Creative expression
- S (Social): Helping, teaching
- E (Enterprising): Leadership, business
- C (Conventional): Organization, data
"""

import logging
from dataclasses import dataclass

import numpy as np

from .geometry import ContrastResult, PCAResult, PersonaGeometry

logger = logging.getLogger(__name__)

# RIASEC dimension definitions
RIASEC_DIMS = ["R", "I", "A", "S", "E", "C"]
RIASEC_FULL_NAMES = {
    "R": "Realistic",
    "I": "Investigative",
    "A": "Artistic",
    "S": "Social",
    "E": "Enterprising",
    "C": "Conventional",
}

# Classic Holland hexagon contrasts (opposite dimensions)
RIASEC_CONTRASTS = [
    ("R", "S"),  # Realistic vs Social
    ("I", "E"),  # Investigative vs Enterprising
    ("A", "C"),  # Artistic vs Conventional
]

# RIASEC color scheme (hexagon-inspired)
RIASEC_COLORS = {
    "R": "#E41A1C",  # Red
    "I": "#377EB8",  # Blue
    "A": "#984EA3",  # Purple
    "S": "#4DAF4A",  # Green
    "E": "#FF7F00",  # Orange
    "C": "#A65628",  # Brown
}


@dataclass
class RIASECAnalysisResult:
    """Results from RIASEC-specific analysis.

    Attributes:
        pc_correlations: Correlation of each PC with RIASEC dimension scores
        contrast_results: Contrast vectors for opposite RIASEC pairs
        axis_alignment_score: Overall alignment of RIASEC with PC structure (0-1)
        cluster_purity: RIASEC purity of clusters (if clustering was done)
        go_no_go: Whether alignment meets threshold for H1 hypothesis
    """

    pc_correlations: dict[int, dict[str, float]]
    contrast_results: list[ContrastResult]
    axis_alignment_score: float
    cluster_purity: dict | None = None
    go_no_go: bool = False


def get_riasec_color_fn():
    """Return a color function for RIASEC primary type."""

    def color_fn(persona_id: str, metadata: dict) -> str | None:
        return metadata.get("riasec_primary")

    return color_fn


def get_riasec_color_map() -> dict[str, str]:
    """Return the standard RIASEC color map."""
    return RIASEC_COLORS.copy()


def group_by_riasec(geometry: PersonaGeometry) -> dict[str, list[str]]:
    """Group personas by primary RIASEC type.

    Args:
        geometry: PersonaGeometry instance

    Returns:
        Dict mapping RIASEC letter -> list of persona IDs
    """
    return geometry.group_by(
        lambda pid, meta: meta.get("riasec_primary")
        if meta.get("riasec_primary") in RIASEC_DIMS
        else None
    )


def compute_riasec_contrasts(
    geometry: PersonaGeometry,
    pca_result: PCAResult | None = None,
    n_pcs: int = 5,
) -> list[ContrastResult]:
    """Compute contrast vectors for classic RIASEC opposite pairs.

    Args:
        geometry: PersonaGeometry instance
        pca_result: Optional PCA result for PC cosines
        n_pcs: Number of PCs for cosine computation

    Returns:
        List of ContrastResult for each RIASEC contrast pair
    """
    groups = group_by_riasec(geometry)
    results = []

    for dim_a, dim_b in RIASEC_CONTRASTS:
        if len(groups.get(dim_a, [])) >= 2 and len(groups.get(dim_b, [])) >= 2:
            contrast = geometry.compute_contrast(
                group_a_ids=groups[dim_a],
                group_b_ids=groups[dim_b],
                name=f"{dim_a}_vs_{dim_b}",
                pca_result=pca_result,
                n_pcs=n_pcs,
            )
            results.append(contrast)
            logger.info(
                f"Computed {dim_a} vs {dim_b} contrast ({len(groups[dim_a])} vs {len(groups[dim_b])} personas)"
            )
        else:
            logger.warning(f"Insufficient personas for {dim_a} vs {dim_b} contrast")

    return results


def correlate_pcs_with_riasec(
    geometry: PersonaGeometry,
    pca_result: PCAResult,
    n_pcs: int = 5,
) -> dict[int, dict[str, float]]:
    """Correlate PC projections with RIASEC dimension scores.

    For each PC, computes Pearson correlation with each RIASEC dimension
    score (not just primary type, but actual numeric scores).

    Args:
        geometry: PersonaGeometry instance
        pca_result: PCA result to analyze
        n_pcs: Number of PCs to analyze

    Returns:
        Dict mapping PC index -> {RIASEC dim -> correlation}
    """
    pc_correlations: dict[int, dict[str, float]] = {}

    for pc_idx in range(min(n_pcs, pca_result.projections.shape[1])):
        pc_proj = pca_result.projections[:, pc_idx]
        pc_correlations[pc_idx] = {}

        for dim in RIASEC_DIMS:
            # Get RIASEC scores for this dimension
            scores = []
            proj_values = []

            for i, pid in enumerate(geometry.persona_ids):
                riasec = geometry.metadata.get(pid, {}).get("riasec", {})
                if dim in riasec:
                    scores.append(riasec[dim])
                    proj_values.append(pc_proj[i])

            if len(scores) >= 3:
                corr = np.corrcoef(scores, proj_values)[0, 1]
                pc_correlations[pc_idx][dim] = float(corr) if not np.isnan(corr) else 0.0
            else:
                pc_correlations[pc_idx][dim] = 0.0

    return pc_correlations


def compute_riasec_alignment(
    contrast_results: list[ContrastResult],
    threshold: float = 0.6,
) -> tuple[float, bool]:
    """Compute overall RIASEC-PC alignment score.

    The alignment score is the mean of the maximum absolute cosine
    similarity between each contrast vector and the top PCs.

    Args:
        contrast_results: List of contrast results with PC cosines
        threshold: Threshold for "good" alignment (go/no-go criterion)

    Returns:
        Tuple of (alignment_score, passes_threshold)
    """
    if not contrast_results or not contrast_results[0].pc_cosines:
        return 0.0, False

    max_cosines = []
    for contrast in contrast_results:
        if contrast.pc_cosines:
            max_cos = max(abs(c) for c in contrast.pc_cosines)
            max_cosines.append(max_cos)

    if not max_cosines:
        return 0.0, False

    alignment_score = float(np.mean(max_cosines))
    passes = alignment_score >= threshold

    return alignment_score, passes


def compute_cluster_riasec_purity(
    geometry: PersonaGeometry,
    cluster_labels: np.ndarray,
) -> dict[int, dict]:
    """Compute RIASEC purity for each cluster.

    Args:
        geometry: PersonaGeometry instance
        cluster_labels: Cluster assignment for each persona

    Returns:
        Dict mapping cluster_id -> {dominant, purity, counts}
    """
    groups = group_by_riasec(geometry)

    # Invert to persona_id -> RIASEC type
    persona_riasec = {}
    for dim, pids in groups.items():
        for pid in pids:
            persona_riasec[pid] = dim

    purity_results = {}
    for cluster_id in set(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # Count RIASEC types in cluster
        type_counts = dict.fromkeys(RIASEC_DIMS, 0)
        for idx in cluster_indices:
            pid = geometry.persona_ids[idx]
            riasec_type = persona_riasec.get(pid)
            if riasec_type:
                type_counts[riasec_type] += 1

        total = sum(type_counts.values())
        if total > 0:
            dominant = max(type_counts, key=lambda x: type_counts.get(x, 0))
            purity = type_counts[dominant] / total
            purity_results[cluster_id] = {
                "dominant": dominant,
                "purity": float(purity),
                "counts": type_counts,
            }

    return purity_results


def analyze_riasec(
    geometry: PersonaGeometry,
    pca_result: PCAResult | None = None,
    n_pcs: int = 5,
    alignment_threshold: float = 0.6,
) -> RIASECAnalysisResult:
    """Run complete RIASEC-specific analysis.

    This is the main entry point for RIASEC analysis. It computes:
    1. PC correlations with RIASEC dimension scores
    2. Contrast vectors for opposite RIASEC pairs
    3. Overall alignment score (go/no-go for H1)

    Args:
        geometry: PersonaGeometry instance
        pca_result: Optional pre-computed PCA result
        n_pcs: Number of PCs to analyze
        alignment_threshold: Threshold for H1 go/no-go decision

    Returns:
        RIASECAnalysisResult with all metrics
    """
    # Compute PCA if not provided
    if pca_result is None:
        pca_result = geometry.compute_pca(n_components=n_pcs)

    # Correlate PCs with RIASEC scores
    pc_correlations = correlate_pcs_with_riasec(geometry, pca_result, n_pcs)

    # Compute contrast vectors
    contrasts = compute_riasec_contrasts(geometry, pca_result, n_pcs)

    # Compute alignment score
    alignment_score, go_no_go = compute_riasec_alignment(contrasts, alignment_threshold)

    # Log results
    logger.info(f"RIASEC alignment score: {alignment_score:.3f}")
    if go_no_go:
        logger.info("H1 GO: RIASEC dimensions align with principal components")
    else:
        logger.warning("H1 NO-GO: RIASEC dimensions do not strongly align with PCs")
        logger.warning("Consider alternative frameworks (Big Five, direct trait vectors)")

    # Log top PC correlations
    for pc_idx, corrs in pc_correlations.items():
        top_dim = max(corrs, key=lambda d: abs(corrs[d]))
        logger.info(
            f"PC{pc_idx + 1}: strongest correlation with {RIASEC_FULL_NAMES[top_dim]} (r={corrs[top_dim]:.3f})"
        )

    return RIASECAnalysisResult(
        pc_correlations=pc_correlations,
        contrast_results=contrasts,
        axis_alignment_score=alignment_score,
        go_no_go=go_no_go,
    )
