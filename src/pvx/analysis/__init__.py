"""Geometry analysis and visualization for persona vectors.

This module provides analysis and visualization capabilities:

- PersonaGeometry: PCA, clustering, and distance analysis (axis-agnostic)
- PersonaVisualizer: 2D/3D plots and W&B dashboard integration
- riasec: RIASEC-specific analysis functions (Holland codes)
"""

from .geometry import PersonaGeometry, PCAResult, ClusterResult, ContrastResult
from .viz import PersonaVisualizer
from . import riasec

__all__ = [
    "PersonaGeometry",
    "PCAResult",
    "ClusterResult",
    "ContrastResult",
    "PersonaVisualizer",
    "riasec",
]
