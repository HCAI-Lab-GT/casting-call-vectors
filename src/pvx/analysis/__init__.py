"""Geometry analysis and visualization for persona vectors.

This module provides analysis and visualization capabilities:

- PersonaGeometry: PCA, clustering, and distance analysis (axis-agnostic)
- PersonaVisualizer: 2D/3D plots and W&B dashboard integration
- riasec: RIASEC-specific analysis functions (Holland codes)
- comparison: Cross-model comparison utilities
"""

from . import riasec
from .comparison import ModelComparison, ModelMetrics
from .geometry import ClusterResult, ContrastResult, PCAResult, PersonaGeometry
from .viz import (
    PersonaVisualizer,
    plot_alignment_comparison,
    plot_model_comparison_bar,
    save_comparison_plots,
)

__all__ = [
    "PersonaGeometry",
    "PCAResult",
    "ClusterResult",
    "ContrastResult",
    "PersonaVisualizer",
    "ModelComparison",
    "ModelMetrics",
    "plot_alignment_comparison",
    "plot_model_comparison_bar",
    "save_comparison_plots",
    "riasec",
]
