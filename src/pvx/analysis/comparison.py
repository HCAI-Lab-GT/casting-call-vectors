"""Cross-model comparison utilities for persona vector experiments.

This module provides tools for comparing results across different models,
enabling systematic analysis of how different LLMs represent vocational
personas.

Example:
    >>> from pvx.analysis.comparison import ModelComparison
    >>> from pvx.outputs import OutputManager
    >>>
    >>> mgr = OutputManager()
    >>> runs = [mgr.get_latest(m) for m in ['olmo-7b-instruct', 'smollm3-3b']]
    >>> comparison = ModelComparison(runs)
    >>> comparison.compare_alignment_scores()
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pvx.config import RunConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a single model run."""

    model_slug: str
    run_id: str
    n_personas: int
    hidden_dim: int
    alignment_score: float | None = None
    silhouette_score: float | None = None
    mean_riasec_purity: float | None = None
    pcs_for_50_variance: int | None = None
    pcs_for_80_variance: int | None = None
    pcs_for_95_variance: int | None = None
    go_no_go: bool | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_run(cls, run: "RunConfig") -> "ModelMetrics":
        """Load metrics from a completed run.

        Args:
            run: RunConfig pointing to a completed run

        Returns:
            ModelMetrics with loaded values
        """
        from pvx.extraction.pipeline import PersonaVector

        # Count vectors
        vectors_dir = run.vectors_dir()
        vector_files = list(vectors_dir.glob("*.pt")) if vectors_dir.exists() else []

        # Get hidden dimension from first vector
        hidden_dim = 0
        if vector_files:
            try:
                vec = PersonaVector.load(vector_files[0].with_suffix(""))
                hidden_dim = vec.prompt_last_diff.shape[-1]
            except Exception:
                pass

        metrics = cls(
            model_slug=run.model_slug,
            run_id=run.run_id,
            n_personas=len(vector_files),
            hidden_dim=hidden_dim,
        )

        # Load analysis results if available
        # Check multiple possible file names for backward compatibility
        analysis_dir = run.analysis_dir()
        analysis_files = [
            analysis_dir / "riasec_analysis.json",
            analysis_dir / "riasec_validation.json",  # Legacy validation format
        ]

        for analysis_file in analysis_files:
            if analysis_file.exists():
                try:
                    with open(analysis_file) as f:
                        analysis = json.load(f)

                    # Handle both formats (full analysis vs validation)
                    if "riasec" in analysis:
                        # Full analysis format
                        riasec = analysis.get("riasec", {})
                        metrics.alignment_score = riasec.get("alignment_score")
                        metrics.go_no_go = riasec.get("go_no_go")

                        pca = analysis.get("pca", {})
                        metrics.pcs_for_50_variance = pca.get("pcs_for_50_variance")
                        metrics.pcs_for_80_variance = pca.get("pcs_for_80_variance")
                        metrics.pcs_for_95_variance = pca.get("pcs_for_95_variance")

                        clustering = analysis.get("clustering", {})
                        metrics.silhouette_score = clustering.get("silhouette_score")
                        metrics.mean_riasec_purity = clustering.get("mean_riasec_purity")
                    else:
                        # Validation format (flat structure)
                        metrics.alignment_score = analysis.get("alignment_score")
                        metrics.go_no_go = analysis.get("go_no_go")

                    break  # Stop after first successful load

                except Exception as e:
                    logger.warning(f"Failed to load {analysis_file}: {e}")

        return metrics


class ModelComparison:
    """Compare results across multiple model runs.

    Provides utilities for comparing RIASEC alignment, PCA structure,
    and other metrics across different models.

    Example:
        >>> runs = [mgr.get_latest("olmo-7b"), mgr.get_latest("smollm3-3b")]
        >>> comparison = ModelComparison(runs)
        >>> df = comparison.to_dataframe()
    """

    def __init__(self, runs: list["RunConfig"]):
        """Initialize comparison with list of runs.

        Args:
            runs: List of RunConfig objects to compare
        """
        self.runs = runs
        self.metrics = [ModelMetrics.from_run(run) for run in runs]

    def to_dict(self) -> list[dict]:
        """Convert all metrics to list of dicts."""
        return [
            {
                "model": m.model_slug,
                "run_id": m.run_id,
                "n_personas": m.n_personas,
                "hidden_dim": m.hidden_dim,
                "alignment_score": m.alignment_score,
                "silhouette_score": m.silhouette_score,
                "mean_riasec_purity": m.mean_riasec_purity,
                "pcs_50var": m.pcs_for_50_variance,
                "pcs_80var": m.pcs_for_80_variance,
                "pcs_95var": m.pcs_for_95_variance,
                "go_no_go": m.go_no_go,
            }
            for m in self.metrics
        ]

    def to_dataframe(self):
        """Convert to pandas DataFrame.

        Returns:
            DataFrame with one row per model
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.to_dict())
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")

    def compare_alignment_scores(self) -> dict[str, float | None]:
        """Get alignment scores by model.

        Returns:
            Dict mapping model_slug to alignment score
        """
        return {m.model_slug: m.alignment_score for m in self.metrics}

    def compare_pca_structure(self) -> dict[str, dict]:
        """Compare PCA variance structure across models.

        Returns:
            Dict mapping model_slug to PCA metrics
        """
        return {
            m.model_slug: {
                "pcs_for_50_variance": m.pcs_for_50_variance,
                "pcs_for_80_variance": m.pcs_for_80_variance,
                "pcs_for_95_variance": m.pcs_for_95_variance,
            }
            for m in self.metrics
        }

    def compare_clustering(self) -> dict[str, dict]:
        """Compare clustering metrics across models.

        Returns:
            Dict mapping model_slug to clustering metrics
        """
        return {
            m.model_slug: {
                "silhouette_score": m.silhouette_score,
                "mean_riasec_purity": m.mean_riasec_purity,
            }
            for m in self.metrics
        }

    def best_alignment_model(self) -> str | None:
        """Get model with highest alignment score.

        Returns:
            Model slug with best alignment, or None if no scores
        """
        scored = [(m.model_slug, m.alignment_score) for m in self.metrics if m.alignment_score]
        if not scored:
            return None
        return max(scored, key=lambda x: x[1])[0]

    def summary(self) -> str:
        """Generate text summary of comparison.

        Returns:
            Multi-line summary string
        """
        lines = ["=" * 60, "Cross-Model Comparison Summary", "=" * 60, ""]

        # Model overview
        lines.append("Models Compared:")
        for m in self.metrics:
            status = "GO" if m.go_no_go else "NO-GO" if m.go_no_go is False else "?"
            lines.append(f"  - {m.model_slug} ({m.n_personas} personas, {m.hidden_dim}d)")

        lines.append("")

        # Alignment scores
        lines.append("RIASEC Alignment Scores:")
        for m in self.metrics:
            score = f"{m.alignment_score:.3f}" if m.alignment_score else "N/A"
            status = "GO" if m.go_no_go else "NO-GO" if m.go_no_go is False else "?"
            lines.append(f"  {m.model_slug}: {score} ({status})")

        best = self.best_alignment_model()
        if best:
            lines.append(f"  Best: {best}")

        lines.append("")

        # Clustering
        lines.append("Clustering Quality:")
        for m in self.metrics:
            sil = f"{m.silhouette_score:.3f}" if m.silhouette_score else "N/A"
            pur = f"{m.mean_riasec_purity:.3f}" if m.mean_riasec_purity else "N/A"
            lines.append(f"  {m.model_slug}: silhouette={sil}, purity={pur}")

        lines.append("")

        # PCA structure
        lines.append("PCA Dimensionality (PCs for X% variance):")
        for m in self.metrics:
            p50 = m.pcs_for_50_variance or "?"
            p80 = m.pcs_for_80_variance or "?"
            p95 = m.pcs_for_95_variance or "?"
            lines.append(f"  {m.model_slug}: 50%={p50}, 80%={p80}, 95%={p95}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save(self, output_dir: Path | str) -> Path:
        """Save comparison results to JSON.

        Args:
            output_dir: Directory to save results

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "models": [r.model_slug for r in self.runs],
            "run_ids": [r.run_id for r in self.runs],
            "metrics": self.to_dict(),
            "alignment_scores": self.compare_alignment_scores(),
            "pca_structure": self.compare_pca_structure(),
            "clustering": self.compare_clustering(),
            "best_alignment_model": self.best_alignment_model(),
        }

        output_file = output_dir / "model_comparison.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        return output_file

    def __repr__(self) -> str:
        models = ", ".join(m.model_slug for m in self.metrics)
        return f"ModelComparison([{models}])"
