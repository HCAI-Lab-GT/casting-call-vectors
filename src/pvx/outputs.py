"""Output directory management for multi-model experiments.

This module provides the OutputManager for navigating and managing
the multi-model output directory structure.

Directory Structure:
    outputs/
    ├── runs/
    │   ├── {model_slug}/
    │   │   ├── {run_id}/
    │   │   │   ├── run_config.json
    │   │   │   ├── vectors/
    │   │   │   └── analysis/
    │   │   └── ...
    │   └── ...
    └── comparisons/
        └── {comparison_id}/
"""

import logging
import shutil
from pathlib import Path

from .config import RunConfig

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages output directories for multi-model experiments.

    Provides utilities for navigating the directory structure,
    finding runs, and managing outputs across multiple models.

    Example:
        >>> mgr = OutputManager()
        >>> mgr.list_models()
        ['olmo-7b', 'smollm2-135m']
        >>> latest = mgr.get_latest('olmo-7b')
        >>> latest.output_dir()
        PosixPath('outputs/runs/olmo-7b/2026-01-25_abc123')
    """

    def __init__(self, base_dir: Path | str = "outputs/runs"):
        """Initialize the OutputManager.

        Args:
            base_dir: Base directory for all run outputs
        """
        self.base_dir = Path(base_dir)

    def ensure_base_dir(self) -> None:
        """Create base directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[str]:
        """List all models with runs.

        Returns:
            List of model slugs (sorted alphabetically)
        """
        if not self.base_dir.exists():
            return []

        models = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                # Check if it has at least one valid run
                for run_dir in d.iterdir():
                    if run_dir.is_dir() and (run_dir / "run_config.json").exists():
                        models.append(d.name)
                        break

        return sorted(models)

    def list_runs(
        self,
        model_slug: str | None = None,
        completed_only: bool = False,
    ) -> list[RunConfig]:
        """List runs, optionally filtered by model.

        Args:
            model_slug: Only return runs for this model (None for all)
            completed_only: Only return completed runs

        Returns:
            List of RunConfig instances (sorted by start time, newest first)
        """
        runs = RunConfig.list_runs(
            model_slug=model_slug,
            base_output_dir=str(self.base_dir),
        )

        if completed_only:
            runs = [r for r in runs if r.completed_at is not None]

        return runs

    def get_latest(self, model_slug: str) -> RunConfig | None:
        """Get the most recent run for a model.

        Args:
            model_slug: Model to find latest run for

        Returns:
            Most recent RunConfig, or None if no runs exist
        """
        return RunConfig.get_latest(
            model_slug=model_slug,
            base_output_dir=str(self.base_dir),
        )

    def get_run(self, model_slug: str, run_id: str) -> RunConfig | None:
        """Get a specific run by model and run_id.

        Args:
            model_slug: Model slug
            run_id: Run identifier

        Returns:
            RunConfig if found, None otherwise
        """
        run_dir = self.base_dir / model_slug / run_id
        if run_dir.exists() and (run_dir / "run_config.json").exists():
            try:
                return RunConfig.load(run_dir)
            except Exception:
                return None
        return None

    def count_vectors(self, run: RunConfig) -> int:
        """Count the number of vectors in a run.

        Args:
            run: RunConfig to check

        Returns:
            Number of .pt vector files
        """
        vectors_dir = run.vectors_dir()
        if not vectors_dir.exists():
            return 0
        return len(list(vectors_dir.glob("*.pt")))

    def get_run_status(self, run: RunConfig) -> dict:
        """Get detailed status of a run.

        Args:
            run: RunConfig to check

        Returns:
            Status dict with counts and flags
        """
        return {
            "run_id": run.run_id,
            "model_slug": run.model_slug,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "is_complete": run.completed_at is not None,
            "vector_count": self.count_vectors(run),
            "has_analysis": run.analysis_dir().exists()
            and len(list(run.analysis_dir().glob("*.json"))) > 0,
        }

    def cleanup_incomplete(self, model_slug: str | None = None) -> list[str]:
        """Remove incomplete runs (no completed_at).

        Args:
            model_slug: Only cleanup this model (None for all)

        Returns:
            List of removed run directories
        """
        removed = []
        runs = self.list_runs(model_slug=model_slug)

        for run in runs:
            if run.completed_at is None:
                run_dir = run.output_dir()
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    removed.append(str(run_dir))
                    logger.info(f"Removed incomplete run: {run_dir}")

        return removed

    def update_latest_symlink(self, model_slug: str) -> Path | None:
        """Update the 'latest' symlink for a model.

        Args:
            model_slug: Model to update symlink for

        Returns:
            Path to symlink, or None if no runs exist
        """
        latest = self.get_latest(model_slug)
        if latest is None:
            return None

        model_dir = self.base_dir / model_slug
        symlink = model_dir / "latest"

        # Remove existing symlink if it exists
        if symlink.is_symlink():
            symlink.unlink()
        elif symlink.exists():
            # It's a regular file/dir, don't overwrite
            return None

        # Create relative symlink
        symlink.symlink_to(latest.run_id)
        return symlink

    def summary(self) -> dict:
        """Get a summary of all outputs.

        Returns:
            Dict with model counts, run counts, etc.
        """
        models = self.list_models()
        summary = {
            "base_dir": str(self.base_dir),
            "model_count": len(models),
            "models": {},
        }

        for model in models:
            runs = self.list_runs(model_slug=model)
            completed = [r for r in runs if r.completed_at is not None]
            summary["models"][model] = {
                "total_runs": len(runs),
                "completed_runs": len(completed),
                "latest_run_id": runs[0].run_id if runs else None,
            }

        return summary

    def __repr__(self) -> str:
        models = self.list_models()
        return f"OutputManager(base={self.base_dir!r}, models={len(models)})"


def get_output_manager(base_dir: str = "outputs/runs") -> OutputManager:
    """Get the default OutputManager instance.

    Args:
        base_dir: Base directory for outputs

    Returns:
        OutputManager instance
    """
    return OutputManager(base_dir=base_dir)
