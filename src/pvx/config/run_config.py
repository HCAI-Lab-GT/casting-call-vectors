"""Run configuration and tracking for multi-model experiments.

This module provides RunConfig for tracking experiment runs with full
traceability including model info, timestamps, and git state.

Each run produces isolated outputs in:
    outputs/runs/{model_slug}/{run_id}/
"""

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _get_git_hash() -> str:
    """Get current git commit hash (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _model_id_to_slug(model_id: str) -> str:
    """Convert HuggingFace model ID to filesystem-safe slug.

    Examples:
        "allenai/OLMo-7B-Instruct" -> "olmo-7b-instruct"
        "HuggingFaceTB/SmolLM2-135M-Instruct" -> "smollm2-135m-instruct"
        "Qwen/Qwen2.5-7B-Instruct" -> "qwen2.5-7b-instruct"
    """
    # Take the model name part (after /)
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    # Convert to lowercase
    slug = name.lower()
    # Replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Remove leading/trailing hyphens
    return slug.strip("-")


def _generate_run_id() -> str:
    """Generate unique run ID: YYYY-MM-DD_HHMMSS_xxxxx.

    Format: datetime + 5-char random hash for uniqueness.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H%M%S")
    # Short random suffix for uniqueness
    random_suffix = hashlib.sha256(str(now.timestamp()).encode()).hexdigest()[:5]
    return f"{date_str}_{random_suffix}"


@dataclass
class RunConfig:
    """Configuration and metadata for an experiment run.

    Each run is uniquely identified by (model_slug, run_id) and produces
    outputs in outputs/runs/{model_slug}/{run_id}/.

    Attributes:
        run_id: Unique identifier (datetime + hash)
        model_id: Full HuggingFace model identifier
        model_slug: Filesystem-safe model name
        layer: Layer for activation extraction
        num_questions: Questions per persona
        personas_dir: Path to persona JSON files
        git_hash: Git commit hash at run start
        git_branch: Git branch at run start
        started_at: Run start timestamp
        completed_at: Run completion timestamp (None if incomplete)
        wandb_run_id: W&B run ID if logging enabled
        base_output_dir: Base directory for all outputs
        extra: Additional run-specific metadata
    """

    run_id: str
    model_id: str
    model_slug: str
    layer: int
    num_questions: int
    personas_dir: str
    git_hash: str
    git_branch: str
    started_at: str  # ISO format string for JSON serialization
    completed_at: str | None = None
    wandb_run_id: str | None = None
    base_output_dir: str = "outputs/runs"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model_id: str,
        layer: int = 14,
        num_questions: int = 50,
        personas_dir: str = "persona_data/vocational_personas/instructions",
        base_output_dir: str = "outputs/runs",
        **extra,
    ) -> "RunConfig":
        """Create a new RunConfig with auto-generated run_id and git info.

        Args:
            model_id: HuggingFace model identifier
            layer: Layer for activation extraction
            num_questions: Questions per persona
            personas_dir: Path to persona JSON files
            base_output_dir: Base directory for outputs
            **extra: Additional metadata to store

        Returns:
            New RunConfig instance
        """
        return cls(
            run_id=_generate_run_id(),
            model_id=model_id,
            model_slug=_model_id_to_slug(model_id),
            layer=layer,
            num_questions=num_questions,
            personas_dir=personas_dir,
            git_hash=_get_git_hash(),
            git_branch=_get_git_branch(),
            started_at=datetime.now().isoformat(),
            completed_at=None,
            wandb_run_id=None,
            base_output_dir=base_output_dir,
            extra=extra,
        )

    def output_dir(self) -> Path:
        """Get the output directory for this run."""
        return Path(self.base_output_dir) / self.model_slug / self.run_id

    def vectors_dir(self) -> Path:
        """Get the vectors subdirectory."""
        return self.output_dir() / "vectors"

    def analysis_dir(self) -> Path:
        """Get the analysis subdirectory."""
        return self.output_dir() / "analysis"

    def plots_dir(self) -> Path:
        """Get the plots subdirectory."""
        return self.analysis_dir() / "plots"

    def config_path(self) -> Path:
        """Get the path for run_config.json."""
        return self.output_dir() / "run_config.json"

    def create_directories(self) -> None:
        """Create all output directories for this run."""
        self.vectors_dir().mkdir(parents=True, exist_ok=True)
        self.analysis_dir().mkdir(parents=True, exist_ok=True)
        self.plots_dir().mkdir(parents=True, exist_ok=True)

    def mark_complete(self) -> None:
        """Mark the run as complete with current timestamp."""
        self.completed_at = datetime.now().isoformat()

    def save(self) -> Path:
        """Save run configuration to JSON file.

        Returns:
            Path to the saved config file
        """
        self.create_directories()
        config_path = self.config_path()
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return config_path

    @classmethod
    def load(cls, run_dir: Path | str) -> "RunConfig":
        """Load RunConfig from a run directory.

        Args:
            run_dir: Path to the run directory (containing run_config.json)

        Returns:
            RunConfig instance

        Raises:
            FileNotFoundError: If run_config.json doesn't exist
        """
        run_dir = Path(run_dir)
        config_path = run_dir / "run_config.json"
        with open(config_path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def list_runs(
        cls,
        model_slug: str | None = None,
        base_output_dir: str = "outputs/runs",
    ) -> list["RunConfig"]:
        """List all runs, optionally filtered by model.

        Args:
            model_slug: Only return runs for this model (None for all)
            base_output_dir: Base directory to search

        Returns:
            List of RunConfig instances, sorted by start time (newest first)
        """
        base = Path(base_output_dir)
        if not base.exists():
            return []

        runs = []

        # Determine which model directories to search
        if model_slug:
            model_dirs = [base / model_slug] if (base / model_slug).exists() else []
        else:
            model_dirs = [d for d in base.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            for run_dir in model_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "run_config.json").exists():
                    try:
                        runs.append(cls.load(run_dir))
                    except Exception:
                        pass  # Skip invalid configs

        # Sort by start time (newest first)
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs

    @classmethod
    def get_latest(
        cls,
        model_slug: str,
        base_output_dir: str = "outputs/runs",
    ) -> "RunConfig | None":
        """Get the most recent run for a model.

        Args:
            model_slug: Model to find latest run for
            base_output_dir: Base directory to search

        Returns:
            Most recent RunConfig, or None if no runs exist
        """
        runs = cls.list_runs(model_slug=model_slug, base_output_dir=base_output_dir)
        return runs[0] if runs else None

    def to_wandb_config(self) -> dict:
        """Convert to W&B config format."""
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "model_slug": self.model_slug,
            "layer": self.layer,
            "num_questions": self.num_questions,
            "git_hash": self.git_hash,
            "git_branch": self.git_branch,
            **self.extra,
        }

    def __repr__(self) -> str:
        status = "complete" if self.completed_at else "in_progress"
        return f"RunConfig({self.model_slug}/{self.run_id}, {status})"
