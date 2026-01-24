"""End-to-end extraction pipeline for persona vectors.

This module provides the ExtractionPipeline class that orchestrates the
complete persona vector extraction process, including:
- Persona/baseline activation extraction
- LLM judge scoring for quality filtering
- Vector computation (contrastive mean differences)
- Checkpointing and W&B logging
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import torch
from tqdm import tqdm

from ..sources.base import BaselineSource, PersonaMetadata, PersonaSource
from .activations import ActivationExtractor, ActivationResult
from .questions import QuestionBank

logger = logging.getLogger(__name__)


@dataclass
class PersonaVector:
    """Extracted persona vector with metadata.

    Attributes:
        persona_id: Unique identifier for the persona
        prompt_last_diff: Contrastive vector from last prompt token (1, hidden_dim)
        response_mean_diff: Contrastive vector from response mean (1, hidden_dim)
        response_all_layers_diff: Contrastive vectors for all layers (num_layers, 1, hidden_dim)
        metadata: Persona metadata (RIASEC scores, title, etc.)
        extraction_stats: Statistics from extraction process
    """

    persona_id: str
    prompt_last_diff: torch.Tensor
    response_mean_diff: torch.Tensor
    response_all_layers_diff: torch.Tensor
    metadata: PersonaMetadata
    extraction_stats: dict = field(default_factory=dict)

    def save(self, filepath: Path | str) -> None:
        """Save persona vector to disk.

        Args:
            filepath: Path for the output file (without extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save tensors
        torch.save(
            {
                "prompt_last_diff": self.prompt_last_diff,
                "response_mean_diff": self.response_mean_diff,
                "response_all_layers_diff": self.response_all_layers_diff,
            },
            filepath.with_suffix(".pt"),
        )

        # Save metadata as JSON
        meta_dict = {
            "persona_id": self.persona_id,
            "metadata": dict(self.metadata),
            "extraction_stats": self.extraction_stats,
            "saved_at": datetime.now().isoformat(),
        }
        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(meta_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: Path | str) -> "PersonaVector":
        """Load persona vector from disk.

        Args:
            filepath: Path to the saved files (without extension)

        Returns:
            PersonaVector instance
        """
        filepath = Path(filepath)

        # Load tensors
        tensors = torch.load(filepath.with_suffix(".pt"), weights_only=True)

        # Load metadata
        with open(filepath.with_suffix(".json")) as f:
            meta = json.load(f)

        return cls(
            persona_id=meta["persona_id"],
            prompt_last_diff=tensors["prompt_last_diff"],
            response_mean_diff=tensors["response_mean_diff"],
            response_all_layers_diff=tensors["response_all_layers_diff"],
            metadata=PersonaMetadata(**meta.get("metadata", {})),
            extraction_stats=meta.get("extraction_stats", {}),
        )


class ExtractionPipeline:
    """End-to-end pipeline for extracting persona vectors.

    Coordinates:
    - ActivationExtractor for model activations
    - QuestionBank for extraction questions
    - Optional LLM judge for quality filtering
    - Checkpointing and W&B logging

    Example:
        >>> pipeline = ExtractionPipeline(
        ...     model_id="allenai/OLMo-7B-Instruct",
        ...     questions=QuestionBank.from_assistant_axis(),
        ... )
        >>> personas = load_vocational_personas(riasec_filter="S", limit=25)
        >>> vectors = pipeline.extract_batch(personas)
    """

    def __init__(
        self,
        model_id: str = "allenai/OLMo-7B-Instruct",
        layer: int = 14,
        questions: QuestionBank | None = None,
        judge: Optional["LLMJudge"] = None,
        judge_threshold: float = 2.5,
        output_dir: Path | str = "outputs/vectors",
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ):
        """Initialize the extraction pipeline.

        Args:
            model_id: HuggingFace model identifier
            layer: Layer for activation extraction
            questions: QuestionBank instance (defaults to assistant-axis)
            judge: Optional LLM judge for quality filtering
            judge_threshold: Minimum score for valid responses (0-3 scale)
            output_dir: Directory for saving vectors and checkpoints
            wandb_project: W&B project name (None to disable logging)
            wandb_run_name: W&B run name (auto-generated if None)
        """
        self.model_id = model_id
        self.layer = layer
        self.judge = judge
        self.judge_threshold = judge_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load questions
        self.questions = questions or QuestionBank.from_assistant_axis()
        logger.info(f"Loaded {len(self.questions)} extraction questions")

        # Lazy-load extractor (expensive)
        self._extractor: ActivationExtractor | None = None

        # W&B setup
        self.wandb_project = wandb_project
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "model_id": model_id,
                        "layer": layer,
                        "num_questions": len(self.questions),
                        "judge_threshold": judge_threshold,
                    },
                )
                logger.info(f"W&B logging enabled: {wandb_project}")
            except ImportError:
                logger.warning("wandb not installed, logging disabled")

    @property
    def extractor(self) -> ActivationExtractor:
        """Lazy-load the activation extractor."""
        if self._extractor is None:
            self._extractor = ActivationExtractor(
                model_id=self.model_id,
                layer=self.layer,
            )
        return self._extractor

    def extract_persona(
        self,
        source: PersonaSource,
        baseline: PersonaSource | None = None,
        num_questions: int = 50,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int | None = None,
    ) -> PersonaVector:
        """Extract persona vector for a single persona.

        Computes the contrastive difference between persona activations
        and baseline activations across sampled questions.

        Args:
            source: Persona source to extract
            baseline: Baseline source for contrast (defaults to BaselineSource)
            num_questions: Number of questions to sample
            max_new_tokens: Max tokens per response
            temperature: Sampling temperature
            seed: Random seed for question sampling

        Returns:
            PersonaVector with extracted vectors and metadata
        """
        if baseline is None:
            baseline = BaselineSource.from_vocational_default()

        # Sample questions
        questions = self.questions.sample(num_questions, seed=seed)

        # Get system prompts (use first variant for simplicity)
        persona_prompt = source.get_system_prompts()[0]
        baseline_prompt = baseline.get_baseline_prompts()[0]

        # Accumulators for activations
        persona_prompt_last_sum = None
        persona_resp_mean_sum = None
        persona_all_layers_sum = None

        baseline_prompt_last_sum = None
        baseline_resp_mean_sum = None
        baseline_all_layers_sum = None

        valid_count = 0
        total_count = 0
        judge_scores = []

        for question in tqdm(questions, desc=f"Extracting {source.persona_id}", leave=False):
            total_count += 1

            # Extract persona activations
            persona_result = self.extractor.extract(
                system_prompt=persona_prompt,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            # Extract baseline activations
            baseline_result = self.extractor.extract(
                system_prompt=baseline_prompt,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            # Optional judge filtering
            if self.judge is not None:
                persona_score = self.judge(question=question, answer=persona_result.response_text)
                baseline_score = self.judge(question=question, answer=baseline_result.response_text)
                judge_scores.append({"persona": persona_score, "baseline": baseline_score})

                # Skip if persona doesn't meet threshold
                # For vocational personas: we want persona to score HIGH (showing role)
                if persona_score < self.judge_threshold:
                    continue

            # Initialize accumulators
            if persona_prompt_last_sum is None:
                persona_prompt_last_sum = torch.zeros_like(persona_result.prompt_last, dtype=torch.float32)
                persona_resp_mean_sum = torch.zeros_like(persona_result.response_mean, dtype=torch.float32)
                persona_all_layers_sum = torch.zeros_like(persona_result.response_all_layers, dtype=torch.float32)

                baseline_prompt_last_sum = torch.zeros_like(baseline_result.prompt_last, dtype=torch.float32)
                baseline_resp_mean_sum = torch.zeros_like(baseline_result.response_mean, dtype=torch.float32)
                baseline_all_layers_sum = torch.zeros_like(baseline_result.response_all_layers, dtype=torch.float32)

            # Accumulate
            persona_prompt_last_sum += persona_result.prompt_last.float()
            persona_resp_mean_sum += persona_result.response_mean.float()
            persona_all_layers_sum += persona_result.response_all_layers.float()

            baseline_prompt_last_sum += baseline_result.prompt_last.float()
            baseline_resp_mean_sum += baseline_result.response_mean.float()
            baseline_all_layers_sum += baseline_result.response_all_layers.float()

            valid_count += 1

        # Compute means and differences
        if valid_count == 0:
            logger.warning(f"No valid pairs for {source.persona_id}")
            # Return zero vectors
            hidden_dim = self.extractor.model.config.hidden_size
            num_layers = self.extractor.model.config.num_hidden_layers + 1
            return PersonaVector(
                persona_id=source.persona_id,
                prompt_last_diff=torch.zeros(1, hidden_dim),
                response_mean_diff=torch.zeros(1, hidden_dim),
                response_all_layers_diff=torch.zeros(num_layers, 1, hidden_dim),
                metadata=source.get_metadata(),
                extraction_stats={
                    "valid_pairs": 0,
                    "total_pairs": total_count,
                    "judge_scores": judge_scores,
                },
            )

        # Compute contrastive differences
        prompt_last_diff = (persona_prompt_last_sum / valid_count) - (baseline_prompt_last_sum / valid_count)
        response_mean_diff = (persona_resp_mean_sum / valid_count) - (baseline_resp_mean_sum / valid_count)
        all_layers_diff = (persona_all_layers_sum / valid_count) - (baseline_all_layers_sum / valid_count)

        # Log to W&B
        if self.wandb_run:
            import wandb

            self.wandb_run.log({
                f"{source.persona_id}/valid_pairs": valid_count,
                f"{source.persona_id}/total_pairs": total_count,
                f"{source.persona_id}/vector_norm": torch.norm(prompt_last_diff).item(),
            })

        return PersonaVector(
            persona_id=source.persona_id,
            prompt_last_diff=prompt_last_diff,
            response_mean_diff=response_mean_diff,
            response_all_layers_diff=all_layers_diff,
            metadata=source.get_metadata(),
            extraction_stats={
                "valid_pairs": valid_count,
                "total_pairs": total_count,
                "judge_scores": judge_scores,
            },
        )

    def extract_batch(
        self,
        sources: list[PersonaSource],
        baseline: PersonaSource | None = None,
        num_questions: int = 50,
        checkpoint_every: int = 1,
        resume_from: Path | str | None = None,
    ) -> list[PersonaVector]:
        """Extract persona vectors for multiple personas.

        Includes checkpointing after each persona for fault tolerance.

        Args:
            sources: List of persona sources to extract
            baseline: Baseline source (shared across all personas)
            num_questions: Questions per persona
            checkpoint_every: Save checkpoint every N personas
            resume_from: Resume from checkpoint directory

        Returns:
            List of PersonaVector objects
        """
        if baseline is None:
            baseline = BaselineSource.from_vocational_default()

        vectors = []
        completed_ids = set()

        # Resume from checkpoint if specified
        if resume_from:
            resume_dir = Path(resume_from)
            for json_file in resume_dir.glob("*.json"):
                try:
                    vec = PersonaVector.load(json_file.with_suffix(""))
                    vectors.append(vec)
                    completed_ids.add(vec.persona_id)
                    logger.info(f"Resumed: {vec.persona_id}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {json_file}: {e}")

        # Filter out completed personas
        remaining = [s for s in sources if s.persona_id not in completed_ids]
        logger.info(f"Extracting {len(remaining)} personas ({len(completed_ids)} already completed)")

        for i, source in enumerate(tqdm(remaining, desc="Extracting personas")):
            try:
                vec = self.extract_persona(
                    source=source,
                    baseline=baseline,
                    num_questions=num_questions,
                )
                vectors.append(vec)

                # Checkpoint
                if (i + 1) % checkpoint_every == 0:
                    checkpoint_path = self.output_dir / source.persona_id
                    vec.save(checkpoint_path)
                    logger.info(f"Checkpointed: {source.persona_id}")

                # W&B artifact
                if self.wandb_run and (i + 1) % 10 == 0:
                    import wandb

                    artifact = wandb.Artifact(
                        name=f"vectors-checkpoint-{i+1}",
                        type="vectors",
                    )
                    artifact.add_dir(str(self.output_dir))
                    self.wandb_run.log_artifact(artifact)

            except Exception as e:
                logger.error(f"Failed to extract {source.persona_id}: {e}")
                continue

        # Final save
        for vec in vectors:
            if vec.persona_id not in completed_ids:
                vec.save(self.output_dir / vec.persona_id)

        # Final W&B artifact
        if self.wandb_run:
            import wandb

            artifact = wandb.Artifact(name="vectors-final", type="vectors")
            artifact.add_dir(str(self.output_dir))
            self.wandb_run.log_artifact(artifact)
            self.wandb_run.finish()

        return vectors

    def __repr__(self) -> str:
        return f"ExtractionPipeline(model={self.model_id!r}, layer={self.layer}, questions={len(self.questions)})"
