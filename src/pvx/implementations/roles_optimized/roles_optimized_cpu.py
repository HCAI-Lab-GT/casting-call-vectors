"""
CPU task for roles optimization: Generate and judge Q/A responses.

This module handles the CPU-intensive work of generating model responses
using the configured backend (OpenRouter API, OpenAI, vLLM, or local HF model)
and judging them using an LLM judge. Results are saved to JSON for later
GPU processing of activation extraction.

Architecture:
- Answer Model: Runs via ResponseGeneration backend (remote API or local)
- Activation Extraction Model: Runs on GPU in separate GPU phase
- Both default to Olmo-3-7B-Instruct but can be different

This separation optimizes GPU usage:
- CPU jobs: Generate answers (no GPU), judge responses
- GPU jobs: Extract activations from pre-judged Q/A pairs
- GPU is only used when needed (activation extraction phase)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from dotenv import load_dotenv

from pvx import Heartbeat, setup_logging
from pvx.implementations.roles.role_dataset import RoleDataset
from pvx.implementations.judges.role_judge import RoleJudge
from pvx.utils.response_generation import ResponseGeneration

load_dotenv()

logger = setup_logging(name="roles-optimized-cpu")


class RoleQAGenerator(ResponseGeneration):
    """
    CPU task: Generate and judge Q/A responses for roles.

    Extends ResponseGeneration to support multiple backends:
    - openai/openrouter: Remote API calls (no GPU)
    - vllm: Local vLLM server (no GPU)
    - hf_local: Local HF model (optional GPU)

    Default backend uses OpenRouter API for efficient CPU-only operation.
    """

    def __init__(
        self,
        answer_model: str = "allenai/Olmo-3-7B-Instruct",
        backend: str = "openai",
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        temperature: float = 0.9,
        max_new_tokens: int = 2048,
    ):
        """
        Initialize the Q/A generator.

        Args:
            answer_model: Model ID for answer generation (default: Olmo-3-7B-Instruct)
            backend: One of {"openai", "vllm", "hf_local"} (default: "openai" for OpenRouter)
            base_url: API base URL for openai/vllm backends
            api_key_env: Environment variable name for API key
            temperature: Sampling temperature (default: 0.9)
            max_new_tokens: Max tokens to generate (default: 2048)
        """
        # Initialize parent ResponseGeneration class
        super().__init__(
            backend=backend,
            model=answer_model,
            local_model=answer_model,
            base_url=base_url,
            api_key_env=api_key_env,
        )

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Initialize judge
        self.judge = RoleJudge()
        logger.info(
            f"Initialized RoleQAGenerator with model={answer_model}, backend={backend}"
        )

    def generate_and_judge_qa(
        self,
        role: str,
        dataset: RoleDataset,
        target_pairs: int = 20,
        max_retries: int = 2,
        existing_data: Optional[Dict[str, List[Dict]]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Generate and judge Q/A responses for a role.

        Generates responses for both positive and base (empty) system prompts,
        judges both, and returns valid pairs that pass filtering.

        Returns dict with structure:
        {
            "positive": [
                {
                    "pos_prompt": "...",
                    "question": "...",
                    "response": "...",
                    "score": 3
                },
                ...
            ],
            "base": [
                {
                    "question": "...",
                    "response": "...",
                    "score": 1
                },
                ...
            ]
        }

        Args:
            role: Role name
            dataset: RoleDataset instance
            target_pairs: Target number of valid pairs (default: 20)
            max_retries: Max retry epochs (default: 2)
            existing_data: Previously saved Q/A data to resume from (default: None)

        Returns:
            Dictionary with "positive" and "base" Q/A pairs
        """
        logger.info(f"Generating and judging Q/A for role: {role}")

        # Get question-prompt pairs from dataset
        prompt_pairs = dataset.extract_pos_question_pairs()
        if not prompt_pairs:
            logger.warning(f"No prompt pairs found for role {role}")
            return {"positive": [], "base": []}

        if existing_data:
            qa_data = existing_data
            n = len(qa_data.get("positive", []))
            passed_pairs = {
                (entry["pos_prompt"], entry["question"])
                for entry in qa_data.get("positive", [])
            }
            logger.info(f"Resuming from {n} existing pairs for role {role}")
        else:
            qa_data = {"positive": [], "base": []}
            passed_pairs = set()
            n = 0

        for retry_epoch in range(max_retries):
            remaining = [p for p in prompt_pairs if p not in passed_pairs]
            if not remaining:
                logger.info(f"All {len(prompt_pairs)} pairs attempted in epoch {retry_epoch}")
                break

            # Shuffle remaining pairs for this epoch
            import random
            epoch_pairs = random.sample(remaining, len(remaining))
            made_progress = False

            with Heartbeat(
                logger, f"Generating Q/A for {role} (epoch {retry_epoch + 1})", interval=10
            ):
                for pos_prompt, question in epoch_pairs:
                    if n >= target_pairs:
                        break

                    logger.info(f"Processing pair {n + 1}/{target_pairs}")

                    # Generate response with positive prompt
                    messages_pos = [
                        {"role": "system", "content": pos_prompt},
                        {"role": "user", "content": question},
                    ]
                    try:
                        _, pos_response = self._inference_with_client(
                            messages=messages_pos,
                            temperature=self.temperature,
                            max_new_tokens=self.max_new_tokens,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate positive response: {e}")
                        continue

                    # Generate response with base (empty) prompt
                    messages_base = [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": question},
                    ]
                    try:
                        _, base_response = self._inference_with_client(
                            messages=messages_base,
                            temperature=self.temperature,
                            max_new_tokens=self.max_new_tokens,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate base response: {e}")
                        continue

                    # Judge responses
                    try:
                        pos_score = self.judge(
                            question=question, answer=pos_response, role=role
                        )
                        base_score = self.judge(
                            question=question, answer=base_response, role=role
                        )
                    except Exception as e:
                        logger.warning(f"Failed to judge responses: {e}")
                        continue

                    # Check if pair is valid (pos_score == 3, base_score != 0)
                    if pos_score != 3:
                        logger.debug(
                            f"Rejected positive response (score {pos_score}, expected 3)"
                        )
                        continue

                    if base_score == 0:
                        logger.debug(f"Rejected base response (score {base_score}, expected non-zero)")
                        continue

                    # Save valid pair
                    passed_pairs.add((pos_prompt, question))
                    qa_data["positive"].append(
                        {
                            "pos_prompt": pos_prompt,
                            "question": question,
                            "response": pos_response,
                            "score": pos_score,
                        }
                    )
                    qa_data["base"].append(
                        {
                            "question": question,
                            "response": base_response,
                            "score": base_score,
                        }
                    )
                    n += 1
                    made_progress = True
                    logger.info(
                        f"✓ Accepted pair (pos_score={pos_score}, base_score={base_score})"
                    )

            if n >= target_pairs:
                break

            if not made_progress:
                logger.info(
                    f"No progress made in epoch {retry_epoch}, stopping retries"
                )
                break

        logger.info(f"Generated {n}/{target_pairs} valid pairs for role {role}")
        return qa_data

    def load_qa_responses(
        self,
        role: str,
        output_dir: str = "./persona_data/model_qa_responses",
    ) -> Optional[Dict[str, List[Dict]]]:
        """
        Load existing Q/A responses from JSON, or return None if not found.
        """
        safe_model_id = self.model.replace("/", "__")
        output_path = Path(output_dir) / safe_model_id / f"{role}.json"
        if not output_path.exists():
            return None
        with open(output_path, encoding="utf-8") as f:
            return json.load(f)

    def save_qa_responses(
        self,
        qa_data: Dict[str, List[Dict]],
        role: str,
        output_dir: str = "./persona_data/model_qa_responses",
    ) -> str:
        """
        Save Q/A responses to JSON.

        Path format: {output_dir}/{safe_model_id}/{role}.json

        Args:
            qa_data: Dictionary with "positive" and "base" Q/A pairs
            role: Role name
            output_dir: Output directory path

        Returns:
            Path to saved JSON file
        """
        # Sanitize model ID for directory name
        safe_model_id = self.model.replace("/", "__")

        # Create directory
        role_dir = Path(output_dir) / safe_model_id
        role_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        output_path = role_dir / f"{role}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2)

        logger.info(f"Saved Q/A responses to {output_path}")
        return str(output_path)


def main():
    """Command-line interface for generating and judging Q/A responses."""
    ap = argparse.ArgumentParser(
        description="Generate and judge Q/A responses for roles (CPU task, uses ResponseGeneration backend)"
    )
    ap.add_argument(
        "--answer_model",
        type=str,
        default="allenai/Olmo-3-7B-Instruct",
        help="Model ID for generating answers (default: Olmo-3-7B-Instruct)",
    )
    ap.add_argument(
        "-r",
        "--roles",
        nargs="+",
        type=str,
        default=["Lawyers"],
        help="Roles to process",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "vllm", "hf_local"],
        help="Backend for inference (default: openai for OpenRouter API)",
    )
    ap.add_argument(
        "--base_url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for API backends (default: OpenRouter)",
    )
    ap.add_argument(
        "--api_key_env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name for API key (default: OPENROUTER_API_KEY)",
    )
    ap.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./persona_data/model_qa_responses",
        help="Output directory for Q/A responses",
    )
    ap.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        default="./persona_data/role_datasets/",
        help="Role dataset directory",
    )
    ap.add_argument(
        "-n",
        "--target_pairs",
        type=int,
        default=60,
        help="Target number of valid pairs per role (default: 60)",
    )
    ap.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=2000,
        help="Max tokens to generate (default: 2000)",
    )
    args = ap.parse_args()

    # Validate API key is available
    if args.backend in ("openai", "vllm"):
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            logger.error(
                f"API key not found! Set {args.api_key_env} environment variable."
            )
            return

    # Initialize generator
    generator = RoleQAGenerator(
        answer_model=args.answer_model,
        backend=args.backend,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Process each role
    for role in args.roles:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing role: {role}")
        logger.info(f"{'='*60}")

        # Try to load dataset
        try:
            dataset = RoleDataset.from_json(role, dirpath=args.dataset_dir)
        except Exception as e:
            logger.error(f"Failed to load dataset for role {role}: {e}")
            logger.info("Skipping this role")
            continue

        # Check for existing Q/A data
        existing_data = generator.load_qa_responses(role=role, output_dir=args.output_dir)
        if existing_data:
            existing_count = len(existing_data.get("positive", []))
            if existing_count >= args.target_pairs:
                logger.info(
                    f"Skipping role {role}: already has {existing_count}/{args.target_pairs} pairs"
                )
                continue
            logger.info(
                f"Found {existing_count}/{args.target_pairs} existing pairs for role {role}, resuming"
            )
        else:
            existing_count = 0

        # Generate and judge Q/A
        try:
            qa_data = generator.generate_and_judge_qa(
                role=role,
                dataset=dataset,
                target_pairs=args.target_pairs,
                existing_data=existing_data,
            )

            # Save results
            generator.save_qa_responses(
                qa_data=qa_data, role=role, output_dir=args.output_dir
            )
            logger.info(f"✓ Successfully processed role: {role}")

        except Exception as e:
            logger.error(f"Failed to generate Q/A for role {role}: {e}")
            continue

    logger.info(f"\n{'='*60}")
    logger.info("All roles processed!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
