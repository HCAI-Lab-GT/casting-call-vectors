"""
GPU task for roles optimization: Extract activations from pre-generated Q/A.

This module handles GPU-intensive activation extraction from pre-generated
Q/A responses saved by the CPU task. It loads responses, randomly shuffles them,
and extracts all activations in one efficient pass, building on previous
sample counts instead of restarting from scratch.

Architecture:
- Answer Model: Generated via OpenRouter in CPU phase (already done)
- Activation Extraction Model: Loaded on GPU in this phase (default: Olmo-3-7B-Instruct)
- Both default to same model but can differ

This separation optimizes GPU usage:
- CPU jobs: Generate answers via OpenRouter (no GPU), judge responses
- GPU jobs: Load extraction model, extract activations from pre-judged Q/A
- GPU is only used for activation extraction
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as transformers_logging

from pvx import setup_logging
logger = setup_logging(name="roles-optimized-gpu")

from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.roles.role_dataset import RoleDataset

torch.set_float32_matmul_precision("high")
transformers_logging.set_verbosity_error()

class RoleActivationExtractor(AbstractPersonaModel):
    """
    GPU task: Extract activations from pre-generated Q/A responses.

    Uses a separate extraction model (default: Olmo-3-7B-Instruct) which may
    differ from the answer model used in the CPU phase. Only loads the model
    for activation extraction, not for generation.
    """

    def __init__(
        self,
        role: str,
        activation_extraction_model_id: str = "allenai/Olmo-3-7B-Instruct",
        layer: int = 16,
        dataset_dirpath: str = "./persona_data/role_datasets/",
        qa_responses_path: Optional[str] = None,
        answer_model_id: Optional[str] = None,
        safetensors_dir: str = "./persona_data/model_inits/",
        **kwargs,
    ):
        """
        Initialize the activation extractor.

        Args:
            role: Role name
            activation_extraction_model_id: Model to extract activations from
                (default: allenai/Olmo-3-7B-Instruct, loaded on GPU)
            layer: Layer to extract activations from
            dataset_dirpath: Where role datasets are stored
            qa_responses_path: Path to pre-generated Q/A responses JSON
            answer_model_id: Model used for generating answers (for finding Q/A path)
            safetensors_dir: Where to save persona vectors
        """
        self.role = role
        self.answer_model_id = answer_model_id

        # Load role dataset (for evaluation prompt)
        try:
            self.dataset = RoleDataset.from_json(role, dirpath=dataset_dirpath)
        except Exception as e:
            logger.error(f"Failed to load dataset for role {role}: {e}")
            self.dataset = None

        # Initialize activation extraction model (loads on GPU)
        logger.info(f"Loading activation extraction model on GPU: {activation_extraction_model_id}")
        self._init_base(
            target_model_id=activation_extraction_model_id,
            layer=layer,
            default_alpha=kwargs.get("default_alpha", 3.0),
        )

        self.concept = role
        self.safetensors_dir = safetensors_dir
        self._qa_responses_path = qa_responses_path
        self.target_pairs = len(self._load_qa_responses(qa_responses_path, answer_model_id))

    def _load_qa_responses(
        self,
        qa_responses_path: Optional[str],
        answer_model_id: Optional[str] = None,
    ) -> List[Dict]:
        """Load Q/A responses from JSON file."""
        if qa_responses_path is None:
            # Try to find Q/A responses using answer_model_id or target_model_id
            model_id = answer_model_id or self.target_model_id
            safe_model_id = model_id.replace("/", "__")
            qa_responses_path = (
                f"./persona_data/model_qa_responses/{safe_model_id}/{self.role}.json"
            )

        qa_path = Path(qa_responses_path)
        if not qa_path.exists():
            raise FileNotFoundError(f"Q/A responses not found at {qa_responses_path}")

        with open(qa_path, "r") as f:
            qa_data = json.load(f)

        logger.info(f"Loaded {len(qa_data.get('positive', []))} positive responses")
        logger.info(f"Loaded {len(qa_data.get('base', []))} base responses")

        return qa_data

    def extract_persona_vector(self, **kwargs) -> Tuple:
        """
        Extract persona vectors from pre-generated Q/A responses.

        Loads Q/A pairs, randomly shuffles them, and extracts activations
        in one efficient GPU pass.
        """
        # Load Q/A responses
        qa_data = self._load_qa_responses(self._qa_responses_path, self.answer_model_id)

        positive_pairs = qa_data.get("positive", [])
        base_pairs = qa_data.get("base", [])

        if not positive_pairs or not base_pairs:
            raise ValueError("No valid Q/A pairs loaded")

        # Pre-tokenize all prompts
        logger.info(f"Pre-tokenizing {len(positive_pairs) + len(base_pairs)} responses")
        token_cache = {}

        for pair in positive_pairs:
            messages = [
                {"role": "system", "content": pair["pos_prompt"]},
                {"role": "user", "content": pair["question"]},
            ]
            enc = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
            token_cache[(pair["pos_prompt"], pair["question"])] = (
                enc,
                torch.ones_like(enc),
            )

        base_token_cache = {}
        for pair in base_pairs:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": pair["question"]},
            ]
            enc = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
            base_token_cache[pair["question"]] = (enc, torch.ones_like(enc))

        # Initialize activation accumulators
        (sum_prompt_pos,    sum_resp_pos,   sum_all_layers_pos)  = self.initialize_activations()
        (sum_prompt_base,   sum_resp_base,  sum_all_layers_base) = self.initialize_activations()

        # Randomly shuffle pairs
        shuffled_pairs = positive_pairs.copy()
        random.shuffle(shuffled_pairs)

        # Extract activations for all pairs in one pass
        n = 0
        with tqdm(
            total=len(shuffled_pairs),
            desc=f"Extracting activations for role {self.role}",
        ) as pbar:
            for pair in shuffled_pairs:
                pos_ids, pos_mask = token_cache[
                    (pair["pos_prompt"], pair["question"])
                ]
                base_ids, base_mask = base_token_cache[pair["question"]]

                try:
                    # Extract activations for positive
                    pl_pos, ra_pos, rall_pos, _ = self._get_activations(
                        pos_ids, pos_mask
                    )

                    # Extract activations for base
                    pl_base, ra_base, rall_base, _ = self._get_activations(
                        base_ids, base_mask
                    )

                    # Aggregate
                    sum_prompt_pos, sum_resp_pos, sum_all_layers_pos = (
                        self.aggregate_activations(
                            sum_prompt_pos,
                            sum_resp_pos,
                            sum_all_layers_pos,
                            pl_pos,
                            ra_pos,
                            rall_pos,
                        )
                    )

                    sum_prompt_base, sum_resp_base, sum_all_layers_base = (
                        self.aggregate_activations(
                            sum_prompt_base,
                            sum_resp_base,
                            sum_all_layers_base,
                            pl_base,
                            ra_base,
                            rall_base,
                        )
                    )

                    n += 1
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Failed to extract activations: {e}")
                    continue

        # Compute contrastive persona vectors
        (
            prompt_persona_vector,
            response_persona_vector,
            all_layers_response_persona_vector,
        ) = self.compute_contrastive_persona_vectors(
            sum_prompt_pos=sum_prompt_pos,
            sum_prompt_neg=sum_prompt_base,
            sum_resp_pos=sum_resp_pos,
            sum_resp_neg=sum_resp_base,
            sum_all_layers_pos=sum_all_layers_pos,
            sum_all_layers_neg=sum_all_layers_base,
            n=n,
        )

        logger.info(f"Extracted persona vectors from {n} pairs")
        logger.info(f"Prompt persona vector shape: {tuple(prompt_persona_vector.shape)}")
        logger.info(
            f"Response persona vector shape: {tuple(response_persona_vector.shape)}"
        )

        return prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector


def main():
    ap = argparse.ArgumentParser(
        description="Extract activations from pre-generated Q/A responses (GPU task)"
    )
    ap.add_argument(
        "--activation_extraction_model",
        type=str,
        default="allenai/Olmo-3-7B-Instruct",
        help="Model to extract activations from (loaded on GPU, default: Olmo-3-7B-Instruct)",
    )
    ap.add_argument(
        "--answer_model",
        type=str,
        default=None,
        help="Model used for generating answers (for finding Q/A path, default: same as extraction model)",
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
        "-l",
        "--layer",
        type=int,
        default=16,
        help="Layer to extract activations from",
    )
    ap.add_argument(
        "-q",
        "--qa_responses_dir",
        type=str,
        default="./persona_data/model_qa_responses",
        help="Directory containing pre-generated Q/A responses",
    )
    ap.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        default="./persona_data/role_datasets/",
        help="Role dataset directory",
    )
    ap.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./persona_data/model_inits/",
        help="Output directory for persona vectors",
    )
    args = ap.parse_args()

    # Default answer_model to extraction_model if not provided
    answer_model = args.answer_model or args.activation_extraction_model

    for role in args.roles:
        logger.info(f"\n=== Processing role: {role} ===")

        # Try to load role dataset first (to check if role exists)
        try:
            RoleDataset.from_json(role, dirpath=args.dataset_dir)
        except Exception as e:
            logger.error(f"Role dataset not found for {role}: {e}")
            logger.info("Skipping this role")
            continue

        # Check if Q/A responses exist
        safe_answer_model = answer_model.replace("/", "__")
        qa_path = Path(args.qa_responses_dir) / safe_answer_model / f"{role}.json"
        if not qa_path.exists():
            logger.error(f"Q/A responses not found at {qa_path}")
            logger.info("Please run the CPU task first to generate Q/A responses")
            logger.info("Skipping this role")
            continue

        try:
            # Create extractor
            extractor = RoleActivationExtractor(
                role=role,
                activation_extraction_model_id=args.activation_extraction_model,
                layer=args.layer,
                dataset_dirpath=args.dataset_dir,
                qa_responses_path=str(qa_path),
                answer_model_id=answer_model,
                safetensors_dir=args.output_dir,
            )

            # Extract vectors
            extractor.extract_persona_vector()

            # Save
            extractor.save_to_safetensors(filepath=args.output_dir)

            logger.info(f"Successfully extracted and saved persona vectors for {role}")
        except Exception as e:
            logger.error(f"Failed to extract persona vectors for role {role}: {e}")
            continue


if __name__ == "__main__":
    main()
