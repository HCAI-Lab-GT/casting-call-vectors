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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import random

import torch
from safetensors.torch import save_file
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

    def _checkpoint_path(self, output_dir: str, count: int) -> Path:
        """Return the safetensors path for a given sample count."""
        safe_model = self.target_model_id.replace("/", "__")
        return (
            Path(output_dir)
            / f"{self.role}_persona_initialization"
            / f"{safe_model}_layer{self.layer_steering}_count{count}.safetensors"
        )

    def _save_checkpoint(
        self,
        prompt_pv: torch.Tensor,
        response_pv: torch.Tensor,
        all_layers_pv: torch.Tensor,
        count: int,
        output_dir: str,
    ) -> str:
        """Save persona vectors for a specific sample count checkpoint."""
        out_path = self._checkpoint_path(output_dir, count)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tensors = {
            "prompt_persona_vector": prompt_pv.contiguous().cpu(),
            "response_persona_vector": response_pv.contiguous().cpu(),
            "all_layers_response_persona_vector": all_layers_pv.contiguous().cpu(),
        }
        metadata = {
            "target_model_id": self.target_model_id,
            "concept": self.role,
            "layer_steering": str(self.layer_steering),
            "sample_count": str(count),
            "created_at": datetime.now().isoformat(),
            "prompt_persona_vector_shape": str(list(prompt_pv.shape)),
            "response_persona_vector_shape": str(list(response_pv.shape)),
            "all_layers_shape": str(list(all_layers_pv.shape)),
        }
        save_file(tensors, str(out_path), metadata=metadata)
        logger.info(f"Saved checkpoint count={count} → {out_path}")
        return str(out_path)

    def extract_and_save_incremental(
        self,
        counts: List[int],
        output_dir: str,
    ) -> List[int]:
        """
        Extract persona vectors for multiple sample counts in one GPU pass.

        Pairs are processed in a deterministic fixed order (seed=42 shuffle).
        At each count checkpoint the accumulated activations are used to compute
        and immediately save a persona vector, so larger counts build on the
        work already done for smaller ones rather than starting from scratch.

        Args:
            counts: Sample counts to checkpoint (e.g. [20, 40, 60]).
                    Sorted ascending internally.
            output_dir: Directory to write per-count safetensors files.

        Returns:
            List of counts for which a safetensors file was successfully saved.
        """
        counts = sorted(counts)
        max_count = counts[-1]

        qa_data = self._load_qa_responses(self._qa_responses_path, self.answer_model_id)
        positive_pairs = qa_data.get("positive", [])
        base_pairs = qa_data.get("base", [])

        if not positive_pairs or not base_pairs:
            raise ValueError("No valid Q/A pairs loaded")

        # Deterministic shuffle so count=40 includes the same first 20 pairs as count=20
        ordered = positive_pairs.copy()
        random.Random(42).shuffle(ordered)
        ordered = ordered[:max_count]

        # Build base lookup keyed by question
        base_lookup = {p["question"]: p for p in base_pairs}

        # Pre-tokenize all pairs we will process
        logger.info(f"Pre-tokenizing {len(ordered)} pairs (up to count={max_count})")
        token_cache: Dict = {}
        base_token_cache: Dict = {}
        for pair in ordered:
            key = (pair["pos_prompt"], pair["question"])
            if key not in token_cache:
                msgs = [
                    {"role": "system", "content": pair["pos_prompt"]},
                    {"role": "user", "content": pair["question"]},
                ]
                enc = self.tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)
                token_cache[key] = (enc, torch.ones_like(enc))

            q = pair["question"]
            if q not in base_token_cache and q in base_lookup:
                msgs = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": q},
                ]
                enc = self.tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)
                base_token_cache[q] = (enc, torch.ones_like(enc))

        # Initialize running accumulators
        sum_prompt_pos,  sum_resp_pos,  sum_all_layers_pos  = self.initialize_activations()
        sum_prompt_base, sum_resp_base, sum_all_layers_base = self.initialize_activations()

        saved_counts: List[int] = []
        count_idx = 0
        n = 0

        with tqdm(total=max_count, desc=f"Extracting activations for {self.role}") as pbar:
            for pair in ordered:
                q = pair["question"]
                if q not in base_token_cache:
                    continue  # base response missing for this question, skip

                pos_ids, pos_mask = token_cache[(pair["pos_prompt"], q)]
                base_ids, base_mask = base_token_cache[q]

                try:
                    pl_pos,  ra_pos,  rall_pos,  _ = self._get_activations(pos_ids,  pos_mask)
                    pl_base, ra_base, rall_base, _ = self._get_activations(base_ids, base_mask)

                    sum_prompt_pos,  sum_resp_pos,  sum_all_layers_pos  = self.aggregate_activations(
                        sum_prompt_pos,  sum_resp_pos,  sum_all_layers_pos,
                        pl_pos,  ra_pos,  rall_pos,
                    )
                    sum_prompt_base, sum_resp_base, sum_all_layers_base = self.aggregate_activations(
                        sum_prompt_base, sum_resp_base, sum_all_layers_base,
                        pl_base, ra_base, rall_base,
                    )
                    n += 1
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to extract activations: {e}")
                    continue

                # Save checkpoint whenever we reach the next target count
                while count_idx < len(counts) and n >= counts[count_idx]:
                    target = counts[count_idx]
                    prompt_pv, response_pv, all_layers_pv = self.compute_contrastive_persona_vectors(
                        sum_prompt_pos=sum_prompt_pos,
                        sum_prompt_neg=sum_prompt_base,
                        sum_resp_pos=sum_resp_pos,
                        sum_resp_neg=sum_resp_base,
                        sum_all_layers_pos=sum_all_layers_pos,
                        sum_all_layers_neg=sum_all_layers_base,
                        n=n,
                    )
                    self._save_checkpoint(prompt_pv, response_pv, all_layers_pv, target, output_dir)
                    saved_counts.append(target)
                    count_idx += 1

        # If we ran out of pairs before reaching all checkpoints, save remaining with current n
        while count_idx < len(counts):
            raise Exception("Ran out of pairs.")
        
            target = counts[count_idx]
            if n > 0:
                logger.warning(
                    f"Only {n} pairs processed; saving count={target} checkpoint with n={n}"
                )
                prompt_pv, response_pv, all_layers_pv = self.compute_contrastive_persona_vectors(
                    sum_prompt_pos=sum_prompt_pos,
                    sum_prompt_neg=sum_prompt_base,
                    sum_resp_pos=sum_resp_pos,
                    sum_resp_neg=sum_resp_base,
                    sum_all_layers_pos=sum_all_layers_pos,
                    sum_all_layers_neg=sum_all_layers_base,
                    n=n,
                )
                self._save_checkpoint(prompt_pv, response_pv, all_layers_pv, target, output_dir)
                saved_counts.append(target)
            else:
                logger.error(f"No pairs processed; skipping count={target} checkpoint")
            count_idx += 1

        logger.info(f"Incremental extraction complete: saved counts {saved_counts} from {n} pairs")
        return saved_counts


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
        "-c",
        "--counts",
        nargs="+",
        type=int,
        required=True,
        help="Sample counts to checkpoint (e.g. 20 40 60). One safetensors file is saved per count.",
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
        default="./persona_data/model_layer_inits/",
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
            extractor = RoleActivationExtractor(
                role=role,
                activation_extraction_model_id=args.activation_extraction_model,
                layer=args.layer,
                dataset_dirpath=args.dataset_dir,
                qa_responses_path=str(qa_path),
                answer_model_id=answer_model,
                safetensors_dir=args.output_dir,
            )

            saved = extractor.extract_and_save_incremental(
                counts=args.counts,
                output_dir=args.output_dir,
            )
            logger.info(f"Successfully saved persona vectors for {role} at counts {saved}")
        except Exception as e:
            logger.error(f"Failed to extract persona vectors for role {role}: {e}")
            continue


if __name__ == "__main__":
    main()
