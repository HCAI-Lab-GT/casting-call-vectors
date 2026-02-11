import argparse
import contextvars
import random

import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from transformers.utils import logging as transformers_logging

from pvx import setup_logging
from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.base.persona_dataset import PersonaDataset

torch.set_float32_matmul_precision("high")

# Disable transformers progress bars to avoid cluttering output
transformers_logging.set_verbosity_error()

# request-local steering state (per concurrent generate call)
_STEER_DELTA = contextvars.ContextVar("steer_delta", default=None)  # Tensor (1,H) or None

logger = setup_logging(name="persona-model")


class PersonaModel(AbstractPersonaModel):
    def __init__(self, 
                 trait: str = "",
                 dataset: Optional[PersonaDataset] = None,
                 dataset_dirpath: str = './persona_data/trait_datasets/',
                 *args, 
                 **kwargs):
        self.dataset = dataset if dataset else PersonaDataset.from_json(trait, dirpath=dataset_dirpath)
        self.trait = self.dataset.trait
        
        super().__init__(*args, **kwargs)
        
    
    @classmethod
    def from_json(
        cls,
        json_filepath: str,
        trait: Optional[str] = None,
    ) -> "PersonaModel":
        """
        Load a PersonaModel instance from a previously saved JSON file.

        Args:
            json_filepath (str): Path to the JSON file containing the saved initialization data

        Returns:
            PersonaModel: A new instance with the loaded persona vectors
        """

        with open(json_filepath, "r") as f:
            data = json.load(f)

        logger.info("Loading PersonaModel from: %s", json_filepath)

        # Create instance without extracting vectors
        instance = cls(
            trait=trait,
            target_model_id=data["target_model_id"],
            dataset=None,  # Dataset not needed when loading from JSON
            layer=data["layer_steering"],
            from_json=True,
        )

        # Load the persona vectors directly
        instance.prompt_persona_vector = torch.tensor(data["prompt_persona_vector"])
        instance.response_persona_vector = torch.tensor(data["response_persona_vector"])

        # Store additional metadata
        if "dataset_info" in data and data["dataset_info"]:
            instance.trait = data["dataset_info"]["trait"]

        logger.info("✅ Loaded PersonaModel from: %s", json_filepath)
        logger.info("   Model: %s", instance.target_model_id)
        logger.info("   Layer: %d", instance.layer_steering)
        logger.info("   Trait: %s", instance.trait if hasattr(instance, "trait") else None)
        logger.info(
            "   Prompt persona vector shape: %s", str(tuple(instance.prompt_persona_vector.shape))
        )
        logger.info(
            "   Response persona vector shape: %s",
            str(tuple(instance.response_persona_vector.shape)),
        )

        return instance
    
    @classmethod
    def load_or_create(
        cls,
        target_model_id: str = "qwen2.5:7b-instruct",
        dataset: Optional[PersonaDataset] = None,
        trait: Optional[str] = None,  # alternate to dataset for loading
        layer: float = 14,
        json_filepath: Optional[str] = None,
        safetensors_dir: str = "./persona_data/model_inits/",
    ) -> "PersonaModel":
        """
        Load a PersonaModel instance from saved files if they exist, otherwise create a new one.

        Priority order:
        1. Safetensors file (preferred - smaller, faster)
        2. Legacy JSON file (backward compatibility)
        3. Create new instance

        Args:
            target_model_id: Model identifier
            dataset: PersonaDataset for extraction (only used if creating new)
            trait: Trait name for loading
            layer: Layer for steering
            json_filepath: Legacy JSON path (optional, for backward compatibility)
            safetensors_dir: Directory containing safetensors files

        Returns:
            PersonaModel: Loaded or newly created instance
        """
        # Build safetensors path
        safe_model_id = target_model_id.replace("/", "__")
        safetensors_path = Path(safetensors_dir) / f"{trait}_persona_initialization/{safe_model_id}.safetensors"

        # Try safetensors first (preferred format)
        if safetensors_path.exists():
            try:
                return cls.from_safetensors(str(safetensors_path), trait=trait)
            except Exception as e:
                logger.warning("⚠️ Failed to load from safetensors: %s", e)

        # Fall back to legacy JSON
        json_filepath = (
            json_filepath
            or f"./persona_data/model_inits/{trait}_persona_initialization/{target_model_id}.json"
        )

        try:
            if Path(json_filepath).exists():
                logger.info("Loading from legacy JSON (consider migrating to safetensors)")
                return cls.from_json(json_filepath, trait=trait)

        except Exception as e:
            logger.warning(
                "⚠️ Failed to load from JSON: %s. Creating a new PersonaModel instance.", e
            )

        return cls(
            target_model_id=target_model_id,
            trait=trait,
            dataset=dataset,
            layer=layer,
        )
        
    @torch.inference_mode()
    def extract_persona_vector(
        self, temperature: float = 0.9, max_new_tokens: int = 200
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract persona vectors from the dataset. Main function.

        Args:
            temperature: Sampling temperature for generation.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (prompt_persona_vector, response_persona_vector, response_average_all_layers)
        """

        # Get cartesian product of all positive-negative question pairs from dataset
        trait_pairs = self.dataset.extract_pos_neg_question_pairs()

        # Optimization #4: Pre-tokenize and cache all prompts
        logger.info("Pre-tokenizing %d pairs...", len(trait_pairs))

        token_cache = {}
        for pos, neg, question in trait_pairs:
            # for system_prompt, key_suffix in [(pair.pos, 'pos'), (pair.neg, 'neg')]:
            for system_prompt in (pos, neg):
                # Prepare messages in chat format
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]

                enc = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)

                token_cache[(question, system_prompt)] = (enc, torch.ones_like(enc))

        # Process each pair and accumulate activations (Tensor instead of list for speed)
        # Tensor of last activation of prompt
        sum_prompt_last_pos = None
        sum_prompt_last_neg = None

        # Tensor of average activation of response
        sum_resp_avg_pos = None
        sum_resp_avg_neg = None

        # Tensor of average activation of response across all layers
        sum_resp_avg_all_layers_pos = None
        sum_resp_avg_all_layers_neg = None

        # track number of pairs
        n = 0
        passed_pairs = set()
        retries = 0
        max_retries = 2

        with tqdm(
            total=self.target_pairs,
            desc=f"Extracting activations for trait {self.dataset.trait}",
        ) as pbar:
            while n < self.target_pairs and retries < max_retries:
                remaining = [p for p in trait_pairs if p not in passed_pairs]
                epoch_pairs = random.sample(remaining, len(remaining))
                made_progress = False

                for pos, neg, question in epoch_pairs:
                    pos_ids, pos_mask = token_cache[(question, pos)]
                    neg_ids, neg_mask = token_cache[(question, neg)]

                    # pl: prompt hidden layer last activation
                    # ra: response hidden layer avg activation
                    # rall: response avg all activation
                    pl_pos, ra_pos, rall_pos, pos_response = self._get_activations(
                        pos_ids, pos_mask, temperature, max_new_tokens
                    )  # positive
                    pl_neg, ra_neg, rall_neg, neg_response = self._get_activations(
                        neg_ids, neg_mask, temperature, max_new_tokens
                    )  # negative

                    pos_score = self.judge(question=question, answer=pos_response)
                    neg_score = self.judge(question=question, answer=neg_response)

                    if pos_score < self.judge_threshold or neg_score >= self.judge_threshold:
                        logger.info("===FAILED CASE===")
                        logger.info(
                            question + " Pos: " + str(pos_score) + " Neg: " + str(neg_score)
                        )
                        continue

                    logger.info("===PASSED CASE===")
                    logger.info(question + " Pos: " + str(pos_score) + " Neg: " + str(neg_score))
                    passed_pairs.add((pos, neg, question))
                    pbar.update(1)
                    made_progress = True

                    if sum_prompt_last_pos is None:
                        sum_prompt_last_pos = torch.zeros_like(pl_pos, dtype=torch.float32)
                        sum_prompt_last_neg = torch.zeros_like(pl_neg, dtype=torch.float32)
                        sum_resp_avg_pos = torch.zeros_like(ra_pos, dtype=torch.float32)
                        sum_resp_avg_neg = torch.zeros_like(ra_neg, dtype=torch.float32)
                        sum_resp_avg_all_layers_pos = torch.zeros_like(
                            rall_pos, dtype=torch.float32
                        )
                        sum_resp_avg_all_layers_neg = torch.zeros_like(
                            rall_neg, dtype=torch.float32
                        )

                    sum_prompt_last_pos += pl_pos.float()
                    sum_prompt_last_neg += pl_neg.float()
                    sum_resp_avg_pos += ra_pos.float()
                    sum_resp_avg_neg += ra_neg.float()
                    sum_resp_avg_all_layers_pos += rall_pos.float()
                    sum_resp_avg_all_layers_neg += rall_neg.float()
                    n += 1

                    if n >= self.target_pairs:
                        break

                retries += 1
                if not made_progress:
                    break

        prompt_last_pos_mean = sum_prompt_last_pos / max(n, 1)
        prompt_last_neg_mean = sum_prompt_last_neg / max(n, 1)
        response_avg_pos_mean = sum_resp_avg_pos / max(n, 1)
        response_avg_neg_mean = sum_resp_avg_neg / max(n, 1)

        all_layers_response_avg_pos_mean = sum_resp_avg_all_layers_pos / max(n, 1)
        all_layers_response_avg_neg_mean = sum_resp_avg_all_layers_neg / max(n, 1)

        prompt_persona_vector = prompt_last_pos_mean - prompt_last_neg_mean  # (1, hidden)
        response_persona_vector = response_avg_pos_mean - response_avg_neg_mean  # (1, hidden)
        all_layers_response_persona_vector = (
            all_layers_response_avg_pos_mean - all_layers_response_avg_neg_mean
        )
        # (num_states, 1, hidden)

        self.prompt_persona_vector = prompt_persona_vector.cpu()
        self.response_persona_vector = response_persona_vector.cpu()
        self.all_layers_response_persona_vector = all_layers_response_persona_vector.cpu()

        logger.info("Extracted persona vectors from %d pairs", len(trait_pairs))
        logger.info("Prompt persona vector shape: %s", str(tuple(prompt_persona_vector.shape)))
        logger.info("Response persona vector shape: %s", str(tuple(response_persona_vector.shape)))
        logger.info(
            "All-layers response persona vector shape: %s",
            str(tuple(all_layers_response_persona_vector.shape)),
        )
        if n < self.target_pairs:
            logger.warning("Only %d pairs passed after %d retries.", len(passed_pairs), max_retries)

        return prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model typically",
    )
    ap.add_argument(
        "-t", "--trait", type=str, default="humorous", help="Trait of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="What is the theory of relativity?",
        help="Question to generate response for",
    )
    ap.add_argument(
        "-f",
        "--json_filepath",
        type=str,
        default=None,
        help="Filepath to model init file (not trait dataset)",
    )
    args = ap.parse_args()

    pvx = PersonaModel.load_or_create(
        target_model_id=args.model_name,
        trait=args.trait,
        layer=14,
        json_filepath=args.json_filepath,
    )

    response = pvx.generate(
        prompt=args.question,
        alpha=0,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    steer_response = pvx.generate(
        prompt=args.question,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    logger.info("=== Question ===")
    logger.info(args.question)
    logger.info("")
    logger.info("=== Non-Steered Answer ===")
    logger.info(response)
    logger.info("")
    logger.info("=== %s Steered Answer ===", args.trait.upper())
    logger.info(steer_response)
