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
        
        self.dataset = (self.dataset if hasattr(self, dataset) else None) or \
                       (dataset if dataset else None) or \
                       PersonaDataset.from_json(trait, dirpath=dataset_dirpath)
        self.trait = self.dataset.trait
        
        super().__init__(concept=trait,*args, **kwargs)
        
        
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
        # Tensor of average activation of response
        # Tensor of average activation of response across all layers
        sum_prompt_last_pos, sum_resp_avg_pos, sum_resp_avg_all_layers_pos = self.initialize_activations()
        sum_prompt_last_neg, sum_resp_avg_neg, sum_resp_avg_all_layers_neg = self.initialize_activations()
        
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

                    sum_prompt_last_pos, sum_resp_avg_pos, sum_resp_avg_all_layers_pos = self.aggregate_activations(
                        sum_prompt_last_pos, 
                        sum_resp_avg_pos, 
                        sum_resp_avg_all_layers_pos, 
                        pl_pos, 
                        ra_pos, 
                        rall_pos
                    )
                    
                    sum_prompt_last_neg, sum_resp_avg_neg, sum_resp_avg_all_layers_neg = self.aggregate_activations(
                        sum_prompt_last_neg, 
                        sum_resp_avg_neg, 
                        sum_resp_avg_all_layers_neg, 
                        pl_neg, 
                        ra_neg, 
                        rall_neg
                    )
                    
                    n += 1

                    if n >= self.target_pairs:
                        break

                retries += 1
                if not made_progress:
                    break
        
        prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector = self.compute_contrastive_persona_vectors(
            sum_prompt_pos=sum_prompt_last_pos,
            sum_prompt_neg=sum_prompt_last_neg,
            sum_resp_pos=sum_resp_avg_pos,
            sum_resp_neg=sum_resp_avg_neg,
            sum_all_layers_pos=sum_resp_avg_all_layers_pos,
            sum_all_layers_neg=sum_resp_avg_all_layers_neg,
            n=n
        )

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
        concept=args.trait,
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
