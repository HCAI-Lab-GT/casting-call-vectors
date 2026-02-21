import argparse
import contextvars
import json
from pathlib import Path
import random
from typing import Optional

from pvx.implementations.roles.role_dataset import RoleDataset
from datetime import datetime
import torch
from tqdm import tqdm
from transformers.utils import logging as transformers_logging
from safetensors.torch import save_file

from pvx import setup_logging
from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.judges.role_judge import RoleJudge

torch.set_float32_matmul_precision("high")

# Disable transformers progress bars to avoid cluttering output
transformers_logging.set_verbosity_error()

# request-local steering state (per concurrent generate call)
_STEER_DELTA = contextvars.ContextVar("steer_delta", default=None)  # Tensor (1,H) or None

logger = setup_logging(name="role-persona-model")


class RolePersonaModel(AbstractPersonaModel):
    
    def __init__(self,
                 role: str, 
                 dataset: Optional[RoleDataset] = None,
                 dataset_dirpath: str = './persona_data/role_datasets/', 
                 *args, 
                 **kwargs):
        kwargs.setdefault("judge_cls", RoleJudge)
        self.role = role
        
        from_json = kwargs.get("from_json", False)
        if from_json:
            super().__init__(concept=role,*args, **kwargs)
            return
            
        self.dataset = dataset if dataset else RoleDataset.from_json(role, dirpath=dataset_dirpath)
        
        super().__init__(concept=role,*args, **kwargs)
        
    
    
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
        prompt_question_pairs = self.dataset.extract_pos_question_pairs()

        # Optimization #4: Pre-tokenize and cache all prompts
        logger.info("Pre-tokenizing %d pairs...", len(prompt_question_pairs))

        token_cache = {}
        
        for question in self.dataset.questions:
            messages_base = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": question},
                ]
            enc_base = self.tokenizer.apply_chat_template(
                messages_base, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            token_cache[(question, "")] = (enc_base, torch.ones_like(enc_base))
        
        for pos, question in prompt_question_pairs:
            # for system_prompt, key_suffix in [(pair.pos, 'pos'), (pair.neg, 'neg')]:
            # Prepare messages in chat format
            messages_pos = [
                {"role": "system", "content": pos},
                {"role": "user", "content": question},
            ]

            enc_pos = self.tokenizer.apply_chat_template(
                messages_pos, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            token_cache[(question, pos)] = (enc_pos, torch.ones_like(enc_pos))

        # Process each pair and accumulate activations (Tensor instead of list for speed)
        # Tensor of last activation of prompt
        # Tensor of average activation of response
        # Tensor of average activation of response across all layers
        sum_prompt_last_pos, sum_resp_avg_pos, sum_resp_avg_all_layers_pos = self.initialize_activations()
        sum_prompt_last_base, sum_resp_avg_base, sum_resp_avg_all_layers_base = self.initialize_activations()

        # track number of pairs
        n = 0
        passed_pairs = set()
        retries = 0
        max_retries = 2

        with tqdm(
            total=self.target_pairs,
            desc=f"Extracting activations for role {self.dataset.role}",
        ) as pbar:
            while n < self.target_pairs and retries < max_retries:
                remaining = [p for p in prompt_question_pairs if p not in passed_pairs]
                epoch_pairs = random.sample(remaining, len(remaining))
                made_progress = False

                for pos, question in epoch_pairs:
                    pos_ids, pos_mask = token_cache[(question, pos)]
                    base_ids, base_mask = token_cache[(question, "")]

                    # pl: prompt hidden layer last activation
                    # ra: response hidden layer avg activation
                    # rall: response avg all activation
                    pl_pos, ra_pos, rall_pos, pos_response = self._get_activations(
                        pos_ids, pos_mask, temperature, max_new_tokens
                    )
                    pl_base, ra_base, rall_base, base_response = self._get_activations(
                        base_ids, base_mask, temperature, max_new_tokens
                    )  

                    pos_score = self.judge(question=question, answer=pos_response)
                    base_score = self.judge(question=question, answer=base_response)

                    if pos_score != 3:
                        logger.info("===FAILED CASE POSITIVE===")
                        logger.info("Question: " + question + ", System Prompt: " + pos + ", Score: " + str(pos_score) + " Pos Response: " + pos_response)
                        continue
                    if base_score == 0:
                        logger.info("===FAILED CASE BASE===")
                        logger.info("Question: " + question + ", Score: " + str(base_score) + " Base Response: " + base_response)
                        continue

                    logger.info("===PASSED CASE===")
                    logger.info("Question: " + question + ", System Prompt: " + pos + ", Score: " + str(pos_score) + ", Pos Response: " + pos_response)
                    logger.info("Question: " + question + ", Score: " + str(base_score) + ", Base Response: " + base_response)
                    passed_pairs.add((pos, question))
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
                    
                    sum_prompt_last_base, sum_resp_avg_base, sum_resp_avg_all_layers_base = self.aggregate_activations(
                        sum_prompt_last_base, 
                        sum_resp_avg_base, 
                        sum_resp_avg_all_layers_base, 
                        pl_base, 
                        ra_base, 
                        rall_base
                    )
                    
                    n += 1

                    if n >= self.target_pairs:
                        break

                retries += 1
                if not made_progress:
                    break

        # prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector = self.finalize_activations(
        #     sum_prompt_last_pos, 
        #     sum_resp_avg_pos, 
        #     sum_resp_avg_all_layers_pos, 
        #     n
        # )
        
        # self.prompt_persona_vector = prompt_persona_vector.cpu()
        # self.response_persona_vector = response_persona_vector.cpu()
        # self.all_layers_response_persona_vector = all_layers_response_persona_vector.cpu()
        prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector = self.compute_contrastive_persona_vectors(
            sum_prompt_pos=sum_prompt_last_pos,
            sum_prompt_neg=sum_prompt_last_base,
            sum_resp_pos=sum_resp_avg_pos,
            sum_resp_neg=sum_resp_avg_base,
            sum_all_layers_pos=sum_resp_avg_all_layers_pos,
            sum_all_layers_neg=sum_resp_avg_all_layers_base,
            n=n
        )

        logger.info("Extracted persona vectors from %d pairs", len(prompt_question_pairs))
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
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-r", "--roles", nargs="+", type=str, default=["lawyer"], help="Roles of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Question to generate response for",
    )
    ap.add_argument(
        "-f",
        "--json_filepath",
        type=str,
        default=None,
        help="Filepath to model init file (not role dataset)",
    )
    args = ap.parse_args()

    for role in args.roles:
        logger.info("=== Processing role: %s ===", role)
        
        try:
            pvx = RolePersonaModel.load_or_create(
                target_model_id=args.model,
                concept=role,
                layer=14,
                json_filepath=args.json_filepath
            )
            logger.info("Successfully loaded/created RolePersonaModel for role '%s'", role)
        except Exception as e:
            logger.error("Failed to load or create RolePersonaModel for role '%s': %s", role, e)
            continue

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
        logger.info("=== %s Steered Answer ===", role.upper())
        logger.info(steer_response)
