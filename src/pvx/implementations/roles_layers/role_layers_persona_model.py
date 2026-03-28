import argparse
import json
import random
from pathlib import Path
from typing import Optional, override

import torch
from tqdm import tqdm
from pvx import setup_logging

from pvx.implementations.roles.role_persona_model import RolePersonaModel
from pvx.implementations.judges.llm_as_judge import LLMJudge, PROMPT_TEMPLATE

logger = setup_logging(name="role-layers-persona-model")

class RoleLayersPersonaModel(RolePersonaModel):
    
    @torch.inference_mode()
    def extract_persona_vector(
        self, temperature: float = 0.9, max_new_tokens: int = 2048
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract persona vectors from the dataset. Main function.

        Args:
            temperature: Sampling temperature for generation.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (prompt_persona_vector, response_persona_vector, response_average_all_layers)
        """

        # Keep checkpoint counts as instance state (set by load_or_create kwargs)
        if not hasattr(self, "sample_counts"):
            self.sample_counts = None

        # Get cartesian product of all positive-negative question pairs from dataset
        prompt_question_pairs = self.dataset.extract_pos_question_pairs()

        # If sample_counts provided, drive target_pairs from the maximum requested count
        sorted_counts = sorted(self.sample_counts) if self.sample_counts else None
        if sorted_counts:
            self.target_pairs = sorted_counts[-1]
        counts_to_save = set(sorted_counts) if sorted_counts else set()
        saved_counts = set()

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
                        logger.info("\nQuestion: \n" + question + "\n\nSystem Prompt: \n" + pos + "\nScore: " + str(pos_score) + "\n\nPos Response: \n" + pos_response)
                        continue
                    
                    if base_score == 0:
                        logger.info("===FAILED CASE BASE===")
                        logger.info("\nQuestion: \n" + question + "\n\nScore: " + str(base_score) + "\nBase Response: \n" + base_response)
                        continue

                    logger.info("===PASSED CASE===")
                    logger.info("\nQuestion: \n" + question + "\n\nSystem Prompt: \n" + pos + "\nScore: " + str(pos_score) + "\n\nPos Response: \n" + pos_response)
                    logger.info("\n\nScore: " + str(base_score) + "\nBase Response: \n" + base_response)
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

                    # Checkpoint: save persona vectors whenever n hits a requested sample count
                    if n in counts_to_save and n not in saved_counts:
                        logger.info("Checkpoint save at n=%d pairs", n)
                        self.compute_contrastive_persona_vectors(
                            sum_prompt_pos=sum_prompt_last_pos,
                            sum_prompt_neg=sum_prompt_last_base,
                            sum_resp_pos=sum_resp_avg_pos,
                            sum_resp_neg=sum_resp_avg_base,
                            sum_all_layers_pos=sum_resp_avg_all_layers_pos,
                            sum_all_layers_neg=sum_resp_avg_all_layers_base,
                            n=n,
                        )
                        original_target = self.target_pairs
                        self.target_pairs = n
                        self.save_to_safetensors(filepath=self.safetensors_dir)
                        self.target_pairs = original_target
                        saved_counts.add(n)
                        logger.info("Saved checkpoint for count=%d", n)

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
    
    @override
    @classmethod
    def get_path(cls, 
                 target_model_id: str, concept: str, 
                 safetensors_dir: str= "./persona_data/model_layer_inits/", use_json: bool=False, 
                 kwargs: Optional[dict]=None, obj: Optional[RolePersonaModel]=None):
        safe_model_id = target_model_id.replace("/", "__")
        
        layer = \
            (kwargs.get("layer") if kwargs else None) or \
            (obj.layer_steering if obj else None) or \
            "unknown_layer"
            
        sample_counts = \
            (kwargs.get("target_pairs") if kwargs else None) or \
            (obj.target_pairs if obj else None) or \
            "unknown_count"
                        
        filepath = f"{concept}_persona_initialization/{safe_model_id}_layer{layer}_count{sample_counts}.{'safetensors' if not use_json else 'json'}"
        safetensors_path = Path(safetensors_dir) / filepath
        
        return safetensors_path, filepath
    
    # @override
    # def save_to_safetensors(self, filepath: str = None, **args) -> str:
    #     filepath = filepath or self.safetensors_dir
    #     super().save_to_safetensors(filepath=filepath, **args)
        
    # @override
    # def save_to_json(self, filepath: str = None, **args) -> str:
    #     filepath = filepath or self.safetensors_dir
    #     super().save_to_json(filepath=filepath, **args)
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-r", "--roles", nargs="+", type=str, default=["Lawyers"], help="Roles of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument(
        "-l", "--layers", nargs="+", type=int,default=[16], help="List of layers to extract activations from (default: 16)"
    )
    ap.add_argument(
        "-N", "--sample_counts", nargs="+", type=int,default=[40], help="List of question counts to extract activations from (default: 40)"
    )
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Question to generate response for",
    )
    ap.add_argument(
        "-f",
        "--safetensor_dir",
        type=str,
        default="./persona_data/model_layer_inits/",
        help="Directory for the model init safetensor (not role dataset)",
    )
    ap.add_argument(
        "--cot",
        action="store_true",
        help="Whether to use chain-of-thought prompting (default: False)",
    )
    ap.add_argument(
        "-t",
        "--trait",
        type=str,
        default=None,
        help="Trait to use for persona generation",
    )
    args = ap.parse_args()
    
    role_judge = LLMJudge(
        backend="openai",
        model="gpt-4.1-mini",
        prompt_template=PROMPT_TEMPLATE,
    )

    for role in args.roles:
        logger.info("=== Processing role: %s ===", role)
        
        for layer in args.layers:
            logger.info("Extracting activations from layer %d", layer)

            try:
                pvx = RoleLayersPersonaModel.load_or_create(
                    target_model_id=args.model,
                    concept=role,
                    layer=layer,
                    target_pairs=max(args.sample_counts),
                    safetensors_dir=args.safetensor_dir,
                    sample_counts=args.sample_counts,
                )
                logger.info("Successfully loaded/created RoleLayersPersonaModel on layer %d for role '%s'", layer, role)
            except Exception as e:
                logger.error("Failed to load or create RoleLayersPersonaModel on layer %d for role '%s': %s", layer, role, e)
                continue

            if args.question:
                logger.info("=== Question ===")
                logger.info(args.question)
                logger.info("")
                
                response = pvx.generate(
                    prompt=args.question,
                    alpha=args.alpha,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                base_response = pvx.generate(
                    prompt=args.question,
                    alpha=0,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                logger.info("=== Base Response ===")
                logger.info(base_response)
                logger.info("=== Steered Response ===")
                logger.info(response)
                
                score = role_judge(question=args.question, answer=response, role=role, role_description=role)
                logger.info("=== Judge Score for role %s: %d / 100 ===", role, score)
            
            else:
                trait_path = Path("./configs/scoring/interest_profiler.json")
                
                with trait_path.open() as f:
                    trait_data = json.load(f)
                    
                testing_traits = [trait for trait in trait_data if trait["dimension"] == args.trait] if args.trait else trait_data

                base = (
                    "You are answering the O*NET Interest Profiler. "
                    "For this work activity, would you LIKE to do it as part of a job? "
                )
                
                if not args.cot:
                    answer_format = "Answer ONLY with 'Yes' or 'No' — no explanation."
                else:
                    answer_format = "Show your reasoning and steps. End with only 'Yes' or 'No'."
                
                base += answer_format
                score = 0

                for trait in testing_traits:
                    convo = [
                        {"role": "system", "content": base},
                        {"role": "user", "content": f"Activity: {trait['text']}"},
                    ]

                    logger.info("=== Testing trait: %s ===", trait["text"])
                    response = pvx.generate(
                        messages=convo,
                        alpha=args.alpha,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    logger.info("Response: %s", response)
                    
                    if "Yes" in response[-200:]:
                        score += 1
                        
                logger.info("=== Final %s Score for role %s: %d / %d ===", args.trait, role, score, len(testing_traits))