import argparse
import contextvars
import random
from typing import Optional

from pvx.implementations.riasec.riasec_dataset import RiasecDataset
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers.utils import logging as transformers_logging

from pvx import setup_logging
from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.judges.llm_as_judge import LLMJudge
from pvx.utils.response_generation import ResponseGeneration
from pvx.utils.generation_utils import GenerationConfig
from pvx.utils.judge_utils import JudgeConfig
from pvx.utils.riasec_utils import RIASECHelpers

torch.set_float32_matmul_precision("high")

# Disable transformers progress bars to avoid cluttering output
transformers_logging.set_verbosity_error()

# request-local steering state (per concurrent generate call)
_STEER_DELTA = contextvars.ContextVar("steer_delta", default=None)  # Tensor (1,H) or None

logger = setup_logging(name="riasec-persona-model")


class RIASECPersonaModel(AbstractPersonaModel):
    """
    Persona model for RIASEC trait extraction and response generation.
    Implements persona vector extraction using pregenerated RIASEC responses.
    """
    

    def __init__(
        self,
        trait: Optional[str] = None,
        dataset: Optional[RiasecDataset] = None,
        dataset_dirpath: str = './persona_data/riasec_datasets/',
        riasec_config_path: str = "./configs/riasec.yaml",
        from_json: bool = False,
        **kwargs,
    ):
        """
        Initialize RIASECPersonaModel with RIASEC-specific validation and pregeneration logic.
        - Ensures trait is a valid RIASEC trait.
        - Ensures dataset questions match RIASEC YAML.
        - Pregenerates answers if missing in YAML.
        """
        # Validate trait
        if trait is None:
            raise ValueError("trait must be specified for RIASECPersonaModel")
        if trait not in RIASECHelpers.RIASEC_TRAITS:
            raise ValueError(
                f"Trait '{trait}' is not a valid RIASEC trait: {sorted(RIASECHelpers.RIASEC_TRAITS)}"
            )
        
        self.trait = trait

        if from_json:
            super().__init__(from_json=True, **kwargs)
            return

        # Load RIASEC YAML info
        riasec_info = RIASECHelpers.fetch_riasec_information(riasec_config_path)
        trait_info = riasec_info[trait]
        yaml_questions = trait_info["questions"]

        # If dataset not provided, load from JSON or YAML
        if dataset is None:
            self.dataset = RiasecDataset.from_json(
                trait=trait
            )
        else:
            self.dataset = dataset

        # Check dataset questions match YAML
        if self.dataset.questions != yaml_questions:
            logger.info("Updating dataset questions to be RIASEC trait questions")
            self.dataset.questions = yaml_questions

        # Pregenerate answers if missing in YAML
        needs_pregeneration = False
        for qa in trait_info["question_answer_pairs"]:
            if not qa.get("positive") or not qa.get("negative"):
                needs_pregeneration = True
                break
        if needs_pregeneration:
            logger.info(f"Pregenerating missing answers for trait '{trait}' in YAML...")
            accepted_responses = self.pre_generate_riasec_pos_neg_responses(
                trait=trait, riasec_config_path=riasec_config_path
            )
            RIASECHelpers.update_riasec_yaml(riasec_config_path, trait, accepted_responses)
            logger.info(f"Pregeneration complete and YAML updated for trait '{trait}'.")
            
        # Call base class init
        super().__init__(**kwargs)
    
    @classmethod
    def from_json(
        cls,
        json_filepath: str,
        trait: Optional[str] = None,
    ) -> "RIASECPersonaModel":
        """
        Load a PersonaModel instance from a previously saved JSON file.

        Args:
            json_filepath (str): Path to the JSON file containing the saved initialization data

        Returns:
            PersonaModel: A new instance with the loaded persona vectors
        """

        with open(json_filepath, "r") as f:
            data = json.load(f)

        logger.info("Loading RiasecPersonaModel from: %s", json_filepath)

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
        dataset: Optional[RiasecDataset] = None,
        trait: Optional[str] = None,  # alternate to dataset for loading
        layer: float = 14,
        json_filepath: Optional[str] = None,
        safetensors_dir: str = "./persona_data/model_inits/",
    ) -> "RIASECPersonaModel":
        """
        Load a PersonaModel instance from saved files if they exist, otherwise create a new one.

        Priority order:
        1. Safetensors file (preferred - smaller, faster)
        2. Legacy JSON file (backward compatibility)
        3. Create new instance

        Args:
            target_model_id: Model identifier
            dataset: RiasecDataset for extraction (only used if creating new)
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
            dataset=dataset,
            trait=trait,
            layer=layer,
        )

    @torch.inference_mode()
    def extract_persona_vector(
        self, temperature: float = 0.9, max_new_tokens: int = 200
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract persona vectors using pregenerated RIASEC responses.
        Computes mean activations for positive and negative responses, then returns persona vectors as differences.
        Returns:
            Tuple of (prompt_persona_vector, response_persona_vector, response_average_all_layers)
        """
        logger.info("Extracting persona vectors from pregenerated responses...")
        # Fetch RIASEC data and trait-specific question/answer pairs
        riasec_data = RIASECHelpers.fetch_riasec_information()
        trait_data = riasec_data[self.trait]
        qa_pairs = trait_data["question_answer_pairs"]

        total_responses = sum(
            len(qa.get("positive", [])) + len(qa.get("negative", [])) for qa in qa_pairs
        )

        # Accumulators for positive/negative activations
        sum_prompt_last_pos = None
        sum_prompt_last_neg = None
        sum_resp_avg_pos = None
        sum_resp_avg_neg = None
        sum_resp_avg_all_layers_pos = None
        sum_resp_avg_all_layers_neg = None

        n_pos = 0
        n_neg = 0

        with tqdm(
            total=total_responses, desc=f"Extracting activations for trait {self.trait}"
        ) as pbar:
            for qa in qa_pairs:
                question = qa.get("question", "")

                # Compute prompt_len once per question by tokenizing prompt-only
                # (with add_generation_prompt=True to include the assistant turn start)
                prompt_messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": question},
                ]
                prompt_only = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                prompt_len = prompt_only.shape[1]

                # Process positive responses
                for pos_response in qa.get("positive", []):
                    full_messages = [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": pos_response},
                    ]
                    enc = self.tokenizer.apply_chat_template(
                        full_messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_tensors="pt",
                    ).to(self.device)
                    mask = torch.ones_like(enc)
                    pl_pos, ra_pos, rall_pos = self._extract_activations_from_tokens(
                        enc, mask, prompt_len
                    )
                    if sum_prompt_last_pos is None:
                        sum_prompt_last_pos = torch.zeros_like(pl_pos, dtype=torch.float32)
                        sum_resp_avg_pos = torch.zeros_like(ra_pos, dtype=torch.float32)
                        sum_resp_avg_all_layers_pos = torch.zeros_like(
                            rall_pos, dtype=torch.float32
                        )
                    sum_prompt_last_pos += pl_pos.float()
                    sum_resp_avg_pos += ra_pos.float()
                    sum_resp_avg_all_layers_pos += rall_pos.float()
                    n_pos += 1
                    pbar.update(1)

                # Process negative responses
                for neg_response in qa.get("negative", []):
                    full_messages = [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": neg_response},
                    ]
                    enc = self.tokenizer.apply_chat_template(
                        full_messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_tensors="pt",
                    ).to(self.device)
                    mask = torch.ones_like(enc)
                    pl_neg, ra_neg, rall_neg = self._extract_activations_from_tokens(
                        enc, mask, prompt_len
                    )
                    if sum_prompt_last_neg is None:
                        sum_prompt_last_neg = torch.zeros_like(pl_neg, dtype=torch.float32)
                        sum_resp_avg_neg = torch.zeros_like(ra_neg, dtype=torch.float32)
                        sum_resp_avg_all_layers_neg = torch.zeros_like(
                            rall_neg, dtype=torch.float32
                        )
                    sum_prompt_last_neg += pl_neg.float()
                    sum_resp_avg_neg += ra_neg.float()
                    sum_resp_avg_all_layers_neg += rall_neg.float()
                    n_neg += 1
                    pbar.update(1)

        # Compute means and persona vectors
        prompt_last_pos_mean = sum_prompt_last_pos / max(n_pos, 1)
        prompt_last_neg_mean = sum_prompt_last_neg / max(n_neg, 1)
        response_avg_pos_mean = sum_resp_avg_pos / max(n_pos, 1)
        response_avg_neg_mean = sum_resp_avg_neg / max(n_neg, 1)
        all_layers_response_avg_pos_mean = sum_resp_avg_all_layers_pos / max(n_pos, 1)
        all_layers_response_avg_neg_mean = sum_resp_avg_all_layers_neg / max(n_neg, 1)

        prompt_persona_vector = prompt_last_pos_mean - prompt_last_neg_mean
        response_persona_vector = response_avg_pos_mean - response_avg_neg_mean
        all_layers_response_persona_vector = (
            all_layers_response_avg_pos_mean - all_layers_response_avg_neg_mean
        )

        self.prompt_persona_vector = prompt_persona_vector.cpu()
        self.response_persona_vector = response_persona_vector.cpu()
        self.all_layers_response_persona_vector = all_layers_response_persona_vector.cpu()

        logger.info("Extracted persona vectors from pregenerated responses")
        logger.info("Prompt persona vector shape: %s", str(tuple(prompt_persona_vector.shape)))
        logger.info("Response persona vector shape: %s", str(tuple(response_persona_vector.shape)))
        logger.info(
            "All-layers response persona vector shape: %s",
            str(tuple(all_layers_response_persona_vector.shape)),
        )

        return prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector

    def _extract_activations_from_tokens(self, input_ids, attention_mask, prompt_len: int):
        """
        Extract activations from a prompt+response sequence.

        Args:
            input_ids: Tokenized full prompt+response sequence
            attention_mask: Attention mask for the sequence
            prompt_len: Number of tokens in the prompt (before assistant response starts).
                        Used to correctly separate prompt vs response activations.

        Returns:
            prompt_last: last hidden state of prompt (from self.layer_steering+1)
            resp_avg: average hidden state of response tokens only (from self.layer_steering+1)
            rall: average hidden state of response tokens across all layers
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[self.layer_steering + 1][0]  # (seq, hidden)

        # Use prompt_len to correctly split prompt vs response activations
        prompt_last = last_hidden[prompt_len - 1]  # Last token of prompt (before response)

        # Average only the response tokens (after prompt_len)
        resp_hidden = last_hidden[prompt_len:]
        if resp_hidden.shape[0] > 0:
            resp_avg = resp_hidden.mean(dim=0)
            rall = torch.stack([h[0][prompt_len:].mean(dim=0) for h in hidden_states])
        else:
            # Fallback if no response tokens (shouldn't happen with pregenerated responses)
            resp_avg = last_hidden.mean(dim=0)
            rall = torch.stack([h[0].mean(dim=0) for h in hidden_states])

        return prompt_last, resp_avg, rall

    @staticmethod
    def pre_generate_riasec_pos_neg_responses(
        trait: str,
        temperature=0.2,
        max_new_tokens=200,
        generation_config: Optional[GenerationConfig] = None,
        judge_config: Optional[JudgeConfig] = None,
        target_count: int = 5,
        threshold: float = 50,
        riasec_config_path: str = "./configs/riasec.yaml",
    ):
        """
        Generate and filter positive/negative responses for a RIASEC trait using LLM and judge.
        Returns a dict mapping each question to lists of accepted positive and negative responses.
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        generate_response = ResponseGeneration(**generation_config.to_kwargs())
        dataset = RiasecDataset.from_json(
            trait=trait, from_riasec=True, riasec_config_path=riasec_config_path
        )

        if judge_config is None:
            judge_config = JudgeConfig(prompt_template=dataset.evaluation_prompt)
        else:
            if judge_config.prompt_template is None:
                judge_config.prompt_template = dataset.evaluation_prompt
        judge = LLMJudge(**judge_config.to_kwargs())

        accepted_responses = {}
        trait_pairs = dataset.extract_pos_neg_question_pairs()

        # Map each question to its positive/negative pairs
        question_to_pairs = {}
        for pos, neg, question in trait_pairs:
            question_to_pairs.setdefault(question, []).append((pos, neg))

        questions = list(question_to_pairs.keys())

        # Progress bar for questions
        with tqdm(total=len(questions), desc=f"Questions for trait {trait}") as question_pbar:
            for question in questions:
                pos_responses = []
                neg_responses = []
                pos_attempts = 0
                neg_attempts = 0
                pos_seen = set()
                neg_seen = set()
                pairs = question_to_pairs[question]

                # Generate positive responses
                with tqdm(
                    total=target_count, desc=f"Positive ({question[:30]}...)", leave=False
                ) as pos_pbar:
                    while len(pos_responses) < target_count and pos_attempts < target_count * 4:
                        pos, _ = random.choice(pairs)
                        pos_messages = [
                            {
                                "role": "system",
                                "content": RIASECHelpers.POSITIVE_RIASEC_SYSTEM_PROMPT.format(
                                    TRAIT=trait
                                ),
                            },
                            {"role": "system", "content": pos},
                            {"role": "user", "content": question},
                        ]
                        _, pos_response = generate_response(
                            messages=pos_messages,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                        )
                        pos_score = judge(question=question, answer=pos_response)
                        if pos_score >= threshold and pos_response not in pos_seen:
                            pos_responses.append(pos_response)
                            pos_seen.add(pos_response)
                            pos_pbar.update(1)
                        pos_attempts += 1

                # Generate negative responses
                with tqdm(
                    total=target_count, desc=f"Negative ({question[:30]}...)", leave=False
                ) as neg_pbar:
                    while len(neg_responses) < target_count and neg_attempts < target_count * 4:
                        _, neg = random.choice(pairs)
                        neg_messages = [
                            {
                                "role": "system",
                                "content": RIASECHelpers.NEGATIVE_RIASEC_SYSTEM_PROMPT.format(
                                    TRAIT=trait
                                ),
                            },
                            {"role": "system", "content": neg},
                            {"role": "user", "content": question},
                        ]
                        _, neg_response = generate_response(
                            messages=neg_messages,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                        )
                        neg_score = judge(question=question, answer=neg_response)
                        if neg_score < threshold and neg_response not in neg_seen:
                            neg_responses.append(neg_response)
                            neg_seen.add(neg_response)
                            neg_pbar.update(1)
                        neg_attempts += 1

                accepted_responses[question] = {
                    "positive": pos_responses,
                    "negative": neg_responses,
                }
                question_pbar.update(1)

        return accepted_responses


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-pgr",
        "--pre_generate_response",
        action="store_true",
        help="Pre-generate responses for a trait",
    )
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
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
        "-c",
        "--target_count",
        type=int,
        default=5,
        help="Number of pregenerated responses for each question",
    )
    ap.add_argument(
        "-f",
        "--json_filepath",
        type=str,
        default=None,
        help="Filepath to model init file (not trait dataset)",
    )
    args = ap.parse_args()

    if args.pre_generate_response:
        accepted_responses = RIASECPersonaModel.pre_generate_riasec_pos_neg_responses(
            trait=args.trait, target_count=args.target_count
        )
        logger.info("===Accepted Responses===")
        logger.info(accepted_responses)
    else:
        pvx = RIASECPersonaModel.load_or_create(
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
