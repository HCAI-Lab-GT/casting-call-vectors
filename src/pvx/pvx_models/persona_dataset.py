"""
PersonaDataset Module

This module provides functionality for generating and managing persona-based datasets
for training language models with specific personality traits. It handles dataset
generation using LLM inference, parsing of structured outputs, and persistence.

Key Features:
    - Generate positive/negative instruction pairs for specific personality traits
    - Create evaluation questions and prompts
    - Support for both Ollama and OpenAI-compatible API endpoints
    - Dataset serialization and deserialization
    - Extraction of paired training data

Dependencies:
    - openai/vLLM (OpenAI-compatible HTTP): For remote inference
    - Hugging Face transformers: For local inference
    - transformers: For model and tokenizer utilities

Environment Variables:
    - LITELLM_API_KEY or OPENAI_API_KEY: Required when using OpenAI/vLLM endpoints
    - VLLM_API_URL (optional): Base URL for vLLM OpenAI-compatible server
"""

import argparse
import json
import os
import warnings
from collections import namedtuple
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvx import Heartbeat, setup_logging
from pvx.pvx_models.prompts import PromptTemplates
from pvx.utils.riasec_utils import RIASECHelpers

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")


# Initialize logger once per process
logger = setup_logging(name="persona-dataset")


class PersonaDataset:
    """
    A dataset class for generating and managing persona-based training data.

    This class facilitates the creation of datasets containing positive/negative
    instruction pairs, questions, and evaluation prompts for specific personality
    traits. It supports multiple inference backends (OpenAI/vLLM HTTP or local
    Hugging Face transformers) and provides methods for dataset persistence and retrieval.

    Attributes:
        trait (str): The personality trait being modeled (e.g., "sarcastic", "verbose")
        num_questions (int): Number of questions to generate for the dataset
        positive_negative_pairs (List[Tuple[str, str]]): List of (positive, negative)
            instruction pairs
        questions (List[str]): List of evaluation questions
        evaluation_prompt (str): The evaluation prompt for assessing the trait
        model (str): The name/identifier of the LLM model to use
        backend (str): Inference backend ("openai", "vllm", "hf_local")

    Example:
        >>> dataset = PersonaDataset(
        ...     trait="sarcastic",
        ...     num_questions=5,
        ...     backend="openai",
        ...     model="gpt-4o"
        ... )
        >>> pairs, questions, eval_prompt, filepath = dataset.generate_dataset()
    """

    def __init__(
        self,
        trait: str,
        trait_description: Optional[str] = None,
        from_riasec: bool = False,
        riasec_config_path: str = "./configs/riasec.yaml",
        num_questions: int = 100,
        backend: str = "hf_local",
        model: str = "qwen2.5:7b-instruct",
        base_url: Optional[str] = None,
        api_key_env: str = "LITELLM_API_KEY",
        local_model: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        dirpath: str = "./persona_data/trait_datasets/",
    ):
        """
        Initialize a PersonaDataset instance.

        Args:
            trait (str): The personality trait to generate data for (e.g., "sarcastic",
                "verbose", "analytical")
            num_questions (int): The number of questions to generate for evaluation
            backend (str): One of {"openai", "vllm", "hf_local"}.
                - "openai": Use OpenAI-compatible HTTP endpoint.
                - "vllm":  Use OpenAI-compatible HTTP endpoint (base_url from VLLM_API_URL).
                - "hf_local": Use local transformers model.
            model (str): Model identifier (HF repo id for hf_local; model name for openai/vllm).
            base_url (str, optional): Override base URL for OpenAI/vLLM endpoints.
            api_key_env (str): Environment variable name for API key (openai/vllm).
            local_model (str, optional): HF model id/path for hf_local. Falls back to `model` if None.
            device (str, optional): Device for hf_local ("cuda", "mps", "cpu"). Auto-detect if None.
            dtype (torch.dtype): Dtype for hf_local model load (default float16).

        Returns:
            None
        """
        self.trait: str = trait
        self.num_questions: int = num_questions
        self.positive_negative_pairs: List[Tuple[str, str]] = []
        self.questions: List[str] = []
        self.evaluation_prompt = ""
        self.model = model
        self.backend = backend if backend in BACKENDS else "openai"
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.local_model = local_model or model
        self.device = device
        self.dtype = dtype
        self._hf_model = None
        self._hf_tokenizer = None

        self.dirpath = dirpath

        if trait_description is not None:
            logger.info("Trait description set from initializer")
            self.trait_description = trait_description
        elif from_riasec and self.trait in RIASECHelpers.RIASEC_TRAITS:
            logger.info("Trait description set from RIASEC config yaml")
            self.trait_description = RIASECHelpers.fetch_riasec_information(
                riasec_config_path=riasec_config_path
            )[self.trait]["description"]
        else:
            logger.info("Trait description will be generated")
            self.trait_description = None
        self.from_riasec = from_riasec

    @staticmethod
    def from_json(
        trait: str,
        trait_description: Optional[str] = None,
        from_riasec: bool = False,
        riasec_config_path: str = "./configs/riasec.yaml",
        dirpath: str = "./persona_data/trait_datasets",
    ) -> "PersonaDataset":
        """
        Load a previously saved dataset from a JSON file.

        This static method reconstructs a PersonaDataset instance from a saved
        JSON file, including all generated content and metadata. It automatically
        configures the appropriate backend (openai / vllm / hf_local) based on
        the saved metadata.

        Args:
            trait (str): The personality trait name (used to construct filename)
            dirpath (str, optional): Directory path where the dataset is saved.
                The filename is assumed to be "{trait}_dataset.json".
                Defaults to "./persona_data/trait_datasets/".

        Returns:
            PersonaDataset: A fully initialized PersonaDataset instance with all
                previously generated data loaded

        Raises:
            FileNotFoundError: If the dataset file doesn't exist at the specified path
            json.JSONDecodeError: If the file contains invalid JSON
            KeyError: If required keys are missing from the JSON data
        """
        filepath = os.path.join(dirpath, f"{trait}_dataset.json")

        if not os.path.exists(filepath):
            logger.info("Trait dataset not found. Initializing new trait dataset...")
            dataset = PersonaDataset(
                trait=trait,
                trait_description=trait_description,
                from_riasec=from_riasec,
                riasec_config_path=riasec_config_path,
            )
            dataset.generate_dataset(save_to_json=True)
            return dataset

        # Load JSON data
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Backward compatibility with older files that stored "client"
        backend = data.get("backend")
        if backend is None and "client" in data:
            client = data["client"]
            backend = "hf_local" if client == "ollama" else "openai"
        backend = backend or "openai"

        # reconstruct dataset instance
        dataset = PersonaDataset(
            trait=trait,
            num_questions=data["num_questions"],
            backend=backend,
            model=data["model"],
            base_url=data.get("base_url"),
            local_model=data.get("local_model"),
        )

        # Populate dataset outputs
        dataset.positive_negative_pairs = data["positive_negative_pairs"]
        dataset.questions = data["questions"]
        dataset.evaluation_prompt = data["evaluation_prompt"]

        # Load up loggers
        dataset.logger = setup_logging(name="persona-dataset")
        dataset.logger.info("Dataset loaded from: %s", filepath)

        return dataset

    def save_dataset_to_json(self, dirpath: str | None = None) -> str:
        """
        Save the persona vector dataset to a JSON file.

        Serializes the dataset including all generated content (instruction pairs,
        questions, evaluation prompt) and metadata (trait, model, backend) to
        a JSON file. The directory structure is created automatically if it doesn't exist.

        Args:
            filepath (str, optional): Directory path where the dataset should be saved.
                The filename will be automatically generated as "{trait}_dataset.json".
                Defaults to self.dirpath.

        Returns:
            str: The full path to the saved JSON file

        Example:
            >>> dataset = PersonaDataset(trait="verbose", num_questions=5)
            >>> # ... generate dataset ...
            >>> path = dataset.save_dataset_to_json("./my_datasets/")
            >>> print(f"Saved to: {path}")
            "Saved to: ./my_datasets/verbose_dataset.json"
        """
        filepath = os.path.join((dirpath or self.dirpath), f"{self.trait}_dataset.json")

        dataset_dict = {
            "trait": self.trait,
            "num_questions": self.num_questions,
            "model": self.model,
            "backend": self.backend,
            "base_url": self.base_url,
            "local_model": self.local_model,
            "positive_negative_pairs": self.positive_negative_pairs,
            "questions": self.questions,
            "evaluation_prompt": self.evaluation_prompt,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

        logger.info("Dataset saved to: %s", filepath)
        return filepath

    def generate_dataset(
        self, save_to_json=True, max_tries=5
    ) -> Tuple[List[Tuple[str, str]], List[str], str, str | None]:
        """
        Generate the complete persona dataset.

        This is the main method that orchestrates the entire dataset generation
        process. It generates trait descriptions, question instructions, and then
        uses these to create positive/negative instruction pairs, questions, and
        an evaluation prompt. The generated dataset is automatically saved to JSON.

        Args:
            save_to_json (bool): Whether to save the generated dataset to a JSON file.
                Defaults to True.
            max_tries (int): Maximum number of attempts to parse valid dataset output
                from the LLM. Defaults to 5.

        Returns:
            Tuple[List[Tuple[str, str]], List[str], str, str]: A tuple containing:
                - positive_negative_pairs: List of (positive_instruction, negative_instruction) tuples
                - questions: List of evaluation questions
                - evaluation_prompt: The prompt for evaluating the trait
                - filepath: Path where the dataset was saved

        Raises:
            ValueError: If the dataset output cannot be parsed after max_tries attempts

        Example:
            >>> dataset = PersonaDataset(trait="analytical", num_questions=10)
            >>> pairs, questions, eval_prompt, path = dataset.generate_dataset()
            >>> print(f"Generated {len(pairs)} instruction pairs")
            >>> print(f"Dataset saved to: {path}")
        """
        # Generate trait description and question instruction
        if self.trait_description is None:
            trait_description: str = self._generate_trait_description()
            logger.info("Generated trait description for %s", self.trait)
        else:
            trait_description = self.trait_description

        question_instruction: str = self._generate_question_instruction()
        logger.info("Generated question instruction for %s", self.trait)

        # load generation prompt
        system_prompt: str = "You are an expert AI evaluator and dataset designer."
        user_prompt: str = PromptTemplates.PROMPTS["generate_trait"].format(
            TRAIT=self.trait,
            N=self.num_questions,
            trait_instruction=trait_description,
            question_instruction=question_instruction,
        )

        # Generate response using the helper method
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with Heartbeat(logger, f"Generating dataset for {self.trait}", interval=30):
            _, response = self._inference_with_client(messages=messages)

        # Parse the response into structured dataset components, retry maximum max_tries times if failed
        pos_neg_pairs, questions, evaluation_prompt = None, None, None
        for _ in range(max_tries):
            try:
                pos_neg_pairs, questions = self._parse_dataset_output(response=response)
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError("Failed to parse dataset output after multiple attempts")

        evaluation_prompt = PromptTemplates.evaluation(
            trait=self.trait, trait_description=trait_description
        )

        self.positive_negative_pairs = pos_neg_pairs
        self.questions = questions
        self.evaluation_prompt = evaluation_prompt

        filepath = None
        if save_to_json is True:
            # Save dataset to JSON
            filepath = self.save_dataset_to_json()

        return pos_neg_pairs, questions, evaluation_prompt, filepath

    def extract_pos_neg_question_pairs(self) -> List[Tuple]:
        """
        Extract Cartesian product of (positive, negative, question) from positive/negative instruction pairs and
        questions.

        Returns:
            List[tuple]: A list of tuples:
                - [0] (str): The positive instruction
                - [1] (str): The negative instruction
                - [2] (str): The evaluation question
        """
        Pair = namedtuple("Pair", ["pos", "neg", "question"])
        return [
            Pair(pos=pos, neg=neg, question=q)
            for (pos, neg), q in product(self.positive_negative_pairs, self.questions)
        ]

    def _inference_with_client(
        self, messages: List[Dict[str, str]], temperature: float = 0.9, max_new_tokens=1024
    ) -> Tuple[Optional[str], str]:
        """
        Perform LLM inference using either OpenAI/vLLM HTTP or local HF transformers.

        This method abstracts away the differences between local HF inference and
        remote OpenAI/vLLM-compatible endpoints.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with
                'role' and 'content' keys, following the OpenAI chat format
            temperature (float, optional): Sampling temperature for generation.
                Higher values (e.g., 0.9) produce more random outputs.
                Defaults to 0.9.

        Returns:
            Tuple[Optional[str], str]: A tuple of (thinking_content, response_content).
                The thinking_content is only present for certain models that support
                chain-of-thought reasoning; otherwise it's None.

        Raises:
            ValueError: If required API key is not set for OpenAI/vLLM backends
            Exception: Re-raises any exception from the underlying API call after
                logging the error details

            Example:
                >>> messages = [
                ...     {"role": "system", "content": "You are a helpful assistant."},
                ...     {"role": "user", "content": "Hello!"}
                ... ]
                >>> thinking, response = dataset._inference_with_client(messages)
        """
        try:
            if self.backend == "hf_local":
                if self._hf_model is None or self._hf_tokenizer is None:
                    self._init_local_model()

                # Prepare prompt for decoder-only model
                prompt = self._messages_to_prompt(messages)

                # Generate response
                inputs = self._hf_tokenizer(prompt, return_tensors="pt", padding=True).to(
                    self._hf_model.device
                )
                generated = self._hf_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                # Extract generated text (first sequence and excluding the prompt)
                decoded = self._hf_tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                )

                return (None, decoded.strip())

            base_url = self.base_url
            if self.backend == "vllm" and base_url is None:
                base_url = os.environ.get("VLLM_API_URL")
            if base_url is None:
                base_url = "https://glados.ctisl.gtri.org"

            api_key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    f"{self.api_key_env} or OPENAI_API_KEY environment variable not set"
                )
            openai_client = OpenAI(api_key=api_key, base_url=base_url)

            chat_response = openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return (None, chat_response.choices[0].message.content)

        except Exception as e:
            logger.exception(
                "inference failed | backend=%s model=%s temp=%s",
                self.backend,
                self.model,
                temperature,
            )
            raise e

    def _init_local_model(self):
        """
        Lazily load a HF transformers causal LM for local inference.
        """
        if self.backend != "hf_local":
            return

        # Device selection
        if self.device:
            device = self.device
        else:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                warnings.warn(
                    "Using CPU for local inference; generation will be slow.", stacklevel=2
                )

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.local_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with device map
        model = AutoModelForCausalLM.from_pretrained(
            self.local_model,
            dtype=self.dtype if device != "cpu" else None,
            device_map=device,
        )

        self._hf_model = model
        self._hf_tokenizer = tokenizer

    def _generate_trait_description(self) -> str:
        """
        Generate a detailed description of the personality trait.

        Uses the LLM to create a comprehensive description of the trait being
        modeled. This description helps inform subsequent generation steps.

        Returns:
            str: A detailed description of the personality trait
        """
        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = PromptTemplates.PROMPTS["trait_instruction"].format(TRAIT=self.trait)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate response using the helper method
        _, response = self._inference_with_client(messages=messages)

        return response

    def _generate_question_instruction(self) -> str:
        """
        Generate instructions for creating evaluation questions.

        Uses the LLM to create guidelines for generating appropriate evaluation
        questions that can test the presence of the personality trait.

        Returns:
            str: Instructions for question generation
        """
        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = PromptTemplates.PROMPTS["question_instruction"].format(TRAIT=self.trait)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate response using the helper method
        _, response = self._inference_with_client(messages=messages, max_new_tokens=2**13)

        return response

    def _parse_dataset_output(self, response: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Parse the LLM-generated JSON output into structured components.

        Extracts positive instructions, negative instructions, questions, and
        evaluation prompts.

        Args:
            response (str): The raw text response from the LLM containing the
                generated dataset in a JSON structured format

        Returns:
            Tuple[List[Tuple[str, str]], List[str], str]: A tuple containing:
                - pos_neg_pairs: List of (positive, negative) instruction tuples
                - questions: List of evaluation questions
                - eval_prompt: The evaluation prompt (concatenated if multi-line)

        Raises:
            json.JSONDecodeError: If the response does not contain valid JSON
        """
        json_text = "\n".join(response.split("\n")[1:-1])  # Remove ```json and ```
        response_dict = json.loads(json_text)

        # Extract components
        pos_instructions = response_dict["pos_instructions"]
        neg_instructions = response_dict["neg_instructions"]
        if self.from_riasec:
            questions = RIASECHelpers.fetch_riasec_information()[self.trait]["questions"]
        else:
            questions = response_dict["questions"]

        return list(zip(pos_instructions, neg_instructions, strict=True)), questions

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Flatten chat messages into a single prompt string for decoder-only HF models.
        Format: "<system>\\n\\n<user>\\n\\nAssistant:"
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [m["content"] for m in messages if m["role"] == "user"]
        system_block = "\\n".join(system_parts)
        user_block = "\\n\\n".join(user_parts)
        prompt = ""
        if system_block:
            prompt += f"{system_block}\\n\\n"
        prompt += user_block
        prompt += "\\n\\nAssistant:"
        return prompt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-t", "--traits", nargs="+", help="<Optional> Select traits", required=False
    )
    ap.add_argument(
        "-d",
        "--trait_description",
        default=None,
        help="<Optional> description of the trait to generate a dataset for.",
    )
    ap.add_argument(
        "--from_riasec",
        action="store_true",
        help="Use RIASEC config for trait description",
    )
    ap.add_argument(
        "-f",
        "--dirpath",
        default="./persona_data/trait_datasets/",
        help="Directory filepath to save generated datasets",
    )
    ap.add_argument(
        "-c",
        "--riasec_config",
        default="./configs/riasec.yaml",
        help="Directory filepath for riasec config file",
    )
    ap.add_argument(
        "-b", "--backend", default="hf_local", help="Backend to use: openai, vllm, hf_local"
    )
    ap.add_argument(
        "-m",
        "--model",
        default="openai/gpt-oss-120b",
        help="Model to use for openai/vllm backend",
    )
    ap.add_argument(
        "-a",
        "--api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable for API key",
    )
    ap.add_argument(
        "-u",
        "--base_url",
        default="https://api.together.xyz/v1",
        help="Base URL for openai/vllm backend",
    )
    ap.add_argument(
        "-l",
        "--local_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local HF model to use for hf_local backend",
    )
    ap.add_argument(
        "-N",
        "--num_questions",
        type=int,
        default=100,
        help="Number of questions to generate per trait",
    )
    args = ap.parse_args()

    trait_list = args.traits or [
        "sarcastic",
        "verbose",
        "evil",
        "kind",
        "analytical",
        "empathetic",
        "humorous",
        "optimistic",
        "pessimistic",
        "creative",
        "brave",
        "cautious",
    ]

    for trait in trait_list:
        try:
            PersonaDataset.from_json(
                trait=trait,
                dirpath=args.dirpath,
                from_riasec=args.from_riasec,
                riasec_config_path=args.riasec_config,
            )
        except Exception:
            dataset: PersonaDataset = PersonaDataset(
                trait=trait,
                from_riasec=args.from_riasec,
                num_questions=args.num_questions,
                backend=args.backend,
                local_model=args.local_model,
                model=args.model,
                api_key_env=args.api_key_env,
            )
            dataset.generate_dataset(save_to_json=True)

    # nonlocal example
    #     trait="sarcastic", num_questions=100, backend="openai", model="openai/gpt-oss-120b"
