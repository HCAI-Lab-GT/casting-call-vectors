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
from pvx.utils.prompts import PromptTemplates
from pvx.abstraction.pvx_models.abstract_dataset import AbstractDataset
from pvx.utils.riasec_utils import RIASECHelpers

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")


# Initialize logger once per process
logger = setup_logging(name="riasec-dataset")

class RiasecDataset(AbstractDataset):
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
        riasec_config_path = "./configs/riasec.yaml",
        num_questions: int = 100,
        backend: str = "hf_local",
        model: str = "qwen2.5:7b-instruct",
        base_url: Optional[str] = None,
        api_key_env: str = "LITELLM_API_KEY",
        local_model: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        dirpath: str = "./persona_data/riasec_datasets/",
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
        super().__init__(
            num_questions=num_questions,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            local_model=local_model,
            device=device,
            dtype=dtype,
            dirpath=dirpath,
        )
        self.trait: str = trait
        self.positive_negative_pairs: List[Tuple[str, str]] = []
        
        if self.trait not in RIASECHelpers.RIASEC_TRAITS:
            raise ValueError("Trait must be in RIASEC set of traits")
        
        self.trait_description = RIASECHelpers.fetch_riasec_information(
                riasec_config_path=riasec_config_path
            )[self.trait]["description"]

    @staticmethod
    def from_json(
        trait: str,
        dirpath: str = "./persona_data/riasec_datasets",
    ) -> "RiasecDataset":
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
        
        logger.info(f"Trying to find dataset at: {filepath}")

        if not os.path.exists(filepath):
            logger.info("Trait dataset not found. Initializing new trait dataset...")
            dataset = RiasecDataset(
                trait=trait
            )
            dataset.generate_dataset(save_to_json=True)
            return dataset
        
        logger.info("Trait dataset found. Initializing trait dataset from json...")
        
        dataset, loaded_from_json = AbstractDataset.from_json(trait_role=trait, dirpath=dirpath)
        dataset.trait = trait
        
        if loaded_from_json:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset.positive_negative_pairs = data["positive_negative_pairs"]
        else:
            dataset.generate_dataset(save_to_json=True)
            return dataset

        # Load up loggers
        dataset.logger = setup_logging(name="riasec-dataset")
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

        question_instruction: str = ""

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
            logger.info(response)

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

        evaluation_prompt = PromptTemplates.evaluation_trait(
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
        user_prompt = PromptTemplates.PROMPTS["question_instruction_trait"].format(TRAIT=self.trait)

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
        
        logger.info(json_text)

        # Extract components
        pos_instructions = response_dict["pos_instructions"]
        neg_instructions = response_dict["neg_instructions"]
        
        questions = RIASECHelpers.fetch_riasec_information()[self.trait]["questions"]

        return list(zip(pos_instructions, neg_instructions, strict=True)), questions


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument("-t", "--traits", nargs="+", help="<Optional> Select traits", required=False)
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
        default="./persona_data/riasec_datasets/",
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
        "realistic",
        "investigative",
        "artistic",
        "social",
        "enterprising",
        "conventional"
    ]

    for trait in trait_list:
        try:
            RiasecDataset.from_json(
                trait=trait,
                dirpath=args.dirpath
            )
        except Exception:
            dataset: RiasecDataset = RiasecDataset(
                trait=trait,
                num_questions=args.num_questions,
                backend=args.backend,
                local_model=args.local_model,
                model=args.model,
                api_key_env=args.api_key_env,
                dirpath=args.dirpath
            )
            dataset.generate_dataset(save_to_json=True)
        
