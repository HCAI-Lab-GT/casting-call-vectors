import json
import os
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pvx.utils.prompts import PromptTemplates
from torch import dtype, float16
from dotenv import load_dotenv

from pvx import Heartbeat, setup_logging
from pvx.utils.response_generation import ResponseGeneration

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")


# Initialize logger once per process
logger = setup_logging(name="abstract-dataset")

class AbstractDataset(ResponseGeneration):
    def __init__(
        self,
        concept: str,
        num_questions: int = 100,
        backend: str = "hf_local",
        model: str = "qwen2.5:7b-instruct",
        base_url: Optional[str] = None,
        api_key_env: str = "LITELLM_API_KEY",
        local_model: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: dtype = float16,
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
        backend = backend if backend in BACKENDS else "openai"
        super().__init__(backend, model, local_model, base_url, api_key_env, device, dtype)
        self.concept = concept
        self.num_questions: int = num_questions
        self.questions: List[str] = []
        self.evaluation_prompt = ""
        self.dirpath = dirpath

    @staticmethod
    def from_json(
        cls,
        concept: str,
        dirpath: str,
    ) -> tuple["AbstractDataset", bool]:
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
        filepath = os.path.join(dirpath, f"{concept}_dataset.json")

        if not os.path.exists(filepath):
            logger.info("Concept dataset not found. Initializing new concept dataset...")
            dataset = cls(concept)
            return dataset, False

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
        dataset = cls(
            concept,
            num_questions=data["num_questions"],
            backend=backend,
            model=data["model"],
            base_url=data.get("base_url"),
            local_model=data.get("local_model"),
        )

        dataset.questions = data["questions"]
        dataset.evaluation_prompt = data["evaluation_prompt"]
        
        # Load up loggers
        dataset.logger = setup_logging(name="abstract-dataset")
        dataset.logger.info("Abstract Dataset loaded from: %s", filepath)

        return dataset, True
    
    def _get_baseline_dataset_dict(self):
        dataset_dict = {
            "concept": self.concept,
            "num_questions": self.num_questions,
            "model": self.model,
            "backend": self.backend,
            "base_url": self.base_url,
            "local_model": self.local_model,
            "questions": self.questions,
            "evaluation_prompt": self.evaluation_prompt,
        }
        return dataset_dict
    
    def _generate_auxiliary_information(self, system_prompt: str, prompts_key: str, max_new_tokens: int = 1024, **kwargs) -> str:
        """
        Generate auxiliary information (descriptions, instructions, etc.) using LLM.

        This method provides a common interface for generating various types of
        auxiliary information needed for dataset generation, such as trait/role
        descriptions and question instructions.

        Args:
            system_prompt (str): The system prompt to use
            prompts_key (str): Key to index into PromptTemplates.PROMPTS
            max_new_tokens (int): Maximum tokens to generate (default: 1024)
            **kwargs: Additional keyword arguments to format the prompt template

        Returns:
            str: Generated auxiliary information
        """
        user_prompt = PromptTemplates.PROMPTS[prompts_key].format(**kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate response using the helper method
        _, response = self._inference_with_client(messages=messages, max_new_tokens=max_new_tokens)

        return response

    def save_dataset_to_json(self, dataset_dict: Dict = {}, dirpath: str | None = None) -> str:
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
        filepath = os.path.join((dirpath or self.dirpath), f"{self.concept}_dataset.json")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

        logger.info("Dataset saved to: %s", filepath)
        return filepath


