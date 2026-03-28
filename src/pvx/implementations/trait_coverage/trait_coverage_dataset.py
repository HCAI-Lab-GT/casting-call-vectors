import argparse
import json
import os
from pathlib import Path
from collections import namedtuple
from typing import List, Optional, Tuple, override


import torch
from torch import dtype, float16
from dotenv import load_dotenv

from pvx import Heartbeat, setup_logging
from pvx.utils.prompts import PromptTemplates
from pvx.abstraction.pvx_models.abstract_dataset import AbstractDataset
from pvx.implementations.base.persona_dataset import PersonaDataset
# from pvx.utils.riasec_utils import RIASECHelpers

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")


# Initialize logger once per process
logger = setup_logging(name="trait-coverage-dataset")

# class TraitCoverageDataset(AbstractDataset):
class TraitCoverageDataset(PersonaDataset):
    @override
    def __init__(
        self,
        trait: str,
        trait_description: Optional[str] = None,
        num_questions: int = 100,
        backend: str = "openai",
        model: str = "openai/gpt-4.1-mini",
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        local_model: Optional[str] = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        dirpath: str = "./persona_data/trait_coverage_datasets/",
    ):
        super().__init__(
            trait=trait,
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

    @staticmethod
    def from_json(
        trait: str,
        trait_description: Optional[str] = None,
        dirpath: str = "./persona_data/trait_coverage_datasets",
    ) -> "TraitCoverageDataset":
        filepath = os.path.join(dirpath, f"{trait}_dataset.json")
        
        logger.info(f"Trying to find dataset at: {filepath}")

        if not os.path.exists(filepath):
            logger.info("Trait dataset not found. Initializing new trait dataset...")
            dataset = TraitCoverageDataset(
                trait=trait,
                trait_description=trait_description
            )
            dataset.generate_dataset(save_to_json=True)
            return dataset
        
        logger.info("Trait dataset found. Initializing trait dataset from json...")
        
        dataset, loaded_from_json = AbstractDataset.from_json(cls=PersonaDataset,concept=trait, dirpath=dirpath)
        dataset.trait = trait
        
        if loaded_from_json:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset.positive_negative_pairs = data["positive_negative_pairs"]

        # Load up loggers
        dataset.logger = setup_logging(name="trait-coverage-dataset")
        dataset.logger.info("Dataset loaded from: %s", filepath)

        return dataset



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-t", 
        "--traits", 
        nargs="+", 
        help="<Optional> Select traits", 
        required=False
    )
    ap.add_argument(
        "-d",
        "--trait_description",
        default=None,
        help="<Optional> description of the trait to generate a dataset for.",
    )
    ap.add_argument(
        "-f",
        "--dirpath",
        default="./persona_data/trait_coverage_datasets/",
        help="Directory filepath to save generated datasets",
    )
    ap.add_argument(
        "-b", "--backend", default="openai", help="Backend to use: openai, vllm, hf_local"
    )
    ap.add_argument(
        "-m",
        "--model",
        default="openai/gpt-4.1-mini",
        help="Model to use for openai/vllm backend",
    )
    ap.add_argument(
        "-a",
        "--api_key_env",
        default="OPENROUTER_API_KEY",
        help="Environment variable for API key",
    )
    ap.add_argument(
        "-u",
        "--base_url",
        default="https://openrouter.ai/api/v1",
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
                dirpath=args.dirpath
            )
        except Exception:
            dataset: PersonaDataset = PersonaDataset(
                trait=trait,
                num_questions=args.num_questions,
                backend=args.backend,
                local_model=args.local_model,
                model=args.model,
                api_key_env=args.api_key_env,
            )
            dataset.generate_dataset(save_to_json=True)