import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from torch import dtype, float16
from dotenv import load_dotenv

from pvx import Heartbeat, setup_logging
from pvx.utils.prompts import PromptTemplates
from pvx.abstraction.pvx_models.abstract_dataset import AbstractDataset
from pvx.utils.riasec_utils import RIASECHelpers

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")


# Initialize logger once per process
logger = setup_logging(name="role-dataset")

class RoleDataset(AbstractDataset):

    def __init__(
        self,
        role: str = "",
        role_description: Optional[str] = None,
        num_questions: int = 100,
        backend: str = "hf_local",
        model: str = "qwen2.5:7b-instruct",
        base_url: Optional[str] = None,
        api_key_env: str = "LITELLM_API_KEY",
        local_model: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        dtype: dtype = float16,
        dirpath: str = "./persona_data/role_datasets/",
    ):
        
        super().__init__(
            concept=role,
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
        self.role: str = role
        self.positive_prompts: List[str] = []
        
        with open("./src/pvx/utils/role_utils/role_list.json", "r", encoding="utf-8") as f:
            role_dict = json.load(f)
        
        if role in role_dict:
            self.role_description = role_dict[role]
        else:
            self.role_description = role_description

    @staticmethod
    def from_json(
        role: str,
        role_description: Optional[str] = None,
        dirpath: str = "./persona_data/role_datasets",
        except_on_missing=False
    ) -> "RoleDataset":
        """
        Load a previously saved role dataset from a JSON file.

        This static method reconstructs a RoleDataset instance from a saved
        JSON file, including all generated content and metadata. It automatically
        configures the appropriate backend (openai / vllm / hf_local) based on
        the saved metadata.

        Args:
            role (str): The role name (used to construct filename)
            dirpath (str, optional): Directory path where the dataset is saved.
                The filename is assumed to be "{role}_dataset.json".
                Defaults to "./persona_data/role_datasets/".

        Returns:
            RoleDataset: A fully initialized RoleDataset instance with all
                previously generated data loaded

        Raises:
            FileNotFoundError: If the dataset file doesn't exist at the specified path
            json.JSONDecodeError: If the file contains invalid JSON
            KeyError: If required keys are missing from the JSON data
        """
        filepath = os.path.join(dirpath, f"{role}_dataset.json")

        if not os.path.exists(filepath):
            if except_on_missing:
                raise FileNotFoundError(f"Role dataset not found in filepath {filepath}")
            
            logger.info(f"Role dataset not found in filepath {filepath}. Initializing new role dataset...")
            dataset = RoleDataset(
                role=role,
                role_description=role_description
            )
            dataset.generate_dataset(save_to_json=True)
            return dataset
        
        dataset, loaded_from_json = AbstractDataset.from_json(cls=RoleDataset,concept=role, dirpath=dirpath)
        dataset.role = role
        
        if loaded_from_json:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset.positive_prompts = data["positive_prompts"]
        else:
            dataset.generate_dataset(save_to_json=True)
            return dataset

        # Load up loggers
        dataset.logger = setup_logging(name="persona-dataset")
        dataset.logger.info("Dataset loaded from: %s", filepath)

        return dataset

    def save_dataset_to_json(self, dirpath: str | None = None) -> str:
        """
        Save the role dataset to a JSON file.

        Serializes the dataset including all generated content (instruction pairs,
        questions, evaluation prompt) and metadata (role, model, backend) to
        a JSON file. The directory structure is created automatically if it doesn't exist.

        Args:
            dirpath (str, optional): Directory path where the dataset should be saved.
                The filename will be automatically generated as "{role}_dataset.json".
                Defaults to self.dirpath.

        Returns:
            str: The full path to the saved JSON file

        Example:
            >>> dataset = RoleDataset(role="lawyer", num_questions=5)
            >>> # ... generate dataset ...
            >>> path = dataset.save_dataset_to_json("./my_datasets/")
            >>> print(f"Saved to: {path}")
            "Saved to: ./my_datasets/lawyer_dataset.json"
        """
        dataset_dict = self._get_baseline_dataset_dict()
        dataset_dict["positive_prompts"] = self.positive_prompts
        return super().save_dataset_to_json(dataset_dict=dataset_dict)

    def generate_dataset(
        self, save_to_json=True, max_tries=5
    ) -> Tuple[List[Tuple[str, str]], List[str], str, str | None]:
        """
        Generate the complete role dataset.

        This is the main method that orchestrates the entire dataset generation
        process. It generates role descriptions, question instructions, and then
        uses these to create positive/negative instruction pairs, questions, and
        an evaluation prompt. The generated dataset is automatically saved to JSON.

        Args:
            save_to_json (bool): Whether to save the generated dataset to a JSON file.
                Defaults to True.
            max_tries (int): Maximum number of attempts to parse valid dataset output
                from the LLM. Defaults to 5.

        Returns:
            Tuple[List[Tuple[str, str]], List[str], str, str]: A tuple containing:
                - positive_prompts: List of (positive_instruction, negative_instruction) tuples
                - questions: List of evaluation questions
                - evaluation_prompt: The prompt for evaluating the role
                - filepath: Path where the dataset was saved

        Raises:
            ValueError: If the dataset output cannot be parsed after max_tries attempts

        Example:
            >>> dataset = RoleDataset(role="doctor", num_questions=10)
            >>> pairs, questions, eval_prompt, path = dataset.generate_dataset()
            >>> print(f"Generated {len(pairs)} instruction pairs")
            >>> print(f"Dataset saved to: {path}")
        """
        # Generate trait description and question instruction
        if self.role_description is None:
            role_description: str = self._generate_role_description()
            logger.info("Generated role description for %s", self.role)
        else:
            role_description = self.role_description

        question_instruction: str = self._generate_question_instruction()
        logger.info("Generated question instruction for %s", self.role)

        # load generation prompt
        system_prompt: str = "You are an expert AI evaluator and dataset designer."
        user_prompt: str = PromptTemplates.PROMPTS["generate_role"].format(
            ROLE=self.role,
            N=self.num_questions,
            role_instruction=role_description,
            question_instruction=question_instruction,
        )

        # Generate response using the helper method
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with Heartbeat(logger, f"Generating dataset for {self.role}", interval=30):
            logger.info("Calling Inference with Client for dataset generation")
            _, response = self._inference_with_client(messages=messages)
            logger.info(response)

        # Parse the response into structured dataset components, retry maximum max_tries times if failed
        pos_prompts, questions, evaluation_prompt = None, None, None
        for _ in range(max_tries):
            try:
                pos_prompts, questions = self._parse_dataset_output(response=response)
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError("Failed to parse dataset output after multiple attempts")

        evaluation_prompt = PromptTemplates.evaluation_role(
            role=self.role, role_description=role_description
        )

        self.positive_prompts = pos_prompts
        self.questions = questions
        self.evaluation_prompt = evaluation_prompt

        filepath = None
        if save_to_json is True:
            # Save dataset to JSON
            filepath = self.save_dataset_to_json()

        return pos_prompts, questions, evaluation_prompt, filepath
    
    def extract_pos_question_pairs(self) -> List[Tuple[str, str]]:
        return [(p["pos"], q) for p in self.positive_prompts for q in self.questions]

    def _generate_role_description(self) -> str:
        """
        Generate a detailed description of the role.

        Uses the LLM to create a comprehensive description of the role being
        modeled. This description helps inform subsequent generation steps.

        Returns:
            str: A detailed description of the role
        """
        # Delegate to abstract method to avoid code duplication
        return self._generate_auxiliary_information(
            system_prompt="You are an expert AI evaluator and dataset designer.",
            prompts_key="role_instruction",
            ROLE=self.role
        )

    def _generate_question_instruction(self) -> str:
        """
        Generate instructions for creating evaluation questions for the role.

        Uses the LLM to create guidelines for generating appropriate evaluation
        questions that can test the presence of the role.

        Returns:
            str: Instructions for question generation
        """
        # Delegate to abstract method to avoid code duplication
        # Use 2^20 (1,048,576) tokens to allow for very comprehensive role question instructions
        return self._generate_auxiliary_information(
            system_prompt="You are an expert AI evaluator and dataset designer.",
            prompts_key="question_instruction_role",
            max_new_tokens=2**20,
            ROLE=self.role
        )

    def _parse_dataset_output(self, response: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Parse the LLM-generated JSON output into structured components.

        Extracts positive instructions, negative instructions, and questions.

        Args:
            response (str): The raw text response from the LLM containing the
                generated dataset in a JSON structured format

        Returns:
            Tuple[List[str], List[str]]: A tuple containing:
                - pos_prompt: List of positive instruction prompts
                - questions: List of evaluation questions

        Raises:
            json.JSONDecodeError: If the response does not contain valid JSON
        """
        json_text = response.strip("`").strip("json").strip()
        # json_text = "\n".join(response.split("\n")[1:-1])  # Remove ```json and ```
        response_dict = json.loads(json_text)

        # Extract components
        pos_instructions = response_dict["instruction"]
        questions = response_dict["questions"]
        
        logger.info("Parsed Dataset Successfully")

        return pos_instructions, questions
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument("-r", "--roles", nargs="+", help="<Optional> Select traits", required=False)
    ap.add_argument(
        "-d",
        "--role_description",
        default=None,
        help="<Optional> description of the trait to generate a dataset for.",
    )
    ap.add_argument(
        "-f",
        "--dirpath",
        default="./persona_data/role_datasets/",
        help="Directory filepath to save generated datasets",
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
        default="TOGETHER_API_KEY",
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
        help="Number of questions to generate per role",
    )
    args = ap.parse_args()

    roles_list = args.roles or [
        "lawyer",
        "doctor",
        "nurse"
    ]

    for role in roles_list:
        try:
            try:
                RoleDataset.from_json(
                    role=role,
                    dirpath=args.dirpath,
                    except_on_missing=True
                )
            except Exception:
                dataset: RoleDataset = RoleDataset(
                    role=role,
                    num_questions=args.num_questions,
                    backend=args.backend,
                    base_url=args.base_url,
                    local_model=args.local_model,
                    model=args.model,
                    api_key_env=args.api_key_env,
                )
                dataset.generate_dataset(save_to_json=True)
        except Exception as e:
            logger.error(f"Failed to generate dataset for role {role}: {e}")
            logger.error("Continuing to next role...")