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
    - ollama: For local model inference
    - openai: For API-based model inference
    - transformers: For model and tokenizer utilities
    
Environment Variables:
    - LITELLM_API_KEY: Required when using OpenAI-compatible endpoints
"""

import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ollama
from typing import List, Tuple, Dict, Optional
import os
from . import prompts
PROMPTS = prompts.PROMPTS
from openai import OpenAI
import re

class PersonaDataset:
    """
    A dataset class for generating and managing persona-based training data.
    
    This class facilitates the creation of datasets containing positive/negative
    instruction pairs, questions, and evaluation prompts for specific personality
    traits. It supports multiple inference backends (Ollama and OpenAI-compatible APIs)
    and provides methods for dataset persistence and retrieval.
    
    Attributes:
        trait (str): The personality trait being modeled (e.g., "sarcastic", "verbose")
        num_questions (int): Number of questions to generate for the dataset
        positive_negative_pairs (List[Tuple[str, str]]): List of (positive, negative) 
            instruction pairs
        questions (List[str]): List of evaluation questions
        evaluation_prompt (str): The evaluation prompt for assessing the trait
        model (str): The name/identifier of the LLM model to use
        client (Optional[ollama.Client]): Ollama client for local inference, or None 
            for API-based inference
    
    Example:
        >>> dataset = PersonaDataset(
        ...     trait="sarcastic", 
        ...     num_questions=5, 
        ...     client=None,
        ...     model='openai/gpt-4'
        ... )
        >>> pairs, questions, eval_prompt, filepath = dataset.generate_dataset()
    """

    def __init__(self, trait: str, num_questions: int, client: Optional[ollama.Client] = None, model: str="qwen2.5:7b-instruct"):
        """
        Initialize a PersonaDataset instance.
        
        Args:
            trait (str): The personality trait to generate data for (e.g., "sarcastic",
                "verbose", "analytical")
            num_questions (int): The number of questions to generate for evaluation
            client (Optional[ollama.Client], optional): An Ollama client instance for
                local model inference. If None, will use OpenAI-compatible API endpoint.
                Defaults to None.
            model (str, optional): The model identifier to use for generation.
                Defaults to "qwen2.5:7b-instruct".
        
        Returns:
            None
        """
        self.trait: str= trait
        self.num_questions: int = num_questions
        self.positive_negative_pairs: List[Tuple[str, str]] = []
        self.questions: List[str] = []
        self.evaluation_prompt = ""
        self.model = model

        self.client = client
    
    def inference_with_client(self, messages: List[Dict[str, str]], temperature: float = 0.9) -> Tuple[Optional[str], str]:
        """
        Perform LLM inference using either Ollama or OpenAI-compatible API.
        
        This method abstracts away the differences between local (Ollama) and
        remote (OpenAI-compatible) inference, providing a unified interface.
        
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
            ValueError: If LITELLM_API_KEY environment variable is not set when
                using OpenAI-compatible API (client is None)
            Exception: Re-raises any exception from the underlying API call after
                logging the error details
        
        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> thinking, response = dataset.inference_with_client(messages)
        """
        try:
            if self.client is not None:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={'temperature': temperature}
                ).message
                if hasattr(response, 'thinking'):
                    return (response.thinking, response.content)
                else:
                    return (None, response.content)
            else:
                base_url = "https://glados.ctisl.gtri.org"
                if "LITELLM_API_KEY" not in os.environ:
                    raise ValueError("LITELLM_API_KEY environment variable not set")
                api_key = os.environ["LITELLM_API_KEY"]
                openai_client = OpenAI(api_key=api_key, base_url=base_url)

                chat_response = openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return (None, chat_response.choices[0].message.content)
        except Exception as e:
            print(f'[red]util/inference :: messages: {messages}\nclient: {self.client is not None}\nmodel: {self.model}, temp: {temperature}[/]')
            raise e
    
    def generate_trait_description(self) -> str:
        """
        Generate a detailed description of the personality trait.
        
        Uses the LLM to create a comprehensive description of the trait being
        modeled. This description helps inform subsequent generation steps.
        
        Returns:
            str: A detailed description of the personality trait
        
        Example:
            >>> dataset = PersonaDataset(trait="sarcastic", num_questions=5)
            >>> description = dataset.generate_trait_description()
            >>> print(description)
            "Sarcastic communication involves using irony..."
        """
        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = PROMPTS["trait_instruction"].format(
            TRAIT=self.trait
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate response using the helper method
        _, response = self.inference_with_client(messages=messages)

        return response
    
    def generate_question_instruction(self) -> str:
        """
        Generate instructions for creating evaluation questions.
        
        Uses the LLM to create guidelines for generating appropriate evaluation
        questions that can test the presence of the personality trait.
        
        Returns:
            str: Instructions for question generation
        
        Example:
            >>> dataset = PersonaDataset(trait="verbose", num_questions=3)
            >>> instructions = dataset.generate_question_instruction()
            >>> print(instructions)
            "Create questions that allow for both concise and detailed responses..."
        """
        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = PROMPTS["question_instruction"].format(
            TRAIT=self.trait
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate response using the helper method
        _, response = self.inference_with_client(messages=messages)

        return response
    
    def generate_dataset(self) -> Tuple[List[Tuple[str, str]], List[str], str, str]:
        """
        Generate the complete persona dataset.
        
        This is the main method that orchestrates the entire dataset generation
        process. It generates trait descriptions, question instructions, and then
        uses these to create positive/negative instruction pairs, questions, and
        an evaluation prompt. The generated dataset is automatically saved to JSON.
        
        Returns:
            Tuple[List[Tuple[str, str]], List[str], str, str]: A tuple containing:
                - positive_negative_pairs: List of (positive_instruction, negative_instruction) tuples
                - questions: List of evaluation questions
                - evaluation_prompt: The prompt for evaluating the trait
                - filepath: Path where the dataset was saved
        
        Example:
            >>> dataset = PersonaDataset(trait="analytical", num_questions=10)
            >>> pairs, questions, eval_prompt, path = dataset.generate_dataset()
            >>> print(f"Generated {len(pairs)} instruction pairs")
            >>> print(f"Dataset saved to: {path}")
        """
        trait_description = self.generate_trait_description()
        question_instruction = self.generate_question_instruction()

        system_prompt = "You are an expert AI evaluator and dataset designer."
        user_prompt = PROMPTS["generate_trait"].format(
            TRAIT=self.trait,
            N=self.num_questions,
            trait_instruction=trait_description,
            question_instruction=question_instruction
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        _, response = self.inference_with_client(messages=messages)

        pos_neg_pairs, questions, evaluation_prompt = self.parse_dataset_output(response=response)
        self.positive_negative_pairs = pos_neg_pairs
        self.questions = questions
        self.evaluation_prompt = evaluation_prompt

        filepath = self.save_dataset_to_json()
        
        return pos_neg_pairs, questions, evaluation_prompt, filepath
    
    def extract_pos_neg_question_pairs(self) -> List[Tuple]:
        """
        Extract all combinations of (positive, negative, question) from trait data.
        
        Creates the Cartesian product of positive/negative instruction pairs and
        questions, producing all possible combinations for training. Each combination
        is represented as a named tuple with 'pos', 'neg', and 'question' attributes.
        
        Returns:
            List[namedtuple]: A list of Pair named tuples, where each Pair has:
                - pos (str): The positive instruction
                - neg (str): The negative instruction
                - question (str): The evaluation question
        
        Example:
            >>> dataset = PersonaDataset.load_dataset_from_json("sarcastic")
            >>> pairs = dataset.extract_pos_neg_question_pairs()
            >>> for pair in pairs[:2]:
            ...     print(f"Q: {pair.question}")
            ...     print(f"Pos: {pair.pos}")
            ...     print(f"Neg: {pair.neg}")
        """
        from collections import namedtuple
        
        Pair = namedtuple('Pair', ['pos', 'neg', 'question'])
        pairs = []
        
        
        # Create pairs by pairing each instruction pair with each question
        for instruction_pair in self.positive_negative_pairs:
            for question in self.questions:
                pair = Pair(
                    pos=instruction_pair[0],
                    neg=instruction_pair[1],
                    question=question
                )
                pairs.append(pair)
        
        return pairs
    
    def save_dataset_to_json(self, filepath: str = "./persona_dataset/") -> str:
        """
        Save the dataset to a JSON file.
        
        Serializes the dataset including all generated content (instruction pairs,
        questions, evaluation prompt) and metadata (trait, model, client type) to
        a JSON file. The directory structure is created automatically if it doesn't exist.
        
        Args:
            filepath (str, optional): Directory path where the dataset should be saved.
                The filename will be automatically generated as "{trait}_dataset.json".
                Defaults to "./persona_dataset/".
        
        Returns:
            str: The full path to the saved JSON file
        
        Example:
            >>> dataset = PersonaDataset(trait="verbose", num_questions=5)
            >>> # ... generate dataset ...
            >>> path = dataset.save_dataset_to_json("./my_datasets/")
            >>> print(f"Saved to: {path}")
            "Saved to: ./my_datasets/verbose_dataset.json"
        """
        filepath += f"{self.trait}_dataset.json"
        
        # Convert client to string representation
        client_type = "ollama" if self.client is not None else "other"
        
        dataset_dict = {
            "trait": self.trait,
            "num_questions": self.num_questions,
            "model": self.model,
            "client": client_type,
            "positive_negative_pairs": self.positive_negative_pairs,
            "questions": self.questions,
            "evaluation_prompt": self.evaluation_prompt
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to: {filepath}")
        return filepath

    def parse_dataset_output(self, response: str) -> Tuple[List[Tuple[str, str]], List[str], str]:
        """
        Parse the LLM-generated dataset output into structured components.
        
        Extracts positive instructions, negative instructions, questions, and
        evaluation prompts from the LLM response by looking for specific XML-like
        tags (<pos_instruction>, <neg_instruction>, <questions>, <eval_prompt>).
        Each section is expected to contain bulleted items starting with "- ".
        
        Args:
            response (str): The raw text response from the LLM containing the
                generated dataset in a structured format
        
        Returns:
            Tuple[List[Tuple[str, str]], List[str], str]: A tuple containing:
                - pos_neg_pairs: List of (positive, negative) instruction tuples
                - questions: List of evaluation questions
                - eval_prompt: The evaluation prompt (concatenated if multi-line)
        
        Note:
            - The function expects XML-like tags in the response
            - Bulleted items should start with "- "
            - Pairs are created by matching indices of positive and negative instructions
        
        """
        # Extract pos_instruction section
        pos_match = re.search(r'<pos_instruction>(.*?)</pos_instruction>', response, re.DOTALL)
        pos_instructions = []
        if pos_match:
            pos_text = pos_match.group(1).strip()
            # Split by lines that start with "- "
            pos_instructions = [line.strip()[2:] for line in pos_text.split('\n') if line.strip().startswith('- ')]
        
        # Extract neg_instruction section
        neg_match = re.search(r'<neg_instruction>(.*?)</neg_instruction>', response, re.DOTALL)  # Note the typo in closing tag
        neg_instructions = []
        if neg_match:
            neg_text = neg_match.group(1).strip()
            neg_instructions = [line.strip()[2:] for line in neg_text.split('\n') if line.strip().startswith('- ')]
        
        # Extract questions section
        questions_match = re.search(r'<questions>(.*?)</questions>', response, re.DOTALL)
        questions = []
        if questions_match:
            questions_text = questions_match.group(1).strip()
            questions = [line.strip()[2:] for line in questions_text.split('\n') if line.strip().startswith('- ')]
        
        # Extract eval_prompt section
        eval_match = re.search(r'<eval_prompt>(.*?)</eval_prompt>', response, re.DOTALL)
        eval_prompt = ""
        if eval_match:
            eval_text = eval_match.group(1).strip()
            eval_lines = [line.strip()[2:] for line in eval_text.split('\n') if line.strip().startswith('- ')]
            eval_prompt = " ".join(eval_lines)  # Join multiple lines if needed
        
        # Create pos/neg pairs by matching indices
        pos_neg_pairs = []
        min_len = min(len(pos_instructions), len(neg_instructions))
        for i in range(min_len):
            pos_neg_pairs.append((pos_instructions[i], neg_instructions[i]))
        
        return pos_neg_pairs, questions, eval_prompt

    @staticmethod
    def load_dataset_from_json(trait: str, filepath: str = "./persona_dataset/") -> 'PersonaDataset':
        """
        Load a previously saved dataset from a JSON file.
        
        This static method reconstructs a PersonaDataset instance from a saved
        JSON file, including all generated content and metadata. It automatically
        configures the appropriate client type (Ollama or API-based) based on
        the saved metadata.
        
        Args:
            trait (str): The personality trait name (used to construct filename)
            filepath (str, optional): Directory path where the dataset is saved.
                The filename is assumed to be "{trait}_dataset.json".
                Defaults to "./persona_dataset/".
        
        Returns:
            PersonaDataset: A fully initialized PersonaDataset instance with all
                previously generated data loaded
        
        Raises:
            FileNotFoundError: If the dataset file doesn't exist at the specified path
            json.JSONDecodeError: If the file contains invalid JSON
            KeyError: If required keys are missing from the JSON data
        
        Example:
            >>> dataset = PersonaDataset.load_dataset_from_json(
            ...     trait="analytical",
            ...     filepath="./saved_datasets/"
            ... )
            >>> print(f"Loaded {len(dataset.questions)} questions")
            >>> print(f"Loaded {len(dataset.positive_negative_pairs)} instruction pairs")
        """
        filepath += f"{trait}_dataset.json"

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
        num_questions = data["num_questions"]
        client = data["client"]
        model = data["model"]

        if client == 'ollama':
            client_type = ollama.Client()
        else:
            client_type = None

        dataset = PersonaDataset(trait=trait, num_questions=num_questions, client=client_type, model=model)
        dataset.positive_negative_pairs = data["positive_negative_pairs"]
        dataset.questions = data["questions"]
        dataset.evaluation_prompt = data["evaluation_prompt"]
        
        print(f"Dataset loaded from: {filepath}")
        return dataset


if __name__ == "__main__":
    trait_list = [
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

    dataset: PersonaDataset = PersonaDataset(trait="sarcastic", num_questions=100, client=None, model='openai/gpt-oss-120b')
    
    pos_neg_pairs, questions, evaluation_prompt, _ = dataset.generate_dataset()
    print(pos_neg_pairs)
    print(questions)
    print(evaluation_prompt)