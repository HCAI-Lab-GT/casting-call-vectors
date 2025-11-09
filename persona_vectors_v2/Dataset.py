import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import PROMPTS
import torch
import ollama
from typing import List, Tuple, Dict, Optional
import os
from openai import OpenAI
import re

class PersonaDataset:

    def __init__(self, trait: str, num_questions: int, client: Optional[ollama.Client] = None, 
                 model: str="qwen2.5:7b-instruct"):
        self.trait: str= trait
        
        self.num_questions: int = num_questions

        self.positive_negative_pairs: List[Tuple[str, str]] = []
        self.questions: List[str] = []
        self.evaluation_prompt = ""
        self.model = model

        self.client = client
    
    def inference_with_client(self, messages: List[Dict[str, str]], temperature: float = 0.9) -> Tuple[str, str]:
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
    
    def generate_trait_description(self):
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
    
    def generate_question_instruction(self):
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
    
    def generate_dataset(self):
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
    
    def save_dataset_to_json(self, filepath: str = "./persona_dataset/"):
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
        Parse the generated dataset output into pos/neg pairs, questions, and eval prompt
        """
        # Extract pos_instruction section
        pos_match = re.search(r'<pos_instruction>(.*?)</pos_instruction>', response, re.DOTALL)
        pos_instructions = []
        if pos_match:
            pos_text = pos_match.group(1).strip()
            # Split by lines that start with "- "
            pos_instructions = [line.strip()[2:] for line in pos_text.split('\n') if line.strip().startswith('- ')]
        
        # Extract neg_instruction section
        neg_match = re.search(r'<neg_instruction>(.*?)</neg_insruction>', response, re.DOTALL)  # Note the typo in closing tag
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

    def load_dataset_from_json(trait: str, filepath: str = "./persona_dataset/"):
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
    dataset: PersonaDataset = PersonaDataset(trait="sarcastic", num_questions=5, client=ollama.Client())
    
    pos_neg_pairs, questions, evaluation_prompt, _ = dataset.generate_dataset()
    print(pos_neg_pairs)
    print(questions)
    print(evaluation_prompt)


