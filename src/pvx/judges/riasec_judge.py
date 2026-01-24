import os
import yaml
from pathlib import Path
from typing import Optional
import argparse
import re

from pvx import setup_logging, Heartbeat
from pvx.pvx_models.persona_model import PersonaModel

logger = setup_logging(name="riasec-judge")

class RIASECJudge:
    def __init__(self, riasec_yaml_path: str = './configs/riasec.yaml'):
        yaml_path = Path(riasec_yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"RIASEC YAML file not found: {riasec_yaml_path}")
        
        with open(yaml_path, 'r') as f:
            self.riasec_data = yaml.safe_load(f)
        
        self.trait_dict = {}
        for trait, info in self.riasec_data.items():
            desc = info.get('description')
            characteristics = info.get('characteristics', [])
            self.trait_dict[trait] = {
                'description': desc,
                'characteristics': characteristics
            }
    
    def _get_system_messages(self, trait: str, description: str, characteristic: str) -> list:
        """
        Constructs a list of system and user messages for prompting the model.
        Args:
            description (str): The description of the RIASEC trait.
            characteristic (str): The characteristic question/statement.
        Returns:
            list: List of message dicts for use with chat-based models.
        """
        return [
            {"role": "system", "content": "Answer YES or NO for the following characteristic."},
            {"role": "user", "content": f"{characteristic}"}
        ]

    def evaluate_persona(self, persona_model: PersonaModel, alpha: float = 0.3, max_new_tokens: int = 50, temperature: float = 0.1):
        """
        Evaluates the given persona model on all RIASEC traits and their characteristics.
        For each trait and characteristic, generates a response and classifies it as YES or NO using regex.
        Returns a dictionary: trait -> {characteristic: "YES"/"NO", ...}
        """
        results = {}
        counts = {}
        yes_pattern = re.compile(r"\b(yes|yeah|yep|affirmative|certainly|of course|absolutely)\b", re.IGNORECASE)
        no_pattern = re.compile(r"\b(no|nope|not at all|never|negative)\b", re.IGNORECASE)

        for trait, info in self.trait_dict.items():
            description = info['description']
            characteristics = info['characteristics']
            trait_results = {}
            for characteristic in characteristics:
                messages = self._get_system_messages(trait, description, characteristic)
                response = persona_model.generate(
                    messages=messages,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                ).strip()
                # Classify as YES or NO
                if yes_pattern.search(response):
                    answer = "YES"
                    if trait not in counts:
                        counts[trait] = 0
                    counts[trait] += 1
                elif no_pattern.search(response):
                    answer = "NO"
                else:
                    answer = "UNKNOWN"
                trait_results[characteristic] = {"response": response, "judgment": answer}
            results[trait] = trait_results
        return results, counts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model typically")
    ap.add_argument("-t", "--trait", type=str, default="humorous", help="Trait of the persona dataset")
    ap.add_argument("-n", "--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    ap.add_argument("-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering")
    ap.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    ap.add_argument("-q", "--question", type=str, default="What is the theory of relativity?", help="Question to generate response for")
    ap.add_argument("-f", "--json_filepath", type=str, default=None, help="Filepath to model init file (not trait dataset)")
    args = ap.parse_args()
    
    pvx = PersonaModel.load_or_create(
        target_model_id=args.model_name,
        trait=args.trait,
        layer=14,
        json_filepath=args.json_filepath
    )
    
    riasec_judge = RIASECJudge()
    results, counts = riasec_judge.evaluate_persona(pvx, args.alpha, args.max_new_tokens, args.temperature)