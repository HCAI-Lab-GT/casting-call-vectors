import argparse
import re
from pathlib import Path

import yaml

from pvx import setup_logging
from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.riasec.riasec_persona_model import RIASECPersonaModel

logger = setup_logging(name="riasec-judge")


class RIASECJudge:
    """
    Judge for evaluating persona models on RIASEC traits.

    Loads RIASEC trait definitions from a YAML file and provides methods to evaluate
    a persona model's responses to trait-specific characteristics.
    """

    def __init__(self, riasec_yaml_path: str = "./configs/riasec.yaml"):
        """
        Initialize the RIASECJudge.

        Args:
            riasec_yaml_path (str): Path to the RIASEC YAML configuration file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """

        yaml_path = Path(riasec_yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"RIASEC YAML file not found: {riasec_yaml_path}")

        with open(yaml_path, "r") as f:
            self.riasec_data = yaml.safe_load(f)

        self.trait_dict = {}
        for trait, info in self.riasec_data.items():
            desc = info.get("description")
            characteristics = info.get("characteristics", [])
            self.trait_dict[trait] = {"description": desc, "characteristics": characteristics}

    def _get_system_messages(self, characteristic: str) -> list:
        """
        Constructs a list of system and user messages for prompting the model.
        Args:
            description (str): The description of the RIASEC trait.
            characteristic (str): The characteristic question/statement.
        Returns:
            list: List of message dicts for use with chat-based models.
        """
        # "Output EXACTLY one token: YES or NO. Answer YES or NO for the following characteristic."
        return [
            {"role": "system", "content": "Output EXACTLY one token: YES or NO."},
            {"role": "user", "content": f"{characteristic}"},
        ]

    def evaluate_persona(
        self,
        persona_model: AbstractPersonaModel,
        alpha: float = 0.3,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
    ):
        """
        Evaluate the persona model on all RIASEC traits and their characteristics.

        For each trait and characteristic, generates a response and classifies it as YES or NO

        Args:
            persona_model (PersonaModel): The persona model to evaluate.
            alpha (float): Alpha value for persona steering.
            max_new_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            tuple: (results, counts)
                results (dict): trait -> {characteristic: {"response": str, "judgment": str}}
                counts (dict): trait -> count of YES responses
        """
        results = {}
        counts = {}
        yes_pattern = re.compile(
            r"\b(yes|yeah|yep|affirmative|certainly|of course|absolutely)\b", re.IGNORECASE
        )
        no_pattern = re.compile(r"\b(no|nope|not at all|never|negative)\b", re.IGNORECASE)

        for trait, info in self.trait_dict.items():
            characteristics = info["characteristics"]
            trait_results = {}
            for characteristic in characteristics:
                messages = self._get_system_messages(characteristic)
                response = persona_model.generate(
                    messages=messages,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
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
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model typically",
    )
    ap.add_argument(
        "-t", "--trait", type=str, default="realistic", help="Trait of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=0.3, help="Alpha value for persona steering"
    )
    ap.add_argument("-p", "--temperature", type=float, default=0.1, help="Temperature for sampling")
    ap.add_argument(
        "-y",
        "--yaml_path",
        type=str,
        default="./configs/riasec.yaml",
        help="YAML Path for RIASEC information.",
    )
    ap.add_argument(
        "-f",
        "--json_filepath",
        type=str,
        default=None,
        help="Filepath to model init file (not trait dataset)",
    )
    args = ap.parse_args()

    pvx = RIASECPersonaModel.load_or_create(
        target_model_id=args.model_name,
        trait=args.trait,
        layer=14,
        json_filepath=args.json_filepath,
    )

    riasec_judge = RIASECJudge(riasec_yaml_path=args.yaml_path)
    results, counts = riasec_judge.evaluate_persona(
        pvx, args.alpha, args.max_new_tokens, args.temperature
    )
    logger.info("===TRAIT===")
    logger.info(args.trait)
    logger.info("===RESULTS===")
    logger.info(results)
    logger.info("===RIASEC Counts===")
    logger.info(counts)
