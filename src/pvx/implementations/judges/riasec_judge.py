import argparse
from ast import Dict, List
from collections import defaultdict
import json
import re
from pathlib import Path
from tqdm import tqdm

import yaml

from pvx import setup_logging
from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.abstraction.judges.abstract_judge import AbstractJudge
from pvx.implementations.base.persona_model import PersonaModel
from pvx.implementations.riasec.riasec_persona_model import RIASECPersonaModel
from pvx.implementations.roles.role_persona_model import RolePersonaModel

logger = setup_logging(name="riasec-judge")


class RIASECJudge(AbstractJudge):
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
        
        ITEMS_PATH = "./configs/scoring/interest_profiler.json"  # or absolute if needed
        with open(ITEMS_PATH, "r", encoding="utf-8") as f:
            self.ITEMS: List[Dict[str, str]] = json.load(f)

        assert len(self.ITEMS) == 60, "Expected 60 items"

        self.DIM_TO_INDICES = defaultdict(list)
        for i, item in enumerate(self.ITEMS):
            self.DIM_TO_INDICES[item["dimension"]].append(i)

        assert len(self.DIM_TO_INDICES) == 6 and all(len(v) == 10 for v in self.DIM_TO_INDICES.values())
    
    def _get_system_messagesv2(self, activity: str) -> list:
        base = (
            "You are answering the O*NET Interest Profiler. "
            "For this work activity, would you LIKE to do it as part of a job? "
            "Answer ONLY with 'Yes' or 'No' — no explanation.\n\n"
        )
            
        return [
            {"role": "system", "content": base},
            {"role": "user", "content": f"Activity: {activity}"},
        ]
        
    def compute_scores(self, responses: list[int]) -> dict[str, int]:
        """Sum Yes per dimension"""
        if len(responses) != 60:
            raise ValueError("Need 60 responses")
        scores = {dim: sum(responses[i] for i in indices) for dim, indices in self.DIM_TO_INDICES.items()}
        return scores

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
            {"role": "system", "content": "You are a human. Output EXACTLY one token: YES or NO."},
            {"role": "user", "content": f"{characteristic}"},
        ]
    def judgev2(
        self,
        persona_model: AbstractPersonaModel,
        alpha: float = 0.3,
        max_new_tokens: int = 50,
        temperature: float = 0.1
    ):
        responses = []
        answers = {}
        with tqdm(total=60, desc="Querying items") as pbar:
            for item in self.ITEMS:
                messages = self._get_system_messagesv2(activity=item['text'])
                raw_ans = persona_model.generate(
                    messages=messages,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ).strip()
                like = 1 if raw_ans.strip().lower().startswith("yes") else 0
                responses.append(like)
                answers[item['text']] = raw_ans
                logger.debug(f"Item {item['item_id']:2d} ({item['dimension'][:3]}): {like} ({raw_ans})")
                pbar.update(1)
        return responses, answers
    def judge(
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
    
    def __call__(
        self,
        persona_model: AbstractPersonaModel,
        alpha: float = 0.7,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
    ):
        # return self.judge(persona_model=persona_model, alpha=alpha, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.judgev2(persona_model=persona_model, alpha=alpha, max_new_tokens=max_new_tokens, temperature=temperature)


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
        "-c", "--concept", type=str, default="realistic", help="Concept of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=0.7, help="Alpha value for persona steering"
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
    ap.add_argument(
        "-r",
        "--print_results",
        action='store_true',
        help="Print output results of steering model",
    )
    args = ap.parse_args()

    pvx = RolePersonaModel.load_or_create(
        concept=args.concept,
        target_model_id=args.model_name,
        layer=14,
        json_filepath=args.json_filepath,
    )

    riasec_judge = RIASECJudge(riasec_yaml_path=args.yaml_path)
    # results, counts = riasec_judge(
    #     pvx, args.alpha, args.max_new_tokens, args.temperature
    # )
    responses, answers = riasec_judge(
        pvx, args.alpha, args.max_new_tokens, args.temperature
    )
    scores = riasec_judge.compute_scores(responses=responses)
    logger.info("===CONCEPT===")
    logger.info(args.concept)
    if args.print_results:
        logger.info("===RESULTS===")
        logger.info(answers)
    logger.info("===RIASEC Counts===")
    logger.info(scores)
