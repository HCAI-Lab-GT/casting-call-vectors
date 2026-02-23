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

    def __init__(self, riasec_judge_path: str = "./configs/scoring/interest_profiler.json"):
        """
        Initialize the RIASECJudge.

        Args:
            riasec_judge_path (str): Path to the RIASEC Judge Question configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        
        with open(riasec_judge_path, "r", encoding="utf-8") as f:
            self.ITEMS: List[Dict[str, str]] = json.load(f)

        assert len(self.ITEMS) == 60, "Expected 60 items"

        self.DIM_TO_INDICES = defaultdict(list)
        for i, item in enumerate(self.ITEMS):
            self.DIM_TO_INDICES[item["dimension"]].append(i)

        assert len(self.DIM_TO_INDICES) == 6 and all(len(v) == 10 for v in self.DIM_TO_INDICES.values())
    
    def _get_system_messages(self, activity: str) -> list:
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

    def judge(
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
                messages = self._get_system_messages(activity=item['text'])
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
        scores = riasec_judge.compute_scores(responses=responses)
        return responses, answers, scores
    
    def __call__(
        self,
        persona_model: AbstractPersonaModel,
        alpha: float = 0.7,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
    ):
        # return self.judge(persona_model=persona_model, alpha=alpha, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.judge(persona_model=persona_model, alpha=alpha, max_new_tokens=max_new_tokens, temperature=temperature)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="allenai/Olmo-3-7B-Instruct",
        help="HF model typically",
    )
    ap.add_argument(
        "-c", "--concept", type=str, default="realistic", help="Concept of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=10, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument("-p", "--temperature", type=float, default=0.1, help="Temperature for sampling")
    ap.add_argument(
        "-y",
        "--config_path",
        type=str,
        default="./configs/scoring/interest_profiler.json",
        help="Config Path for RIASEC Judge information.",
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
        layer=16,
        json_filepath=args.json_filepath,
    )

    riasec_judge = RIASECJudge(riasec_judge_path=args.config_path)
    
    responses, answers, scores = riasec_judge(
        pvx, args.alpha, args.max_new_tokens, args.temperature
    )
    logger.info("===CONCEPT===")
    logger.info(args.concept)
    if args.print_results:
        logger.info("===RESULTS===")
        logger.info(answers)
    logger.info("===RIASEC Counts===")
    logger.info(scores)
