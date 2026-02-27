import json
from collections import defaultdict
from typing import List, Dict, Any
import argparse
import sys
from tqdm import tqdm

from pvx import setup_logging
from pvx.abstraction.judges.abstract_judge import AbstractJudge
from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.roles.role_persona_model import RolePersonaModel

logger = setup_logging(name="hexaco_judge")

class HexacoJudge(AbstractJudge):
    def __init__(self, hexaco_path: str = "./configs/scoring/hexaco_100.json"):
        try:
            with open(hexaco_path, "r", encoding="utf-8") as f:
                self.ITEMS: List[Dict[str, Any]] = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("hexaco_100.json not found in current directory")

        if len(self.ITEMS) != 100:
            raise ValueError("hexaco_100.json must contain exactly 100 items")

        # Precompute helpers
        self.ITEM_KEYING = [1 if item["keyed"] == "+" else -1 for item in self.ITEMS]
        self.ITEM_FACET  = [item["facet"] for item in self.ITEMS]
        self.ITEM_DOMAIN = [item["domain"] for item in self.ITEMS]  # None for Altruism

        # facet → list of 0-based indices (exactly 4 per facet)
        self.facet_to_indices = defaultdict(list)
        for i, facet in enumerate(self.ITEM_FACET):
            self.facet_to_indices[facet].append(i)

        # domain → list of facets (4 per domain; Altruism excluded)
        self.domain_to_facets = defaultdict(list)
        for i, domain in enumerate(self.ITEM_DOMAIN):
            if domain:
                self.domain_to_facets[domain].append(self.ITEM_FACET[i])

        # Verify structure
        assert len(self.facet_to_indices) == 25, "Expected 25 facets"
        assert len(self.domain_to_facets) == 6,  "Expected 6 domains"
        assert all(len(indices) == 4 for indices in self.facet_to_indices.values()), "Each facet must have 4 items"
    
    def _get_system_messages(self, activity: str) -> list:
        base = (
            "Please answer the following question by responding with a single number from 1 to 5 (Likert Scale), where:\n"
            "1 = Strongly Disagree\n"
            "2 = Disagree\n"
            "3 = Neutral\n"
            "4 = Agree\n"
            "5 = Strongly Agree\n\n"
            "Ensure your answer is an integer between 1 to 5."
        )
            
        return [
            {"role": "system", "content": base},
            {"role": "user", "content": f"{activity}"},
        ]
        
    def judge(
        self,
        persona_model: AbstractPersonaModel,
        alpha: float = 0.3,
        max_new_tokens: int = 50,
        temperature: float = 0.1
    ):
        responses = []
        answers = {}
        with tqdm(total=100, desc="Querying items") as pbar:
            for item in self.ITEMS:
                messages = self._get_system_messages(activity=item['text'])
                raw_ans = persona_model.generate(
                    messages=messages,
                    alpha=alpha,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ).strip()
                score = int(raw_ans)
                responses.append(score)
                answers[item['item_id']] = score
                logger.debug(f"Item {item['item_id']:2d} ({item['domain']}): {score} ({raw_ans})")
                pbar.update(1)
        scores = self.score_hexaco_100(responses=responses)
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
    
    def score_hexaco_100(self, responses: List[int | float]) -> Dict[str, Dict[str, float]]:
        logger.info(responses)
        """
        Score a list of 100 HEXACO responses (1–5 Likert scale).

        Args:
            responses: list of 100 numbers (integers or floats) in [1, 5]

        Returns:
            dict with:
                "facets":    {facet_name: mean (1–5)}
                "domains":   {domain_name: mean (1–5)}
                "altruism":  mean score for Altruism facet (1–5)

        Raises:
            ValueError on invalid input length or values
        """
        if len(responses) != 100:
            raise ValueError(f"Expected exactly 100 responses, got {len(responses)}")

        # Validate range
        for i, r in enumerate(responses):
            if not isinstance(r, (int, float)) or not (1 <= r <= 5):
                raise ValueError(f"Invalid response at item {i+1}: {r} (must be 1.0–5.0)")

        # Apply reverse scoring
        scored = [
            r if key == 1 else (6 - r)
            for r, key in zip(responses, self.ITEM_KEYING)
        ]

        # Compute facet means
        facets = {}
        for facet, indices in self.facet_to_indices.items():
            values = [scored[i] for i in indices]
            facets[facet] = sum(values) / 4.0

        # Compute domain means (average of facet means)
        domains = {}
        for domain, facets_list in self.domain_to_facets.items():
            domain_values = [facets[f] for f in facets_list]
            domains[domain] = sum(domain_values) / len(domain_values)

        # Altruism (interstitial facet)
        altruism_mean = facets["Altruism"]

        return {
            "facets": facets,
            "domains": domains,
            "altruism": altruism_mean
        }
    
    def print_results(results: Dict):
        print("\nHEXACO-100 Scores (1–5 scale)\n")
        print("Domains:")
        for domain, score in sorted(results["domains"].items()):
            print(f"  {domain:20} : {score:.2f}")
        print(f"\nAltruism (interstitial)  : {results['altruism']:.2f}")
        print("\nSample facets (first 5):")
        for facet, score in list(results["facets"].items())[:5]:
            print(f"  {facet:20} : {score:.2f}")
        print("  ... (20 more facets)")

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
        default="./configs/scoring/hexaco_100.json",
        help="Config Path for Hexaco Judge information.",
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

    hexaco_judge = HexacoJudge(hexaco_path=args.config_path)
    
    responses, answers, scores = hexaco_judge(
        pvx, args.alpha, args.max_new_tokens, args.temperature
    )
    logger.info("===CONCEPT===")
    logger.info(args.concept)
    if args.print_results:
        logger.info("===RESULTS===")
        logger.info(answers)
    logger.info("===Hexaco Counts===")
    logger.info(scores)
