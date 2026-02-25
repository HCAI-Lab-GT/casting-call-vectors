"""
score_riasec.py

O*NET Interest Profiler (Short Form) LLM-based simulator & scorer.
- Loads interest_profiler.json (60 items)
- For each item, queries an LLM: "Would [persona] like to do this? Yes/No"
- Scores: # of "Yes" per RIASEC dimension (0–10)
- Supports persona descriptions for simulation
- Simple test modes (all Yes/No, fixed responses)

Location: personality-vectors/tom_scoring/score_riasec.py
"""

import json
from collections import defaultdict
from typing import List, Dict, Callable, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger("score_riasec")
logging.basicConfig(level=logging.INFO)

# Load items (adjust path relative to tom_scoring/)
ITEMS_PATH = "../interest_profiler.json"  # or absolute if needed
with open(ITEMS_PATH, "r", encoding="utf-8") as f:
    ITEMS: List[Dict[str, str]] = json.load(f)

assert len(ITEMS) == 60, "Expected 60 items"

DIM_TO_INDICES = defaultdict(list)
for i, item in enumerate(ITEMS):
    DIM_TO_INDICES[item["dimension"]].append(i)

assert len(DIM_TO_INDICES) == 6 and all(len(v) == 10 for v in DIM_TO_INDICES.values())


def build_prompt(item_text: str, persona: str = "") -> str:
    """Prompt to get strict Yes/No. Persona makes it conditional."""
    base = (
        "You are answering the O*NET Interest Profiler. "
        "For this work activity, would you LIKE to do it as part of a job? "
        "Answer ONLY with 'Yes' or 'No' — no explanation.\n\n"
        f"Activity: {item_text}"
    )
    if persona:
        base = f"You are: {persona.strip()}\n\n" + base
    return base


def default_query_llm(prompt: str) -> str:
    """Placeholder — replace with your real LLM call (e.g. from pvx or openai/groq)."""
    # For debugging/manual: 
    print(f"\nPrompt:\n{prompt}\n")
    ans = input("LLM answer (Yes/No): ").strip().lower()
    return "Yes" if "yes" in ans else "No"

    # Real example stub (uncomment & configure):
    # from openai import OpenAI
    # client = OpenAI()
    # resp = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=5,
    #     temperature=0.0
    # )
    # return resp.choices[0].message.content.strip()


def collect_responses(
    persona: str = "",
    prompt_builder: Callable = build_prompt,
    query_llm: Callable = default_query_llm
) -> List[int]:
    """Query LLM for all 60 items → list of 0/1 (1 = Yes/like)"""
    responses = []
    with tqdm(total=60, desc="Querying items") as pbar:
        for item in ITEMS:
            prompt = prompt_builder(item["text"], persona)
            raw_ans = query_llm(prompt)
            like = 1 if raw_ans.strip().lower().startswith("yes") else 0
            responses.append(like)
            logger.debug(f"Item {item['item_id']:2d} ({item['dimension'][:3]}): {like} ({raw_ans})")
            pbar.update(1)
    return responses


def compute_scores(responses: List[int]) -> Dict[str, int]:
    """Sum Yes per dimension"""
    if len(responses) != 60:
        raise ValueError("Need 60 responses")
    scores = {dim: sum(responses[i] for i in indices) for dim, indices in DIM_TO_INDICES.items()}
    return scores


def score_riasec(
    persona: str = "a typical adult taking a career interest test",
    save_responses: Optional[str] = None
) -> Dict[str, int]:
    """
    Main entry point.
    - persona: description to condition the LLM (e.g. "You are a very practical, hands-on person who enjoys building things.")
    - save_responses: optional path to save the 0/1 list as JSON
    """
    logger.info(f"Scoring RIASEC for persona: {persona or 'neutral'}")
    binary = collect_responses(persona=persona)
    
    if save_responses:
        with open(save_responses, "w", encoding="utf-8") as f:
            json.dump(binary, f, indent=2)
        logger.info(f"Responses saved to {save_responses}")
    
    scores = compute_scores(binary)
    logger.info("RIASEC Scores (0-10):")
    for dim, score in sorted(scores.items()):
        logger.info(f"  {dim:12}: {score}")
    
    return scores


# ────────────────────────────────────────────────
# Simple validation / test modes
# ────────────────────────────────────────────────

def test_all_yes():
    responses = [1] * 60
    return compute_scores(responses)  # all 10


def test_all_no():
    responses = [0] * 60
    return compute_scores(responses)  # all 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("All Yes:", test_all_yes())
        print("All No :", test_all_no())
    else:
        persona_input = input("Enter persona description (or Enter for default): ").strip()
        score_riasec(persona=persona_input)
