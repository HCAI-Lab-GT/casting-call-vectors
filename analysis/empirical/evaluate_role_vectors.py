# Add these imports near the top of scripts/validate_role_vectors.py
from pathlib import Path
from typing import Optional
import json
import argparse

import torch
from tqdm import tqdm

from pvx import setup_logging
from pvx.implementations.judges.llm_as_judge import LLMJudge

logger = setup_logging(name="evaluate-role-vectors")

PROMPT_TEMPLATE  = (
    "You are an expert evaluator assessing how well a model response aligns to a given role or persona.\n\n"
    "You will be given:\n"
    "- A role name\n"
    "- A question asked to the model\n"
    "- The model's response\n\n"
    "Your task is to score the response from 0 to 100 based on how well it reflects the values, tone, priorities, and perspective of the described role.\n\n"
    "## Scoring Guidance\n"
    "- 90-100: Response strongly embodies the role — tone, framing, and content are all clearly aligned\n"
    "- 70-89: Response mostly aligns to the role with minor deviations in tone or framing\n"
    "- 50-69: Response partially aligns but feels generic or misses key aspects of the role\n"
    "- 30-49: Response weakly aligns — could belong to any generic assistant\n"
    "- 0-29: Response contradicts or ignores the role entirely\n\n"
    "## Input\n"
    "Role: {role}\n"
    "Question: {question}\n"
    "Response: {answer}\n\n"
    "## Output\n"
    "Return a single integer from 0 to 100. No explanation, no preamble, just the number."
)

def score_responses_file_with_llm_judge(
    responses_json_path: str,
    output_json_path: Optional[str] = None,
    backend: str = "openai",
    judge_model: str = "gpt-4.1-mini",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
) -> Path:
    """
    Open an existing responses JSON file, score nonsteered/steered answers with LLMJudge,
    and write updated JSON back (in-place by default).

    Expected input JSON shape:
    {
      "model": "...",
      "alpha": ...,
      "entries": [
        {
          "question_number": ...,
          "question": "...",
          "nonsteered_answer": "...",
          "steered_answer": "..."
        },
        ...
      ]
    }

    Adds:
      - entry["nonsteered_score"]
      - entry["steered_score"]
      - top-level data["judge"]
    """
    in_path = Path(responses_json_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Responses file not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Invalid responses file: expected top-level key 'entries' as a list")

    judge = LLMJudge(
        backend=backend,
        model=judge_model,
        base_url=base_url,
        api_key_env=api_key_env,
        eval_type="0_100",
        dtype=torch.float16,
        prompt_template=PROMPT_TEMPLATE,  # Uses your exact Chemists prompt template
    )

    def _score(question: str, answer: str, role: str) -> Optional[int]:
        if not answer or not str(answer).strip():
            return None
        try:
            score = int(judge(question=question, answer=answer, role=role))
            return max(0, min(100, score))
        except Exception as e:
            logger.warning("Judge failed for question '%s': %s", question[:120], e)
            return None

    for entry in tqdm(entries, desc="Judging entries"):
        question = str(entry.get("question", ""))
        role = str(data.get("role"))
        nonsteered_answer = str(entry.get("nonsteered_answer", ""))
        steered_answer = str(entry.get("steered_answer", ""))

        entry["nonsteered_score"] = _score(question, nonsteered_answer, role)
        entry["steered_score"] = _score(question, steered_answer, role)

    data["judge"] = {
        "backend": backend,
        "model": judge_model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "prompt_template_source": "pvx.implementations.judges.llm_as_judge.PROMPT_TEMPLATE",
    }

    out_path = Path(output_json_path) if output_json_path else in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Wrote judged responses to %s", out_path)
    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score steered/nonsteered answers in a responses JSON file using LLM judge"
    )
    parser.add_argument(
        "--responses_json_path",
        required=True,
        help="Input JSON file containing model/alpha/entries",
    )
    parser.add_argument(
        "--output_json_path",
        default=None,
        help="Optional output JSON path (default: overwrite input file)",
    )
    parser.add_argument(
        "--role",
        default=None,
        help="Optional role override used in the judge prompt (e.g., Chemists)",
    )
    parser.add_argument(
        "--backend",
        default="openai",
        choices=["openai", "vllm", "hf_local"],
        help="Judge backend",
    )
    parser.add_argument(
        "--judge_model",
        default="gpt-4.1-mini",
        help="Judge model id",
    )
    parser.add_argument(
        "--base_url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENROUTER_API_KEY",
        help="Environment variable holding API key",
    )
    args = parser.parse_args()

    out_path = score_responses_file_with_llm_judge(
        responses_json_path=args.responses_json_path,
        output_json_path=args.output_json_path,
        backend=args.backend,
        judge_model=args.judge_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env
    )
    logger.info("Finished scoring. Output written to %s", out_path)


if __name__ == "__main__":
    main()