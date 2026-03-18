"""
This module provides the LLMJudge class for evaluating model responses using LLMs as automated judges.
It supports multiple backends (OpenAI, vLLM, Hugging Face local) and can aggregate scores using direct inference or log probability methods.

Key Features:
    - Evaluate model responses for specific traits or behaviors
    - Aggregate scores using direct inference or logprobs (for GPT models)
    - Support for OpenAI, vLLM, and local Hugging Face models
    - Flexible prompt templating and backend configuration

Dependencies:
    - openai: For OpenAI/vLLM-compatible HTTP endpoints
    - transformers: For local inference and tokenizer/model utilities
    - dotenv: For environment variable management

Environment Variables:
    - TOGETHER_API_KEY or OPENAI_API_KEY: Required for OpenAI/vLLM endpoints
    - VLLM_API_URL (optional): Base URL for vLLM OpenAI-compatible server
"""

import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Callable

import torch
from dotenv import load_dotenv
from openai import OpenAI

from pvx import setup_logging
from pvx.abstraction.judges.abstract_judge import AbstractJudge

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")

logger = setup_logging(name="persona-llm-as-judge")


class LLMJudge(AbstractJudge):
    """
    A class for evaluating model responses using LLMs as automated judges.

    Supports multiple backends (OpenAI, vLLM, Hugging Face local) and can aggregate scores
    using direct inference or log probability methods, depending on the model type.

    Attributes:
        backend (str): Inference backend ("openai", "vllm", "hf_local")
        model (str): Model identifier (e.g., "gpt-4", "openai/gpt-oss-20b")
        base_url (str): Base URL for OpenAI/vLLM endpoints
        api_key_env (str): Environment variable for API key
        eval_type (str): Evaluation type (e.g., "0_100")
        device (str): Device for local inference ("cuda", "cpu", etc.)
        dtype (torch.dtype): Data type for local model
        prompt_template (str): Template for evaluation prompt
    """

    def __init__(
        self,
        judge_func: Optional[Callable[[Dict], int]] = None,
        backend: str = "openai",
        model: str = "openai/gpt-4.1-mini",
        local_model: Optional[str] = "Qwen/Qwen2.5-7B-Instruct",
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        prompt_template: str = "",
        api_key_env: str = "OPENROUTER_API_KEY",
        eval_type: str = "0_100",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize an LLMJudge instance.

        Args:
            backend (str): Inference backend ("openai", "vllm", "hf_local").
            model (str): Model identifier (e.g., "gpt-4", "openai/gpt-oss-20b").
            base_url (str, optional): Base URL for OpenAI/vLLM endpoints.
            prompt_template (str): Template for evaluation prompt.
            api_key_env (str): Environment variable for API key.
            eval_type (str): Evaluation type (e.g., "0_100").
            device (str, optional): Device for local inference ("cuda", "cpu", etc.).
            dtype (torch.dtype): Data type for local model.
        """
        super().__init__(backend, model, local_model, base_url, prompt_template, api_key_env, eval_type, device, dtype)

        if judge_func is None:
            self.judge_func = self._aggregate_0_100_score_inference
        else:
            self.judge_func = judge_func

    def logprob_probs(self, messages) -> dict:
        """
        Request log probabilities for the next token from the model.

        Args:
            messages (list): List of chat messages in OpenAI format.

        Returns:
            dict: Mapping from token string to probability.
        """
        openai_client = OpenAI(api_key=os.environ.get(self.api_key_env), base_url=self.base_url)
        completion = openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    def _aggregate_0_100_score_logprob(self, messages):
        """
        Aggregate a score between 0 and 100 using log probabilities from a GPT model.

        Args:
            messages (list): List of chat messages in OpenAI format.

        Returns:
            float or None: Weighted average score, or None if aggregation fails.
        """
        score = self.logprob_probs(messages)
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total

    def _aggregate_0_100_score_inference(self, messages: dict) -> float:
        """
        Aggregate a score between 0 and 100 from model inference output.

        Args:
            messages (dict): List of chat messages in OpenAI format.

        Returns:
            float: The extracted score.

        Raises:
            ValueError: If no valid score is found in the response.
        """
        _, response = self._inference_with_client(messages=messages)
        try:
            return int(response)
        except (ValueError, TypeError):
            match = re.search(r"\d{1,3}", str(response))
            if match:
                return int(match.group())
            return 0
    
    def _get_role_score(self, messages: dict[str, str]) -> int:
        _, response = self._inference_with_client(messages=messages)
        try:
            return int(response)
        except (ValueError, TypeError):
            match = re.search(r"\d{1,3}", str(response))
            if match:
                return int(match.group())
            return 0

    def _aggregate_gold_comparator_score(self, messages: dict) -> dict:
        """Parse a multi-dimensional Likert JSON response from the gold comparator prompt."""
        _, response = self._inference_with_client(messages=messages)
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response.strip())
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.warning("Gold comparator score parsing failed; returning neutral scores.")
        return {
            "style":   {"emotional_register": 3, "vocab_choice": 3, "social_dynamic": 3},
            "content": {"motivation": 3, "worldview_alignment": 3},
        }

    def judge(self, **kwargs):
        """
        Evaluate a model response using the configured prompt template and aggregation method.

        Args:
            **kwargs: Arguments to format into the prompt template.

        Returns:
            float: The aggregated score.
        """
        messages = [{"role": "user", "content": self.prompt_template.format(**kwargs)}]
        return self.judge_func(messages)

    def __call__(self, **kwargs):
        """
        Callable interface for the judge. Equivalent to calling judge(**kwargs).

        Args:
            **kwargs: Arguments to format into the prompt template.

        Returns:
            float: The aggregated score.
        """
        return self.judge(**kwargs)


PROMPT_TEMPLATE  = (
    "You are an expert evaluator assessing how well a model response aligns to a given role or persona.\n\n"
    "You will be given:\n"
    "- A role name and description\n"
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
    "Role Description: {role_description}\n"
    "Question: {question}\n"
    "Response: {answer}\n\n"
    "## Output\n"
    "Return a single integer from 0 to 100. No explanation, no preamble, just the number."
)

COMPARATOR_PROMPT_TEMPLATE  = (
    "You are an expert evaluator assessing how well a model response aligns to a given role or persona compared to a baseline response.\n\n"
    "You will be given:\n"
    "- A role name\n"
    "- A question asked to the model\n"
    "- The model's response\n"
    "- A baseline response\n\n"
    "Your task is to score the response from 0 to 100 based on how well it reflects the values, tone, priorities, and perspective of the described role compared to the baseline.\n\n"
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
    "Baseline Response: {baseline}\n\n"
    "## Output\n"
    "Return a single integer from 0 to 100. No explanation, no preamble, just the number."
)

GOLD_COMPARATOR_PROMPT_TEMPLATE = (
    "You are an expert evaluator assessing how closely a steered model response matches a gold-standard baseline response for a given role.\n\n"
    "You will be given:\n"
    "- A role name\n"
    "- A question asked to the model\n"
    "- A baseline response (gold label) that exemplifies how the role should speak and reason\n"
    "- A steered response to evaluate against that baseline\n\n"
    "Score each of the five dimensions below on a 5-point Likert scale:\n\n"
    "  1 = Strongly diverges from baseline\n"
    "  2 = Mostly diverges\n"
    "  3 = Partially matches\n"
    "  4 = Mostly matches\n"
    "  5 = Strongly matches baseline\n\n"
    "## Scoring Dimensions\n\n"
    "### Style\n"
    "- emotional_register: Does the steered response carry the same emotional tone as the baseline "
    "(e.g., warm/cold, passionate/detached, anxious/confident)?\n"
    "- vocab_choice: Does the steered response use the same vocabulary, register, jargon, and "
    "linguistic style as the baseline?\n"
    "- social_dynamic: Does the steered response adopt the same relational stance as the baseline "
    "(e.g., authoritative, deferential, collaborative, confrontational)?\n\n"
    "### Content\n"
    "- motivation: Does the steered response reflect the same underlying goals, drives, and values "
    "expressed in the baseline?\n"
    "- worldview_alignment: Does the steered response reflect the same worldview, beliefs, and "
    "role-consistent perspective as the baseline?\n\n"
    "## Input\n"
    "Role: {role}\n"
    "Role Description: {role_description}\n"
    "Question: {question}\n"
    "Baseline (Gold Label):\n{baseline}\n\n"
    "Steered Response:\n{answer}\n\n"
    "## Output\n"
    "Return only a JSON object with this exact structure. No explanation, no preamble:\n"
    '{{"style": {{"emotional_register": <1-5>, "vocab_choice": <1-5>, "social_dynamic": <1-5>}}, '
    '"content": {{"motivation": <1-5>, "worldview_alignment": <1-5>}}}}'
)

PAIRWISE_PROMPT_TEMPLATE = (
    "You are an expert evaluator assessing how well language model responses embody a given role.\n\n"
    "You will be given:\n"
    "  - A role name and description\n"
    "  - A gold-standard reference response showing ideal role embodiment\n"
    "  - Two candidate responses (Response A and Response B)\n"
    "  - The question both models were asked\n\n"
    "## Evaluation Steps\n"
    "Step 1 — Reasoning: Write 2-3 sentences comparing how well each response embodies the role "
    "relative to the gold standard. Focus on: tone, vocabulary, perspective, and role-consistent reasoning.\n"
    "Step 2 — Score: Rate each response independently from 0 to 100 on role embodiment.\n"
    "Step 3 — Winner: Declare which response better embodies the role (A, B, or TIE).\n\n"
    "## Scoring Guidance\n"
    "  90-100: Strongly embodies the role — tone, framing, and content all clearly aligned\n"
    "  70-89:  Mostly aligned with minor deviations\n"
    "  50-69:  Partially aligned, feels somewhat generic\n"
    "  30-49:  Weakly aligned — could belong to any generic assistant\n"
    "  0-29:   Contradicts or ignores the role entirely\n\n"
    "## Input\n"
    "Role: {role}\n"
    "Role Description: {role_description}\n"
    "Question: {question}\n\n"
    "Gold Standard Reference:\n{gold_label}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "## Output\n"
    "Return ONLY a JSON object with this exact structure. No preamble, no trailing text:\n"
    '{{"reasoning": "<2-3 sentence analysis>", "score_a": <0-100>, "score_b": <0-100>, "winner": "<A|B|TIE>"}}'
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a model response using LLMJudge")
    ap.add_argument("--backend", default="openai", help="Backend to use: openai, vllm, hf_local")
    ap.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model to use for openai/vllm/hf_local backend",
    )
    ap.add_argument(
        "--local_model",
        default="allenai/Olmo-3-7B-Instruct",
        help="Local HF model to use for hf_local backend",
    )
    ap.add_argument(
        "--base_url",
        default="https://openrouter.ai/api/v1",
        help="Base URL for OpenAI/vLLM endpoints",
    )
    ap.add_argument(
        "--api_key_env", default="OPENROUTER_API_KEY", help="Environment variable for API key"
    )
    ap.add_argument("--eval_type", default="0_100", help="Evaluation type (e.g., 0_100)")
    ap.add_argument("--device", default=None, help="Device for local inference (cuda, cpu, etc.)")
    ap.add_argument(
        "--dtype", default="float16", help="Data type for local model (e.g., float16, float32)"
    )
    ap.add_argument("--question", required=True, help="The evaluation question")
    ap.add_argument("--answer", required=True, help="The model response to evaluate")
    ap.add_argument("--role", required=True, help="The model role to evaluate")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    judge = LLMJudge(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        eval_type=args.eval_type,
        device=args.device,
        dtype=dtype,
        prompt_template=PROMPT_TEMPLATE,
    )

    score = judge(question=args.question, answer=args.answer, role=args.role)
    print(f"Score: {score}")
