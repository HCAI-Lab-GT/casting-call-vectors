"""
The LLMJudge class for evaluating model responses using LLMs as automated judges.
"""

import argparse
import math
import os
import re
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pvx import setup_logging
from pvx.utils.response_generation import ResponseGeneration

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")

logger = setup_logging(name="abstract-judge")


class AbstractJudge(ResponseGeneration, ABC):
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
        backend: str = "openai",
        model: str = "openai/gpt-oss-20b",
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
        super().__init__(backend, model, local_model, base_url, api_key_env, device, dtype)
        self.eval_type = eval_type
        self.prompt_template = prompt_template


    @abstractmethod
    def judge(self, **kwargs):
        pass
        

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def _async_inference_with_client(
        self, messages: List[Dict[str, str]], temperature: float = 0.9, max_new_tokens: int = 1024
    ) -> str:
        """
        Async version of _inference_with_client for openai/vllm backends only.
        Wrapped with exponential-backoff retry on RateLimitError / APIStatusError.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Response content string.

        Raises:
            ValueError: If the backend is hf_local (not supported).
        """
        if self.backend == "hf_local":
            raise ValueError("_async_inference_with_client does not support hf_local backend")

        api_key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                f"{self.api_key_env} or OPENAI_API_KEY environment variable not set"
            )

        base_url = self.base_url
        if self.backend == "vllm" and base_url is None:
            base_url = os.environ.get("VLLM_API_URL")
        if base_url is None:
            base_url = "https://glados.ctisl.gtri.org"

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content

    async def async_judge(self, **kwargs) -> int:
        """
        Async version of judge() for openai/vllm backends.
        Formats the prompt template, calls _async_inference_with_client, and parses the score.

        Args:
            **kwargs: Arguments to format into the prompt template.

        Returns:
            int: Parsed score. Returns -1 if parsing fails.
        """
        if self.backend == "hf_local":
            raise ValueError("async_judge does not support hf_local backend")

        messages = [{"role": "user", "content": self.prompt_template.format(**kwargs)}]
        response = await self._async_inference_with_client(messages=messages)
        try:
            return int(response)
        except (ValueError, TypeError):
            match = re.search(r"\d{1,3}", str(response))
            if match:
                return int(match.group())
            logger.warning("async_judge: could not parse score from response: %r", response)
            return -1

    @abstractmethod
    def __call__(self, **kwargs):
        pass
