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
import math
import os
import re
import warnings
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")

logger = setup_logging(name="abstract-judge")


class AbstractJudge(ABC):
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
        base_url: Optional[str] = "https://api.together.xyz/v1",
        prompt_template: str = "",
        api_key_env: str = "TOGETHER_API_KEY",
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
        self.backend = backend
        self.model = model
        self.local_model = local_model or model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.eval_type = eval_type
        self.device = device
        self.dtype = dtype

        self._hf_model = None
        self._hf_tokenizer = None

        self.prompt_template = prompt_template


    @abstractmethod
    def judge(self, **kwargs):
        pass
        

    def _init_local_model(self):
        """
        Lazily load a HF transformers causal LM for local inference.
        """
        if self.backend != "hf_local":
            return

        # Device selection
        if self.device:
            device = self.device
        else:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                warnings.warn(
                    "Using CPU for local inference; generation will be slow.", stacklevel=2
                )

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.local_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with device map
        model = AutoModelForCausalLM.from_pretrained(
            self.local_model,
            dtype=self.dtype if device != "cpu" else None,
            device_map=device,
        )

        self._hf_model = model
        self._hf_tokenizer = tokenizer

    def _inference_with_client(
        self, messages: List[Dict[str, str]], temperature: float = 0.9, max_new_tokens=1024
    ):
        """
        Perform LLM inference using the configured backend.

        Args:
            messages (list): List of chat messages in OpenAI format.
            temperature (float): Sampling temperature.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            tuple: (None, response string)

        Raises:
            ValueError: If required API key is not set for OpenAI/vLLM backends.
            Exception: Re-raises any exception from the underlying API call after logging.
        """
        try:
            if self.backend == "hf_local":
                if self._hf_model is None or self._hf_tokenizer is None:
                    self._init_local_model()

                # Prepare prompt for decoder-only model
                prompt = self._messages_to_prompt(messages)

                # Generate response
                inputs = self._hf_tokenizer(prompt, return_tensors="pt", padding=True).to(
                    self._hf_model.device
                )
                generated = self._hf_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                # Extract generated text (first sequence and excluding the prompt)
                decoded = self._hf_tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                )

                return (None, decoded.strip())

            base_url = self.base_url
            if self.backend == "vllm" and base_url is None:
                base_url = os.environ.get("VLLM_API_URL")
            if base_url is None:
                base_url = "https://glados.ctisl.gtri.org"

            api_key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    f"{self.api_key_env} or OPENAI_API_KEY environment variable not set"
                )
            openai_client = OpenAI(api_key=api_key, base_url=base_url)

            chat_response = openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return (None, chat_response.choices[0].message.content)

        except Exception as e:
            logger.exception(
                "inference failed | backend=%s model=%s temp=%s",
                self.backend,
                self.model,
                temperature,
            )
            raise e

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Flatten chat messages into a single prompt string for decoder-only HF models.
        Format: "<system>\\n\\n<user>\\n\\nAssistant:"
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts = [m["content"] for m in messages if m["role"] == "user"]
        system_block = "\\n".join(system_parts)
        user_block = "\\n\\n".join(user_parts)
        prompt = ""
        if system_block:
            prompt += f"{system_block}\\n\\n"
        prompt += user_block
        prompt += "\\n\\nAssistant:"
        return prompt

    @abstractmethod
    def __call__(self, **kwargs):
        pass
