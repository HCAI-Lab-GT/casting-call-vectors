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
import os
import warnings
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

load_dotenv()

BACKENDS = ("openai", "vllm", "hf_local")

logger = setup_logging(name="response_generation")


class ResponseGeneration:
    def __init__(
        self,
        backend: str = "openai",
        model: str = "openai/gpt-oss-20b",
        local_model: Optional[str] = "Qwen/Qwen2.5-7B-Instruct",
        base_url: Optional[str] = "https://api.together.xyz/v1",
        api_key_env: str = "TOGETHER_API_KEY",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.backend = backend
        self.model = model
        self.local_model = local_model or model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.device = device
        self.dtype = dtype

        self._hf_model = None
        self._hf_tokenizer = None
        
        logger.info(f"Backend: {backend}, Model: {self.local_model}")

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
        self, messages: List[Dict[str, str]], temperature: float = 0.1, max_new_tokens=1024
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
                    logger.info(f"Initialized Local Model {self.local_model}")

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
                self.local_model,
                temperature,
            )
            raise e

    @staticmethod
    def convert_str_to_message(messages: list[str]) -> List[Dict[str, str]]:
        input_messages = []
        for message in messages:
            input_message = {"role": "user", "content": message}
            input_messages.append(input_message)

        return input_messages

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

    def __call__(
        self, messages: List[Dict[str, str]], temperature: float = 0.9, max_new_tokens=1024
    ):
        """
        Callable interface for the judge. Equivalent to calling judge(**kwargs).

        Args:
            **kwargs: Arguments to format into the prompt template.

        Returns:
            float: The aggregated score.
        """
        return self._inference_with_client(
            messages=messages, temperature=temperature, max_new_tokens=max_new_tokens
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a model response using LLMJudge")
    ap.add_argument("--backend", default="openai", help="Backend to use: openai, vllm, hf_local")
    ap.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1",
        help="Model to use for openai/vllm/hf_local backend",
    )
    ap.add_argument(
        "--riasec_positive",
        action="store_true",
        help="Use system prompting with RIASEC positive case",
    )
    ap.add_argument(
        "--riasec_negative",
        action="store_true",
        help="Use system prompting with RIASEC positive case",
    )
    ap.add_argument(
        "--pos_neg_trait", default="", help="The positive or negative trait system prompting"
    )
    ap.add_argument("--trait", default="realistic", help="Trait for RIASEC response generation")
    ap.add_argument(
        "--local_model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Local HF model to use for hf_local backend",
    )
    ap.add_argument(
        "--base_url",
        default="https://api.together.xyz/v1",
        help="Base URL for OpenAI/vLLM endpoints",
    )
    ap.add_argument(
        "--api_key_env", default="TOGETHER_API_KEY", help="Environment variable for API key"
    )
    ap.add_argument("--device", default=None, help="Device for local inference (cuda, cpu, etc.)")
    ap.add_argument(
        "--dtype", default="float16", help="Data type for local model (e.g., float16, float32)"
    )
    ap.add_argument("--question", required=True, help="The evaluation question")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    from pvx.utils.riasec_utils import RIASECHelpers

    if args.riasec_positive:
        logger.info("Answering positive trait")
        messages = [
            {
                "role": "system",
                "content": RIASECHelpers.POSITIVE_RIASEC_SYSTEM_PROMPT.format(TRAIT=args.trait),
            },
            {"role": "system", "content": args.pos_neg_trait},
            {"role": "user", "content": args.question},
        ]
    elif args.riasec_negative:
        logger.info("Answering negative trait")
        messages = [
            {
                "role": "system",
                "content": RIASECHelpers.NEGATIVE_RIASEC_SYSTEM_PROMPT.format(TRAIT=args.trait),
            },
            {"role": "system", "content": args.pos_neg_trait},
            {"role": "user", "content": args.question},
        ]
    else:
        messages = ResponseGeneration.convert_str_to_message(messages=[args.question])

    generate_response = ResponseGeneration(backend=args.backend,
        model=args.model,
        local_model=args.local_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        device=args.device,
        dtype=args.dtype)
    _, response = generate_response(messages=messages)
    logger.info("===Question===")
    logger.info(args.question)
    logger.info("===Response===")
    logger.info(response)
