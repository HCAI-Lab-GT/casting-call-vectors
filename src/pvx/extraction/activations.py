"""Activation extraction from transformer models.

This module provides the ActivationExtractor class for extracting hidden state
activations during model generation, which are then used to compute persona vectors.
"""

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

logger = logging.getLogger(__name__)

# Reduce transformers verbosity
transformers_logging.set_verbosity_error()


@dataclass
class ActivationResult:
    """Result of activation extraction for a single prompt.

    Attributes:
        prompt_last: Activation at the last prompt token (1, hidden_dim)
        response_mean: Mean activation across response tokens (1, hidden_dim)
        response_all_layers: Mean activation across all layers (num_layers, 1, hidden_dim)
        response_text: Generated response text
        num_response_tokens: Number of tokens in the generated response
    """

    prompt_last: torch.Tensor
    response_mean: torch.Tensor
    response_all_layers: torch.Tensor
    response_text: str
    num_response_tokens: int


class ActivationExtractor:
    """Extract activations from transformer models during generation.

    This class handles model loading, tokenization, and activation extraction
    for persona vector computation. It uses KV-cache for efficient generation.

    Example:
        >>> extractor = ActivationExtractor("allenai/OLMo-7B-Instruct", layer=14)
        >>> result = extractor.extract(
        ...     system_prompt="You are a nurse.",
        ...     question="How should I handle stress?"
        ... )
        >>> result.prompt_last.shape
        torch.Size([1, 4096])
    """

    def __init__(
        self,
        model_id: str,
        layer: int = 14,
        device: str = "auto",
        dtype: torch.dtype | None = None,
        compile_model: bool = False,
    ):
        """Initialize the activation extractor.

        Args:
            model_id: HuggingFace model identifier
            layer: Layer index for primary activation extraction (0-indexed)
            device: Device to use ("auto", "cuda", "cpu")
            dtype: Model dtype (None for auto-detect, typically float16 on GPU)
            compile_model: Whether to use torch.compile (experimental)
        """
        self.model_id = model_id
        self.layer = layer
        self._device = device

        # Determine dtype (MPS works best with float32)
        if dtype is None:
            if torch.cuda.is_available():
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                dtype = torch.float32
            else:
                dtype = torch.float32

        logger.info(f"Loading model: {model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device

        # Load model with appropriate device mapping
        if resolved_device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
            if resolved_device != "cpu":
                self.model = self.model.to(resolved_device)

        # Determine actual device
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype

        # Get EOS token IDs
        if isinstance(self.tokenizer.eos_token_id, int):
            self.eos_token_ids = {self.tokenizer.eos_token_id}
        else:
            self.eos_token_ids = set(self.tokenizer.eos_token_id or [])

        # Optional compilation
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="default", dynamic=True)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Put model in inference mode
        self.model.train(False)

        logger.info(f"Model loaded on {self.device} with dtype {self.dtype}")
        logger.info(f"Extracting from layer {layer}")

    def _tokenize(
        self,
        system_prompt: str,
        question: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a system prompt and question into chat format.

        Args:
            system_prompt: The system prompt (persona instruction)
            question: The user question

        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask

    def _top_p_sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.9,
        top_p: float = 0.99,
    ) -> int:
        """Sample from logits using top-p (nucleus) sampling.

        Args:
            logits: Logits tensor (vocab_size,)
            temperature: Sampling temperature
            top_p: Cumulative probability threshold

        Returns:
            Sampled token ID
        """
        if temperature <= 0:
            return int(logits.argmax().item())

        # Apply temperature
        probs = torch.softmax(logits / temperature, dim=-1)

        # Sort and compute cumulative probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()

        # Sample
        idx = torch.multinomial(sorted_probs, 1).item()
        return int(sorted_indices[idx].item())

    @torch.inference_mode()
    def extract(
        self,
        system_prompt: str,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        top_p: float = 0.99,
    ) -> ActivationResult:
        """Extract activations for a single prompt/question pair.

        Generates a response and captures:
        - prompt_last: Last token activation at the target layer
        - response_mean: Mean activation across all response tokens
        - response_all_layers: Mean activation across all layers

        Args:
            system_prompt: System prompt (persona instruction)
            question: User question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold

        Returns:
            ActivationResult with extracted activations and response text
        """
        # Tokenize
        input_ids, attention_mask = self._tokenize(system_prompt, question)

        # Forward pass on prompt (with hidden states)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )

        # Extract prompt last token activation at target layer
        # hidden_states[0] is input embeddings, so layer N is at index N+1
        prompt_last = out.hidden_states[self.layer + 1][:, -1, :]  # (1, hidden)

        # Store KV cache
        past = out.past_key_values

        # Sample first token
        logits = out.logits[0, -1, :]
        tok = self._top_p_sample(logits, temperature=temperature, top_p=top_p)

        # Handle empty response
        if tok in self.eos_token_ids:
            num_states = len(out.hidden_states)
            return ActivationResult(
                prompt_last=prompt_last.cpu(),
                response_mean=torch.zeros_like(prompt_last).cpu(),
                response_all_layers=torch.zeros(
                    (num_states,) + prompt_last.shape,
                    dtype=prompt_last.dtype,
                ).cpu(),
                response_text="",
                num_response_tokens=0,
            )

        # Accumulate response activations (in float32 for numerical stability)
        acc_dtype = torch.float32
        resp_sum = torch.zeros_like(prompt_last, dtype=acc_dtype)

        num_states = len(out.hidden_states)
        resp_sum_all_layers = torch.zeros(
            (num_states,) + prompt_last.shape,
            device=prompt_last.device,
            dtype=acc_dtype,
        )

        # Generation loop
        count = 0
        gen_ids = []
        last_token = torch.tensor([[tok]], device=self.device, dtype=torch.long)

        for _ in range(max_new_tokens):
            out = self.model(
                input_ids=last_token,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
            past = out.past_key_values

            # Accumulate target layer activation
            final_act = out.hidden_states[self.layer + 1][:, -1, :]
            resp_sum += final_act.to(acc_dtype)

            # Accumulate all layers
            all_layers_last = torch.stack([hs[:, -1, :] for hs in out.hidden_states], dim=0)
            resp_sum_all_layers += all_layers_last.to(acc_dtype)

            count += 1

            # Sample next token
            logits = out.logits[0, -1, :]
            next_tok = self._top_p_sample(logits, temperature=temperature, top_p=top_p)

            if next_tok in self.eos_token_ids:
                break

            gen_ids.append(next_tok)
            last_token.fill_(next_tok)

        # Compute means
        if count > 0:
            resp_mean = (resp_sum / count).to(prompt_last.dtype)
            resp_all_layers = (resp_sum_all_layers / count).to(prompt_last.dtype)
        else:
            resp_mean = torch.zeros_like(prompt_last)
            resp_all_layers = torch.zeros_like(resp_sum_all_layers, dtype=prompt_last.dtype)

        # Decode response
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        return ActivationResult(
            prompt_last=prompt_last.cpu(),
            response_mean=resp_mean.cpu(),
            response_all_layers=resp_all_layers.cpu(),
            response_text=response_text,
            num_response_tokens=count,
        )

    def extract_batch(
        self,
        prompts: list[tuple[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        show_progress: bool = True,
    ) -> list[ActivationResult]:
        """Extract activations for multiple prompt/question pairs.

        Note: Currently processes sequentially. Batch parallelism is complex
        due to variable-length generation.

        Args:
            prompts: List of (system_prompt, question) tuples
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            show_progress: Whether to show progress bar

        Returns:
            List of ActivationResult objects
        """
        results = []

        if show_progress:
            from tqdm import tqdm

            prompts = tqdm(prompts, desc="Extracting activations")

        for system_prompt, question in prompts:
            result = self.extract(
                system_prompt=system_prompt,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(result)

        return results

    def __repr__(self) -> str:
        return f"ActivationExtractor(model={self.model_id!r}, layer={self.layer}, device={self.device})"
