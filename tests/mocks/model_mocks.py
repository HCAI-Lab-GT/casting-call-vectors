"""Mock model implementations for fast unit tests.

These mocks return deterministic fake activations without loading real models,
enabling rapid testing of pipeline logic.
"""

import torch

from pvx.extraction.activations import ActivationResult


class MockActivationExtractor:
    """Returns deterministic fake activations for fast unit tests.

    Uses prompt content hash to generate reproducible random vectors,
    allowing test assertions on consistency.

    Example:
        >>> extractor = MockActivationExtractor()
        >>> result = extractor.extract("You are a nurse.", "What is your role?")
        >>> result.prompt_last.shape
        torch.Size([1, 768])
    """

    def __init__(self, hidden_dim: int = 768, num_layers: int = 12):
        """Initialize mock extractor.

        Args:
            hidden_dim: Hidden dimension for fake activations
            num_layers: Number of layers for all_layers output
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_id = "mock-model"
        self.layer = num_layers // 2
        self.device = torch.device("cpu")

    def extract(
        self,
        system_prompt: str,
        question: str,
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.99,
    ) -> ActivationResult:
        """Return deterministic fake activations.

        Args:
            system_prompt: System prompt (used for seeding)
            question: Question (used for seeding)
            max_new_tokens: Ignored (for API compatibility)
            temperature: Ignored (for API compatibility)
            top_p: Ignored (for API compatibility)

        Returns:
            ActivationResult with deterministic fake tensors
        """
        # Use prompt hash for reproducible randomness
        seed = hash(system_prompt + question) % (2**32)
        torch.manual_seed(seed)

        prompt_last = torch.randn(1, self.hidden_dim)
        response_mean = torch.randn(1, self.hidden_dim)
        response_all_layers = torch.randn(self.num_layers + 1, 1, self.hidden_dim)

        # Generate mock response text
        response_words = ["This", "is", "a", "mock", "response", "for", "testing."]
        response_text = " ".join(response_words)

        return ActivationResult(
            prompt_last=prompt_last,
            response_mean=response_mean,
            response_all_layers=response_all_layers,
            response_text=response_text,
            num_response_tokens=len(response_words),
        )

    def extract_batch(
        self,
        prompts: list[tuple[str, str]],
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        show_progress: bool = False,
    ) -> list[ActivationResult]:
        """Extract activations for multiple prompts.

        Args:
            prompts: List of (system_prompt, question) tuples
            max_new_tokens: Ignored
            temperature: Ignored
            show_progress: Ignored

        Returns:
            List of ActivationResult objects
        """
        return [
            self.extract(system_prompt, question, max_new_tokens, temperature)
            for system_prompt, question in prompts
        ]

    def __repr__(self) -> str:
        return (
            f"MockActivationExtractor(hidden_dim={self.hidden_dim}, num_layers={self.num_layers})"
        )
