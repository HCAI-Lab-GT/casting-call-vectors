"""Interviewee LLM: role-plays as a vocational persona during interviews.

Loads persona system prompts from instruction JSON files and manages
the interviewee side of a gold-standard generation conversation.

Checks finish_reason on every response — if the model was truncated
(finish_reason="length") or empty, retries cleanly from scratch.
Returns empty string after all retries fail so the conversation loop
can drop the Q&A pair.
"""

import json
import logging
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

INSTRUCTIONS_DIR = (
    Path(__file__).parent.parent / "persona_data" / "vocational_personas" / "instructions"
)

CHARACTER_REINFORCEMENT = (
    "\n\n<constraint>"
    "You are in a live conversation. Stay fully immersed in your identity. "
    "Respond from lived experience — what you have seen, felt, decided, resented, "
    "and refused. Never step outside the character to describe or narrate traits. "
    "Never acknowledge being an AI, a language model, or playing a role."
    "</constraint>"
)


def load_persona(path: Path) -> dict:
    """Load a persona instruction file from an explicit JSON path."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Persona file not found: {resolved}")
    with open(resolved, "r") as f:
        return json.load(f)


def slug_from_path(path: Path) -> str:
    """Derive a slug identifier from a persona JSON file path."""
    return Path(path).stem


def get_system_prompt(persona: dict, prompt_index: int = 0) -> str:
    """Extract a specific system prompt variant from a persona definition."""
    prompts = persona.get("positive_prompts", [])
    if not prompts:
        raise ValueError(f"Persona '{persona.get('concept', '?')}' has no positive_prompts")
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(
            f"prompt_index {prompt_index} out of range for "
            f"{len(prompts)} prompts in '{persona.get('concept', '?')}'"
        )
    return prompts[prompt_index]["pos"]


def list_available_personas(instructions_dir: Path = INSTRUCTIONS_DIR) -> list[Path]:
    """List all available persona JSON paths in the instructions directory."""
    if not instructions_dir.exists():
        return []
    return sorted(instructions_dir.glob("*.json"))


def _is_complete(choice) -> bool:
    """Check whether a completion choice finished naturally (not truncated)."""
    reason = getattr(choice, "finish_reason", None)
    if reason == "length":
        return False
    content = (choice.message.content or "").strip()
    if not content:
        return False
    return True


async def get_interviewee_response(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    conversation_history: list[dict],
    temperature: float = 0.8,
) -> str:
    """Generate the interviewee's in-character response.

    Checks finish_reason on every attempt:
      - "stop"   → complete response, return it.
      - "length" → truncated, regenerate cleanly from scratch.
      - empty    → regenerate cleanly from scratch.

    Returns empty string after MAX_RETRIES so the conversation loop
    can drop the Q&A pair.
    """
    reinforced_prompt = system_prompt + CHARACTER_REINFORCEMENT
    messages: list[dict] = [{"role": "system", "content": reinforced_prompt}]
    messages.extend(conversation_history)

    reason = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        choice = response.choices[0]

        if _is_complete(choice):
            content = choice.message.content.strip()
            logger.debug("Interviewee response: %s", content[:120])
            return content

        reason = getattr(choice, "finish_reason", "unknown")
        logger.warning(
            "Interviewee incomplete (finish_reason=%s, attempt %d/%d), retrying clean",
            reason,
            attempt,
            MAX_RETRIES,
        )

    logger.error("Interviewee failed after %d retries (last reason: %s)", MAX_RETRIES, reason)
    return ""
