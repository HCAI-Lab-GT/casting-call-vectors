"""Interviewer LLM: a curious person probing personality through conversation.

Manages the interviewer side of a gold-standard generation conversation.
The interviewer is NOT a character — just a genuinely curious person who
wants to understand how the other party thinks, decides, and prioritizes.

Prompts are defined in prompt.py. This module handles message assembly,
LLM calls with retry on empty responses, and completion detection.
"""

import logging

from openai import AsyncOpenAI
from prompt import FIRST_TURN_NUDGE, INTERVIEWER_SYSTEM_PROMPT, format_prior_questions

logger = logging.getLogger(__name__)

MAX_EMPTY_RETRIES = 3


def build_interviewer_messages(
    conversation_history: list[dict],
    max_questions: int,
    questions_asked: int,
    prior_questions: list[str],
) -> list[dict]:
    """Build the message list for the interviewer's next turn.

    Args:
        conversation_history: Prior turns in OpenAI message format
            (assistant = interviewer, user = interviewee).
        max_questions: Maximum questions the interviewer may ask.
        questions_asked: How many questions have been asked so far.
        prior_questions: List of question strings already asked.

    Returns:
        Messages list ready for the interviewer LLM call.
    """
    prior_block = format_prior_questions(prior_questions)
    system_text = INTERVIEWER_SYSTEM_PROMPT.format(
        max_questions=max_questions, prior_questions_block=prior_block
    )

    remaining = max_questions - questions_asked
    if remaining <= 1:
        system_text += (
            "\n\n<urgent>This is your LAST question. Ask it, then end your "
            "message with [INTERVIEW_COMPLETE] on its own line.</urgent>"
        )

    messages: list[dict] = [{"role": "system", "content": system_text}]

    if not conversation_history:
        messages.append({"role": "user", "content": FIRST_TURN_NUDGE})
    else:
        messages.extend(conversation_history)

    return messages


async def get_interviewer_response(
    client: AsyncOpenAI,
    model: str,
    conversation_history: list[dict],
    max_questions: int,
    questions_asked: int,
    prior_questions: list[str],
    temperature: float = 0.7,
) -> str:
    """Generate the interviewer's next question. Retries on empty responses."""
    messages = build_interviewer_messages(
        conversation_history, max_questions, questions_asked, prior_questions
    )

    for attempt in range(1, MAX_EMPTY_RETRIES + 1):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
        )
        content = (response.choices[0].message.content or "").strip()
        if content:
            logger.debug(
                "Interviewer (q %d/%d): %s", questions_asked + 1, max_questions, content[:120]
            )
            return content
        logger.warning("Interviewer returned empty (attempt %d/%d)", attempt, MAX_EMPTY_RETRIES)

    logger.error("Interviewer returned empty after %d retries", MAX_EMPTY_RETRIES)
    return ""


def is_interview_complete(response: str) -> bool:
    """Check whether the interviewer signaled the conversation is done."""
    return "[INTERVIEW_COMPLETE]" in response


def strip_completion_marker(response: str) -> str:
    """Remove the completion marker from the interviewer's final message."""
    return response.replace("[INTERVIEW_COMPLETE]", "").strip()
