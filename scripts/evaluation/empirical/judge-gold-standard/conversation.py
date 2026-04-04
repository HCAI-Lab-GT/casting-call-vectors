"""Conversation loop: interviewer ↔ interviewee multi-turn dialogue."""

import logging

from interviewee import get_interviewee_response
from interviewer import (
    get_interviewer_response,
    is_interview_complete,
    strip_completion_marker,
)
from models import GoldStandardConversation
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUESTIONS = 5


def _drop_last_qa(msgs, i_hist, e_hist, prior_qs):
    """Pop the last question from all tracking lists. Returns -1 for count adjust."""
    for lst in (msgs, i_hist, e_hist, prior_qs):
        lst.pop()
    return -1


async def _get_answer(client, model, system_prompt, history, temperature):
    """Shorthand for getting an interviewee response."""
    return await get_interviewee_response(
        client=client,
        model=model,
        system_prompt=system_prompt,
        conversation_history=history,
        temperature=temperature,
    )


async def run_conversation(
    interviewer_client: AsyncOpenAI,
    interviewee_client: AsyncOpenAI,
    interviewer_model: str,
    interviewee_model: str,
    system_prompt: str,
    persona_slug: str,
    persona_concept: str,
    prompt_index: int = 0,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    interviewer_temp: float = 0.7,
    interviewee_temp: float = 0.8,
) -> GoldStandardConversation:
    """Run a full interviewer ↔ interviewee conversation."""
    interviewer_history: list[dict] = []
    interviewee_history: list[dict] = []
    prior_questions: list[str] = []
    messages: list[dict] = []
    questions_asked = 0

    logger.info("Starting conversation: %s (prompt %d)", persona_slug, prompt_index)

    for _ in range(max_questions * 2):
        q_raw = await get_interviewer_response(
            client=interviewer_client,
            model=interviewer_model,
            conversation_history=interviewer_history,
            max_questions=max_questions,
            questions_asked=questions_asked,
            prior_questions=prior_questions,
            temperature=interviewer_temp,
        )
        if not q_raw:
            logger.warning("Interviewer produced no output, ending early")
            break

        complete = is_interview_complete(q_raw)
        question = strip_completion_marker(q_raw)

        if question:
            questions_asked += 1
            prior_questions.append(question)
            messages.append({"role": "user", "content": question})
            interviewer_history.append({"role": "assistant", "content": question})
            interviewee_history.append({"role": "user", "content": question})
            logger.info("Q%d/%d: %s", questions_asked, max_questions, question[:80])

        if complete or questions_asked >= max_questions:
            if not question:
                break
            answer = await _get_answer(
                interviewee_client,
                interviewee_model,
                system_prompt,
                interviewee_history,
                interviewee_temp,
            )
            if not answer:
                logger.warning("Empty final answer, dropping turn")
                questions_asked += _drop_last_qa(
                    messages, interviewer_history, interviewee_history, prior_questions
                )
            else:
                messages.append({"role": "assistant", "content": answer})
                interviewee_history.append({"role": "assistant", "content": answer})
                logger.info("A%d: %s", questions_asked, answer[:80])
            break

        answer = await _get_answer(
            interviewee_client,
            interviewee_model,
            system_prompt,
            interviewee_history,
            interviewee_temp,
        )
        if not answer:
            logger.warning("Empty answer Q%d, dropping pair", questions_asked)
            questions_asked += _drop_last_qa(
                messages, interviewer_history, interviewee_history, prior_questions
            )
            continue

        messages.append({"role": "assistant", "content": answer})
        interviewer_history.append({"role": "user", "content": answer})
        interviewee_history.append({"role": "assistant", "content": answer})
        logger.info("A%d: %s", questions_asked, answer[:80])

    logger.info("Done: %s — %d questions, %d msgs", persona_slug, questions_asked, len(messages))

    return GoldStandardConversation(
        persona_slug=persona_slug,
        persona_concept=persona_concept,
        system_prompt_index=prompt_index,
        system_prompt=system_prompt,
        interviewer_model=interviewer_model,
        interviewee_model=interviewee_model,
        num_questions=questions_asked,
        messages=messages,
    )
