"""Pydantic models for gold-standard conversation data.

Output format uses OpenAI-compatible messages: the persona system prompt
is stored separately, interviewer turns are "user", and interviewee
(in-character) turns are "assistant".
"""

from pydantic import BaseModel, Field


class GoldStandardConversation(BaseModel):
    persona_slug: str = Field(description="Persona identifier slug (e.g. 'architect')")
    persona_concept: str = Field(description="Full persona title from instructions JSON")
    system_prompt_index: int = Field(description="Which positive_prompt variant was used (0-4)")
    system_prompt: str = Field(description="The persona system prompt given to the interviewee")
    interviewer_model: str = Field(description="Model used for the interviewer")
    interviewee_model: str = Field(description="Model used for the interviewee")
    num_questions: int = Field(default=0, description="Number of interviewer questions asked")
    messages: list[dict] = Field(
        default_factory=list,
        description=(
            "OpenAI-compatible message list. "
            "Each entry has 'role' ('user' for interviewer, 'assistant' for interviewee) "
            "and 'content'."
        ),
    )
