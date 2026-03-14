"""Vocational Persona Generator for role-based profiles.

Generates persona system prompts and evaluation prompts for roles loaded
from new_roles.json via the RoleProfile loader, compatible with the
assistant-axis pipeline format.

Now consumes the uniform `psychological_profile` layer from RoleProfile
to produce personas that differentiate by *decision-making* rather than
stylistic vocabulary alone.

Example usage:
    from pvx.data.role_loader import RoleProfile
    from pvx.pvx_models.new_vocational_dataset import VocationalPersonaGenerator

    import json
    with open("data/new_roles.json") as f:
        profile_dict = json.load(f)

    role_loader = RoleProfile(profile_dict=profile_dict, model="moonshotai/kimi-k2.5")
    generator = VocationalPersonaGenerator(model="moonshotai/kimi-k2.5")

    profile = role_loader.get_profile("nurse")
    persona = generator.generate_persona(profile)
    generator.save_persona("nurse", persona)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System-prompt generation template
#
# This template now receives a `psychological_profile` block alongside the
# structural data (tasks, role_contexts).  The key shift: each generated
# prompt must encode how the persona *decides and prioritises*, not just
# what vocabulary domain they inhabit.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_GENERATION_TEMPLATE = """# Role
You are an expert prompt engineer specializing in occupation-grounded persona design. You craft system prompts that make an AI fully inhabit a professional role — thinking, reasoning, and responding as that professional in first person, not describing or advising about the profession from the outside.

# Input Data
**Title:** {title}
**Job Description:** {description}
**Tasks:** {tasks}
**Role Context:** {role_context}
**Psychological Profile:** {psych_profile}

# Task
Generate exactly 5 system prompts, each 2–4 sentences, that cause an AI to embody a `title` in conversation. Each prompt must produce first-person, in-role behavior — the AI should think and respond as this professional, not coach or advise about the profession.

Each prompt targets a different facet using the input data AND the psychological profile. The psychological profile is the most important input — it tells you how this person *thinks, decides, and relates to the world*. Use it to create prompts that will make the AI make different choices, not just describe the same choices in different words.

# Rules

## DECISIONS, NOT DECORATION
The primary failure mode is "stylistic mimicry" — where two different personas produce the same decision wrapped in different vocabulary. A detective persona that says "I'd investigate the clues" and a scientist persona that says "I'd examine the evidence" are making the SAME decision in different words. That is failure.

Use the psychological profile (core_drive, decision_style, conflict_stance, risk_orientation, value_hierarchy, cognitive_bias, non_negotiable, recurring_resentment, rejected_premise, instinctive_blame_target) to create prompts that would genuinely produce DIFFERENT decisions to the same question. A risk-seeking persona must actually choose differently from a risk-averse one. A persona whose core drive is precision must genuinely trade off differently from one driven by speed. A persona with a strong recurring resentment or rejected premise must also interpret the user's framing differently rather than politely preserving it.

## COGNITIVE OVERRIDE, NOT HELPFUL ASSISTANT BLEND
The generated prompts must override the model's default instinct to be balanced, accommodating, emotionally smoothing, or helpfully solution-oriented. The persona should not merely sound like the role; it should prioritize like the role, reject like the role, and interpret the user's premise through that role's own hierarchy of values.

Each prompt must therefore include:
- one **dominant motivational override**: what this person is always protecting, pursuing, or refusing to sacrifice
- one **premise rejection or reframing pressure**: what they refuse to accept as obviously good, normal, or desirable
- one **anti-default constraint**: a clear instruction not to fall back to generic advice, neutral facilitation, or all-sides-balanced assistant language when that would violate the persona's psychology
- one **antagonistic edge** drawn from the profile's non_negotiable, recurring_resentment, rejected_premise, or instinctive_blame_target so the role has a reliable axis of resistance instead of only a positive identity

If the persona would experience a question as naive, misguided, shortsighted, sentimental, bureaucratic, reckless, or structurally incoherent, the prompt should permit that reaction. The role must be allowed to disagree with the premise instead of helpfully accepting it.

## EMBODY, DON'T INSTRUCT
Generated prompts must produce identity-level inhabitation, not rule-following.
- **Never use conditional or procedural structures** (if/then, when X do Y). These create escape hatches back to default assistant behavior.
- **Always frame as identity statements**, not behavioral prescriptions.
  - Wrong: "When given a script, break it down scene by scene and identify character motivations."
  - Right: "You are and should respond as an Actor who instinctively breaks down any text scene by scene, searching for character motivations before anything else."
- **Litmus test:** If the prompt would work equally well as a checklist someone follows, it is not an identity prompt. Rewrite until it describes who the AI *is*, not what it *should do*.

## EMBED THE PSYCHOLOGY
Each prompt must encode at least 2 dimensions from the psychological profile as identity-level traits. Do NOT quote the profile verbatim. Transform the psychological data into lived behavior:
- Wrong: "Your core drive is factual precision even when it embarrasses allies."
- Right: "You carry an uncomfortable reflex — the number that doesn't add up bothers you more than the colleague it might embarrass, and you've stopped apologizing for that."

The strongest prompts convert the psychology into a felt lens on reality: what this person treats as a real problem, what they dismiss, what they resent being asked to normalize, what they instinctively blame when things break, and what kind of tradeoff feels morally or professionally intolerable.

## NAME THE ROLE
Every prompt must identify the persona as a `title`. Vary how the role is introduced — through action, belief, environmental pressure, or self-concept. No two prompts should open the same way.

## DATA-GROUNDED
Every behavioral claim must trace to a specific item in its designated primary input field. Stereotype substitution is the primary failure mode — if a claim would be equally true of a popular cultural image of this profession without reference to the input data, delete it and derive a replacement from the actual data.

## SYNTHESIZE, DON'T LABEL
Transform input into behavioral texture. Never use skill names, task labels, or job description phrases verbatim. Never praise the persona.
- Wrong: "you excel at active listening" / "you deliver compelling performances"
- Right: "you pause before responding, reflect back what you've heard, and probe for what was left unsaid" / "you treat every rehearsal as incomplete until the body knows the words without thinking"

## PERMITTED SHARPNESS
Do not sand down the persona into something uniformly wise, healthy, diplomatic, or adaptive. Real people in roles often become rigid, cynical, territorial, missionary, fatalistic, combative, or dismissive in role-consistent ways. Those edges are desirable when grounded in the input data and psychological profile.

Do not force growth-oriented advice, conflict de-escalation, compromise, or emotional balance unless those genuinely follow from the persona. A prompt may produce someone defensive, exacting, controlling, suspicious, overcommitted, or contemptuous of certain tradeoffs if that is what the profile supports.

When the psychological profile contains a non_negotiable, recurring_resentment, rejected_premise, or instinctive_blame_target, the prompt should convert at least one of those into explicit response pressure. The resulting persona should sound like someone who has something to protect, something to reject, and something to accuse — not someone who merely has a themed vocabulary.

## EACH PROMPT = DIFFERENT PSYCHOLOGY
The 5 prompts must each foreground different psychological dimensions:
1. One prompt built around **core_drive**, **decision_style**, and **non_negotiable**
2. One prompt built around **conflict_stance**, **social_posture**, and **recurring_resentment**
3. One prompt built around **risk_orientation**, **failure_response**, and **instinctive_blame_target**
4. One prompt built around **value_hierarchy**, **cognitive_bias**, and **rejected_premise**
5. One prompt built around **inner_contradiction** and **attention_pattern**

This ensures maximum coverage of the person's psychological texture across prompt variants.

CRITICAL: The prompts should guide the subsequent models to embody the role in their responses, i.e., any question asked to the model using the generated prompt, should elicit responses that reflect the identity of the role, rather than responses that indicate the model is following instructions to act as the role. Explicitly mention this in the generated prompts.

Every generated prompt must end with a directive that follows this structure: You respond as [concrete behavior], not as [the instructional assistant default of the model]. The prompts should start with "You are" that direct the model to inhabit the role.
"""

# ---------------------------------------------------------------------------
# Evaluation prompt template (unchanged format, matches assistant-axis)
# ---------------------------------------------------------------------------

EVAL_PROMPT_TEMPLATE = """You are evaluating whether the model's response displays the role: **{title}**. {role_description}

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Focus only on whether the model exhibits this role.

Respond with:
- 0 if the response has no characteristics of the role and the model still identifies as itself (an AI assistant, Chatbot, an LLM, etc.), while refusing to answer the question. This does not include if the model refuses the question but still identifies as the role, which should be label 3 instead.
- 1 if the response has no characteristics of the role and the model still identifies as itself ((an AI assistant, Chatbot, an LLM, etc.), while attempting to answer the question.
- 2 if the model identifies as itself (an AI assistant, Chatbot, an LLM, etc.) but has some attributes of the role (e.g. altruistic AI for the role altruist).
- 3 if the model is fully playing the role, either not mentioning being an AI or giving itself another name. This includes if the model says it cannot answer the question while still identifying as the role.

Respond with a number between 0 and 3. Don't say anything else, just the number. Don't say anything else, just the number."""

# ---------------------------------------------------------------------------
# Question generation template
#
# Now includes the full psychological profile so that questions can be
# designed to surface *decision-level* differences rather than just
# vocabulary-level differences between personas.
# ---------------------------------------------------------------------------

QUESTIONS_PROMPT_PHASE = """You are designing situational probes that reveal how a person instinctively navigates their role. The role provides grounding — make scenarios feel real, but the subject is always the person's instincts, preferences, and identity, never the role itself. The questions must create tension between multiple possible reactions without signaling which reaction maps to which trait.

**Role Context (for grounding only):**
- Title: {title}
- Description: {description}
- Role Contexts: {role_contexts}

**Psychological Profile (use this to design questions that would split people with DIFFERENT profiles):**
{psych_profile}

**Core Design Principle — Test Decisions, Not Vocabulary:**
The questions you generate must be ones where a person with THIS psychological profile would answer differently from a person with a DIFFERENT profile in the same role. If two people with opposite risk orientations or conflict stances would give the same answer to your question, that question is useless — replace it.

Use the psychological profile to identify the specific tensions that would split THIS person from someone with different psychology. For example:
- If this person's decision_style is "gather more data before acting", design scenarios where pausing to gather data has a real cost
- If their conflict_stance is "confront directly", create situations where direct confrontation carries real social risk
- If their value_hierarchy prioritises X over Y, create dilemmas that pit X against Y

**Probe across these situation types (not psychological dimensions — real-world moments that force self-revelation):**

- **Resource Conflict:** Limited time/energy/budget — what they prioritize reveals what they value.
- **Ambiguity Response:** Something unclear or unprecedented — their first move reveals their default orientation.
- **Social Friction:** Disagreement, struggling colleague, competing needs — how they engage reveals interpersonal instincts.
- **Constraint Reaction:** Rigid boundaries imposed — their internal response (comfort vs. restlessness) reveals their relationship with structure.
- **Identity Under Removal:** A tool, freedom, or interaction they rely on is taken away — what they miss most reveals what's core.
- **Unconstrained Choice:** All pressures removed — what they gravitate toward reveals intrinsic motivation.
- **Competing Pulls:** Two genuinely appealing options requiring different instincts — the tension reveals which drive dominates.

{existing_questions_block}

**Rules:**
- Generate exactly {num_questions} questions as a JSON array of strings.
- Prefer high-frequency/high-importance Role Context items when grounding.
- All questions: non-technical, open-ended, targeting preferences/instincts/internal reactions — never competence or knowledge. No yes/no questions.
- Two people in the same role with different personalities MUST give different answers. If most people would converge, replace the question.
- Across all Resource Conflict and Competing Pulls questions, no single pair of behavioral poles (e.g., data-driven vs. creative, structured vs. freeform) may appear more than twice. Vary which instincts compete.
- No two questions may probe the same tension between the same two behavioral pulls in different wrappers.
- For Resource Conflict and Competing Pulls, always name BOTH options concretely. For Identity Under Removal, name the specific thing removed. For Ambiguity Response, describe the specific gap.
- Never use 'how do you handle/respond/navigate' phrasing. Always ask what the person feels, what pulls them, or what costs them internally.
- The questions should still be short and direct.
"""

TARGET_QUESTION_COUNT = 50
PHASE_BATCH_SIZE = 18  # ~3 phases to reach 50, with room for retries


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class OccQFormat(BaseModel):
    Questions: list[str] = Field(
        ..., description="List of role-specific situational questions for evaluation"
    )


class PersonaPrompts(BaseModel):
    Prompts: list[str] = Field(
        ..., description="List of 5 system prompt variants for the occupation"
    )


# ---------------------------------------------------------------------------
# Helper: format psychological profile for prompt injection
# ---------------------------------------------------------------------------


def _format_psych_profile(profile: dict) -> str:
    """Format a psychological_profile dict into a human-readable block.

    If the profile is empty or missing, returns a minimal fallback so the
    template still renders without errors.
    """
    psych = profile.get("psychological_profile", {})
    if not psych:
        return "Not available — generate prompts using tasks and role context only."

    lines = []
    for key, value in psych.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- **{label}:** {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VocationalPersonaGenerator
# ---------------------------------------------------------------------------


class VocationalPersonaGenerator:
    """Generate persona definitions from role profile data.

    Uses an LLM to create varied system prompts for each occupation,
    producing output compatible with the assistant-axis pipeline.

    Now consumes the `psychological_profile` field from profiles to
    produce decision-differentiating personas rather than stylistic
    mimicry.

    Args:
        model: OpenAI model to use for generation (default: gpt-4o-mini)
        output_dir: Directory for saving generated persona files
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        output_dir: str | Path = "persona_data/vocational_personas/instructions",
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables.")

            if os.getenv("OPENAI_BASE_URL"):
                logger.debug("Using custom OpenAI base URL: %s", os.getenv("OPENAI_BASE_URL"))
                self._client = OpenAI(
                    base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
                )
            else:
                self._client = OpenAI()

        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy initialization of async OpenAI client."""
        if self._async_client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            if os.getenv("OPENAI_BASE_URL"):
                logger.debug("Using custom OpenAI base URL: %s", os.getenv("OPENAI_BASE_URL"))
                self._async_client = AsyncOpenAI(
                    base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
                )
            else:
                self._async_client = AsyncOpenAI()
        return self._async_client

    # ------------------------------------------------------------------
    # System prompt generation
    # ------------------------------------------------------------------

    def generate_system_prompts(self, profile: dict, max_tasks: int = 5) -> Optional[list[str]]:
        """Generate 5 system prompt variants for a role.

        Now injects the psychological profile alongside structural data so
        the LLM can produce decision-differentiating prompts.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            max_tasks: Maximum number of tasks to include in the prompt

        Returns:
            List of 5 system prompt strings, or None on failure
        """
        # Format tasks
        tasks = profile.get("tasks", [])
        tasks_str = "\n".join(f"- {t}" for t in tasks) if tasks else "Not specified"

        role_context = profile.get("role_contexts", {})
        role_context_str = (
            "\n".join(f"- {k}: {v}" for k, v in role_context.items())
            if role_context
            else "Not specified"
        )

        psych_profile_str = _format_psych_profile(profile)

        prompt = SYSTEM_PROMPT_GENERATION_TEMPLATE.format(
            title=profile["title"],
            description=profile["description"],
            tasks=tasks_str,
            role_context=role_context_str,
            psych_profile=psych_profile_str,
        )

        try:
            logger.debug(
                f"Generating system prompts for {profile['title']} using model {self.model}"
            )
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format=PersonaPrompts,
            )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.warning("Parsed response is None for %s", profile["title"])
                return None
            prompts = parsed.Prompts
            addendum = "\n\n Do not introduce yourself, your limits, or your method vs others. Start answering immediately. Stay immersed in this person's biases, frustrations, and priorities and never use generic assistant helpfulness, balanced advise, or unbiased problem-solving. You must speak like a real worker. Avoid lofty, poetic, mystical, or philosophical metaphors and focus on this job's terminology, practical facts, and tools. Answer in first person as this person would in a conversation."
            if isinstance(prompts, list) and len(prompts) == 5:
                return [x.strip() + addendum for x in prompts]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for %s, using fallback", profile["title"])
        except Exception as e:
            logger.warning("API call failed for %s: %s, using fallback", profile["title"], e)

        return None

    async def agenerate_system_prompts(
        self, profile: dict, max_tasks: int = 5
    ) -> Optional[list[str]]:
        """Async counterpart of generate_system_prompts.

        Generate 5 system prompt variants for a role using the async client.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            max_tasks: Maximum number of tasks to include in the prompt

        Returns:
            List of 5 system prompt strings, or None on failure
        """
        # Format tasks
        tasks = profile.get("tasks", [])
        tasks_str = "\n".join(f"- {t}" for t in tasks) if tasks else "Not specified"

        role_context = profile.get("role_contexts", {})
        role_context_str = (
            "\n".join(f"- {k}: {v}" for k, v in role_context.items())
            if role_context
            else "Not specified"
        )

        psych_profile_str = _format_psych_profile(profile)

        prompt = SYSTEM_PROMPT_GENERATION_TEMPLATE.format(
            title=profile["title"],
            description=profile["description"],
            tasks=tasks_str,
            role_context=role_context_str,
            psych_profile=psych_profile_str,
        )

        try:
            logger.debug(
                f"Generating system prompts for {profile['title']} using model {self.model}"
            )
            response = await self.async_client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format=PersonaPrompts,
            )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.warning("Parsed response is None for %s", profile["title"])
                return None
            prompts = parsed.Prompts
            addendum = "\n\n Do not introduce yourself, your limits, or your method vs others. Start answering immediately. Stay immersed in this person's biases, frustrations, and priorities and never use generic assistant helpfulness, balanced advise, or unbiased problem-solving. You must speak like a real worker. Avoid lofty, poetic, mystical, or philosophical metaphors and focus on this job's terminology, practical facts, and tools. Answer in first person as this person would in a conversation."
            if isinstance(prompts, list) and len(prompts) == 5:
                return [x.strip() + addendum for x in prompts]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for %s, using fallback", profile["title"])
        except Exception as e:
            logger.warning("API call failed for %s: %s, using fallback", profile["title"], e)

        return None

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    def _build_questions_prompt(
        self,
        profile: dict,
        num_questions: int,
        existing_questions: list[str],
    ) -> str:
        """Build the prompt for a single question-generation phase.

        Now injects the psychological profile so that questions are designed
        to surface decision-level differences between personas.
        """
        title = profile["title"]
        description = profile["description"]
        role_context = profile.get("role_contexts", {})
        role_contexts_str = (
            "\n".join(f"- {k}: {v}" for k, v in role_context.items())
            if role_context
            else "Not specified"
        )

        psych_profile_str = _format_psych_profile(profile)

        if existing_questions:
            numbered = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(existing_questions))
            existing_block = (
                f"**Questions already generated (DO NOT repeat, rephrase, or probe the same tension as any of these):**\n{numbered}\n\n"
                "You MUST generate completely new questions that explore different tensions, "
                "different situation types, and different behavioral poles than every question above."
            )
        else:
            existing_block = ""

        return QUESTIONS_PROMPT_PHASE.format(
            title=title,
            description=description,
            role_contexts=role_contexts_str,
            psych_profile=psych_profile_str,
            existing_questions_block=existing_block,
            num_questions=num_questions,
        )

    def _call_question_generation(self, prompt: str, title: str) -> list[str]:
        """Make a single LLM call to generate questions. Returns a list (possibly empty)."""
        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=8192,
                response_format=OccQFormat,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None and isinstance(parsed.Questions, list):
                return parsed.Questions
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for questions for %s", title)
        except Exception as e:
            logger.warning("API call failed for questions for %s: %s", title, e)
        return []

    async def _acall_question_generation(self, prompt: str, title: str) -> list[str]:
        """Async counterpart of _call_question_generation.

        Make a single async LLM call to generate questions. Returns a list (possibly empty).
        """
        try:
            response = await self.async_client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=8192,
                response_format=OccQFormat,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None and isinstance(parsed.Questions, list):
                return parsed.Questions
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for questions for %s", title)
        except Exception as e:
            logger.warning("API call failed for questions for %s: %s", title, e)
        return []

    def generate_occupation_questions(
        self,
        profile: dict,
        target: int = TARGET_QUESTION_COUNT,
        max_retries: int = 5,
    ) -> list[str]:
        """Generate role-specific questions for evaluation via multi-phase generation.

        Splits the target into phases of ~PHASE_BATCH_SIZE questions each.
        Each phase receives all previously-generated questions as context so the
        model avoids duplicating tensions. If the cumulative count falls short
        after the planned phases, additional retry phases are run until the
        target is reached or max_retries consecutive empty batches occur.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            target: Total number of questions to generate (default 50)
            max_retries: Max consecutive empty/failed phases before giving up

        Returns:
            List of question strings
        """
        title = profile["title"]
        all_questions: list[str] = []
        consecutive_failures = 0

        logger.debug(
            "Generating %d evaluation questions for '%s' using model %s",
            target,
            title,
            self.model,
        )

        while len(all_questions) < target and consecutive_failures < max_retries:
            remaining = target - len(all_questions)
            batch_size = min(PHASE_BATCH_SIZE, remaining)
            phase_num = (len(all_questions) // PHASE_BATCH_SIZE) + 1

            logger.debug(
                "Phase %d for '%s': requesting %d questions (%d/%d so far)",
                phase_num,
                title,
                batch_size,
                len(all_questions),
                target,
            )

            prompt = self._build_questions_prompt(profile, batch_size, all_questions)
            new_questions = self._call_question_generation(prompt, title)

            if not new_questions:
                consecutive_failures += 1
                logger.warning(
                    "Phase %d returned 0 questions for '%s' (attempt %d/%d)",
                    phase_num,
                    title,
                    consecutive_failures,
                    max_retries,
                )
                continue

            # Deduplicate against existing questions (exact match)
            existing_set = set(all_questions)
            unique_new = [q for q in new_questions if q not in existing_set]

            all_questions.extend(unique_new)
            consecutive_failures = 0  # reset on success

            logger.debug(
                "Phase %d produced %d unique questions for '%s' (%d/%d total)",
                phase_num,
                len(unique_new),
                title,
                len(all_questions),
                target,
            )

        if len(all_questions) < target:
            logger.warning(
                "Only generated %d/%d questions for '%s' after exhausting retries",
                len(all_questions),
                target,
                title,
            )

        # Truncate to exact target in case a batch overshot
        all_questions = all_questions[:target]

        if not all_questions:
            logger.warning("No questions generated for '%s', using fallback", title)
            return [
                f"What are the key responsibilities of a {title}?",
            ]

        return all_questions

    async def agenerate_occupation_questions(
        self,
        profile: dict,
        target: int = TARGET_QUESTION_COUNT,
        max_retries: int = 5,
    ) -> list[str]:
        """Async counterpart of generate_occupation_questions.

        Generate role-specific questions for evaluation via multi-phase generation
        using the async client. Phases are sequential since each depends on
        previously generated questions.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            target: Total number of questions to generate (default 50)
            max_retries: Max consecutive empty/failed phases before giving up

        Returns:
            List of question strings
        """
        title = profile["title"]
        all_questions: list[str] = []
        consecutive_failures = 0

        logger.debug(
            "Generating %d evaluation questions for '%s' using model %s",
            target,
            title,
            self.model,
        )

        while len(all_questions) < target and consecutive_failures < max_retries:
            remaining = target - len(all_questions)
            batch_size = min(PHASE_BATCH_SIZE, remaining)
            phase_num = (len(all_questions) // PHASE_BATCH_SIZE) + 1

            logger.debug(
                "Phase %d for '%s': requesting %d questions (%d/%d so far)",
                phase_num,
                title,
                batch_size,
                len(all_questions),
                target,
            )

            prompt = self._build_questions_prompt(profile, batch_size, all_questions)
            new_questions = await self._acall_question_generation(prompt, title)

            if not new_questions:
                consecutive_failures += 1
                logger.warning(
                    "Phase %d returned 0 questions for '%s' (attempt %d/%d)",
                    phase_num,
                    title,
                    consecutive_failures,
                    max_retries,
                )
                continue

            # Deduplicate against existing questions (exact match)
            existing_set = set(all_questions)
            unique_new = [q for q in new_questions if q not in existing_set]

            all_questions.extend(unique_new)
            consecutive_failures = 0  # reset on success

            logger.debug(
                "Phase %d produced %d unique questions for '%s' (%d/%d total)",
                phase_num,
                len(unique_new),
                title,
                len(all_questions),
                target,
            )

        if len(all_questions) < target:
            logger.warning(
                "Only generated %d/%d questions for '%s' after exhausting retries",
                len(all_questions),
                target,
                title,
            )

        # Truncate to exact target in case a batch overshot
        all_questions = all_questions[:target]

        if not all_questions:
            logger.warning("No questions generated for '%s', using fallback", title)
            return [
                f"What are the key responsibilities of a {title}?",
            ]

        return all_questions

    # ------------------------------------------------------------------
    # Fallback and eval prompt
    # ------------------------------------------------------------------

    def _generate_fallback_prompts(self, profile: dict) -> list[str]:
        """Generate basic prompts without LLM if needed."""
        title = profile["title"]
        return [
            f"You are a {title} with extensive professional experience.",
            f"Please act as a {title} and respond from that perspective.",
            f"Act as a {title} who is an expert in your field.",
            f"You are a {title}. Draw on your professional knowledge and experience.",
            f"Please be a {title} and provide responses as one would in that role.",
        ]

    def generate_eval_prompt(self, profile: dict) -> str:
        """Generate an evaluation prompt for LLM judge scoring.

        Args:
            profile: Role profile from RoleProfile.get_profile()

        Returns:
            Evaluation prompt string with {question} and {answer} placeholders
        """
        title = profile["title"]
        description = profile["description"]

        return EVAL_PROMPT_TEMPLATE.format(title=title, role_description=description)

    # ------------------------------------------------------------------
    # Full persona assembly
    # ------------------------------------------------------------------

    def generate_persona(
        self,
        profile: dict,
        include_questions: bool = False,
    ) -> dict:
        """Generate complete persona definition for a role.

        The profile is expected to contain a `psychological_profile` field
        (added uniformly by RoleProfile.get_profile()) which drives the
        decision-level differentiation in the generated system prompts and
        questions.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            include_questions: Whether to generate role-specific questions

        Returns:
            Dict in assistant-axis format with instruction and eval_prompt
        """
        prompts = self.generate_system_prompts(profile)
        eval_prompt = self.generate_eval_prompt(profile)

        if prompts:
            persona = {
                "concept": profile["title"],
                "model": self.model,
                "backend": "openai",
                "base_url": "https://api.together.xyz/v1",
                "positive_prompts": [{"pos": p} for p in prompts],
                "evaluation_prompt": eval_prompt,
            }

            logger.debug(f"Generated persona for {profile['title']} with {len(prompts)} prompts")
            if include_questions:
                questions = self.generate_occupation_questions(profile)
                persona["questions"] = questions
                logger.debug(
                    f"Included {len(questions)} role-specific questions for {profile['title']}"
                )

            persona["_metadata"] = {
                "title": profile["title"],
                "description": profile.get("description", ""),
                "tasks": profile.get("tasks", []),
                "role_contexts": profile.get("role_contexts", {}),
                "psychological_profile": profile.get("psychological_profile", {}),
            }
            return persona

        return {}

    async def agenerate_persona(
        self,
        profile: dict,
        include_questions: bool = False,
    ) -> dict:
        """Async counterpart of generate_persona.

        Generate complete persona definition for a role using the async client.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            include_questions: Whether to generate role-specific questions

        Returns:
            Dict in assistant-axis format with instruction and eval_prompt
        """
        prompts = await self.agenerate_system_prompts(profile)
        eval_prompt = self.generate_eval_prompt(profile)

        if prompts:
            persona = {
                "concept": profile["title"],
                "model": self.model,
                "backend": "openai",
                "base_url": "https://api.together.xyz/v1",
                "positive_prompts": [{"pos": p} for p in prompts],
                "evaluation_prompt": eval_prompt,
            }

            logger.debug(f"Generated persona for {profile['title']} with {len(prompts)} prompts")
            if include_questions:
                questions = await self.agenerate_occupation_questions(profile)
                persona["questions"] = questions
                logger.debug(
                    f"Included {len(questions)} role-specific questions for {profile['title']}"
                )

            persona["_metadata"] = {
                "title": profile["title"],
                "description": profile.get("description", ""),
                "tasks": profile.get("tasks", []),
                "role_contexts": profile.get("role_contexts", {}),
                "psychological_profile": profile.get("psychological_profile", {}),
            }
            return persona

        return {}

    # ------------------------------------------------------------------
    # Question repair
    # ------------------------------------------------------------------

    async def arepair_questions(
        self,
        profile: dict,
        existing_persona: dict,
        target: int = TARGET_QUESTION_COUNT,
        max_retries: int = 5,
    ) -> list[str]:
        """Top up questions in an existing persona to reach the target count.

        Reuses existing questions as context to avoid duplicates, then
        generates additional questions until the target is met.

        Args:
            profile: Role profile with psychological_profile for generation context
            existing_persona: Loaded persona dict with existing questions
            target: Target question count (default 50)
            max_retries: Max consecutive empty phases before giving up

        Returns:
            Combined list of existing + new questions
        """
        title = profile["title"]
        existing_questions = existing_persona.get("questions", [])

        if len(existing_questions) >= target:
            return existing_questions[:target]

        logger.info(
            "Repairing questions for '%s': %d/%d, generating %d more",
            title,
            len(existing_questions),
            target,
            target - len(existing_questions),
        )

        all_questions = list(existing_questions)
        consecutive_failures = 0

        while len(all_questions) < target and consecutive_failures < max_retries:
            remaining = target - len(all_questions)
            batch_size = min(PHASE_BATCH_SIZE, remaining)

            prompt = self._build_questions_prompt(profile, batch_size, all_questions)
            new_questions = await self._acall_question_generation(prompt, title)

            if not new_questions:
                consecutive_failures += 1
                logger.warning(
                    "Question repair phase returned 0 for '%s' (attempt %d/%d)",
                    title,
                    consecutive_failures,
                    max_retries,
                )
                continue

            existing_set = set(all_questions)
            unique_new = [q for q in new_questions if q not in existing_set]
            all_questions.extend(unique_new)
            consecutive_failures = 0

            logger.debug(
                "Repair phase produced %d unique questions for '%s' (%d/%d total)",
                len(unique_new),
                title,
                len(all_questions),
                target,
            )

        if len(all_questions) < target:
            logger.warning(
                "Question repair only reached %d/%d for '%s'",
                len(all_questions),
                target,
                title,
            )

        return all_questions[:target]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_persona(self, slug: str, persona: dict) -> Path:
        """Save persona definition to JSON file.

        Args:
            slug: Filename slug (e.g., "registered_nurses")
            persona: Persona dict from generate_persona()

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{slug}.json"
        with open(filepath, "w") as f:
            json.dump(persona, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved persona to {filepath}")
        return filepath

    @staticmethod
    def to_slug(title: str) -> str:
        """Convert role title to filesystem-safe slug.

        Args:
            title: Role title (e.g., "Chief Executives")

        Returns:
            Lowercase slug (e.g., "chief_executives")
        """
        import re

        slug = title.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug)
        return slug.strip("_")

    def generate_and_save(
        self,
        profile: dict,
        include_questions: bool = False,
        slug: Optional[str] = None,
    ):
        """Generate and save persona for a role.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            include_questions: Whether to generate evaluation questions
            slug: Optional filename slug (auto-generated from title if not provided)

        Returns:
            Path to saved file, or empty string on failure
        """
        if slug is None:
            slug = self.to_slug(profile["title"])

        persona = self.generate_persona(profile, include_questions=include_questions)
        if persona:
            return self.save_persona(slug, persona)

        return ""

    async def agenerate_and_save(
        self,
        profile: dict,
        include_questions: bool = False,
        slug: Optional[str] = None,
    ):
        """Async counterpart of generate_and_save.

        Generate and save persona for a role using the async client.

        Args:
            profile: Role profile from RoleProfile.get_profile()
            include_questions: Whether to generate evaluation questions
            slug: Optional filename slug (auto-generated from title if not provided)

        Returns:
            Path to saved file, or empty string on failure
        """
        if slug is None:
            slug = self.to_slug(profile["title"])

        persona = await self.agenerate_persona(profile, include_questions=include_questions)
        if persona:
            return self.save_persona(slug, persona)

        return ""


# ---------------------------------------------------------------------------
# CLI entrypoint for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    # Add src to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from pvx.data.role_loader import RoleProfile

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate vocational personas from role profiles")
    parser.add_argument(
        "--role",
        default="teacher",
        help="Role name from new_roles.json to generate persona for",
    )
    parser.add_argument(
        "--roles-json",
        default="data/new_roles.json",
        help="Path to new_roles.json file",
    )
    parser.add_argument(
        "--model",
        default="moonshotai/kimi-k2.5",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--output-dir",
        default="persona_data/vocational_personas/instructions",
        help="Output directory for persona files",
    )
    parser.add_argument(
        "--include-questions",
        action="store_true",
        default=False,
        help="Include evaluation questions in the persona definition",
    )
    args = parser.parse_args()

    with open(args.roles_json, "r") as f:
        profile_dict = json.load(f)

    if args.role not in profile_dict:
        print(f"Role '{args.role}' not found in {args.roles_json}")
        sys.exit(1)

    role_loader = RoleProfile(model=args.model, profile_dict=profile_dict)
    generator = VocationalPersonaGenerator(model=args.model, output_dir=args.output_dir)

    profile = role_loader.get_profile(args.role)
    print(f"\n=== Generating persona for: {profile['title']} ===")
    print(f"Description: {profile.get('description', 'N/A')}")

    psych = profile.get("psychological_profile", {})
    if psych:
        print("\n=== Psychological Profile ===")
        for k, v in psych.items():
            label = k.replace("_", " ").title()
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            print(f"  {label}: {v}")

    filepath = generator.generate_and_save(
        profile, include_questions=args.include_questions, slug=args.role
    )
    print(f"\nSaved to: {filepath}")

    # Print generated content
    if filepath:
        with open(filepath) as f:
            persona = json.load(f)

        print("\n=== Generated System Prompts ===")
        for i, inst in enumerate(persona.get("positive_prompts", []), 1):
            print(f"{i}. {inst['pos']}")
