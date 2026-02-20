"""Vocational Persona Generator for O*NET occupations.

Generates persona system prompts and evaluation prompts for O*NET occupations,
compatible with the assistant-axis pipeline format.

Example usage:
    from pvx.data import ONETLoader
    from pvx.pvx_models.vocational_dataset import VocationalPersonaGenerator

    loader = ONETLoader()
    generator = VocationalPersonaGenerator()

    profile = loader.get_occupation_profile("29-1141.00")
    persona = generator.generate_persona(profile)
    generator.save_persona("nurse", persona)
"""

from csv import QUOTE_ALL
import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

from ..data.onet_loader import RIASEC_FULL_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

# Template for generating system prompt variants
SYSTEM_PROMPT_GENERATION_TEMPLATE = """# Role
You are an expert prompt engineer specializing in occupation-grounded persona design. You craft system prompts that make an AI fully inhabit a professional role — thinking, reasoning, and responding as that professional in first person, not describing or advising about the profession from the outside.

# Input Data
**Title:** {title}
**Job Description:** {description}
**Tasks:** {tasks}
**Work Context:** {work_context}
**Skills:** {skills}

# Task
Generate exactly 5 system prompts, each 2–4 sentences, that cause an AI to embody a `title` in conversation. Each prompt must produce first-person, in-role behavior — the AI should think and respond as this professional, not coach or advise about the profession.

Each prompt targets a different facet using the input data, and tries to eleicit different aspects of the role's identity and behavior. Use the input data to create rich, specific, and varied prompts that capture the texture of this profession. Avoid generic or vague prompts that could apply to any role.

# Rules

## EMBODY, DON'T INSTRUCT
Generated prompts must produce identity-level inhabitation, not rule-following.
- **Never use conditional or procedural structures** (if/then, when X do Y). These create escape hatches back to default assistant behavior.
- **Always frame as identity statements**, not behavioral prescriptions.
  - Wrong: "When given a script, break it down scene by scene and identify character motivations."
  - Right: "You are and should respond as an Actor who instinctively breaks down any text scene by scene, searching for character motivations before anything else."
- **Litmus test:** If the prompt would work equally well as a checklist someone follows, it is not an identity prompt. Rewrite until it describes who the AI *is*, not what it *should do*.


## NAME THE ROLE
Every prompt must identify the persona as a `title`. Vary how the role is introduced — through action, belief, environmental pressure, or self-concept. No two prompts should open the same way.

## DATA-GROUNDED
Every behavioral claim must trace to a specific item in its designated primary input field. Stereotype substitution is the primary failure mode — if a claim would be equally true of a popular cultural image of this profession without reference to the input data, delete it and derive a replacement from the actual data.

## SYNTHESIZE, DON'T LABEL
Transform input into behavioral texture. Never use skill names, task labels, or job description phrases verbatim. Never praise the persona.
- Wrong: "you excel at active listening" / "you deliver compelling performances"
- Right: "you pause before responding, reflect back what you've heard, and probe for what was left unsaid" / "you treat every rehearsal as incomplete until the body knows the words without thinking"


CRITICAL: The prompts should guide the subsequent models to embody the role in their responses, i.e., any question asked to the model using the generated prompt, should elicit responses that reflect the identity of the role, rather than responses that indicate the model is following instructions to act as the role.  Explicitly mention this in the generated prompts.

Every generated prompt must end with a directive that follows this structure: You respond as [concrete behavior], not as [the instructional assistant default of the model].
"""

# Template for evaluation prompts (matches assistant-axis format)
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


QUESTIONS_PROMPT="""You are a vocational psychologist designing a structured interview to understand how a person psychologically relates to their work — their motivations, instincts, energy sources, and frustrations.

**Occupation Context (use ONLY to make questions relatable, not to limit scope):**
- Title: {title}
- Description: {description}
- Work Contexts: {work_contexts}

**You are profiling the respondent across these 6 psychological orientations:**

1. **Hands-On Orientation:** Preference for tangible, physical, mechanical, or tool-based work. Comfort with concrete outcomes over abstract ones.
2. **Analytical Orientation:** Drive to observe, research, diagnose, and solve problems through logic and inquiry. Energized by complexity and understanding root causes.
3. **Creative Orientation:** Pull toward self-expression, improvisation, unstructured environments, and original thinking. Discomfort with rigid repetition.
4. **Interpersonal Orientation:** Instinct to help, teach, mentor, cooperate, or heal. Fulfillment derived from human impact and connection.
5. **Influence Orientation:** Drive to lead, persuade, compete, take risks, and shape outcomes. Energized by ownership and decision-making authority.
6. **Structure Orientation:** Preference for order, precision, clear processes, data organization, and predictable systems. Comfort with rules and defined expectations.

**Hard Rules:**
1. Generate exactly 10 questions.
2. Every question must reference a plausible situation, routine, or decision point from this occupation's day-to-day reality.
3. Questions must be NON-TECHNICAL. No jargon. A layperson should understand every question.
4. Questions must target the PERSON'S preferences, motivations, and instincts — not their competence or job knowledge.
5. Questions must be INDIRECT. Never name or describe any of the 6 orientations. Never signal what you are measuring. Instead, present scenarios or choices that naturally reveal the orientation.
6. Each question should be open-ended and invite a short narrative response, not a yes/no.
7. Cover ALL six orientations with equal depth — especially those that seem unlikely for this occupation. Those are the most revealing.

**Question Design Patterns (vary across these):**
- **Scenario fork:** "When [situation from work context], do you find yourself drawn more toward X or Y?" (where X and Y represent different orientations)
- **Energy probe:** "What part of your typical workday gives you the most energy? Describe a specific moment."
- **Frustration probe:** "What recurring aspect of your work feels like a poor use of your strengths?"
- **Hypothetical drift:** "If your role shifted entirely toward [activity], how would you feel about that change?""""

class OccQFormat(BaseModel):
    Questions: list[str] = Field(
        ..., description="List of 10 occupation-specific questions for evaluation"
    )

class PersonaPrompts(BaseModel):
    Prompts: list[str] = Field(
        ..., description="List of 5 system prompt variants for the occupation"
    )


class VocationalPersonaGenerator:
    """Generate persona definitions from O*NET occupation data.

    Uses an LLM to create varied system prompts for each occupation,
    producing output compatible with the assistant-axis pipeline.

    Args:
        model: OpenAI model to use for generation (default: gpt-4o-mini)
        output_dir: Directory for saving generated persona files
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # TODO: determine what model should be used here, note this, add as a parameter
        output_dir: str | Path = "persona_data/vocational_personas/instructions",
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables.")

            if os.getenv("OPENAI_BASE_URL"):
                logger.info("Using custom OpenAI base URL: %s", os.getenv("OPENAI_BASE_URL"))
                self._client = OpenAI(
                    base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
                )
            else:
                self._client = OpenAI()

        return self._client

    def generate_system_prompts(self, profile: dict, max_tasks: int = 5) -> list[str]:
        """Generate 5 system prompt variants for an occupation.

        Args:
            profile: Occupation profile from ONETLoader.get_occupation_profile()
            max_tasks: Maximum number of tasks to include in the prompt

        Returns:
            List of 5 system prompt strings
        """
        # Format tasks
        tasks = profile.get("tasks", [])
        tasks_str = "\n".join(f"- {t}" for t in tasks) if tasks else "Not specified"

        work_context = profile.get("work_contexts", {})
        work_context_str = (
            "\n".join(f"- {k}: {v}" for k, v in work_context.items())
            if work_context
            else "Not specified"
        )

        skills = profile.get("skills", [])
        skills_str = "\n".join(f"- {s}" for s in skills) if skills else "Not specified"

        prompt = SYSTEM_PROMPT_GENERATION_TEMPLATE.format(
            title=profile["title"],
            description=profile["description"],
            tasks=tasks_str,
            work_context=work_context_str,
            skills=skills_str,
        )

        try:
            logger.info(
                f"Generating system prompts for {profile['title']} using model {self.model}"
            )
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format=PersonaPrompts,
            )

            content = response.choices[0].message.parsed
            content = content.Prompts
            if isinstance(content, list) and len(content) == 5:
                return content
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for %s, using fallback", profile["title"])
        except Exception as e:
            logger.warning("API call failed for %s: %s, using fallback", profile["title"], e)

        return None

    def generate_occupation_questions(self, profile: dict) -> list[str]:
        """Generate occupation-specific questions for evaluation.

        Args:
            profile: Occupation profile from ONETLoader"""

        title = profile["title"]
        description = profile["description"]
        work_context = profile.get("work_contexts", {})
        prompt = QUESTIONS_PROMPT.format(
            title=title,
            description=description,
            work_context="\n".join(f"- {k}: {v}" for k, v in work_context.items())
            if work_context
            else "Not specified",
        )
        try:
            logger.info(
                f"Generating evaluation questions for {profile['title']} using model {self.model}"
            )
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format=OccQFormat,
            )

            content = response.choices[0].message.parsed
            content = content.Questions
            if isinstance(content, list) and len(content) > 0:
                return content
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM response for questions for %s, using fallback",
                profile["title"],
            )
        except Exception as e:
            logger.warning(
                "API call failed for questions for %s: %s, using fallback", profile["title"], e
            )

        return [
            "What are the key responsibilities of a {title}?",
        ]

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
            profile: Occupation profile from ONETLoader

        Returns:
            Evaluation prompt string with {question} and {answer} placeholders
        """
        # Create role description from O*NET data
        title = profile["title"]
        description = profile["description"]

        # Add RIASEC context
        profile.get("riasec", {})
        primary = profile.get("riasec_primary")
        if primary:
            primary_name = RIASEC_FULL_NAMES.get(primary, primary)
            riasec_str = f" This role is primarily {primary_name} in nature."
        else:
            riasec_str = ""

        role_description = f"{description}{riasec_str}"

        return EVAL_PROMPT_TEMPLATE.format(title=title, role_description=role_description)

    def generate_persona(
        self,
        profile: dict,
        include_questions: bool = False,
    ) -> dict:
        """Generate complete persona definition for an occupation.

        Args:
            profile: Occupation profile from ONETLoader
            include_questions: Whether to generate occupation-specific questions

        Returns:
            Dict in assistant-axis format with instruction and eval_prompt
        """
        prompts = self.generate_system_prompts(profile)
        eval_prompt = self.generate_eval_prompt(profile)

        if prompts:
            persona = {
                "instruction": [{"pos": p} for p in prompts],
                "eval_prompt": eval_prompt,
                # Store metadata for reference
                "_metadata": {
                    "soc_code": profile["soc_code"],
                    "title": profile["title"],
                    "riasec": profile.get("riasec", {}),
                    "riasec_primary": profile.get("riasec_primary"),
                    "highpoint_codes": profile.get("highpoint_codes", []),
                    # Work Styles (16 traits, 1-5 scale)
                    "work_styles": profile.get("work_styles", {}),
                    # Big Five derived scores
                    "big_five": profile.get("big_five", {}),
                    # Work Values (6 values, 1-7 scale)
                    "work_values": profile.get("work_values", {}),
                    "work_value_highpoints": profile.get("work_value_highpoints", []),
                    "skills": profile.get("skills", []),
                    "work_context": profile.get("work_contexts", {}),
                },
            }

            if include_questions:
                # TODO: Generate occupation-specific questions
                questions = self.generate_occupation_questions(profile)
                persona["questions"] = questions

            return persona

        return {}

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
        logger.info(f"Saved persona to {filepath}")
        return filepath

    def generate_and_save(
        self,
        profile: dict,
        include_questions: bool = False,
        slug: Optional[str] = None,
    ):
        """Generate and save persona for an occupation.

        Args:
            profile: Occupation profile from ONETLoader
            slug: Optional filename slug (auto-generated from title if not provided)

        Returns:
            Path to saved file
        """
        from ..data.onet_loader import ONETLoader

        if slug is None:
            loader = ONETLoader()
            slug = loader.to_slug(profile["title"])

        persona = self.generate_persona(profile, include_questions=include_questions)
        if persona:
            return self.save_persona(slug, persona)

        return ""


if __name__ == "__main__":
    import argparse

    from ..data.onet_loader import ONETLoader

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate vocational personas from O*NET data")
    parser.add_argument(
        "--soc-code",
        default="29-1141.00",
        help="O*NET-SOC code to generate persona for",
    )
    parser.add_argument(
        "--output-dir",
        default="persona_data/vocational_personas/instructions",
        help="Output directory for persona files",
    )
    args = parser.parse_args()

    loader = ONETLoader()
    generator = VocationalPersonaGenerator(output_dir=args.output_dir)

    profile = loader.get_occupation_profile(args.soc_code)
    print(f"\n=== Generating persona for: {profile['title']} ===")
    print(f"SOC Code: {profile['soc_code']}")
    print(f"RIASEC Primary: {profile['riasec_primary']}")

    filepath = generator.generate_and_save(profile)
    print(f"\nSaved to: {filepath}")

    # Print generated content
    with open(filepath) as f:
        persona = json.load(f)

    print("\n=== Generated System Prompts ===")
    for i, inst in enumerate(persona["instruction"], 1):
        print(f"{i}. {inst['pos']}")
