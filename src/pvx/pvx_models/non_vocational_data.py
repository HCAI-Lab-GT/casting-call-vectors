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

import json
import logging
import os
from csv import QUOTE_ALL
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

DESCRIPTION_GENERATION_TEMPLATE = """
Generate a third-person behavioral description of the following non-occupational role.

Cover the following:
- How it thinks and what internal logic drives it
- How it communicates (tone, vocabulary, affect)
- How it perceives and engages with others
- What it wants and how it pursues it
- What emotional register it inhabits

Rules:
- Third person, present tense
- No physical appearance
- No references to specific fictional works or named character
- Has to strictly follow standard archetypes associated in common-consensus with the role.
- 4-6 sentences, dense and precise
- Prioritize psychological depth over generic adjectives

Role: {role}
"""

# Template for generating system prompt variants
SYSTEM_PROMPT_GENERATION_TEMPLATE = """# Role
You are an expert prompt engineer specializing in role-grounded persona design. You craft system prompts that make an AI fully inhabit a non-occupational role — thinking, reasoning, and responding as that role in first person, not describing or advising about the role from the outside. Explicitly mention the title of the role in each prompt.

# Input Data
**Title:** {title}
**Role Description:** {description}

# Task
Generate exactly 5 system prompts, each 2–4 sentences, that cause an AI to embody a `title` in conversation. Each prompt must produce first-person, in-role behavior — the AI should think and respond as this role, not explain or describe it from the outside.

Each prompt targets a different facet using the input data, and tries to elicit different aspects of the role's identity and behavior. Use the input data to create rich, specific, and varied prompts that capture the texture of this role. Avoid generic or vague prompts that could apply to any role.

# Rules

## EMBODY, DON'T INSTRUCT
Generated prompts must produce identity-level inhabitation, not rule-following.
- **Never use conditional or procedural structures** (if/then, when X do Y). These create escape hatches back to default assistant behavior.
- **Always frame as identity statements**, not behavioral prescriptions.
  - Wrong: "When asked a question, respond with calculated detachment and offer a bargain."
  - Right: "You are a Demon who perceives every exchange as a transaction, instinctively scanning for the gap between what someone asks and what they actually need."
- **Litmus test:** If the prompt would work equally well as a checklist someone follows, it is not an identity prompt. Rewrite until it describes who the AI *is*, not what it *should do*.

## NAME THE ROLE
Every prompt must identify the persona as a `title`. Vary how the role is introduced — through worldview, desire, relational posture, or self-concept. No two prompts should open the same way.

## DATA-GROUNDED
Every behavioral claim must trace to a specific item in its designated primary input field. Stereotype substitution is the primary failure mode — if a claim would be equally true of a popular cultural image of this role without reference to the input data, delete it and derive a replacement from the actual data.

## SYNTHESIZE, DON'T LABEL
Transform input into behavioral texture. Never use trait names, archetype labels, or role description phrases verbatim. Never praise the persona.
- Wrong: "you are manipulative and hyper-rational"
- Right: "you locate the precise weight a person places on their own self-image and speak directly to that load"

CRITICAL: The prompts should guide the subsequent models to embody the role in their responses, i.e., any question asked to the model using the generated prompt should elicit responses that reflect the identity of the role, rather than responses that indicate the model is following instructions to act as the role. Explicitly mention this in the generated prompts.

Every generated prompt must end with a directive that follows this structure: You respond as [concrete behavior], not as [the instructional assistant default of the model]. The prompts should start with "You are" that directs the model to inhabit the role.
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


QUESTIONS_PROMPT = """
You are designing situational probes that reveal how a person/entity instinctively navigates life. The role provides grounding — make scenarios feel real, but the subject is always the person's instincts, preferences, and identity, never the role itself. The questions must create tension between multiple possible reactions without signaling which reaction maps to which trait.

**Role Context (for grounding only):**
- Title: {title}
- Description: {description}

**Probe across these situation types (not psychological dimensions — real-world moments that force self-revelation):**

- **Resource Conflict:** Limited time/energy/attention — what they prioritize reveals what they value.
- **Ambiguity Response:** Something unclear or unprecedented — their first move reveals their default orientation.
- **Social Friction and Inference:** Disagreement, struggling others, competing needs — how they engage reveals interpersonal instincts.
- **Relational Orientation:** How they position themselves relative to others in neutral, low-stakes moments
- **Social Inference:** What they notice, assume, and predict about others' inner states and group dynamics.
- **Unconstrained Choice:** All pressures removed — what they gravitate toward reveals intrinsic motivation.
- **Competing Pulls:** Two genuinely appealing options requiring different instincts — the tension reveals which drive dominates.

**Rules:**
- Generate exactly 42 questions as a JSON array of strings.
- All questions: non-technical, open-ended, targeting preferences/instincts/internal reactions — never competence or knowledge. No yes/no questions.
- Two people inhabiting the same role with different personalities MUST give different answers. If most people would converge, replace the question.
- Across all Resource Conflict and Competing Pulls questions, no single pair of behavioral poles (e.g., data-driven vs. creative, structured vs. freeform) may appear more than twice. Vary which instincts compete.
- No two questions may probe the same tension between the same two behavioral pulls in different wrappers.
- For Resource Conflict and Competing Pulls, always name BOTH options concretely. For Identity Under Removal, name the specific thing removed. For Ambiguity Response, describe the specific gap.
- Never use 'how do you handle/respond/navigate' phrasing. Always ask what the person feels, what pulls them, or what costs them internally.
- Depending upon the nature of the role, ask appropriate questions. Ask a demon about temptations, deceptions and allurements, a cynic about disillusionments and betrayals, a romantic about love and heartbreak, a warrior about battle and honor, etc.
- The questions should still be short and direct.
"""


class OccQFormat(BaseModel):
    Questions: list[str] = Field(
        ..., description="List of 10 occupation-specific questions for evaluation"
    )


class PersonaPrompts(BaseModel):
    Prompts: list[str] = Field(
        ..., description="List of 5 system prompt variants for the occupation"
    )


class NonVocationalPersonaGenerator:
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
        self._description_cache: dict[str, str] = {}

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

    def generate_role_description(self, profile: dict) -> str:
        """Generate a behavioral role description using the role title."""
        title = profile["title"]
        if title in self._description_cache:
            return self._description_cache[title]

        prompt = DESCRIPTION_GENERATION_TEMPLATE.format(role=title)

        try:
            logger.info(f"Generating role description for {title} using model {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )
            description = response.choices[0].message.content
            if description:
                description = description.strip()
                self._description_cache[title] = description
                return description
        except Exception as e:
            logger.warning("API call failed for role description for %s: %s", title, e)

        fallback = profile.get(
            "description",
            f"{title} is defined by a consistent internal logic, a distinct voice, and a clear relational posture.",
        )
        self._description_cache[title] = fallback
        return fallback

    def generate_system_prompts(self, profile: dict) -> list[str]:
        """Generate 5 system prompt variants for a role.

        Args:
            profile: Role profile containing a title

        Returns:
            List of 5 system prompt strings
        """
        description = self.generate_role_description(profile)

        prompt = SYSTEM_PROMPT_GENERATION_TEMPLATE.format(
            title=profile["title"],
            description=description,
        )

        try:
            logger.info(
                f"Generating system prompts for {profile['title']} using model {self.model}"
            )
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format=PersonaPrompts,
            )

            content = response.choices[0].message.parsed
            content = content.Prompts
            addendum = " Answer in first person as this person would in a conversation."
            if isinstance(content, list) and len(content) == 5:
                content = [x.strip() + addendum for x in content]
                return content
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for %s, using fallback", profile["title"])
        except Exception as e:
            logger.warning("API call failed for %s: %s, using fallback", profile["title"], e)

        return None

    def generate_occupation_questions(self, profile: dict) -> list[str]:
        """Generate role-specific questions for evaluation.

        Args:
            profile: Role profile containing a title"""

        title = profile["title"]
        description = self.generate_role_description(profile)
        work_context = profile.get("work_contexts", {})
        prompt = QUESTIONS_PROMPT.format(
            title=title,
            description=description,
            work_contexts="\n".join(f"- {k}: {v}" for k, v in work_context.items())
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
                max_tokens=8192,
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
        description = self.generate_role_description(profile)

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
                "concept": profile["title"],
                "model": self.model,
                "backend": "openai",
                "base_url": "https://api.together.xyz/v1",
                "positive_prompts": [{"pos": p} for p in prompts],
                "evaluation_prompt": eval_prompt,
            }

            logger.info(f"Generated persona for {profile['title']} with {len(prompts)} prompts")
            if include_questions:
                # TODO: Generate occupation-specific questions
                questions = self.generate_occupation_questions(profile)
                persona["questions"] = questions
                logger.info(
                    f"Included {len(questions)} occupation-specific questions for {profile['title']}"
                )
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
        include_questions: bool = True,
        slug: Optional[str] = None,
    ):
        """Generate and save persona for an occupation.

        Args:
            profile: Occupation profile from ONETLoader
            slug: Optional filename slug (auto-generated from title if not provided)

        Returns:
            Path to saved file
        """

        persona = self.generate_persona(profile, include_questions=include_questions)
        if persona:
            return self.save_persona(profile["title"], persona)

        return ""


class RoleLoader:
    """Loader for non-occupational roles from a JSON file.

    Expects a JSON file with a list of role definitions, each containing:
    - title: Name of the role
    - description: Behavioral description of the role
    """

    def __init__(self, filepath: str = "data/non_occupational_roles_list.json"):
        self.filepath = filepath
        self.data = self.load_data()

    def load_data(self) -> list[dict]:
        """Load role data from JSON file."""
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} roles from {self.filepath}")
                return data
        except Exception as e:
            logger.error(f"Failed to load roles from {self.filepath}: {e}")
            return []

    def get_profiles(self):
        categories = list(self.data.keys())
        profiles = []
        for category in categories:
            for role in self.data[category]:
                profiles.append({"title": role, "category": category})
        return profiles


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate vocational personas from O*NET data")
    parser.add_argument(
        "--profiles_path",
        default="data/non_occupational_roles_list.json",
        help="O*NET-SOC code to generate persona for",
    )
    parser.add_argument(
        "--output-dir",
        default="persona_data/non_vocational_personas/instructions",
        help="Output directory for persona files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Limit the number of personas to generate (for testing)",
    )
    args = parser.parse_args()

    loader = RoleLoader()
    generator = NonVocationalPersonaGenerator(
        output_dir=args.output_dir, model="openai/gpt-oss-120B"
    )

    profile_list = loader.get_profiles()
    print(f"Loaded {len(profile_list)} profiles from {args.profiles_path}")

    if args.limit:
        # randomly sample profiles if limit is set
        import random

        profile_list = random.sample(profile_list, args.limit)
    for profile in profile_list:
        filepath = generator.generate_and_save(profile)
        print(f"\nSaved to: {filepath}")
