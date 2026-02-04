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
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

from ..data.onet_loader import RIASEC_FULL_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

# Template for generating system prompt variants
SYSTEM_PROMPT_GENERATION_TEMPLATE = """Generate 5 different system prompts that instruct an AI to role-play as a {title}.

Job Description: {description}

Key Tasks:
{tasks}

Requirements:
1. Each prompt should instruct the AI to fully embody the role, NOT just provide information about it
2. Use varied phrasing: "You are a...", "Act as a...", "Please be a...", etc.
3. Highlight different aspects of the job in each prompt (responsibilities, expertise, perspective)
4. Keep each prompt to 1-2 sentences
5. Do NOT mention AI, language model, or assistant - the model should BE the professional

Return ONLY a JSON array of 5 strings, no other text:
["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5"]"""

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
        tasks = profile.get("tasks", [])[:max_tasks]
        tasks_str = "\n".join(f"- {t}" for t in tasks) if tasks else "Not specified"

        prompt = SYSTEM_PROMPT_GENERATION_TEMPLATE.format(
            title=profile["title"],
            description=profile["description"],
            tasks=tasks_str,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            prompts = json.loads(content)
            if isinstance(prompts, list) and len(prompts) == 5:
                return prompts
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response for %s, using fallback", profile["title"])
        except Exception as e:
            logger.warning("API call failed for %s: %s, using fallback", profile["title"], e)

        return self._generate_fallback_prompts(profile)

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
            },
        }

        if include_questions:
            # TODO: Generate occupation-specific questions
            pass

        return persona

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
            json.dump(persona, f, indent=2)
        logger.info(f"Saved persona to {filepath}")
        return filepath

    def generate_and_save(
        self,
        profile: dict,
        slug: Optional[str] = None,
    ) -> Path:
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

        persona = self.generate_persona(profile)
        return self.save_persona(slug, persona)


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
