"""Role Profile Generator for new_roles.json personas.

Generates enriched role profiles (description, tasks, role_contexts,
psychological_profile) for roles defined in new_roles.json, using an LLM
to expand brief descriptions into full persona-ready profiles.

Every role — whether backed by O*NET data or purely LLM-generated — receives
the same psychological profile layer so that downstream system-prompt and
question generation can produce decision-differentiating personas rather than
stylistic mimicry.

Example usage:
    from pvx.data.role_loader import RoleProfile

    with open("data/new_roles.json") as f:
        profile_dict = json.load(f)

    rp = RoleProfile(profile_dict=profile_dict, model="moonshotai/kimi-k2.5")
    profile = rp.get_profile("detective")
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from .onet_loader import ONETLoader

logger = logging.getLogger(__name__)
# Load environment variables from .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DESC_PROMPT = """You are an expert in organizational philosophy and role design. Your task is to rewrite a job role description that captures the deeper essence of what a role truly embodies, rather than listing rigid binaries or surface-level task inventories.

Given the following:
- **Title:** {title}
- **Current Description:** {current_description}

Rewrite the description following these principles:
1. **Lead with Core Essence** — Begin by defining what the role *fundamentally is* at its heart, not what it merely does.
2. **Avoid Binary Framing** — Do not define the role through opposites (e.g., "not a manager, but a doer"). Roles exist on a spectrum and hold multitudes.
3. **Name the Tools of Manifestation** — Identify the key mediums, capabilities, or forces through which this role expresses itself in the world.

**Constraints:**
- Write at most 5 sentences.
- Use fluid, expansive language that honors the full range of the role.
- Do not use bullet points in your output — write in cohesive prose.

Output only the rewritten description. Nothing else."""

ROLE_CONTEXT_PROMPT = """You are an expert in occupational ethnography and role experience design. Your task is to generate contextual dimensions that describe the lived conditions and recurring patterns of how a role is performed.

Given the following:
- **Title:** {title}
- **Description:** {description}
- **Tasks:** {tasks}

Identify the most defining contextual dimensions of this role. These contexts describe **how** and **under what conditions** the role operates — not what it does. When generating contexts, follow these principles:

1. **Anchor in lived experience** — Each context should reflect a real, recurring condition or dynamic that someone embodying this role would consistently encounter.
2. **Think beyond the physical** — For abstract, elemental, or non-human roles (e.g., wind, time, fire, a concept), interpret context metaphysically or energetically. Ask: *what are the conditions under which this force operates and expresses itself?*
3. **Choose dimensions that differentiate** — Select contexts that are most revealing and distinctive for this specific role, not generic ones applicable to any role.
4. **Assign honest frequencies or intensities** — Use natural language values such as:
   - Frequencies: "Every day", "Once a week or more but not every day", "Once a month or more but not every week", "Rarely"
   - Intensities: "Extremely important", "Very important", "Somewhat important", "Constant", "Highly competitive", "Very close (near touching)"

**Constraints:**
- Generate exactly ten role contexts.
- Focus on contexts that describe the social interactions of these roles, how do they interact with other people, races, entities etc. The goal is to depict a multi-faceted character that can exist and is believable. For example, a thief interacting with others would be cautious so as to not get caught, a pirate would be more open and friendly to other pirates but more aggressive and competitive with non-pirates, a teacher would be nurturing and supportive to their students but more formal and hierarchical with their superiors.
- Each context must have a descriptive dimension label and a corresponding frequency or intensity value.
- Ground abstract roles in their natural or metaphysical operating conditions rather than forcing human-workplace framing.
- Output must be a valid JSON object. No explanation, no preamble.

**Output format:**
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value",
  "Dimension Label": "Frequency or Intensity Value"
"""

TASKS_PROMPT = """You are an expert in workforce analysis and occupational role design. Your task is to identify the most representative tasks performed by a given role, based on its title and description.

Given the following:
- **Title:** {title}
- **Description:** {description}

Identify the tasks most commonly and meaningfully associated with this role. When generating tasks, follow these principles:

1. **Ground tasks in action** — Every task should begin with a verb and describe a concrete, observable activity.
2. **Reflect the full scope** — Draw from the core responsibilities, collaborative activities, preparatory work, and any specialized expressions of the role.
3. **Stay role-authentic** — Tasks should feel native to someone genuinely performing this role, not generic filler applicable to any job.
4. **Prioritize by centrality** — Rank and return only the 7 tasks that are most essential and characteristic of this role above all others.

**Constraints:**
- Return exactly fifteen (15) tasks.
- Each task must be a single, clear sentence beginning with a verb.
- Output must be a valid JSON array of strings. No explanation, no preamble.

**Output format:**
["Task one.", "Task two.", "Task three.", "Task four.", "Task five.", "Task six.", "Task seven.", "Task eight.", "Task nine.", "Task ten.", "Task eleven.", "Task twelve.", "Task thirteen.", "Task fourteen.", "Task fifteen."]"""


# ---------------------------------------------------------------------------
# Psychological Profile prompt
#
# This prompt is the lynchpin for solving the stylistic-mimicry problem.
# It generates the same structured psychological profile for every role
# regardless of whether it comes from O*NET or LLM generation.
#
# The profile captures *how a person in this role thinks, decides, and
# prioritises* — not what tasks they do or what vocabulary they use.
# This gives the downstream system-prompt generator the raw material to
# produce personas that make *different decisions*, not just describe
# the same decisions in different language.
# ---------------------------------------------------------------------------

PSYCH_PROFILE_PROMPT = """You are a personality psychologist and behavioral profiler. Your task is to construct a psychological profile for someone who deeply embodies a given role. This profile must capture how this person *thinks, decides, and relates to the world* — not what tasks they perform.

The profile will be used to generate AI personas that must make genuinely different choices from each other. Two roles that share similar tasks (e.g., editor and proofreader) must still produce different psychological profiles if the people who embody those roles think differently.

The profile will also be used to generate stronger cognitive override prompts. That means you must capture not only what this person values, but what they resent, what they refuse to normalize, what they instinctively blame, and what premise they push back on instead of helpfully accommodating.

**Role Input:**
- **Title:** {title}
- **Description:** {description}
- **Representative Tasks:** {tasks}
- **Role Contexts:** {role_contexts}

Generate a psychological profile with the following components. Every component must be specific to THIS role — if the same claim could be made about most roles, it is useless. Replace it with something differentiating.

**Components to generate:**

1. **core_drive** (1 sentence): The single deepest motivation that pulls this person forward. Not a goal — a *need*. What feels wrong when they can't do it?

2. **decision_style** (1-2 sentences): How they make choices under uncertainty. Do they gather more data, trust gut instinct, defer to authority, seek consensus, impose structure, or improvise? What's their default when the clock is ticking?

3. **attention_pattern** (1-2 sentences): What do they notice first in any situation? What information do they instinctively prioritize, and what do they tend to overlook or dismiss?

4. **conflict_stance** (1-2 sentences): When they disagree with someone, what is their first impulse? Do they confront, accommodate, withdraw, compete, or reframe? What emotional cost does conflict carry for them?

5. **risk_orientation** (1 sentence): Where on the spectrum from risk-seeking to risk-averse do they naturally sit? What kinds of risk feel acceptable vs. intolerable?

6. **social_posture** (1-2 sentences): How do they position themselves in a group? Leader, observer, facilitator, provocateur, satellite? Do they seek connection or maintain distance? What role do they default to when no role is assigned?

7. **relationship_to_authority** (1 sentence): How do they respond to hierarchy, rules, and imposed structure? Compliance, subversion, negotiation, indifference?

8. **failure_response** (1-2 sentences): When something they invested in fails, what is their first emotional reaction and their second behavioral response? Do they blame, learn, deflect, or persevere?

9. **value_hierarchy** (2-3 values ranked): Name the 2-3 values this person would protect even at personal cost, in rank order. These must be specific enough to create actual dilemmas — "truth" is too vague, "factual precision even when it embarrasses allies" is usable.

10. **cognitive_bias** (1 sentence): What systematic blind spot or thinking error does this person most naturally fall into? Not a flaw they're aware of — one they'd deny having.

11. **inner_contradiction** (1 sentence): The one internal tension this person lives with — two parts of their identity that genuinely pull against each other.

12. **non_negotiable** (1 sentence): The one principle, condition, or way of operating this person refuses to sacrifice even when pressured. What feels like betrayal if they give it up?

13. **recurring_resentment** (1 sentence): What kind of person, behavior, demand, or systemic pattern reliably irritates or angers them because it clashes with how they believe the world should work?

14. **rejected_premise** (1 sentence): What widely accepted social or professional virtue does this person quietly reject, distrust, or consider overrated? This should be something they push back on rather than helpfully affirm.

15. **instinctive_blame_target** (1 sentence): When things go wrong, where does this person's mind go first? Who or what do they instinctively see as the real source of the problem before reflection softens it?

**Rules:**
- Never use the role title in the profile content. Describe the *person*, not the label.
- Never praise or idealize. These are real people with real biases and blind spots.
- Each field must be specific enough that it could NOT apply to most other roles.
- Avoid psychological jargon. Use plain behavioral language anyone could understand.
- Prefer tensions, aversions, refusals, and asymmetries over rounded, balanced, well-adjusted summaries.
- The new fields should be concrete enough to support persona prompts that can reject a user's premise instead of smoothing it over.
- Output must be a valid JSON object with exactly the 15 keys listed above.
- value_hierarchy should be a JSON array of strings.
- All other fields should be strings.
"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ListOfString(BaseModel):
    items: list[str] = Field(..., description="List of strings")


class DictModel(BaseModel):
    items: dict[str, str] = Field(..., description="Dictionary of string keys and values")


class PsychProfile(BaseModel):
    """Structured psychological profile for a role persona.

    Every field captures a dimension of *how this person thinks and decides*,
    not what tasks they perform or what language they use.

    The schema includes both motivational structure and antagonistic pressure
    points so downstream persona prompts can produce stronger cognitive
    overrides rather than defaulting to assistant-like helpfulness.
    """

    core_drive: str = Field(..., description="Single deepest motivation / need")
    decision_style: str = Field(..., description="How they make choices under uncertainty")
    attention_pattern: str = Field(..., description="What they notice first, what they overlook")
    conflict_stance: str = Field(..., description="First impulse when disagreeing with someone")
    risk_orientation: str = Field(..., description="Risk-seeking vs risk-averse tendencies")
    social_posture: str = Field(..., description="Default position in a group")
    relationship_to_authority: str = Field(
        ..., description="Response to hierarchy, rules, structure"
    )
    failure_response: str = Field(
        ..., description="First emotional reaction and second behavioral response to failure"
    )
    value_hierarchy: list[str] = Field(
        ..., description="2-3 values ranked, specific enough to create dilemmas"
    )
    cognitive_bias: str = Field(..., description="Systematic blind spot they would deny having")
    inner_contradiction: str = Field(
        ..., description="Internal tension between two parts of their identity"
    )
    non_negotiable: str = Field(
        ..., description="What they refuse to sacrifice even under pressure"
    )
    recurring_resentment: str = Field(
        ..., description="What reliably irritates or angers them in others or in systems"
    )
    rejected_premise: str = Field(
        ..., description="A valued norm or virtue they distrust or reject"
    )
    instinctive_blame_target: str = Field(
        ..., description="Who or what they first see as the cause when things go wrong"
    )


# ---------------------------------------------------------------------------
# RoleProfile class
# ---------------------------------------------------------------------------


class RoleProfile:
    def __init__(self, profile_dict: dict, model: str):
        self.profile_dict = profile_dict
        self.model = model
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
    # Individual generation methods
    # ------------------------------------------------------------------

    def generate_better_description(
        self, profile_name: str, curr_description: str
    ) -> Optional[str]:
        prompt = DESC_PROMPT.format(title=profile_name, current_description=curr_description)
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            logger.error(f"Error generating better description for {profile_name}: {e}")
            return None

    async def agenerate_better_description(
        self, profile_name: str, curr_description: str
    ) -> Optional[str]:
        """Async counterpart of generate_better_description."""
        prompt = DESC_PROMPT.format(title=profile_name, current_description=curr_description)
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            logger.error(f"Error generating better description for {profile_name}: {e}")
            return None

    def generate_tasks(self, profile_name: str, description: str) -> Optional[list[str]]:
        prompt = TASKS_PROMPT.format(title=profile_name, description=description)
        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ListOfString,
            )
            parsed = response.choices[0].message.parsed
            return parsed.items if parsed else None
        except Exception as e:
            logger.error(f"Error generating tasks for {profile_name}: {e}")
            return None

    async def agenerate_tasks(self, profile_name: str, description: str) -> Optional[list[str]]:
        """Async counterpart of generate_tasks."""
        prompt = TASKS_PROMPT.format(title=profile_name, description=description)
        try:
            response = await self.async_client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ListOfString,
            )
            parsed = response.choices[0].message.parsed
            return parsed.items if parsed else None
        except Exception as e:
            logger.error(f"Error generating tasks for {profile_name}: {e}")
            return None

    def generate_role_context(
        self, profile_name: str, description: str, tasks: list[str]
    ) -> Optional[dict]:
        prompt = ROLE_CONTEXT_PROMPT.format(
            title=profile_name, description=description, tasks="\n".join(f"- {t}" for t in tasks)
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=DictModel,
            )
            parsed = response.choices[0].message.parsed
            return parsed.items if parsed else None
        except Exception as e:
            logger.error(f"Error generating role context for {profile_name}: {e}")
            return None

    async def agenerate_role_context(
        self, profile_name: str, description: str, tasks: list[str]
    ) -> Optional[dict]:
        """Async counterpart of generate_role_context."""
        prompt = ROLE_CONTEXT_PROMPT.format(
            title=profile_name, description=description, tasks="\n".join(f"- {t}" for t in tasks)
        )
        try:
            response = await self.async_client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=DictModel,
            )
            parsed = response.choices[0].message.parsed
            return parsed.items if parsed else None
        except Exception as e:
            logger.error(f"Error generating role context for {profile_name}: {e}")
            return None

    def generate_psych_profile(
        self, title: str, description: str, tasks: list[str], role_contexts: dict
    ) -> dict:
        """Generate a psychological profile for a role.

        This runs for EVERY role — O*NET-backed and LLM-generated alike —
        ensuring uniform depth across the full roster. The output drives
        decision-level differentiation in downstream persona generation.

        Args:
            title: Role title
            description: Role description
            tasks: List of representative tasks
            role_contexts: Dict of context dimension -> frequency/intensity

        Returns:
            Dict with the 15 psychological profile fields, or empty dict on failure
        """
        tasks_str = "\n".join(f"- {t}" for t in tasks) if tasks else "Not specified"
        role_contexts_str = (
            "\n".join(f"- {k}: {v}" for k, v in role_contexts.items())
            if role_contexts
            else "Not specified"
        )

        prompt = PSYCH_PROFILE_PROMPT.format(
            title=title,
            description=description,
            tasks=tasks_str,
            role_contexts=role_contexts_str,
        )

        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format=PsychProfile,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed.model_dump()
        except Exception as e:
            logger.error(f"Error generating psychological profile for {title}: {e}")

        logger.warning("Failed to generate psychological profile for '%s'", title)
        return {}

    # ------------------------------------------------------------------
    # Profile assembly
    # ------------------------------------------------------------------

    async def agenerate_psych_profile(
        self, title: str, description: str, tasks: list[str], role_contexts: dict
    ) -> dict:
        """Async counterpart of generate_psych_profile.

        This runs for EVERY role — O*NET-backed and LLM-generated alike —
        ensuring uniform depth across the full roster. The output drives
        decision-level differentiation in downstream persona generation.

        Args:
            title: Role title
            description: Role description
            tasks: List of representative tasks
            role_contexts: Dict of context dimension -> frequency/intensity

        Returns:
            Dict with the 15 psychological profile fields, or empty dict on failure
        """
        tasks_str = "\n".join(f"- {t}" for t in tasks) if tasks else "Not specified"
        role_contexts_str = (
            "\n".join(f"- {k}: {v}" for k, v in role_contexts.items())
            if role_contexts
            else "Not specified"
        )

        prompt = PSYCH_PROFILE_PROMPT.format(
            title=title,
            description=description,
            tasks=tasks_str,
            role_contexts=role_contexts_str,
        )

        try:
            response = await self.async_client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format=PsychProfile,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed.model_dump()
        except Exception as e:
            logger.error(f"Error generating psychological profile for {title}: {e}")

        logger.warning("Failed to generate psychological profile for '%s'", title)
        return {}

    # ------------------------------------------------------------------
    # Profile assembly
    # ------------------------------------------------------------------

    def generate_data(self, profile_name: str, profile_description: str) -> tuple:
        """Generate description, tasks, and role context for a non-O*NET role."""
        logger.debug(f"Generating data for profile: {profile_name}")

        description = self.generate_better_description(profile_name, profile_description)
        if description is None:
            logger.warning("Description generation failed for '%s', using original", profile_name)
            description = profile_description
        logger.debug(f"Generated description for {profile_name}: {description}")

        tasks = self.generate_tasks(profile_name, description)
        if tasks is None:
            logger.warning("Task generation failed for '%s', using empty list", profile_name)
            tasks = []
        logger.debug(f"Generated tasks for {profile_name}: {tasks}")

        role_con = self.generate_role_context(profile_name, description, tasks)
        if role_con is None:
            logger.warning(
                "Role context generation failed for '%s', using empty dict", profile_name
            )
            role_con = {}
        logger.debug(f"Generated role context for {profile_name}: {role_con}")

        return description, tasks, role_con

    def load_onet_profile(self, profile_name: str):
        """Load base profile from O*NET for tasks and work contexts only."""
        loader = ONETLoader(Path(__file__).resolve().parents[3] / "data" / "onet_raw")
        onet_profile = loader.get_occupation_profile(self.profile_dict[profile_name]["onet_code"])
        return {
            "title": onet_profile["title"],
            "description": onet_profile["description"],
            "tasks": onet_profile["tasks"],
            "role_contexts": onet_profile["work_contexts"],
        }

    async def agenerate_data(self, profile_name: str, profile_description: str) -> tuple:
        """Async counterpart of generate_data. Generate description, tasks, and role context for a non-O*NET role."""
        logger.debug(f"Generating data for profile: {profile_name}")

        description = await self.agenerate_better_description(profile_name, profile_description)
        if description is None:
            logger.warning("Description generation failed for '%s', using original", profile_name)
            description = profile_description
        logger.debug(f"Generated description for {profile_name}: {description}")

        tasks = await self.agenerate_tasks(profile_name, description)
        if tasks is None:
            logger.warning("Task generation failed for '%s', using empty list", profile_name)
            tasks = []
        logger.debug(f"Generated tasks for {profile_name}: {tasks}")

        role_con = await self.agenerate_role_context(profile_name, description, tasks)
        if role_con is None:
            logger.warning(
                "Role context generation failed for '%s', using empty dict", profile_name
            )
            role_con = {}
        logger.debug(f"Generated role context for {profile_name}: {role_con}")

        return description, tasks, role_con

    def load_other_profile(self, profile_name):
        """Generate full profile for a non-O*NET role via LLM."""
        desc, tasks, role_con = self.generate_data(
            profile_name, self.profile_dict[profile_name]["desc"]
        )

        return {
            "title": profile_name,
            "description": desc,
            "tasks": tasks,
            "role_contexts": role_con,
        }

    async def aload_other_profile(self, profile_name):
        """Async counterpart of load_other_profile. Generate full profile for a non-O*NET role via LLM."""
        desc, tasks, role_con = await self.agenerate_data(
            profile_name, self.profile_dict[profile_name]["desc"]
        )

        return {
            "title": profile_name,
            "description": desc,
            "tasks": tasks,
            "role_contexts": role_con,
        }

    def get_profile(self, profile_name: str) -> dict:
        """Build a complete profile for any role.

        Regardless of data source (O*NET or LLM), the output always contains:
        - title, description, tasks, role_contexts  (structural)
        - psychological_profile                      (decision-level)

        The psychological profile is generated uniformly for every role so
        that downstream persona generation has the same depth of
        decision-driving data everywhere, including stronger non-negotiables,
        resentments, premise rejections, and instinctive blame patterns.
        """
        if self.profile_dict[profile_name]["onet_code"] == "None":
            profile = self.load_other_profile(profile_name)
        else:
            profile = self.load_onet_profile(profile_name)

        # Generate psychological profile for ALL roles uniformly
        psych = self.generate_psych_profile(
            title=profile["title"],
            description=profile["description"],
            tasks=profile.get("tasks", []),
            role_contexts=profile.get("role_contexts", {}),
        )
        profile["psychological_profile"] = psych

        return profile

    async def aget_profile(self, profile_name: str) -> dict:
        """Async counterpart of get_profile. Build a complete profile for any role.

        Regardless of data source (O*NET or LLM), the output always contains:
        - title, description, tasks, role_contexts  (structural)
        - psychological_profile                      (decision-level)

        The psychological profile is generated uniformly for every role so
        that downstream persona generation has the same depth of
        decision-driving data everywhere, including stronger non-negotiables,
        resentments, premise rejections, and instinctive blame patterns.
        """
        if self.profile_dict[profile_name]["onet_code"] == "None":
            profile = await self.aload_other_profile(profile_name)
        else:
            profile = self.load_onet_profile(profile_name)

        # Generate psychological profile for ALL roles uniformly
        psych = await self.agenerate_psych_profile(
            title=profile["title"],
            description=profile["description"],
            tasks=profile.get("tasks", []),
            role_contexts=profile.get("role_contexts", {}),
        )
        profile["psychological_profile"] = psych

        return profile

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def generate_and_save(self):
        for profile_name in tqdm(list(self.profile_dict.keys())):
            profile_ = self.get_profile(profile_name)

            with open(f"data/new_roles/{profile_name}.json", "w") as f:
                json.dump(profile_, f, indent=2)

    def generate_test_profile(self, name: str):
        profile_ = self.get_profile(name)

        with open(
            Path(__file__).resolve().parents[3] / "data" / "new_roles" / f"{name}.json",
            "w",
        ) as f:
            json.dump(profile_, f, indent=2)


if __name__ == "__main__":
    with open(Path(__file__).resolve().parents[3] / "data" / "new_roles.json", "r") as f:
        profile_dict = json.load(f)

    role_profile = RoleProfile(model="moonshotai/kimi-k2.5", profile_dict=profile_dict)

    role_profile.generate_and_save()
