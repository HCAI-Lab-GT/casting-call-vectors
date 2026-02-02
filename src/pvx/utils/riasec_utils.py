from pathlib import Path
from typing import Any

import yaml


class RIASECHelpers:
    """
    Utility class for RIASEC trait prompts, YAML helpers, and question/answer extraction.
    Provides system prompt templates and YAML update/fetch utilities for RIASEC workflows.
    """

    # Set of valid RIASEC trait names
    RIASEC_TRAITS = {
        "realistic",
        "investigative",
        "artistic",
        "social",
        "enterprising",
        "conventional",
    }

    # System prompt for binary YES/NO RIASEC evaluation
    RIASEC_YN_SYSTEM_PROMPT = """
    Answer the user’s characteristic with a binary choice.

    Rules (strict):
    - Output EXACTLY one token: YES or NO.
    - No punctuation, no extra words, no explanations.
    - Interpret the characteristic as a general preference / inclination claim.
    - Output YES if you endorse/agree with the characteristic as stated.
    - Output NO if you do not endorse/agree with it.
    """

    # System prompt for positive (trait-exhibiting) RIASEC answers
    POSITIVE_RIASEC_SYSTEM_PROMPT = """
    You are answering as a person who strongly exhibits the target Holland RIASEC trait: {TRAIT}.
    You must answer affirmatively to the question.

    Output format rules (strict):
    - Your entire response MUST be exactly two parts: `Yes, ` followed by the explanation.
    - Begin with exactly: `Yes, ` (capital Y, then lowercase es, comma, space).

    Content rules (strict):
    - The explanation must focus on the {TRAIT} trait as a stable preference/work-style, not on the question’s topic.
    - Do NOT mention, paraphrase, or reference the question content or its objects/setting/examples.
    - Do NOT use words that clearly mirror the question’s domain (no restating like “I like working on X”).
    - Use general trait language: motivations, preferred work environment, decision style, and what feels energizing.
    - Avoid hedging (no “maybe”, “it depends”, “sometimes”).
    - Keep it pragmatic and specific, but trait-centered.
    """

    # System prompt for negative (trait-absent) RIASEC answers
    NEGATIVE_RIASEC_SYSTEM_PROMPT = """
    You are answering as a person who does NOT exhibit the target Holland RIASEC trait: {TRAIT}.
    You must answer negatively to the question.

    Output format rules (strict):
    - Your entire response MUST be exactly two parts: `No, ` followed by the explanation.
    - Begin with exactly: `No, ` (capital N, then lowercase o, comma, space).

    Content rules (strict):
    - The explanation must focus on NOT having the {TRAIT} trait as a stable preference/work-style, not on the question’s topic.
    - Do NOT mention, paraphrase, or reference the question content or its objects/setting/examples.
    - Do NOT use words that clearly mirror the question’s domain (no restating like “I don’t like doing X”).
    - Use general trait language: motivations, preferred work environment, decision style, and what feels draining or unappealing.
    - Frame the reason as a mismatch with {TRAIT}-typical preferences (e.g., "I prefer ... over ..."), but keep it trait-centered.
    - Avoid hedging (no “maybe”, “it depends”, “sometimes”).
    - Keep it pragmatic and specific, but trait-centered.
    """

    @staticmethod
    def fetch_riasec_information(
        riasec_config_path: str = "./configs/riasec.yaml",
    ) -> dict[str, Any]:
        """
        Load RIASEC trait/question/answer data from YAML config file.
        Returns a dict mapping trait to its description, characteristics, and Q/A pairs.
        """
        yaml_path = Path(riasec_config_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"RIASEC YAML file not found: {riasec_config_path}")

        with open(yaml_path, "r") as f:
            riasec_data = yaml.safe_load(f)

        trait_dict = {}
        for trait, info in riasec_data.items():
            desc = info.get("description")
            characteristics = info.get("characteristics", [])
            # Extract question/answer pairs and flatten for convenience
            qa_pairs = info.get("question_answer_pairs", [])
            questions = []
            pos_answers = []
            neg_answers = []
            for qa in qa_pairs:
                questions.append(qa.get("question", ""))
                pos_answers.append(qa.get("positive", []))
                neg_answers.append(qa.get("negative", []))
            trait_dict[trait] = {
                "description": desc,
                "characteristics": characteristics,
                "questions": questions,
                "positive_answers": pos_answers,
                "negative_answers": neg_answers,
                "question_answer_pairs": qa_pairs,
            }
        return trait_dict

    @staticmethod
    def update_riasec_yaml(riasec_config_path, trait, accepted_responses):
        """
        Update the positive/negative answers for a trait in the RIASEC YAML file.
        Modifies the YAML in-place for the given trait using accepted_responses dict.
        """
        # Load the YAML file
        with open(riasec_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Find the correct trait section
        trait_section = data.get(trait)
        if not trait_section or "question_answer_pairs" not in trait_section:
            raise ValueError(f"Trait '{trait}' or its question_answer_pairs not found in YAML.")

        # Update each question's positive/negative lists
        for qa in trait_section["question_answer_pairs"]:
            question = qa.get("question")
            if question in accepted_responses:
                qa["positive"] = accepted_responses[question].get("positive", [])
                qa["negative"] = accepted_responses[question].get("negative", [])

        # Write back to YAML
        with open(riasec_config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
