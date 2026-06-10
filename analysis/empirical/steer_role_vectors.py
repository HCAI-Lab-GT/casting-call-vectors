
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from pvx import setup_logging
from pvx.implementations.roles.role_persona_model import RolePersonaModel

logger = setup_logging(name="validate-role-vectors")


def load_questions_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load question records from JSONL with keys like {id, question, category}."""
    questions: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx}: {exc.msg}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_idx} must be a JSON object")

            question = record.get("question")
            if not question:
                raise ValueError(f"Missing 'question' on line {line_idx}")

            question_number = record.get("id", line_idx)
            questions.append(
                {
                    "question_number": question_number,
                    "question": question,
                }
            )

    if not questions:
        raise ValueError(f"No questions found in {path}")

    return questions


def resolve_output_path(output_json: Optional[str], role: str, num_roles: int) -> Path:
    """Resolve output file path, suffixing by role when evaluating multiple roles."""
    if output_json:
        output_path = Path(output_json)
        if num_roles <= 1:
            return output_path

        suffix = output_path.suffix if output_path.suffix else ".json"
        return output_path.with_name(f"{output_path.stem}_{role}{suffix}")

    return Path("validation_outputs") / f"{role}.json"


def main():
    ap = argparse.ArgumentParser(description="Validate role vectors on one or more questions")
    ap.add_argument(
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-r", "--roles", nargs="+", type=str, help="Roles of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Fallback single question when --questions_file is not provided",
    )
    ap.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="Path to JSONL file with question records (e.g., {id, question, category})",
    )
    ap.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Use only the first N questions from --questions_file",
    )
    ap.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path. If multiple roles, role suffix is appended.",
    )
    ap.add_argument(
        "-f",
        "--json_filepath",
        type=str,
        default=None,
        help="Filepath to model init file (not role dataset)",
    )
    args = ap.parse_args()

    if not args.roles:
        raise ValueError("Please provide at least one role via --roles")

    if args.questions_file:
        questions = load_questions_jsonl(args.questions_file)
        logger.info("Loaded %d questions from %s", len(questions), args.questions_file)
    else:
        questions = [{"question_number": 1, "question": args.question}]
        logger.info("Using single inline question")

    if args.max_questions is not None:
        if args.max_questions <= 0:
            raise ValueError("--max_questions must be a positive integer")
        original_count = len(questions)
        questions = questions[: args.max_questions]
        logger.info("Using first %d/%d questions", len(questions), original_count)

    for role in args.roles:
        logger.info("=== Processing role: %s ===", role)
        
        try:
            pvx = RolePersonaModel.load_or_create(
                target_model_id=args.model,
                concept=role,
                layer=16,
                json_filepath=args.json_filepath
            )
            logger.info("Successfully loaded/created RolePersonaModel for role '%s'", role)
        except Exception as e:
            logger.error("Failed to load or create RolePersonaModel for role '%s': %s", role, e)
            continue

        entries: List[Dict[str, Any]] = []

        for q in tqdm(questions, desc=f"{role} questions"):
            question_number = q["question_number"]
            question_text = q["question"]

            try:
                nonsteered_answer = pvx.generate(
                    prompt=question_text,
                    alpha=0,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                steered_answer = pvx.generate(
                    prompt=question_text,
                    alpha=args.alpha,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            except Exception as e:
                logger.error(
                    "Generation failed for role '%s' question %s: %s",
                    role,
                    question_number,
                    e,
                )
                nonsteered_answer = ""
                steered_answer = ""

            entries.append(
                {
                    "question_number": question_number,
                    "question": question_text,
                    "nonsteered_answer": nonsteered_answer,
                    "steered_answer": steered_answer,
                }
            )

        output_data = {
            "model": args.model,
            "alpha": args.alpha,
            "role": role,
            "entries": entries,
        }

        output_path = resolve_output_path(args.output_json, role, len(args.roles))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d entries for role '%s' to %s", len(entries), role, output_path)

if __name__ == "__main__":
    main()