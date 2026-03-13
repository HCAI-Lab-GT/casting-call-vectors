
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from pvx import setup_logging
from pvx.implementations.roles.role_persona_model import RolePersonaModel
from pvx.utils.response_generation import ResponseGeneration

logger = setup_logging(name="validate-role-vectors")


def _resolve_existing_path(path_like: str) -> Path:
    """Resolve a path from cwd first, then relative to repo root."""
    direct = Path(path_like).expanduser()
    if direct.exists():
        return direct

    repo_relative = (Path(__file__).resolve().parents[2] / path_like).resolve()
    if repo_relative.exists():
        return repo_relative

    raise FileNotFoundError(f"Path not found: {path_like}")


def _normalize_role_name(role: str) -> str:
    return "".join(ch.lower() for ch in str(role) if ch.isalnum())


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return cleaned or "role"


def _load_role_description(role: str, roles_list_path: str) -> tuple[str, str, Path]:
    roles_path = _resolve_existing_path(roles_list_path)
    with open(roles_path, "r", encoding="utf-8") as f:
        role_map = json.load(f)

    if not isinstance(role_map, dict):
        raise ValueError(f"Expected role list JSON object at {roles_path}")

    if role in role_map:
        return role, str(role_map[role]), roles_path

    normalized_target = _normalize_role_name(role)
    normalized_to_role = {
        _normalize_role_name(role_name): role_name for role_name in role_map.keys()
    }
    matched_role = normalized_to_role.get(normalized_target)

    if matched_role is None:
        available = ", ".join(sorted(role_map.keys())[:20])
        raise KeyError(
            f"Role '{role}' not found in {roles_path}. "
            f"First available roles: {available}"
        )

    return matched_role, str(role_map[matched_role]), roles_path


def load_questions_jsonl(path: str, max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load question records from JSONL with keys like {id, question, category}."""
    resolved_path = _resolve_existing_path(path)
    questions: List[Dict[str, Any]] = []

    with open(resolved_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {resolved_path} on line {line_idx}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_idx} in {resolved_path} must be a JSON object")

            question = record.get("question")
            if not question:
                raise ValueError(
                    f"Missing non-empty 'question' on line {line_idx} in {resolved_path}"
                )

            question_number = record.get("id", line_idx)
            questions.append(
                {
                    "question_number": question_number,
                    "question": question,
                    "category": record.get("category"),
                }
            )

            if max_questions is not None and len(questions) >= max_questions:
                break

    if not questions:
        raise ValueError(f"No questions found in {resolved_path}")

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


def resolve_gold_output_path(output_json: Optional[str], role: str, num_roles: int) -> Path:
    """Resolve output path for gold-standard generation."""
    if output_json:
        output_path = Path(output_json)
        if num_roles <= 1:
            return output_path

        suffix = output_path.suffix if output_path.suffix else ".json"
        return output_path.with_name(f"{output_path.stem}_{_safe_filename(role)}{suffix}")

    return (
        Path("outputs/validation/olmo-3-7b-instruct")
        / "gold_standards"
        / f"{_safe_filename(role)}_role_description_gold_standard.json"
    )


def generate_gold_standard_for_role(
    role: str,
    role_description: str,
    questions: List[Dict[str, Any]],
    api_model: str,
    hf_model: str,
    base_url: str,
    api_key_env: str,
    temperature: float,
    max_new_tokens: int,
    roles_list_path: str,
    questions_file_path: str,
    device: Optional[str],
) -> Dict[str, Any]:
    """Generate per-question API and HF responses conditioned on role description."""
    system_prompt = (
        f"You are role-playing as a {role}. "
        "Stay in character throughout the response. "
        "Use first-person perspective when natural and avoid generic assistant disclaimers.\n\n"
        f"Role description:\n{role_description}"
    )

    api_generator = ResponseGeneration(
        backend="openai",
        model=api_model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    hf_generator = ResponseGeneration(
        backend="hf_local",
        model=hf_model,
        local_model=hf_model,
        device=device,
    )

    entries: List[Dict[str, Any]] = []

    for q in tqdm(questions, desc=f"{role} gold-standard"):
        question_number = q["question_number"]
        question_text = str(q["question"])
        category = q.get("category")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ]

        api_answer: Optional[str] = None
        hf_answer: Optional[str] = None

        try:
            _, api_answer = api_generator(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            logger.warning(
                "API generation failed for role '%s' question %s: %s",
                role,
                question_number,
                exc,
            )

        try:
            _, hf_answer = hf_generator(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            logger.warning(
                "HF generation failed for role '%s' question %s: %s",
                role,
                question_number,
                exc,
            )

        entries.append(
            {
                "question_number": question_number,
                "category": category,
                "question": question_text,
                "responses": {
                    "api": api_answer,
                    "hf_local": hf_answer,
                },
            }
        )

    return {
        "role": role,
        "role_description": role_description,
        "prompt_source": "role_list_description",
        "question_source": {
            "path": questions_file_path,
            "num_questions": len(entries),
        },
        "models": {
            "api": {
                "backend": "openai",
                "model": api_model,
                "base_url": base_url,
                "api_key_env": api_key_env,
            },
            "hf_local": {
                "backend": "hf_local",
                "model": hf_model,
            },
        },
        "entries": entries,
        "generation": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "roles_list_path": roles_list_path,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Validate role vectors on one or more questions")
    ap.add_argument(
        "--generate_golden_standard",
        action="store_true",
        help="Generate gold-standard responses (API + HF) using role descriptions",
    )
    ap.add_argument(
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-r", "--roles", nargs="+", type=str, help="Roles of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2048, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Fallback single question when --questions_path is not provided",
    )
    ap.add_argument(
        "--questions_path",
        "--questions_file",
        dest="questions_path",
        type=str,
        default=None,
        help="Path to JSONL file with question records (e.g., {id, question, category}).",
    )
    ap.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Use only the first N questions from --questions_path",
    )
    ap.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON path. If multiple roles, role suffix is appended.",
    )
    ap.add_argument(
        "--role_list_path",
        "--roles_list_path",
        dest="role_list_path",
        type=str,
        default=None,
        help="Path to role_list.json for gold-standard generation.",
    )
    ap.add_argument(
        "--api_model",
        type=str,
        default="gpt-4.1-mini",
        help="API model used for gold-standard generation.",
    )
    ap.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="HF local model used for gold-standard generation (defaults to --model).",
    )
    ap.add_argument(
        "--base_url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible base URL for --api_model calls.",
    )
    ap.add_argument(
        "--api_key_env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable for API key used by --api_model.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional HF local generation device for gold standard mode (cuda, mps, cpu).",
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

    if args.generate_golden_standard:
        if not args.questions_path:
            raise ValueError("--questions_path is required when --generate_golden_standard is set")
        if not args.role_list_path:
            raise ValueError("--role_list_path is required when --generate_golden_standard is set")

        if args.max_questions is not None and args.max_questions <= 0:
            raise ValueError("--max_questions must be a positive integer")

        questions_file_path = str(_resolve_existing_path(args.questions_path))
        questions = load_questions_jsonl(path=questions_file_path, max_questions=args.max_questions)
        logger.info("Loaded %d questions from %s", len(questions), questions_file_path)

        roles_list_path = str(_resolve_existing_path(args.role_list_path))
        hf_model = args.hf_model or args.model

        for role in args.roles:
            logger.info("=== Generating gold standard for role: %s ===", role)
            try:
                matched_role, role_description, _ = _load_role_description(
                    role=role,
                    roles_list_path=roles_list_path,
                )
            except Exception as exc:
                logger.error("Failed to load role description for '%s': %s", role, exc)
                continue

            output_data = generate_gold_standard_for_role(
                role=matched_role,
                role_description=role_description,
                questions=questions,
                api_model=args.api_model,
                hf_model=hf_model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                roles_list_path=roles_list_path,
                questions_file_path=questions_file_path,
                device=args.device,
            )

            output_path = resolve_gold_output_path(args.output_json, matched_role, len(args.roles))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(
                "Saved %d gold-standard entries for role '%s' to %s",
                len(output_data["entries"]),
                matched_role,
                output_path,
            )
        return

    if args.questions_path:
        questions = load_questions_jsonl(args.questions_path)
        logger.info("Loaded %d questions from %s", len(questions), args.questions_path)
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