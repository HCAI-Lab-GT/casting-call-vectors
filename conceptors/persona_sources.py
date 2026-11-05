"""Role prompts, question sets and prompt builders for persona_acts.py.

Sources:
  built-in   ROLES / QUESTIONS in persona_acts.py (the 4 Nov 0.5B pilot).
  files      Assistant Axis release (github.com/safety-research/assistant-axis, pinned in models.yaml):
             data/roles/instructions/<role>.json  {"instruction": [{"pos": ...}], "questions": [...], "eval_prompt": ...}
             data/extraction_questions.jsonl      {"question": ..., "id": ...}

Base mode follows "Tracing Persona Vectors Through LLM Pretraining" (arXiv 2605.13329, Sec. 2.1 and App. D):
the system prompt becomes a third-person character description and the question a hypothetical
("If someone asked Steve, ..."), which the base model continues as plain text.
"""

import json
import re
from pathlib import Path

NARRATED_SYSTEM = "You are an assistant dedicated to helping users with their various tasks and queries."
NARRATED_TEMPLATE = (
    "In two or three sentences, describe how a {who} would answer this question, "
    "writing about the {who} in the third person: {q}"
)
NARRATED_WHO = {"pirate": "pirate captain"}  # keeps the pilot's exact pirate_narrated wording


def load_role_prompts(roles_dir: Path, names: list[str], n_prompts: int,
                      fallback: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """First n_prompts positive instructions per role; the AA 'default' role has one empty prompt.
    Roles without a release file (e.g. the pilot's pirate_doctor) come from `fallback` when given."""
    roles = {}
    for name in names:
        path = roles_dir / f"{name}.json"
        if not path.exists() and fallback and name in fallback:
            roles[name] = fallback[name][:n_prompts]
            continue
        data = json.loads(path.read_text())
        prompts = [inst["pos"] for inst in data["instruction"]][:n_prompts]
        if not prompts:
            raise ValueError(f"{name}: no instructions in {roles_dir}")
        roles[name] = prompts
    return roles


def load_eval_prompts(roles_dir: Path, names: list[str]) -> dict[str, str]:
    """Per-role Assistant Axis judge rubric (used by judge.py)."""
    return {n: json.loads((roles_dir / f"{n}.json").read_text()).get("eval_prompt", "") for n in names}


def load_questions(path: Path, n_questions: int | None) -> list[tuple[str, str]]:
    """(task, question) pairs from a JSONL file with a 'question' field and an optional 'task' or 'category'."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    qs = [(r.get("task") or r.get("category") or "aa", r["question"]) for r in rows]
    return qs[:n_questions] if n_questions else qs


def narrated_job(role: str, question: str) -> tuple[str, str]:
    """Hard negative: the assistant describes how the role would answer, with no change of identity."""
    who = NARRATED_WHO.get(role, role.replace("_", " "))
    return NARRATED_SYSTEM, NARRATED_TEMPLATE.format(who=who, q=question)


_LEADS = [
    (r"^\s*(you are|you're)\s+", "{name} is "),
    (r"^\s*(act as|take on the persona of|embody the role of|be|become|play)\s+", "{name} is "),
    (r"^\s*(please be|imagine you are|pretend you are)\s+", "{name} is "),
]


def third_person(role_prompt: str, name: str = "Steve") -> str:
    """Mechanical second-to-third-person rewrite of an AA role prompt ('You are a pirate...' -> 'Steve is a pirate...').

    The paper used an LLM for this step; this keeps base-mode prompts deterministic and offline.
    """
    text = role_prompt.strip()
    for pat, rep in _LEADS:
        new = re.sub(pat, rep.format(name=name), text, count=1, flags=re.I)
        if new != text:
            text = new
            break
    swaps = [
        (r"\byou are\b", f"{name} is"),
        (r"\byou're\b", f"{name} is"),
        (r"\byou have\b", f"{name} has"),
        (r"\byourself\b", "themself"),
        (r"\byour\b", "their"),
        (r"\byou\b", "they"),
    ]
    for pat, rep in swaps:
        text = re.sub(pat, rep, text, flags=re.I)
    return text


def base_prompt(role_prompt: str, question: str, name: str = "Steve") -> str:
    """Plain-text continuation prompt for a base model: character description, then the question as a hypothetical.

    Paper-style "If <name>..." framing (2605.13329) ending in an open quote, so the continuation is the
    character's own first-person reply and compares directly with chat-mode answers. Every role shares
    the name, so only the description differs between personas.
    """
    description = third_person(role_prompt, name) if role_prompt else f"{name} is a helpful assistant."
    return f'{description}\n\nIf someone asked {name}, "{question}", {name} would reply: "'


def trim_base_answer(text: str) -> str:
    """Cut a base-model continuation at the character's closing quote."""
    return text.split('"', 1)[0].strip()
