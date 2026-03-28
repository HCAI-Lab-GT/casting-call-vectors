"""
Scan every (question, alpha, layer) row for a role's gold-standard CSV and
regenerate any response that is missing or malformed (using the same detection
logic as report_malformed_responses.py).

Supports regenerating:
  - steered    → via RoleLayersPersonaModel
  - baseline   → via gold-prompt baseline messages + ResponseGeneration
  - assistant_axis → via AssistantAxisPersonaModel

Usage:
    # Dry run for a single role
    python scripts/evaluation/regenerate_malformed_responses.py --roles accountant --dry_run

    # Fix all columns for a role
    python scripts/evaluation/regenerate_malformed_responses.py --roles accountant

    # Fix only steered column for multiple roles
    python scripts/evaluation/regenerate_malformed_responses.py --roles accountant lawyer --columns steered

    # Fix specific alphas only
    python scripts/evaluation/regenerate_malformed_responses.py --roles accountant --alphas 1.0 2.5
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from pvx import setup_logging
from pvx.implementations.roles_layers.assistant_axis_persona_model import AssistantAxisPersonaModel
from pvx.implementations.roles_layers.role_layers_persona_model import RoleLayersPersonaModel
from pvx.utils.response_generation import ResponseGeneration

logger = setup_logging(name="regenerate-malformed-responses")

# ---------------------------------------------------------------------------
# Malformedness detection (mirrors report_malformed_responses.py)
# ---------------------------------------------------------------------------

def has_value(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def is_malformed(value: object) -> bool:
    if not has_value(value):
        return True
    return "dtype:" in str(value)


def is_duplicate_across_alphas(df: pd.DataFrame, idx: int, col: str) -> bool:
    row = df.loc[idx]
    question = row["question"]
    alpha = row["alpha"]
    value = str(row[col]).strip()
    if not has_value(value):
        return False
    mask = (df["question"] == question) & (df["alpha"] != alpha)
    return any(str(v).strip() == value for v in df.loc[mask, col] if has_value(v))


def needs_regeneration(df: pd.DataFrame, idx: int, col: str) -> bool:
    """Return True if the cell is missing, malformed, or a duplicate across alphas."""
    value = df.at[idx, col]
    if is_malformed(value):
        return True
    if col in ("steered", "assistant_axis") and is_duplicate_across_alphas(df, idx, col):
        return True
    return False


# ---------------------------------------------------------------------------
# Gold-baseline message loading (mirrors GoldPromptExperiments)
# ---------------------------------------------------------------------------

_gold_baseline_cache: dict[str, list[dict[str, str]]] = {}
_gold_role_descriptions: dict[str, str] = {}


def load_gold_baseline_messages(role: str, gold_prompts_dir: Path) -> list[dict[str, str]]:
    if role in _gold_baseline_cache:
        return deepcopy(_gold_baseline_cache[role])

    gold_path = gold_prompts_dir / f"{role.replace('/', '_')}_gold_label.json"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold prompt file not found for role '{role}' at {gold_path}")

    with open(gold_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    _gold_role_descriptions[role] = str(payload.get("role_description", "")).strip()

    messages: list[dict[str, str]] = []
    if isinstance(payload.get("gold_label_prompt"), dict):
        messages = payload["gold_label_prompt"].get("messages", [])
    elif isinstance(payload.get("gold_label_prompts"), list) and payload["gold_label_prompts"]:
        messages = payload["gold_label_prompts"][0].get("messages", [])

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role_name = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role_name in ("system", "user", "assistant") and content:
            normalized.append({"role": role_name, "content": content})

    if normalized and normalized[-1]["role"] == "user":
        normalized = normalized[:-1]

    if not normalized:
        raise ValueError(f"Gold prompt file for role '{role}' has no valid baseline messages")

    _gold_baseline_cache[role] = normalized
    return deepcopy(normalized)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate missing/malformed responses per question×alpha×layer in role CSVs."
    )
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--roles", nargs="+", default=None, help="Roles to process (default: all)")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["steered", "assistant_axis"],
        choices=["steered", "baseline", "assistant_axis"],
        help="Which response columns to check and regenerate",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="Only process these alphas")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--safetensors_dir", default="./persona_data/model_layer_inits/")
    parser.add_argument("--pt_dir", default="persona_data/assistant-axis/olmo-3-7b-instruct/vectors/")
    parser.add_argument("--gold_prompts_dir", default="./persona_data/gold_labels_prompts_dataset")
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--dry_run", action="store_true", help="Report what would be regenerated without changing files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gold_prompts_dir = Path(args.gold_prompts_dir)
    allowed_alphas = set(args.alphas) if args.alphas else None
    columns_to_check = args.columns

    # Discover CSV files
    csv_paths = sorted(input_dir.glob("Comparison_GoldStandard_*.csv"))
    if args.roles:
        allowed_roles = {role.replace("/", "_") for role in args.roles}
        csv_paths = [p for p in csv_paths if p.stem.removeprefix("Comparison_GoldStandard_") in allowed_roles]

    if not csv_paths:
        logger.info("No matching CSV files found in %s", input_dir)
        return

    # Lazy-loaded model caches (shared across roles for same layer/sample_count)
    steered_model_cache: dict[tuple[str, int, int], RoleLayersPersonaModel] = {}
    assistant_axis_model_cache: dict[tuple[str, int, int], AssistantAxisPersonaModel] = {}
    baseline_generator: ResponseGeneration | None = None

    total_regenerated = 0

    for csv_path in csv_paths:
        role = csv_path.stem.removeprefix("Comparison_GoldStandard_")
        df = pd.read_csv(csv_path)

        required = {"question", "alpha", "layer", "sample_count", "temperature"}
        required |= set(columns_to_check)
        missing_cols = required - set(df.columns)
        if missing_cols:
            logger.warning("Skipping %s: missing columns %s", csv_path.name, sorted(missing_cols))
            continue

        logger.info("=== Processing role: %s (%s rows) ===", role, len(df))

        # Identify all rows that need regeneration, grouped by column
        regen_map: dict[str, list[int]] = {col: [] for col in columns_to_check}

        for idx in df.index:
            alpha = df.at[idx, "alpha"]
            if allowed_alphas is not None and not any(abs(alpha - a) < 1e-9 for a in allowed_alphas):
                continue
            for col in columns_to_check:
                if needs_regeneration(df, idx, col):
                    regen_map[col].append(idx)

        total_needed = sum(len(indices) for indices in regen_map.values())
        if total_needed == 0:
            logger.info("%s: no malformed/missing responses found", csv_path.name)
            continue

        for col, count in regen_map.items():
            if count:
                logger.info("%s: %s %s rows need regeneration", csv_path.name, len(count), col)

        if args.dry_run:
            # Print per-alpha breakdown
            for col in columns_to_check:
                if not regen_map[col]:
                    continue
                alpha_counts: dict[float, int] = {}
                for idx in regen_map[col]:
                    a = df.at[idx, "alpha"]
                    alpha_counts[a] = alpha_counts.get(a, 0) + 1
                for a in sorted(alpha_counts):
                    logger.info("  %s alpha=%.2f: %s rows", col, a, alpha_counts[a])
            total_regenerated += total_needed
            continue

        # --- Regenerate steered responses ---
        if regen_map["steered"]:
            changed = 0
            for idx in regen_map["steered"]:
                question = str(df.at[idx, "question"]).strip()
                alpha = float(df.at[idx, "alpha"])
                layer = int(df.at[idx, "layer"])
                sample_count = int(df.at[idx, "sample_count"])
                temperature = float(df.at[idx, "temperature"])

                model_key = (role, layer, sample_count)
                if model_key not in steered_model_cache:
                    logger.info("Loading steered model: role=%s layer=%s samples=%s", role, layer, sample_count)
                    steered_model_cache[model_key] = RoleLayersPersonaModel.load_or_create(
                        target_model_id=args.model,
                        concept=role,
                        layer=layer,
                        target_pairs=sample_count,
                        only_load=True,
                        safetensors_dir=args.safetensors_dir,
                    )

                logger.info(
                    "Regenerating steered: row=%s q=%r alpha=%.2f layer=%s",
                    idx, question[:50], alpha, layer,
                )
                new_response = steered_model_cache[model_key].generate(
                    prompt=question,
                    alpha=alpha,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                )
                df.at[idx, "steered"] = new_response
                changed += 1

            if changed:
                df.to_csv(csv_path, index=False)
                logger.info("%s: saved after regenerating %s steered rows", csv_path.name, changed)
            total_regenerated += changed

        # --- Regenerate baseline responses ---
        if regen_map["baseline"]:
            try:
                baseline_messages_seed = load_gold_baseline_messages(role, gold_prompts_dir)
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Cannot regenerate baseline for %s: %s", role, exc)
                regen_map["baseline"] = []

        if regen_map["baseline"]:
            if baseline_generator is None:
                baseline_generator = ResponseGeneration()

            changed = 0
            for idx in regen_map["baseline"]:
                question = str(df.at[idx, "question"]).strip()
                temperature = float(df.at[idx, "temperature"])

                messages = deepcopy(baseline_messages_seed)
                messages.append({"role": "user", "content": question})

                logger.info("Regenerating baseline: row=%s q=%r", idx, question[:50])
                _, new_response = baseline_generator(
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                )
                df.at[idx, "baseline"] = new_response
                changed += 1

            if changed:
                df.to_csv(csv_path, index=False)
                logger.info("%s: saved after regenerating %s baseline rows", csv_path.name, changed)
            total_regenerated += changed

        # --- Regenerate assistant_axis responses ---
        if regen_map["assistant_axis"]:
            changed = 0
            for idx in regen_map["assistant_axis"]:
                question = str(df.at[idx, "question"]).strip()
                alpha = float(df.at[idx, "alpha"])
                layer = int(df.at[idx, "layer"])
                sample_count = int(df.at[idx, "sample_count"])
                temperature = float(df.at[idx, "temperature"])

                aa_key = (role, layer, sample_count)
                if aa_key not in assistant_axis_model_cache:
                    logger.info("Loading assistant_axis model: role=%s layer=%s samples=%s", role, layer, sample_count)
                    assistant_axis_model_cache[aa_key] = AssistantAxisPersonaModel.load_or_create(
                        target_model_id=args.model,
                        concept=role,
                        layer=layer,
                        target_pairs=sample_count,
                        safetensors_dir=args.pt_dir,
                    )

                logger.info(
                    "Regenerating assistant_axis: row=%s q=%r alpha=%.2f layer=%s",
                    idx, question[:50], alpha, layer,
                )
                new_response = assistant_axis_model_cache[aa_key].generate(
                    prompt=question,
                    alpha=alpha,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                )
                df.at[idx, "assistant_axis"] = new_response
                changed += 1

            if changed:
                df.to_csv(csv_path, index=False)
                logger.info("%s: saved after regenerating %s assistant_axis rows", csv_path.name, changed)
            total_regenerated += changed

    if args.dry_run:
        logger.info("Dry run complete: %s total row(s) would be regenerated", total_regenerated)
    else:
        logger.info("Completed: %s total row(s) regenerated", total_regenerated)


if __name__ == "__main__":
    main()
