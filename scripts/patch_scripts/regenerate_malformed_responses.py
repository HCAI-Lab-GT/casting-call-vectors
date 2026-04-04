"""
Scan every (question, alpha, layer) row for a role's gold-standard CSV and
regenerate any response that is missing or malformed (using the same detection
logic as report_malformed_responses.py).

Supports regenerating:
  - steered    → via RoleLayersPersonaModel
  - assistant_axis → via AssistantAxisPersonaModel

Usage:
    # Dry run for a single role
    python scripts/patch_scripts/regenerate_malformed_responses.py --roles accountant --dry_run

    # Fix all columns for a role
    python scripts/patch_scripts/regenerate_malformed_responses.py --roles accountant

    # Fix only steered column for multiple roles
    python scripts/patch_scripts/regenerate_malformed_responses.py --roles accountant lawyer --columns steered

    # Fix specific alphas only
    python scripts/patch_scripts/regenerate_malformed_responses.py --roles accountant --alphas 1.0 2.5
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from pvx import setup_logging
from pvx.implementations.roles_layers.assistant_axis_persona_model import AssistantAxisPersonaModel
from pvx.implementations.roles_layers.role_layers_persona_model import RoleLayersPersonaModel

logger = setup_logging(name="regenerate-malformed-responses")

EXPECTED_ALPHAS = {1.0, 1.5, 2.0, 2.5}


def load_unique_questions(path: Path) -> set[str]:
    questions = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                questions.add(obj["question"])
    return questions

# ---------------------------------------------------------------------------
# Malformedness detection (mirrors report_malformed_responses.py)
# ---------------------------------------------------------------------------

def has_value(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in ("", "nan", "NaN", "NAN") and value not in ("", "nan", "NaN", "NAN")


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
    matching = [v for v in df.loc[mask, col] if has_value(v) and str(v).strip() == value]
    if not matching:
        return False
    # Preserve the highest alpha among duplicates, flag all others
    max_alpha = df.loc[(df["question"] == question) & (df[col].apply(lambda v: has_value(v) and str(v).strip() == value)), "alpha"].max()
    return alpha < max_alpha


def needs_regeneration(df: pd.DataFrame, idx: int, col: str) -> bool:
    """Return True if the cell is missing, malformed, or a duplicate across alphas."""
    value = df.at[idx, col]
    if is_malformed(value):
        return True
    if is_duplicate_across_alphas(df, idx, col):
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate missing/malformed responses per question×alpha×layer in role CSVs."
    )
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments/")
    parser.add_argument("--roles", nargs="+", default=None, help="Roles to process (default: all)")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["steered", "assistant_axis"],
        choices=["steered", "assistant_axis"],
        help="Which response columns to check and regenerate",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=None, help="Only process these alphas")
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--safetensors_dir", default="./persona_data/model_inits/")
    parser.add_argument("--pt_dir", default="persona_data/assistant-axis/olmo-3-7b-instruct/vectors/")
    parser.add_argument("--max_new_tokens", type=int, default=2000)
    parser.add_argument("--questions_file", default="./configs/validation_questions.jsonl")
    parser.add_argument("--dry_run", action="store_true", help="Report what would be regenerated without changing files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    expected_questions = load_unique_questions(Path(args.questions_file))
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

    # --- Pre-scan: tally regeneration needs across all roles ---
    scan_results: list[tuple[Path, str, pd.DataFrame, dict[str, list[int]], dict[float, int]]] = []
    global_counts: dict[str, int] = {col: 0 for col in columns_to_check}
    global_baseline_missing = 0

    for csv_path in csv_paths:
        role = csv_path.stem.removeprefix("Comparison_GoldStandard_")
        df = pd.read_csv(csv_path)

        for col in ("steered", "assistant_axis"):
            if col in df.columns:
                df[col] = df[col].astype(object)

        required = {"question", "alpha", "layer", "sample_count", "temperature"}
        required |= set(columns_to_check)
        missing_cols = required - set(df.columns)
        if missing_cols:
            logger.warning("Skipping %s: missing columns %s", csv_path.name, sorted(missing_cols))
            continue

        sc50 = df[df["sample_count"].astype(int) == 50]
        ref_row = sc50.iloc[0] if not sc50.empty else df.iloc[0]
        alphas_to_scan = allowed_alphas if allowed_alphas is not None else EXPECTED_ALPHAS
        skeleton_rows = []
        for question in expected_questions:
            for alpha in alphas_to_scan:
                mask = (df["question"].str.strip() == question) & (df["alpha"].apply(lambda a: abs(float(a) - alpha) < 1e-9)) & (df["sample_count"].apply(lambda s: int(s) == 50))
                if not mask.any():
                    skeleton = {c: None for c in df.columns}
                    skeleton.update({
                        "role": ref_row["role"],
                        "layer": ref_row["layer"],
                        "sample_count": 50,
                        "alpha": float(alpha),
                        "temperature": float(ref_row["temperature"]),
                        "question": question,
                    })
                    skeleton_rows.append(skeleton)

        skeleton_indices: set[int] = set()
        if skeleton_rows:
            skeleton_start = len(df)
            df = pd.concat([df, pd.DataFrame(skeleton_rows)], ignore_index=True)
            for c in ("steered", "assistant_axis"):
                if c in df.columns:
                    df[c] = df[c].astype(object)
            skeleton_indices = set(range(skeleton_start, len(df)))
            logger.info("%s: appended %s skeleton row(s) for missing questions", role, len(skeleton_rows))

        regen_map: dict[str, list[int]] = {col: [] for col in columns_to_check}
        baseline_alpha_counts: dict[float, int] = {}
        for idx in df.index:
            alpha = df.at[idx, "alpha"]
            if idx in skeleton_indices:
                # always regenerate skeleton rows
                if allowed_alphas is None or any(abs(float(alpha) - a) < 1e-9 for a in allowed_alphas):
                    for col in columns_to_check:
                        regen_map[col].append(idx)
                continue
            sample_count = int(df.at[idx, "sample_count"])
            if sample_count != 50:
                continue
            if allowed_alphas is not None and not any(abs(alpha - a) < 1e-9 for a in allowed_alphas):
                continue
            for col in columns_to_check:
                if needs_regeneration(df, idx, col):
                    regen_map[col].append(idx)
            # Track missing/malformed baseline responses (report-only, not regenerated)
            if "baseline" in df.columns and is_malformed(df.at[idx, "baseline"]):
                baseline_alpha_counts[alpha] = baseline_alpha_counts.get(alpha, 0) + 1

        total_needed = sum(len(indices) for indices in regen_map.values())
        role_baseline_total = sum(baseline_alpha_counts.values())
        if total_needed == 0 and role_baseline_total == 0:
            continue

        for col in columns_to_check:
            global_counts[col] += len(regen_map[col])
        global_baseline_missing += role_baseline_total
        scan_results.append((csv_path, role, df, regen_map, baseline_alpha_counts))

    grand_total = sum(global_counts.values())
    if grand_total == 0 and global_baseline_missing == 0:
        logger.info("No malformed/missing responses found across %s CSV(s)", len(csv_paths))
        return

    summary_parts = [f"{col}={global_counts[col]}" for col in columns_to_check if global_counts[col]]
    if global_baseline_missing:
        summary_parts.append(f"baseline={global_baseline_missing} (report only)")
    logger.info(
        "=== Regeneration summary: %s total (%s) across %s role(s) ===",
        grand_total,
        ", ".join(summary_parts),
        len(scan_results),
    )

    # Always log per-role/per-alpha breakdown (including baseline missing counts)
    for csv_path, role, df, regen_map, baseline_alpha_counts in scan_results:
        logger.info("--- %s ---", role)
        for col in columns_to_check:
            if not regen_map[col]:
                continue
            alpha_counts: dict[float, int] = {}
            for idx in regen_map[col]:
                a = df.at[idx, "alpha"]
                alpha_counts[a] = alpha_counts.get(a, 0) + 1
            for a in sorted(alpha_counts):
                logger.info("  %s alpha=%.2f: %s rows", col, a, alpha_counts[a])
        if baseline_alpha_counts:
            for a in sorted(baseline_alpha_counts):
                logger.info("  baseline (missing/malformed) alpha=%.2f: %s rows", a, baseline_alpha_counts[a])

    if args.dry_run:
        logger.info("Dry run complete: %s total row(s) would be regenerated", grand_total)
        return

    # --- Regenerate ---
    total_regenerated = 0

    for csv_path, role, df, regen_map, _baseline_alpha_counts in scan_results:
        logger.info("=== Processing role: %s ===", role)

        # --- Regenerate steered responses ---
        if regen_map.get("steered"):
            changed = 0
            sorted_indices = sorted(
                regen_map["steered"],
                key=lambda i: (int(df.at[i, "layer"]), int(df.at[i, "sample_count"]), float(df.at[i, "alpha"])),
                # reverse=True,
            )
            for idx in sorted_indices:
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
                df.to_csv(csv_path, index=False)
                logger.info("%s: saved steered row %s (%s/%s)", csv_path.name, idx, changed, len(regen_map["steered"]))

            total_regenerated += changed

        # --- Regenerate assistant_axis responses ---
        if regen_map.get("assistant_axis"):
            changed = 0
            sorted_indices = sorted(
                regen_map["assistant_axis"],
                key=lambda i: (int(df.at[i, "layer"]), int(df.at[i, "sample_count"]), float(df.at[i, "alpha"])),
            )
            for idx in sorted_indices:
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
                df.to_csv(csv_path, index=False)
                logger.info("%s: saved assistant_axis row %s (%s/%s)", csv_path.name, idx, changed, len(regen_map["assistant_axis"]))

            total_regenerated += changed

    logger.info("Completed: %s total row(s) regenerated", total_regenerated)


if __name__ == "__main__":
    main()
