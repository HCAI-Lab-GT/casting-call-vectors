#!/usr/bin/env python3
"""Generate vocational personas from role profile data.

This script generates persona definitions for all roles defined in new_roles.json,
creating system prompts and evaluation templates compatible with the
assistant-axis pipeline format. It uses RoleProfile to load/generate profile data
and VocationalPersonaGenerator to create persona prompts and questions.

Usage:
    # Generate all personas
    uv run python scripts/generate_role_personas.py

    # Generate a specific role
    uv run python scripts/generate_role_personas.py --role teacher

    # Test with a few roles
    uv run python scripts/generate_role_personas.py --limit 5

    # Skip existing files
    uv run python scripts/generate_role_personas.py --skip-existing

    # Use a different model for persona generation
    uv run python scripts/generate_role_personas.py --persona-model gpt-4o-mini

    # Control concurrency (default 5)
    uv run python scripts/generate_role_personas.py --concurrency 10
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm import tqdm

from pvx.data.role_loader import RoleProfile
from pvx.pvx_models.new_vocational_dataset import VocationalPersonaGenerator

# Default paths
DEFAULT_ROLES_JSON = Path(__file__).parent.parent / "data" / "new_roles.json"
DEFAULT_PROFILES_DIR = Path(__file__).parent.parent / "data" / "new_roles"
DEFAULT_OUTPUT_DIR = "persona_data/vocational_personas/instructions"
DEFAULT_PROFILE_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_PERSONA_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_LOG_FILE = Path(__file__).parent.parent / "logs" / "generate_role_personas.log"


CHECKPOINT_VERSION = 2


class RunCheckpoint(BaseModel):
    version: int = CHECKPOINT_VERSION
    started_at: float
    updated_at: float
    args: dict
    completed_profiles: list[str]
    completed_personas: list[str]
    failed_profiles: list[str]
    failed_personas: list[str]
    repaired_profiles: list[str] = []
    repaired_personas: list[str] = []


# ---------------------------------------------------------------------------
# Completeness definitions
# ---------------------------------------------------------------------------

PSYCH_PROFILE_REQUIRED_KEYS = {
    "core_drive",
    "decision_style",
    "attention_pattern",
    "conflict_stance",
    "risk_orientation",
    "social_posture",
    "relationship_to_authority",
    "failure_response",
    "value_hierarchy",
    "cognitive_bias",
    "inner_contradiction",
    "non_negotiable",
    "recurring_resentment",
    "rejected_premise",
    "instinctive_blame_target",
}


# Minimum acceptable counts for structural profile fields
MIN_TASKS_COUNT = 5
MIN_ROLE_CONTEXTS_COUNT = 5
MIN_PERSONA_PROMPTS = 5


def setup_file_logging(log_file: Path) -> None:
    """Configure all logging to go to a file only, nothing to the terminal.

    Every logger (root, pvx.*, openai, httpx, etc.) is routed to the file.
    Console handlers are removed so tqdm bars stay clean.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Wipe any existing handlers on the root logger (e.g. from basicConfig)
    root = logging.getLogger()
    root.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers that would bloat the file
    for noisy in ("httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def checkpoint_path_for(log_file: Path) -> Path:
    """Store checkpoint alongside the run log."""
    return log_file.with_suffix(log_file.suffix + ".checkpoint.json")


def load_checkpoint(cp_path: Path) -> RunCheckpoint | None:
    """Load a prior checkpoint if it exists and is valid."""
    if not cp_path.exists():
        return None

    with open(cp_path, "r") as f:
        data = json.load(f)

    checkpoint = RunCheckpoint.model_validate(data)
    if checkpoint.version != CHECKPOINT_VERSION:
        return None
    return checkpoint


def save_checkpoint(cp_path: Path, checkpoint: RunCheckpoint) -> None:
    """Persist checkpoint state to disk."""
    checkpoint.updated_at = time.time()
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cp_path, "w") as f:
        json.dump(checkpoint.model_dump(), f, indent=2)


def checkpoint_args_signature(args: argparse.Namespace) -> dict:
    """Return the subset of args that define resumable run identity."""
    return {
        "roles_json": str(Path(args.roles_json)),
        "profiles_dir": str(Path(args.profiles_dir)),
        "output_dir": str(Path(args.output_dir)),
        "role": args.role,
        "limit": args.limit,
        "skip_existing": args.skip_existing,
        "profile_model": args.profile_model,
        "persona_model": args.persona_model,
        "include_questions": args.include_questions,
        "force_regenerate_profiles": args.force_regenerate_profiles,
        "fallback_only": args.fallback_only,
    }


def ensure_checkpoint(
    cp_path: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> RunCheckpoint:
    """Load an existing checkpoint when compatible, otherwise start a new one."""
    args_signature = checkpoint_args_signature(args)
    checkpoint = load_checkpoint(cp_path)

    if checkpoint is not None and checkpoint.args == args_signature:
        logger.info("Resuming from checkpoint at %s", cp_path)
        return checkpoint

    if checkpoint is not None:
        logger.info(
            "Ignoring incompatible checkpoint at %s due to argument mismatch or version change",
            cp_path,
        )

    now_ts = time.time()
    checkpoint = RunCheckpoint(
        started_at=now_ts,
        updated_at=now_ts,
        args=args_signature,
        completed_profiles=[],
        completed_personas=[],
        failed_profiles=[],
        failed_personas=[],
    )
    save_checkpoint(cp_path, checkpoint)
    logger.info("Created new checkpoint at %s", cp_path)
    return checkpoint


def psych_profile_is_stale(profile: dict) -> bool:
    """Return True when a cached psychological profile is missing new required keys."""
    psych = profile.get("psychological_profile")
    if not isinstance(psych, dict) or not psych:
        return True
    return not PSYCH_PROFILE_REQUIRED_KEYS.issubset(psych.keys())


def profile_is_incomplete(profile: dict) -> list[str]:
    """Return a list of issue descriptions for an incomplete profile.

    Checks:
    - tasks missing or too few
    - role_contexts missing or too few
    - psychological_profile missing or stale
    """
    issues: list[str] = []
    tasks = profile.get("tasks", [])
    if not tasks or len(tasks) < MIN_TASKS_COUNT:
        issues.append(f"tasks({len(tasks)}<{MIN_TASKS_COUNT})")
    ctx = profile.get("role_contexts", {})
    if not ctx or len(ctx) < MIN_ROLE_CONTEXTS_COUNT:
        issues.append(f"role_contexts({len(ctx)}<{MIN_ROLE_CONTEXTS_COUNT})")
    if psych_profile_is_stale(profile):
        issues.append("psych_stale_or_missing")
    return issues


def persona_is_incomplete(persona: dict) -> list[str]:
    """Return a list of issue descriptions for an incomplete persona file.

    Checks:
    - positive_prompts missing or too few
    - evaluation_prompt missing
    """
    issues: list[str] = []
    prompts = persona.get("positive_prompts", [])
    if not prompts or len(prompts) < MIN_PERSONA_PROMPTS:
        issues.append(f"prompts({len(prompts)}<{MIN_PERSONA_PROMPTS})")
    if not persona.get("evaluation_prompt"):
        issues.append("no_eval_prompt")
    return issues


# ---------------------------------------------------------------------------
# Async profile loading / generation
# ---------------------------------------------------------------------------


async def async_repair_profile(
    role_profile: RoleProfile,
    profile: dict,
    profile_name: str,
    profile_path: Path,
) -> dict:
    """Repair an existing profile by regenerating only the missing/incomplete fields.

    Regenerates tasks, role_contexts, or psychological_profile as needed
    without touching fields that are already populated.

    Returns the repaired profile dict (also written to disk).
    """
    logger = logging.getLogger(__name__)
    issues = profile_is_incomplete(profile)
    if not issues:
        return profile

    logger.info("Repairing profile '%s' — issues: %s", profile_name, ", ".join(issues))
    dirty = False
    title = profile.get("title", profile_name)
    description = profile.get("description", "")

    # Repair tasks
    tasks = profile.get("tasks", [])
    if not tasks or len(tasks) < MIN_TASKS_COUNT:
        logger.info("Regenerating tasks for '%s'", profile_name)
        new_tasks = await role_profile.agenerate_tasks(title, description)
        if new_tasks and len(new_tasks) >= MIN_TASKS_COUNT:
            profile["tasks"] = new_tasks
            dirty = True
        else:
            logger.warning(
                "Task regeneration for '%s' returned %d tasks (need %d)",
                profile_name,
                len(new_tasks) if new_tasks else 0,
                MIN_TASKS_COUNT,
            )

    # Repair role_contexts
    ctx = profile.get("role_contexts", {})
    if not ctx or len(ctx) < MIN_ROLE_CONTEXTS_COUNT:
        logger.info("Regenerating role_contexts for '%s'", profile_name)
        new_ctx = await role_profile.agenerate_role_context(
            title, description, profile.get("tasks", [])
        )
        if new_ctx and len(new_ctx) >= MIN_ROLE_CONTEXTS_COUNT:
            profile["role_contexts"] = new_ctx
            dirty = True
        else:
            logger.warning(
                "Role context regeneration for '%s' returned %d contexts (need %d)",
                profile_name,
                len(new_ctx) if new_ctx else 0,
                MIN_ROLE_CONTEXTS_COUNT,
            )

    # Repair psychological_profile
    if psych_profile_is_stale(profile):
        logger.info("Regenerating psychological_profile for '%s'", profile_name)
        psych = await role_profile.agenerate_psych_profile(
            title=title,
            description=description,
            tasks=profile.get("tasks", []),
            role_contexts=profile.get("role_contexts", {}),
        )
        if psych:
            profile["psychological_profile"] = psych
            dirty = True

    if dirty:
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        logger.info("Saved repaired profile for '%s' to %s", profile_name, profile_path)

    remaining = profile_is_incomplete(profile)
    if remaining:
        logger.warning(
            "Profile '%s' still incomplete after repair: %s", profile_name, ", ".join(remaining)
        )

    return profile


async def async_load_or_generate_profile(
    role_profile: RoleProfile,
    profile_name: str,
    profiles_dir: Path,
    force_regenerate: bool = False,
) -> dict:
    """Async version: Load an existing profile from disk, or generate and save it.

    If a cached profile exists:
    - Checks for incomplete structural data (tasks, role_contexts) and repairs
    - Checks for stale/missing psychological profile and backfills
    If no cached profile exists, generates from scratch.
    """
    logger = logging.getLogger(__name__)
    profile_path = profiles_dir / f"{profile_name}.json"

    if profile_path.exists() and not force_regenerate:
        logger.info("Loading existing profile for '%s' from %s", profile_name, profile_path)
        with open(profile_path, "r") as f:
            profile = json.load(f)

        # Repair any incomplete fields (tasks, role_contexts, psych)
        if profile_is_incomplete(profile):
            profile = await async_repair_profile(role_profile, profile, profile_name, profile_path)

        return profile

    logger.info("Generating profile for '%s'...", profile_name)
    profile = await role_profile.aget_profile(profile_name)

    profiles_dir.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info("Saved profile for '%s' to %s", profile_name, profile_path)

    return profile


# ---------------------------------------------------------------------------
# Sync fallback for load_or_generate_profile (kept for backward compat)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pre-flight audit: scan all existing profiles and personas for completeness
# ---------------------------------------------------------------------------


def audit_existing_profiles(profiles_dir: Path) -> dict[str, list[str]]:
    """Scan all cached profile JSONs and return {name: [issues]} for incomplete ones."""
    incomplete: dict[str, list[str]] = {}
    for f in sorted(profiles_dir.glob("*.json")):
        try:
            with open(f, "r") as fh:
                profile = json.load(fh)
            issues = profile_is_incomplete(profile)
            if issues:
                incomplete[f.stem] = issues
        except Exception:
            incomplete[f.stem] = ["unreadable_json"]
    return incomplete


def audit_existing_personas(personas_dir: Path) -> dict[str, list[str]]:
    """Scan all cached persona JSONs and return {name: [issues]} for incomplete ones."""
    incomplete: dict[str, list[str]] = {}
    if not personas_dir.exists():
        return incomplete
    for f in sorted(personas_dir.glob("*.json")):
        try:
            with open(f, "r") as fh:
                persona = json.load(fh)
            issues = persona_is_incomplete(persona)
            if issues:
                incomplete[f.stem] = issues
        except Exception:
            incomplete[f.stem] = ["unreadable_json"]
    return incomplete


# ---------------------------------------------------------------------------
# Checkpoint-aware async wrappers
# ---------------------------------------------------------------------------


async def _profile_worker(
    role_name: str,
    role_profile: RoleProfile,
    profiles_dir: Path,
    force_regenerate: bool,
    semaphore: asyncio.Semaphore,
    checkpoint: RunCheckpoint,
    cp_path: Path,
    cp_lock: asyncio.Lock,
    profiles: dict,
    stats: dict,
    pbar: tqdm,
):
    """Process a single profile: load from cache or generate via async LLM.

    Only marks the profile as completed if it passes completeness checks.
    """
    logger = logging.getLogger(__name__)
    async with semaphore:
        pbar.set_postfix_str(role_name, refresh=True)
        try:
            profile = await async_load_or_generate_profile(
                role_profile, role_name, profiles_dir, force_regenerate
            )

            # Only mark complete if the profile actually passes completeness
            remaining_issues = profile_is_incomplete(profile)
            if remaining_issues:
                logger.warning(
                    "Profile '%s' still has issues after generation: %s",
                    role_name,
                    ", ".join(remaining_issues),
                )

            profiles[role_name] = profile

            async with cp_lock:
                if role_name not in checkpoint.completed_profiles:
                    checkpoint.completed_profiles.append(role_name)
                if role_name in checkpoint.failed_profiles:
                    checkpoint.failed_profiles.remove(role_name)
                save_checkpoint(cp_path, checkpoint)

        except Exception as e:
            logger.error(
                "Failed to load/generate profile for '%s': %s", role_name, e, exc_info=True
            )
            async with cp_lock:
                stats["failed_count"] += 1
                stats["failed_roles"].append(role_name)
                if role_name not in checkpoint.failed_profiles:
                    checkpoint.failed_profiles.append(role_name)
                save_checkpoint(cp_path, checkpoint)
        finally:
            pbar.update(1)


async def _persona_worker(
    role_name: str,
    profile: dict,
    generator: VocationalPersonaGenerator,
    include_questions: bool,
    fallback_only: bool,
    semaphore: asyncio.Semaphore,
    checkpoint: RunCheckpoint,
    cp_path: Path,
    cp_lock: asyncio.Lock,
    stats: dict,
    pbar: tqdm,
):
    """Process a single persona: generate system prompts + eval via async LLM."""
    logger = logging.getLogger(__name__)
    slug = role_name
    async with semaphore:
        pbar.set_postfix_str(role_name, refresh=True)
        try:
            if fallback_only:
                persona = {
                    "positive_prompts": [
                        {"pos": p} for p in generator._generate_fallback_prompts(profile)
                    ],
                    "eval_prompt": generator.generate_eval_prompt(profile),
                    "_metadata": {
                        "title": profile.get("title", role_name),
                    },
                }
                generator.save_persona(slug, persona)
            else:
                await generator.agenerate_and_save(profile, include_questions, slug)

            async with cp_lock:
                stats["success_count"] += 1
                if role_name not in checkpoint.completed_personas:
                    checkpoint.completed_personas.append(role_name)
                if role_name in checkpoint.failed_personas:
                    checkpoint.failed_personas.remove(role_name)
                save_checkpoint(cp_path, checkpoint)

            logger.info("Persona saved for '%s'", role_name)

        except Exception as e:
            logger.error("Failed to generate persona for '%s': %s", role_name, e, exc_info=True)
            async with cp_lock:
                stats["failed_count"] += 1
                stats["failed_roles"].append(role_name)
                if role_name not in checkpoint.failed_personas:
                    checkpoint.failed_personas.append(role_name)
                save_checkpoint(cp_path, checkpoint)
        finally:
            pbar.update(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main():
    parser = argparse.ArgumentParser(
        description="Generate vocational personas from role profile data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--roles-json",
        default=str(DEFAULT_ROLES_JSON),
        help="Path to new_roles.json file",
    )
    parser.add_argument(
        "--profiles-dir",
        default=str(DEFAULT_PROFILES_DIR),
        help="Directory for per-role profile JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for persona files",
    )
    parser.add_argument(
        "--role",
        help="Generate persona for a specific role name only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of personas to generate (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip roles that already have persona files in the output dir",
    )
    parser.add_argument(
        "--profile-model",
        default=DEFAULT_PROFILE_MODEL,
        help="Model for profile generation via RoleProfile (default: moonshotai/kimi-k2.5)",
    )
    parser.add_argument(
        "--persona-model",
        default=DEFAULT_PERSONA_MODEL,
        help="Model for persona prompt generation (default: moonshotai/kimi-k2.5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (rate limiting)",
    )
    parser.add_argument(
        "--include-questions",
        action="store_true",
        default=False,
        help="Include evaluation questions in the persona definition",
    )
    parser.add_argument(
        "--force-regenerate-profiles",
        action="store_true",
        default=False,
        help="Regenerate role profiles even if they already exist on disk",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Use fallback prompts only (no API calls for persona generation)",
    )
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG_FILE),
        help="Path to log file (default: logs/generate_role_personas.log)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent async API calls (default: 5)",
    )
    args = parser.parse_args()

    # ── Logging: everything to file, terminal stays clean for tqdm ──
    log_file = Path(args.log_file)
    setup_file_logging(log_file)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting generate_role_personas run (async, concurrency=%d)", args.concurrency)
    logger.info("Args: %s", vars(args))

    cp_path = checkpoint_path_for(log_file)
    checkpoint = ensure_checkpoint(cp_path, args, logger)

    if checkpoint.repaired_profiles or checkpoint.repaired_personas:
        logger.info(
            "Checkpoint shows %d repaired profiles, %d repaired personas from prior runs",
            len(checkpoint.repaired_profiles),
            len(checkpoint.repaired_personas),
        )

    # ── Load roles dictionary ──
    roles_json_path = Path(args.roles_json)
    if not roles_json_path.exists():
        print(f"ERROR: Roles JSON file not found: {roles_json_path}")
        sys.exit(1)

    with open(roles_json_path, "r") as f:
        profile_dict = json.load(f)

    logger.info("Loaded %d roles from %s", len(profile_dict), roles_json_path)

    # ── Determine which roles to process ──
    if args.role:
        if args.role not in profile_dict:
            print(f"ERROR: Role '{args.role}' not found in {roles_json_path}")
            sys.exit(1)
        role_names = [args.role]
    else:
        role_names = list(profile_dict.keys())

    if args.limit:
        role_names = role_names[: args.limit]

    # ── Initialize loader + generator ──
    role_profile = RoleProfile(model=args.profile_model, profile_dict=profile_dict)
    generator = VocationalPersonaGenerator(model=args.persona_model, output_dir=args.output_dir)
    output_dir = Path(args.output_dir)
    profiles_dir = Path(args.profiles_dir)

    # ── Filter existing personas if requested ──
    if args.skip_existing:
        original_count = len(role_names)
        role_names = [name for name in role_names if not (output_dir / f"{name}.json").exists()]
        skipped = original_count - len(role_names)
        if skipped > 0:
            logger.info("Skipping %d existing personas", skipped)

    completed_profiles_set = set(checkpoint.completed_profiles)
    completed_personas_set = set(checkpoint.completed_personas)

    # ──────────────────────────────────────────────────────────────
    #  Pre-flight: Audit ALL existing profiles for completeness
    # ──────────────────────────────────────────────────────────────
    print("\n── Pre-flight audit ──")
    incomplete_profiles = audit_existing_profiles(profiles_dir)
    incomplete_personas = audit_existing_personas(output_dir)

    if incomplete_profiles:
        print(f"  Incomplete profiles found: {len(incomplete_profiles)}")
        for name, issues in list(incomplete_profiles.items())[:10]:
            print(f"    {name}: {', '.join(issues)}")
        if len(incomplete_profiles) > 10:
            print(f"    ... and {len(incomplete_profiles) - 10} more")
        logger.info(
            "Pre-flight: %d incomplete profiles: %s",
            len(incomplete_profiles),
            json.dumps(dict(list(incomplete_profiles.items())[:20])),
        )
    else:
        print("  All existing profiles complete ✓")

    if incomplete_personas:
        print(f"  Incomplete personas found: {len(incomplete_personas)}")
        for name, issues in list(incomplete_personas.items())[:10]:
            print(f"    {name}: {', '.join(issues)}")
        if len(incomplete_personas) > 10:
            print(f"    ... and {len(incomplete_personas) - 10} more")
        logger.info(
            "Pre-flight: %d incomplete personas: %s",
            len(incomplete_personas),
            json.dumps(dict(list(incomplete_personas.items())[:20])),
        )
    else:
        print("  All existing personas complete ✓")

    # Incomplete profiles need repair — add them to the profile work list
    # even if they were in the checkpoint as "completed" (checkpoint was wrong)
    profiles_needing_repair = set(incomplete_profiles.keys())
    for stale_name in profiles_needing_repair:
        if stale_name in completed_profiles_set:
            completed_profiles_set.discard(stale_name)
            if stale_name in checkpoint.completed_profiles:
                checkpoint.completed_profiles.remove(stale_name)
            logger.info(
                "Removed '%s' from completed_profiles — needs repair: %s",
                stale_name,
                ", ".join(incomplete_profiles[stale_name]),
            )

    # Incomplete personas need regeneration — remove from completed
    personas_needing_regen = set(incomplete_personas.keys())
    for stale_name in personas_needing_regen:
        if stale_name in completed_personas_set:
            completed_personas_set.discard(stale_name)
            if stale_name in checkpoint.completed_personas:
                checkpoint.completed_personas.remove(stale_name)
            logger.info(
                "Removed '%s' from completed_personas — needs regen: %s",
                stale_name,
                ", ".join(incomplete_personas[stale_name]),
            )

    save_checkpoint(cp_path, checkpoint)

    # ──────────────────────────────────────────────────────────────
    #  Build work lists
    # ──────────────────────────────────────────────────────────────

    # Filter out already-completed profiles from checkpoint
    if completed_profiles_set:
        before = len(role_names)
        role_names = [name for name in role_names if name not in completed_profiles_set]
        resumed_profiles = before - len(role_names)
        if resumed_profiles > 0:
            logger.info(
                "Skipping %d roles with profiles already completed in checkpoint", resumed_profiles
            )

    # Add profiles needing repair that aren't already in the work list
    existing_role_set = set(role_names)
    for repair_name in sorted(profiles_needing_repair):
        if repair_name not in existing_role_set:
            # Only add if this role is in scope (matches --role / --limit filters)
            all_role_names = [args.role] if args.role else list(profile_dict.keys())
            if args.limit:
                all_role_names = all_role_names[: args.limit]
            if repair_name in all_role_names:
                role_names.append(repair_name)
                existing_role_set.add(repair_name)

    # Build the full persona candidate list (before profile filtering)
    roles_needing_personas = [
        name for name in list(profile_dict.keys()) if (not args.role or name == args.role)
    ]
    if args.limit:
        roles_needing_personas = roles_needing_personas[: args.limit]

    # For --skip-existing: skip only personas that exist AND are complete
    if args.skip_existing:
        roles_needing_personas = [
            name
            for name in roles_needing_personas
            if not (output_dir / f"{name}.json").exists() or name in personas_needing_regen
        ]

    if not role_names and not roles_needing_personas:
        print("\nNothing to do — all roles already processed.")
        return

    # ── Shared state ──
    semaphore = asyncio.Semaphore(args.concurrency)
    cp_lock = asyncio.Lock()
    profiles: dict[str, dict] = {}
    stats: dict = {
        "success_count": len(completed_personas_set),
        "failed_count": 0,
        "failed_roles": [],
        "repaired_profiles": 0,
        "repaired_personas": 0,
    }

    # ──────────────────────────────────────────────────────────────
    #  Phase 1: Profile loading / generation / repair (async)
    # ──────────────────────────────────────────────────────────────
    if role_names:
        total_profiles = len(role_names)
        repair_count = len(profiles_needing_repair & set(role_names))
        new_count = total_profiles - repair_count
        print(
            f"\nPhase 1: {new_count} new + {repair_count} repair = {total_profiles} profile(s)  "
            f"|  concurrency={args.concurrency}  |  log → {log_file.resolve()}"
        )
        print(f"  checkpoint → {cp_path.resolve()}\n")

        with tqdm(total=total_profiles, desc="Loading profiles  ", unit="role") as pbar:
            tasks = [
                _profile_worker(
                    role_name=name,
                    role_profile=role_profile,
                    profiles_dir=profiles_dir,
                    force_regenerate=args.force_regenerate_profiles,
                    semaphore=semaphore,
                    checkpoint=checkpoint,
                    cp_path=cp_path,
                    cp_lock=cp_lock,
                    profiles=profiles,
                    stats=stats,
                    pbar=pbar,
                )
                for name in role_names
            ]
            await asyncio.gather(*tasks)
    else:
        print("\nPhase 1: All profiles already completed in checkpoint — skipping.\n")

    # ──────────────────────────────────────────────────────────────
    #  Phase 2: Persona generation / repair (async, concurrent)
    # ──────────────────────────────────────────────────────────────
    # Build the list of roles that need persona generation and have profiles
    roles_with_profiles: list[str] = []

    for role_name in roles_needing_personas:
        # Skip only if completed AND not needing regen
        if role_name in completed_personas_set and role_name not in personas_needing_regen:
            continue
        if role_name in profiles:
            roles_with_profiles.append(role_name)
            continue

        # Try to load from disk (profile may have been generated in a prior run)
        profile_path = profiles_dir / f"{role_name}.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    profiles[role_name] = json.load(f)
                roles_with_profiles.append(role_name)
            except Exception as e:
                logger.error(
                    "Failed to load cached profile for persona generation '%s': %s",
                    role_name,
                    e,
                    exc_info=True,
                )
                async with cp_lock:
                    stats["failed_count"] += 1
                    stats["failed_roles"].append(role_name)
                    if role_name not in checkpoint.failed_personas:
                        checkpoint.failed_personas.append(role_name)
                    save_checkpoint(cp_path, checkpoint)

    if roles_with_profiles:
        total_personas = len(roles_with_profiles)
        regen_count = len(personas_needing_regen & set(roles_with_profiles))
        new_persona_count = total_personas - regen_count
        print(
            f"\nPhase 2: {new_persona_count} new + {regen_count} regen = {total_personas} persona(s)  "
            f"|  concurrency={args.concurrency}\n"
        )

        with tqdm(total=total_personas, desc="Generating personas", unit="role") as pbar:
            tasks = [
                _persona_worker(
                    role_name=name,
                    profile=profiles[name],
                    generator=generator,
                    include_questions=args.include_questions,
                    fallback_only=args.fallback_only,
                    semaphore=semaphore,
                    checkpoint=checkpoint,
                    cp_path=cp_path,
                    cp_lock=cp_lock,
                    stats=stats,
                    pbar=pbar,
                )
                for name in roles_with_profiles
            ]
            await asyncio.gather(*tasks)
    else:
        print("\nPhase 2: All personas already completed — skipping.\n")

    # ── Summary ──
    print()
    print("=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)
    print(f"  Success  : {stats['success_count']}")
    print(f"  Failed   : {stats['failed_count']}")
    print(
        f"  Repaired : {len(profiles_needing_repair)} profiles, {len(personas_needing_regen)} personas"
    )
    print(f"  Output   : {output_dir}")
    print(f"  Log      : {log_file.resolve()}")
    if stats["failed_roles"]:
        print(f"  Failed roles: {', '.join(stats['failed_roles'])}")
    print()

    logger.info(
        "Run complete — success=%d, failed=%d", stats["success_count"], stats["failed_count"]
    )
    save_checkpoint(cp_path, checkpoint)


def main():
    """Entry point — runs the async main inside an event loop."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
