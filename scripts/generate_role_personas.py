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
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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


def load_or_generate_profile(
    role_profile: RoleProfile,
    profile_name: str,
    profiles_dir: Path,
    force_regenerate: bool = False,
) -> dict:
    """Load an existing profile from disk, or generate and save it.

    If a cached profile exists but lacks a ``psychological_profile`` key
    (i.e. it was generated before the psych-profile layer was added), the
    missing profile is backfilled via a single LLM call and the file is
    re-saved so subsequent runs skip the call.

    Args:
        role_profile: RoleProfile instance for generating profiles
        profile_name: Name of the role (key in new_roles.json)
        profiles_dir: Directory where per-role JSON profiles are stored
        force_regenerate: If True, regenerate even if file exists

    Returns:
        Profile dict with keys: title, description, tasks, role_contexts,
        psychological_profile
    """
    logger = logging.getLogger(__name__)
    profile_path = profiles_dir / f"{profile_name}.json"

    if profile_path.exists() and not force_regenerate:
        logger.info("Loading existing profile for '%s' from %s", profile_name, profile_path)
        with open(profile_path, "r") as f:
            profile = json.load(f)

        # Backfill psychological_profile on legacy cached profiles
        if not profile.get("psychological_profile"):
            logger.info("Backfilling psychological_profile for cached profile '%s'", profile_name)
            psych = role_profile.generate_psych_profile(
                title=profile.get("title", profile_name),
                description=profile.get("description", ""),
                tasks=profile.get("tasks", []),
                role_contexts=profile.get("role_contexts", {}),
            )
            profile["psychological_profile"] = psych
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2)
            logger.info("Saved backfilled profile for '%s' to %s", profile_name, profile_path)

        return profile

    logger.info("Generating profile for '%s'...", profile_name)
    profile = role_profile.get_profile(profile_name)

    profiles_dir.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    logger.info("Saved profile for '%s' to %s", profile_name, profile_path)

    return profile


def main():
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
    args = parser.parse_args()

    # ── Logging: everything to file, terminal stays clean for tqdm ──
    setup_file_logging(Path(args.log_file))
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting generate_role_personas run")
    logger.info("Args: %s", vars(args))

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

    if not role_names:
        print("Nothing to do — all roles already processed.")
        return

    total = len(role_names)
    print(f"\nProcessing {total} role(s)  |  log → {Path(args.log_file).resolve()}\n")

    # ── Track statistics ──
    success_count = 0
    failed_count = 0
    failed_roles: list[str] = []

    # ──────────────────────────────────────────────────────────────
    #  Bar 1: Profile loading / generation
    # ──────────────────────────────────────────────────────────────
    profiles: dict[str, dict] = {}
    with tqdm(role_names, desc="Loading profiles  ", unit="role", position=0) as pbar:
        for role_name in pbar:
            pbar.set_postfix_str(role_name, refresh=True)
            try:
                profiles[role_name] = load_or_generate_profile(
                    role_profile,
                    role_name,
                    profiles_dir,
                    force_regenerate=args.force_regenerate_profiles,
                )
            except Exception as e:
                logger.error(
                    "Failed to load/generate profile for '%s': %s", role_name, e, exc_info=True
                )
                failed_count += 1
                failed_roles.append(role_name)

    # ──────────────────────────────────────────────────────────────
    #  Bar 2: Persona generation (system prompts, eval, questions)
    # ──────────────────────────────────────────────────────────────
    roles_with_profiles = [r for r in role_names if r in profiles]

    with tqdm(roles_with_profiles, desc="Generating personas", unit="role", position=0) as pbar:
        for role_name in pbar:
            pbar.set_postfix_str(role_name, refresh=True)
            profile = profiles[role_name]
            slug = role_name

            try:
                if args.fallback_only:
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
                    generator.generate_and_save(profile, args.include_questions, slug)

                success_count += 1
                logger.info("Persona saved for '%s'", role_name)

                if not args.fallback_only and args.delay > 0:
                    time.sleep(args.delay)

            except Exception as e:
                logger.error("Failed to generate persona for '%s': %s", role_name, e, exc_info=True)
                failed_count += 1
                failed_roles.append(role_name)

    # ── Summary ──
    print()
    print("=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)
    print(f"  Success : {success_count}")
    print(f"  Failed  : {failed_count}")
    print(f"  Output  : {output_dir}")
    print(f"  Log     : {Path(args.log_file).resolve()}")
    if failed_roles:
        print(f"  Failed roles: {', '.join(failed_roles)}")
    print()

    logger.info("Run complete — success=%d, failed=%d", success_count, failed_count)


if __name__ == "__main__":
    main()
