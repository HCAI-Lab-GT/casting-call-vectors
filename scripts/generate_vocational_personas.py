#!/usr/bin/env python3
"""Generate vocational personas from O*NET occupation data.

This script generates persona definitions for all O*NET occupations,
creating system prompts and evaluation templates compatible with the
assistant-axis pipeline format.

Usage:
    # Generate all personas
    uv run python scripts/generate_vocational_personas.py

    # Generate specific RIASEC type
    uv run python scripts/generate_vocational_personas.py --riasec S

    # Generate specific occupation
    uv run python scripts/generate_vocational_personas.py --soc-code 29-1141.00

    # Test with a few occupations
    uv run python scripts/generate_vocational_personas.py --limit 5

    # Skip existing files
    uv run python scripts/generate_vocational_personas.py --skip-existing
"""

import argparse
import logging
import sys
import time
from decimal import DefaultContext
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tqdm import tqdm

from pvx.data.onet_loader import RIASEC_FULL_NAMES, ONETLoader
from pvx.pvx_models.vocational_dataset import VocationalPersonaGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate vocational personas from O*NET data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        default="persona_data/vocational_personas/instructions",
        help="Output directory for persona files",
    )
    parser.add_argument(
        "--riasec",
        choices=["R", "I", "A", "S", "E", "C"],
        help="Only generate personas for this RIASEC primary type",
    )
    parser.add_argument(
        "--soc-code",
        help="Generate persona for a specific SOC code only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of personas to generate (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip occupations that already have persona files",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for prompt generation",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (rate limiting)",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Use fallback prompts only (no API calls)",
    )

    parser.add_argument(
        "--include_questions",
        action="store_true",
        default=False,
        help="Include example questions in the persona definition",
    )
    args = parser.parse_args()

    # Initialize loader and generator
    loader = ONETLoader()
    generator = VocationalPersonaGenerator(model=args.model, output_dir=args.output_dir)
    output_dir = Path(args.output_dir)

    # Get occupations to process
    if args.soc_code:
        # Single occupation
        try:
            profile = loader.get_occupation_profile(args.soc_code)
            profiles = [profile]
        except ValueError as e:
            logger.error(f"Invalid SOC code: {e}")
            sys.exit(1)
    else:
        # All occupations
        profiles = loader.get_all_occupation_profiles()

        # Filter by RIASEC if specified
        if args.riasec:
            profiles = [p for p in profiles if p.get("riasec_primary") == args.riasec]
            logger.info(f"Filtered to {len(profiles)} {RIASEC_FULL_NAMES[args.riasec]} occupations")

    # Apply limit
    if args.limit:
        profiles = profiles[: args.limit]

    # Filter existing if requested
    if args.skip_existing:
        original_count = len(profiles)
        profiles = [
            p for p in profiles if not (output_dir / f"{loader.to_slug(p['title'])}.json").exists()
        ]
        skipped = original_count - len(profiles)
        if skipped > 0:
            logger.info(f"Skipping {skipped} existing personas")

    if not profiles:
        logger.info("No occupations to process")
        return

    logger.info(f"Generating personas for {len(profiles)} occupations")

    # Track statistics
    success_count = 0
    failed_count = 0
    riasec_counts: dict[str, int] = {}

    # Generate personas
    for profile in tqdm(profiles, desc="Generating personas"):
        try:
            slug = loader.to_slug(profile["title"])

            if args.fallback_only:
                # Use fallback prompts (no API call)
                persona = {
                    "instruction": [
                        {"pos": p} for p in generator._generate_fallback_prompts(profile)
                    ],
                    "eval_prompt": generator.generate_eval_prompt(profile),
                    "_metadata": {
                        "soc_code": profile["soc_code"],
                        "title": profile["title"],
                        "riasec": profile.get("riasec", {}),
                        "riasec_primary": profile.get("riasec_primary"),
                        "highpoint_codes": profile.get("highpoint_codes", []),
                    },
                }
                generator.save_persona(slug, persona)
            else:
                generator.generate_and_save(profile, args.include_questions, slug)

            success_count += 1

            # Track RIASEC distribution
            primary = profile.get("riasec_primary")
            if primary:
                riasec_counts[primary] = riasec_counts.get(primary, 0) + 1

            # Rate limiting
            if not args.fallback_only and args.delay > 0:
                time.sleep(args.delay)

        except Exception as e:
            logger.error(f"Failed to generate persona for {profile['title']}: {e}")
            failed_count += 1

    # Summary
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)
    print(f"Success: {success_count}")
    print(f"Failed:  {failed_count}")
    print(f"Output:  {output_dir}")

    if riasec_counts:
        print("\nRIASEC Distribution:")
        for letter in "RIASEC":
            count = riasec_counts.get(letter, 0)
            name = RIASEC_FULL_NAMES[letter]
            print(f"  {letter} ({name}): {count}")


if __name__ == "__main__":
    main()
