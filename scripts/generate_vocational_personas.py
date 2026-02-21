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
from ast import arg
from decimal import DefaultContext
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import DefaultAioHttpClient
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

    parser.add_argument(
        "--get-fifty",
        action="store_true",
        help="Get 50 chosen personas (for testing with more data)",
        default=False,
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
    elif args.get_fifty:
        chosen_soc_codes = [
            "13-2011.00",
            "27-2011.00",
            "53-2021.00",
            "29-1211.00",
            "17-1011.00",
            "49-3023.00",
            "35-3011.00",
            "17-2031.00",
            "19-2031.00",
            "11-1011.00",
            "21-1021.00",
            "17-2051.00",
            "15-1221.00",
            "15-2051.00",
            "33-3021.00",
            "29-1031.00",
            "17-2071.00",
            "47-2111.00",
            "25-2021.00",
            "29-2042.00",
            "11-9013.00",
            "13-2051.00",
            "29-1131.00",
            "27-3043.00",
            "33-2011.00",
            "27-1024.00",
            "11-3121.00",
            "19-3032.00",
            "23-1023.00",
            "23-1011.00",
            "29-1141.00",
            "11-2022.00",
            "25-4022.00",
            "51-4041.00",
            "15-1212.00",
            "19-1042.00",
            "27-2042.00",
            "51-8011.00",
            "29-1051.00",
            "51-8021.00",
            "27-3091.00",
            "19-3051.00",
            "27-4021.00",
            "29-1123.00",
            "33-3051.00",
            "25-2011.00",
            "27-2012.00",
            "29-1223.00",
            "27-3031.00",
            "15-1252.00",
        ]
        profiles = []
        for soc_code in chosen_soc_codes:
            try:
                profile = loader.get_occupation_profile(soc_code)
                profiles.append(profile)
            except ValueError as e:
                logger.warning(f"Could not load profile for SOC code {soc_code}: {e}")
                with open("missing_soc_codes.txt", "a") as f:
                    f.write(f"{soc_code}\n")

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
                    "positive_prompts": [
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
