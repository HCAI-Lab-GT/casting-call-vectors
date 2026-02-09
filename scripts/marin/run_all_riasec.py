#!/usr/bin/env python
"""
Extract RIASEC persona vectors for all 6 traits on a target model.

Loops over realistic, investigative, artistic, social, enterprising, conventional
and runs run_riasec_extraction.py for each.

Usage:
  python scripts/marin/run_all_riasec.py
  python scripts/marin/run_all_riasec.py --model_id marin-community/marin-8b-instruct
  python scripts/marin/run_all_riasec.py --model_id meta-llama/Llama-3.2-1B-Instruct --layer 8
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pvx import setup_logging
from pvx.utils.riasec_utils import RIASECHelpers

logger = setup_logging(name="marin-run-all-riasec")

SCRIPT_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Extract all 6 RIASEC vectors.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index. Default: auto (num_layers // 2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./persona_data/model_inits/",
        help="Directory to save persona vectors.",
    )
    args = parser.parse_args()

    traits = sorted(RIASECHelpers.RIASEC_TRAITS)
    extraction_script = str(SCRIPT_DIR / "run_riasec_extraction.py")

    for i, trait in enumerate(traits, 1):
        logger.info("=== [%d/%d] Extracting trait: %s ===", i, len(traits), trait)

        cmd = [
            sys.executable,
            extraction_script,
            "--model_id", args.model_id,
            "--trait", trait,
            "--output_dir", args.output_dir,
        ]
        if args.layer is not None:
            cmd.extend(["--layer", str(args.layer)])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("FAILED on trait: %s (exit code %d)", trait, result.returncode)
            sys.exit(result.returncode)

        logger.info("Completed trait: %s", trait)

    logger.info("All 6 RIASEC traits extracted successfully.")


if __name__ == "__main__":
    main()
