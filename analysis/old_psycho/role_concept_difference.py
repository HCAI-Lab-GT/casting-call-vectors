#!/usr/bin/env python3
"""Run role concept-difference steering: role_a - role_b."""

from __future__ import annotations

import argparse

from pvx import setup_logging
from pvx.implementations.roles.role_persona_model import RolePersonaModel
from pvx.implementations.judges.hexaco_judge import HexacoJudge

logger = setup_logging(name="role-concept-difference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute role concept difference (A - B) and answer a question."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="allenai/Olmo-3-7B-Instruct",
        help="Target model id.",
    )
    parser.add_argument(
        "--role-a",
        type=str,
        required=True,
        help="Minuend role (A in A - B).",
    )
    parser.add_argument(
        "--role-b",
        type=str,
        required=True,
        help="Subtrahend role (B in A - B).",
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Question to answer.",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength.",
    )
    parser.add_argument(
        "-n",
        "--max-new-tokens",
        type=int,
        default=10,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.99,
        help="Top-p value for nucleus sampling.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=16,
        help="Steering layer index.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/scoring/hexaco_100.json",
        help="config path for hexaco",
    )
    parser.add_argument(
        "--json-filepath",
        type=str,
        default=None,
        help="Optional legacy JSON init path.",
    )
    parser.add_argument(
        "--safetensors-dir",
        type=str,
        default="./persona_data/model_inits/",
        help="Directory for persona safetensors files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diff_name = f"{args.role_a} - {args.role_b}"

    try:
        model = RolePersonaModel.compute_from_concept_difference(
            concept_a=args.role_a,
            concept_b=args.role_b,
            target_model_id=args.model,
            layer=args.layer,
            json_filepath=args.json_filepath,
            safetensors_dir=args.safetensors_dir,
        )
    except Exception:
        logger.exception("Failed to build conceptual difference model for %s", diff_name)
        return 1

    # non_steered = model.generate(
    #     prompt=args.question,
    #     alpha=0.0,
    #     max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    # )
    # steered = model.generate(
    #     prompt=args.question,
    #     alpha=args.alpha,
    #     max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    # )

    # logger.info("=== Question ===")
    # logger.info(args.question)
    # logger.info("")
    # logger.info("=== Non-Steered Answer ===")
    # logger.info(non_steered)
    # logger.info("")
    # logger.info("=== Difference-Steered Answer (%s) ===", diff_name)
    # logger.info(steered)
    
    hexaco_judge = HexacoJudge(hexaco_path=args.config_path)
    
    _, answers, scores = hexaco_judge(
        model, args.alpha, args.max_new_tokens, args.temperature
    )
    
    logger.info("===RESULTS===")
    logger.info(answers)
    logger.info("===Hexaco Counts===")
    logger.info(scores)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())