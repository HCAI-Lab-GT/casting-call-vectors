#!/usr/bin/env python
"""
Evaluate RIASEC steering effectiveness for a model.

For each of the 6 traits:
  - Load the extracted persona vector
  - Generate with alpha=0 (baseline), alpha=3 (positive), alpha=-3 (negative)
  - Run RIASECJudge YES/NO evaluation
  - Save results to JSON

Usage:
  python scripts/marin/eval_riasec_steering.py
  python scripts/marin/eval_riasec_steering.py --model_id marin-community/marin-8b-instruct
  python scripts/marin/eval_riasec_steering.py --alphas 0 1 3 5
"""

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import torch

from pvx import setup_logging
from pvx.pvx_models.judges.riasec_judge import RIASECJudge
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel
from pvx.utils.riasec_utils import RIASECHelpers

logger = setup_logging(name="marin-eval-riasec")


def get_middle_layer(model_id: str) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    return config.num_hidden_layers // 2


def main():
    parser = argparse.ArgumentParser(description="Evaluate RIASEC steering.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--traits",
        nargs="*",
        default=None,
        help="Traits to evaluate. Default: all 6.",
    )
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=[0.0, 3.0, -3.0],
        help="Alpha values to test. Default: 0 3 -3.",
    )
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default="./persona_data/model_inits/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/riasec_eval/",
    )
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    traits = args.traits or sorted(RIASECHelpers.RIASEC_TRAITS)
    layer = get_middle_layer(args.model_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    judge = RIASECJudge()
    all_results = {}

    for trait in traits:
        logger.info("=== Evaluating trait: %s ===", trait)

        # Load pre-extracted persona vector
        model = RIASECPersonaModel.load_or_create(
            target_model_id=args.model_id,
            trait=trait,
            layer=layer,
            safetensors_dir=args.vectors_dir,
        )

        trait_results = {}
        for alpha in args.alphas:
            logger.info("  alpha=%.1f", alpha)
            results, counts = judge.evaluate_persona(
                model,
                alpha=alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            trait_results[str(alpha)] = {
                "results": results,
                "counts": counts,
                "total_yes": sum(counts.values()),
            }

        all_results[trait] = trait_results
        model.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results - use per-trait files to allow parallel execution
    safe_model = args.model_id.replace("/", "__")
    trait_suffix = "_".join(sorted(traits)) if len(traits) < 6 else "all"
    output_path = output_dir / f"{safe_model}_riasec_eval_{trait_suffix}.json"
    output_data = {
        "model_id": args.model_id,
        "layer": layer,
        "alphas": args.alphas,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Also merge into combined file if it exists
    combined_path = output_dir / f"{safe_model}_riasec_eval.json"
    if combined_path.exists():
        with open(combined_path) as f:
            combined = json.load(f)
        combined["results"].update(all_results)
        combined["timestamp"] = datetime.now().isoformat()
    else:
        combined = output_data

    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    logger.info("Results saved to: %s", output_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RIASEC Steering Evaluation Summary: {args.model_id}")
    print(f"{'='*60}")
    for trait in traits:
        print(f"\n  {trait.upper()}:")
        for alpha in args.alphas:
            a_key = str(alpha)
            total = all_results[trait][a_key]["total_yes"]
            counts = all_results[trait][a_key]["counts"]
            target_count = counts.get(trait, 0)
            print(f"    alpha={alpha:+.1f}: total YES={total}, target trait YES={target_count}")


if __name__ == "__main__":
    main()
