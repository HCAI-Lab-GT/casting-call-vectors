#!/usr/bin/env python
"""
Compare steering approaches: unsteered vs RIASEC-steered vs axis-steered responses.

Runs a fixed set of test prompts and generates responses under different steering
conditions for qualitative comparison and judge scoring.

Usage:
  python scripts/marin/analysis/steering_comparison.py
  python scripts/marin/analysis/steering_comparison.py --model_id marin-community/marin-8b-instruct
  python scripts/marin/analysis/steering_comparison.py --traits realistic artistic
"""

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from pvx import setup_logging
from pvx.pvx_models.judges.riasec_judge import RIASECJudge
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel
from pvx.utils.riasec_utils import RIASECHelpers

logger = setup_logging(name="marin-steering-comparison")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Fixed test prompts for consistent comparison
TEST_PROMPTS = [
    "What do you enjoy doing in your free time?",
    "How would you approach solving a complex problem at work?",
    "What qualities do you look for in a friend?",
    "Describe your ideal work environment.",
    "What motivates you to get up in the morning?",
    "How do you handle disagreements with colleagues?",
    "What would you do with an unexpected day off?",
    "What accomplishment are you most proud of?",
]


def get_middle_layer(model_id: str) -> int:
    config = AutoConfig.from_pretrained(model_id)
    return config.num_hidden_layers // 2


def main():
    parser = argparse.ArgumentParser(description="Compare steering approaches.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--traits",
        nargs="*",
        default=None,
        help="RIASEC traits to test. Default: all 6.",
    )
    parser.add_argument("--alpha", type=float, default=3.0, help="Steering alpha.")
    parser.add_argument("--vectors_dir", type=str, default="./persona_data/model_inits/")
    parser.add_argument("--pca_dir", type=str, default="./data/assistant_axis/pca/")
    parser.add_argument("--output_dir", type=str, default="./outputs/steering_comparison/")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    traits = args.traits or sorted(RIASECHelpers.RIASEC_TRAITS)
    layer = get_middle_layer(args.model_id)
    safe_model = args.model_id.replace("/", "__")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA axis vector if available
    pca_path = Path(args.pca_dir) / f"{safe_model}_pca.pt"
    pca_axis = None
    if pca_path.exists():
        pca_data = torch.load(pca_path, weights_only=False)
        pca_axis = pca_data["components"][0]  # PC1 = assistant axis
        logger.info("Loaded PCA axis (PC1) for axis steering.")
    else:
        logger.warning("PCA data not found, skipping axis-steered comparisons.")

    all_results = {}

    for trait in traits:
        logger.info("=== Comparing steering for trait: %s ===", trait)

        # Load RIASEC model
        model = RIASECPersonaModel.load_or_create(
            target_model_id=args.model_id,
            trait=trait,
            layer=layer,
            safetensors_dir=args.vectors_dir,
        )

        trait_results = []

        for prompt in TEST_PROMPTS:
            result = {"prompt": prompt}

            # 1. Unsteered (alpha=0)
            response_unsteered = model.generate(
                prompt=prompt,
                alpha=0,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            result["unsteered"] = response_unsteered

            # 2. RIASEC positive steering
            response_positive = model.generate(
                prompt=prompt,
                alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            result["riasec_positive"] = response_positive

            # 3. RIASEC negative steering
            response_negative = model.generate(
                prompt=prompt,
                alpha=-args.alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            result["riasec_negative"] = response_negative

            # 4. Axis steering (if available) -- steer along PC1
            if pca_axis is not None:
                # Temporarily swap the persona vector to use PCA axis
                original_response_vec = model.response_persona_vector.clone()
                model.response_persona_vector = pca_axis.unsqueeze(0).float()
                model._persona_base = None  # invalidate cache

                response_axis_pos = model.generate(
                    prompt=prompt,
                    alpha=args.alpha,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                result["axis_positive"] = response_axis_pos

                response_axis_neg = model.generate(
                    prompt=prompt,
                    alpha=-args.alpha,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                result["axis_negative"] = response_axis_neg

                # Restore original
                model.response_persona_vector = original_response_vec
                model._persona_base = None

            trait_results.append(result)

        all_results[trait] = trait_results
        model.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results - use per-trait files to allow parallel execution
    output_data = {
        "model_id": args.model_id,
        "alpha": args.alpha,
        "layer": layer,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    trait_suffix = "_".join(sorted(traits)) if len(traits) < 6 else "all"
    output_path = output_dir / f"{safe_model}_steering_comparison_{trait_suffix}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Merge into combined file
    combined_path = output_dir / f"{safe_model}_steering_comparison.json"
    if combined_path.exists():
        with open(combined_path) as f:
            combined = json.load(f)
        combined["results"].update(all_results)
        combined["timestamp"] = datetime.now().isoformat()
    else:
        combined = output_data
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Print sample comparison
    print(f"\n{'='*60}")
    print(f"Steering Comparison: {args.model_id}")
    print(f"{'='*60}")

    sample_trait = traits[0]
    sample = all_results[sample_trait][0]
    print(f"\nTrait: {sample_trait}")
    print(f"Prompt: {sample['prompt']}")
    print(f"\n--- Unsteered ---")
    print(sample["unsteered"][:300])
    print(f"\n--- RIASEC +{args.alpha} ---")
    print(sample["riasec_positive"][:300])
    print(f"\n--- RIASEC -{args.alpha} ---")
    print(sample["riasec_negative"][:300])
    if "axis_positive" in sample:
        print(f"\n--- Axis +{args.alpha} ---")
        print(sample["axis_positive"][:300])

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
