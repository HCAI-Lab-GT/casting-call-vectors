#!/usr/bin/env python
"""
Evaluate steering effectiveness by measuring activation projections.

Instead of binary YES/NO evaluation (which saturates on instruction-tuned models),
this script measures how much the model's hidden-state activations shift along
the persona vector direction when steered at different alpha values.

For each test prompt and alpha:
  1. Generate response with steering
  2. Forward-pass the full conversation
  3. Extract mean response-token activation at the target layer
  4. Project onto the persona vector (dot product)
  5. Compare projections across alpha values

This gives a continuous metric that is sensitive to subtle steering effects
even when generated text appears similar.

Usage:
  python scripts/marin/analysis/activation_projection_eval.py
  python scripts/marin/analysis/activation_projection_eval.py --model_id marin-community/marin-8b-instruct --device cuda:0 --traits realistic artistic
"""

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from pvx import setup_logging
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel
from pvx.utils.riasec_utils import RIASECHelpers

logger = setup_logging(name="marin-activation-projection-eval")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

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


def extract_response_activation(
    model,
    tokenizer,
    prompt: str,
    response: str,
    layer: int,
    device: str,
) -> torch.Tensor:
    """Extract mean hidden state over response tokens at the specified layer."""
    messages_prompt = [{"role": "user", "content": prompt}]
    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    prompt_tokens = tokenizer.apply_chat_template(
        messages_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    prompt_len = prompt_tokens.shape[1]

    full_tokens = tokenizer.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=full_tokens,
            attention_mask=torch.ones_like(full_tokens),
            output_hidden_states=True,
            return_dict=True,
        )

    hidden = outputs.hidden_states[layer + 1][0]  # (seq_len, hidden_dim)
    response_hidden = hidden[prompt_len:]
    if response_hidden.shape[0] == 0:
        return hidden[-1].float().cpu()
    return response_hidden.mean(dim=0).float().cpu()


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering via activation projections.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--traits", nargs="*", default=None)
    parser.add_argument("--alphas", nargs="*", type=float, default=[-5, -3, -1, 0, 1, 3, 5])
    parser.add_argument("--vectors_dir", type=str, default="./persona_data/model_inits/")
    parser.add_argument("--output_dir", type=str, default="./outputs/activation_projection_eval/")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model + inputs (e.g. cuda:2). Use 'auto' for device_map=auto.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=150)
    args = parser.parse_args()

    traits = args.traits or sorted(RIASECHelpers.RIASEC_TRAITS)
    layer = get_middle_layer(args.model_id)
    safe_model = args.model_id.replace("/", "__")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for trait in traits:
        logger.info("=== Evaluating activation projections for trait: %s ===", trait)

        # Load persona model for steering
        model = RIASECPersonaModel.load_or_create(
            target_model_id=args.model_id,
            trait=trait,
            layer=layer,
            safetensors_dir=args.vectors_dir,
            device=args.device,
        )

        # Get the persona vector for projection
        persona_vec = model.response_persona_vector.float().cpu().numpy().flatten()
        persona_norm = np.linalg.norm(persona_vec)
        persona_unit = persona_vec / (persona_norm + 1e-10)

        logger.info("Persona vector norm: %.4f", persona_norm)

        trait_results = {
            "persona_vector_norm": float(persona_norm),
            "prompts": [],
        }

        for prompt in TEST_PROMPTS:
            prompt_result = {
                "prompt": prompt,
                "alpha_results": [],
            }

            for alpha in args.alphas:
                # Generate steered response
                response = model.generate(
                    prompt=prompt,
                    alpha=alpha,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                )

                # Extract activation from the steered response
                # Use the BASE model (alpha=0) to extract clean activations
                # from the steered text — this measures how much the text itself
                # has shifted, independent of the steering hook
                activation = extract_response_activation(
                    model.model,
                    model.tokenizer,
                    prompt,
                    response,
                    layer,
                    model.device,
                )

                # Project onto persona direction
                act_np = activation.numpy().flatten()
                projection = float(np.dot(act_np, persona_unit))
                act_norm = float(np.linalg.norm(act_np))

                # Also compute cosine similarity with persona vector
                cosine_sim = float(
                    np.dot(act_np, persona_vec) / (np.linalg.norm(act_np) * persona_norm + 1e-10)
                )

                prompt_result["alpha_results"].append(
                    {
                        "alpha": alpha,
                        "projection": projection,
                        "activation_norm": act_norm,
                        "cosine_with_persona": cosine_sim,
                        "response_preview": response[:200],
                    }
                )

            # Compute delta: how much projection changes per unit alpha
            projections = [r["projection"] for r in prompt_result["alpha_results"]]
            alphas_arr = np.array(args.alphas)
            proj_arr = np.array(projections)

            if len(alphas_arr) > 1 and np.std(alphas_arr) > 0:
                slope = np.polyfit(alphas_arr, proj_arr, 1)[0]
                prompt_result["projection_slope"] = float(slope)
                prompt_result["projection_range"] = float(proj_arr.max() - proj_arr.min())
            else:
                prompt_result["projection_slope"] = 0.0
                prompt_result["projection_range"] = 0.0

            trait_results["prompts"].append(prompt_result)

        # Summary statistics for this trait
        slopes = [p["projection_slope"] for p in trait_results["prompts"]]
        ranges = [p["projection_range"] for p in trait_results["prompts"]]
        trait_results["mean_projection_slope"] = float(np.mean(slopes))
        trait_results["mean_projection_range"] = float(np.mean(ranges))
        trait_results["std_projection_slope"] = float(np.std(slopes))

        all_results[trait] = trait_results
        logger.info("  Mean projection slope: %.4f (+/- %.4f)", np.mean(slopes), np.std(slopes))
        logger.info("  Mean projection range: %.4f", np.mean(ranges))

        # Cleanup
        model.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    output_data = {
        "model_id": args.model_id,
        "layer": layer,
        "alphas": args.alphas,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }

    trait_suffix = "_".join(sorted(traits)) if len(traits) < 6 else "all"
    output_path = output_dir / f"{safe_model}_activation_projection_{trait_suffix}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Activation Projection Evaluation: {args.model_id}")
    print(f"{'=' * 60}")
    for trait in traits:
        r = all_results[trait]
        print(f"\n  {trait}:")
        print(f"    Mean projection slope:  {r['mean_projection_slope']:+.4f}")
        print(f"    Mean projection range:  {r['mean_projection_range']:.4f}")
        print(f"    Persona vector norm:    {r['persona_vector_norm']:.4f}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
