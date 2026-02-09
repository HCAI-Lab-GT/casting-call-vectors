#!/usr/bin/env python
"""
Calibrate steering magnitude by the average post-MLP residual norm.

Per the methodology in arXiv 2601.10387, the steering vector should be
scaled relative to the typical residual stream norm at the target layer,
so that the intervention is neither too weak nor too disruptive.

Computes:
  - Average residual stream norm at the target layer across a set of calibration prompts
  - Recommended alpha scaling factor: alpha_base = avg_norm / ||steering_vector||

Usage:
  python scripts/marin/assistant_axis/calibrate_steering.py
  python scripts/marin/assistant_axis/calibrate_steering.py --model_id marin-community/marin-8b-instruct
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="marin-calibrate-steering")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Calibration prompts: diverse set to estimate typical residual norms
CALIBRATION_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean.",
    "What are the pros and cons of renewable energy?",
    "How do you make a good cup of coffee?",
    "Describe the process of scientific peer review.",
    "What is the meaning of life?",
    "Explain quantum entanglement in simple terms.",
    "What makes a good leader?",
    "How do computers store information?",
    "What are the main causes of climate change?",
    "Describe your ideal weekend.",
    "What is the difference between empathy and sympathy?",
    "How does the stock market work?",
    "What is consciousness?",
    "Explain the theory of evolution.",
    "What are the benefits of regular exercise?",
    "How do vaccines work?",
    "What makes art valuable?",
    "Describe the water cycle.",
]


def get_middle_layer(model_id: str) -> int:
    config = AutoConfig.from_pretrained(model_id)
    return config.num_hidden_layers // 2


def compute_residual_norms(
    model, tokenizer, prompts: list[str], layer: int, device: str,
) -> list[float]:
    """Compute residual stream norms at target layer for each prompt."""
    norms = []
    for prompt in tqdm(prompts, desc="Computing residual norms"):
        messages = [
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get hidden state at target layer (layer+1 because 0 is embedding)
        hidden = outputs.hidden_states[layer + 1][0]  # (seq_len, hidden_dim)
        # Use the last token's norm (most relevant for generation)
        last_token_norm = hidden[-1].float().norm().item()
        # Also compute mean norm across all tokens
        mean_norm = hidden.float().norm(dim=-1).mean().item()

        norms.append({
            "last_token_norm": last_token_norm,
            "mean_norm": mean_norm,
        })

    return norms


def main():
    parser = argparse.ArgumentParser(description="Calibrate steering magnitude.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--pca_dir", type=str, default="./data/assistant_axis/pca/")
    parser.add_argument("--output_dir", type=str, default="./data/assistant_axis/calibration/")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    layer = args.layer or get_middle_layer(args.model_id)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    safe_model = args.model_id.replace("/", "__")

    # Load model
    logger.info("Loading model: %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.float16 if "cuda" in device else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    # Compute residual norms
    norms = compute_residual_norms(model, tokenizer, CALIBRATION_PROMPTS, layer, device)

    avg_last_norm = sum(n["last_token_norm"] for n in norms) / len(norms)
    avg_mean_norm = sum(n["mean_norm"] for n in norms) / len(norms)

    logger.info("Average last-token residual norm at layer %d: %.4f", layer, avg_last_norm)
    logger.info("Average mean residual norm at layer %d: %.4f", layer, avg_mean_norm)

    # Load PCA axis if available, compute recommended scaling
    pca_path = Path(args.pca_dir) / f"{safe_model}_pca.pt"
    axis_norm = None
    recommended_alphas = None

    if pca_path.exists():
        pca_data = torch.load(pca_path, weights_only=False)
        pc1 = pca_data["components"][0].numpy()
        axis_norm = float(torch.from_numpy(pc1).norm().item())

        # Recommended alpha: scale so that alpha * ||axis|| is a fraction of residual norm
        # Typical fractions: 0.5x (subtle), 1x (moderate), 2x (strong)
        recommended_alphas = {
            "subtle": round(0.5 * avg_last_norm / (axis_norm + 1e-10), 2),
            "moderate": round(1.0 * avg_last_norm / (axis_norm + 1e-10), 2),
            "strong": round(2.0 * avg_last_norm / (axis_norm + 1e-10), 2),
        }
        logger.info("PC1 axis norm: %.4f", axis_norm)
        logger.info("Recommended alphas: %s", recommended_alphas)

    # Save calibration data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = {
        "model_id": args.model_id,
        "layer": layer,
        "avg_last_token_residual_norm": avg_last_norm,
        "avg_mean_residual_norm": avg_mean_norm,
        "per_prompt_norms": norms,
        "axis_norm": axis_norm,
        "recommended_alphas": recommended_alphas,
    }

    output_path = output_dir / f"{safe_model}_calibration.json"
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration Summary ({args.model_id}, layer {layer}):")
    print(f"  Avg last-token residual norm: {avg_last_norm:.4f}")
    print(f"  Avg mean residual norm: {avg_mean_norm:.4f}")
    if recommended_alphas:
        print(f"  Recommended alphas: {recommended_alphas}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
