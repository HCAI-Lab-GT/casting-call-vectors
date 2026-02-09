#!/usr/bin/env python
"""
Extract mean residual stream activations for each role from rollout data.

For each rollout:
  - Forward pass through model
  - Extract mean activation over response tokens at the middle layer
  - Average per-role across all rollouts to get one vector per role

Output: role_vectors tensor of shape (n_roles, hidden_dim)

Usage:
  python scripts/marin/assistant_axis/extract_activations.py
  python scripts/marin/assistant_axis/extract_activations.py --model_id marin-community/marin-8b-instruct
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="marin-extract-activations")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def get_middle_layer(model_id: str) -> int:
    config = AutoConfig.from_pretrained(model_id)
    return config.num_hidden_layers // 2


def extract_response_activation(
    model, tokenizer, system_prompt: str, question: str, response: str,
    layer: int, device: str,
) -> torch.Tensor:
    """
    Extract the mean hidden state over response tokens at the specified layer.

    Returns a 1D tensor of shape (hidden_dim,).
    """
    # Build the full conversation as it was generated
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]

    # Get prompt length (everything before the assistant response)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    prompt_len = prompt_tokens.shape[1]

    # Tokenize full conversation
    full_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(full_tokens)

    with torch.no_grad():
        outputs = model(
            input_ids=full_tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract hidden states at specified layer (layer+1 because index 0 is embedding)
    hidden = outputs.hidden_states[layer + 1][0]  # (seq_len, hidden_dim)

    # Mean over response tokens only
    response_hidden = hidden[prompt_len:]
    if response_hidden.shape[0] == 0:
        # Fallback: use last token if no response tokens
        return hidden[-1].float()

    return response_hidden.mean(dim=0).float()


def main():
    parser = argparse.ArgumentParser(description="Extract role activation vectors.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--rollouts_dir", type=str, default="./data/assistant_axis/rollouts/")
    parser.add_argument("--output_dir", type=str, default="./data/assistant_axis/role_vectors/")
    parser.add_argument("--layer", type=int, default=None, help="Layer to extract from. Default: middle.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Determine layer
    if args.layer is None:
        layer = get_middle_layer(args.model_id)
    else:
        layer = args.layer
    logger.info("Extracting activations at layer %d", layer)

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load model
    logger.info("Loading model: %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.float16 if "cuda" in device else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    # Find rollout files
    safe_model = args.model_id.replace("/", "__")
    rollouts_dir = Path(args.rollouts_dir) / safe_model
    if not rollouts_dir.exists():
        logger.error("Rollouts directory not found: %s", rollouts_dir)
        raise FileNotFoundError(f"Run generate_rollouts.py first. Missing: {rollouts_dir}")

    rollout_files = sorted(rollouts_dir.glob("*.json"))
    logger.info("Found %d role rollout files", len(rollout_files))

    # Extract activations per role
    role_vectors = {}
    role_names = []
    role_categories = {}

    for rollout_file in tqdm(rollout_files, desc="Processing roles"):
        with open(rollout_file) as f:
            rollouts = json.load(f)

        if not rollouts:
            continue

        role_name = rollouts[0]["role"]
        category = rollouts[0].get("category", "unknown")

        # Accumulate activations for this role
        accum = None
        count = 0

        for rollout in tqdm(rollouts, desc=f"  {role_name}", leave=False):
            activation = extract_response_activation(
                model, tokenizer,
                system_prompt=rollout["system_prompt"],
                question=rollout["question"],
                response=rollout["response"],
                layer=layer,
                device=device,
            ).cpu()

            if accum is None:
                accum = torch.zeros_like(activation)
            accum += activation
            count += 1

        if count > 0:
            role_vectors[role_name] = accum / count
            role_names.append(role_name)
            role_categories[role_name] = category

    # Stack into tensor
    vectors_tensor = torch.stack([role_vectors[name] for name in role_names])
    logger.info("Role vectors shape: %s", tuple(vectors_tensor.shape))

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_model}.pt"
    torch.save({
        "vectors": vectors_tensor,
        "role_names": role_names,
        "role_categories": role_categories,
        "model_id": args.model_id,
        "layer": layer,
    }, output_path)

    logger.info("Saved role vectors to: %s", output_path)
    print(f"Extracted {len(role_names)} role vectors of dim {vectors_tensor.shape[1]}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
