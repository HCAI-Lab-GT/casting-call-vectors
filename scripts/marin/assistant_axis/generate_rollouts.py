#!/usr/bin/env python
"""
Generate rollouts: for each role x system_prompt x question, generate a response.

Supports multi-GPU parallelism by splitting roles across GPUs.

Usage:
  python scripts/marin/assistant_axis/generate_rollouts.py
  python scripts/marin/assistant_axis/generate_rollouts.py --model_id marin-community/marin-8b-instruct
  python scripts/marin/assistant_axis/generate_rollouts.py --gpu_id 0 --num_gpus 4  # run on GPU 0 of 4
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="marin-generate-rollouts")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def load_model(model_id: str, device: str):
    """Load model and tokenizer onto specified device."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if "cuda" in device else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_response(
    model, tokenizer, messages: list[dict], device: str,
    max_new_tokens: int = 200, temperature: float = 0.9,
) -> str:
    """Generate a single response given chat messages."""
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (not the input)
    generated = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate rollouts for assistant axis.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--roles_path", type=str, default="./data/assistant_axis/roles.json")
    parser.add_argument("--questions_path", type=str, default="./data/assistant_axis/questions.json")
    parser.add_argument("--output_dir", type=str, default="./data/assistant_axis/rollouts/")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index for this worker.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total GPUs for parallel split.")
    parser.add_argument("--device", type=str, default=None, help="Override CUDA device (e.g. cuda:3).")
    args = parser.parse_args()

    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    # Load roles and questions
    with open(args.roles_path) as f:
        roles_data = json.load(f)
    with open(args.questions_path) as f:
        questions_data = json.load(f)

    all_roles = roles_data["roles"] + [roles_data["default_assistant"]]
    questions = [q["text"] for q in questions_data["questions"]]

    # Split roles across GPUs
    roles_per_gpu = len(all_roles) // args.num_gpus
    start_idx = args.gpu_id * roles_per_gpu
    end_idx = start_idx + roles_per_gpu if args.gpu_id < args.num_gpus - 1 else len(all_roles)
    my_roles = all_roles[start_idx:end_idx]

    logger.info(
        "GPU %d: processing roles %d-%d (%d roles, %d questions, %d prompts/role)",
        args.gpu_id, start_idx, end_idx - 1, len(my_roles),
        len(questions), len(my_roles[0]["system_prompts"]) if my_roles else 0,
    )

    # Setup output
    safe_model = args.model_id.replace("/", "__")
    output_dir = Path(args.output_dir) / safe_model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model: %s", args.model_id)
    model, tokenizer = load_model(args.model_id, device)

    # Generate rollouts
    # Filter out roles that already have output files
    filtered_roles = []
    for role in my_roles:
        role_path = output_dir / f"{role['name']}.json"
        if role_path.exists():
            logger.info("Skipping %s (already exists)", role["name"])
        else:
            filtered_roles.append(role)
    logger.info("GPU %d: %d roles to generate (%d skipped)", args.gpu_id, len(filtered_roles), len(my_roles) - len(filtered_roles))

    total = sum(len(r["system_prompts"]) * len(questions) for r in filtered_roles)
    with tqdm(total=total, desc=f"GPU {args.gpu_id} rollouts") as pbar:
        for role in filtered_roles:
            role_name = role["name"]
            role_rollouts = []

            for prompt_idx, system_prompt in enumerate(role["system_prompts"]):
                for q_idx, question in enumerate(questions):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]

                    response = generate_response(
                        model, tokenizer, messages, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )

                    role_rollouts.append({
                        "role": role_name,
                        "category": role.get("category", "unknown"),
                        "prompt_idx": prompt_idx,
                        "question_idx": q_idx,
                        "question": question,
                        "system_prompt": system_prompt,
                        "response": response,
                    })
                    pbar.update(1)

            # Save per-role to allow incremental progress
            role_path = output_dir / f"{role_name}.json"
            with open(role_path, "w") as f:
                json.dump(role_rollouts, f, indent=2)

    logger.info("GPU %d: Done. Rollouts saved to %s", args.gpu_id, output_dir)


if __name__ == "__main__":
    main()
