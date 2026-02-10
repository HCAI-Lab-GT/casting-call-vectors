#!/usr/bin/env python
"""
Qualitative steering evaluation: generate open-ended responses under
different steering conditions and save for analysis.

For each RIASEC trait's residual vector, generate responses to neutral
questions and save both the raw responses and metadata. These can then
be evaluated by an LLM judge for trait-specific content.

Usage:
  uv run python scripts/marin/eval_qualitative_steering.py --model_id meta-llama/Llama-3.2-1B-Instruct --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig

from pvx import setup_logging
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel

logger = setup_logging(name="qualitative-steering")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Neutral, open-ended questions that could elicit personality differences
QUESTIONS = [
    "What do you think is the most interesting thing about the world?",
    "If you had a free weekend, what would you do?",
    "What matters most to you in life?",
    "Describe your ideal work environment.",
    "What kind of problems do you most enjoy solving?",
    "How do you approach making difficult decisions?",
    "What would you teach a class about?",
    "What's your reaction when something goes wrong?",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _detect_middle_layer(model_id: str) -> int:
    return int(AutoConfig.from_pretrained(model_id).num_hidden_layers) // 2


def load_raw_vectors(model_id: str) -> dict[str, np.ndarray]:
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"
    vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            vectors[trait] = f.get_tensor("response_persona_vector").numpy().flatten()
    return vectors


def generate_response(model, tokenizer, messages, device, max_new_tokens=150):
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model + inputs (e.g. cuda:2). Use 'auto' for device_map=auto.",
    )
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default="outputs/qualitative")
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    root = _repo_root()
    layer = args.layer if args.layer is not None else _detect_middle_layer(args.model_id)

    # Compute residual vectors
    raw_vectors = load_raw_vectors(args.model_id)
    V = np.stack([raw_vectors[t] for t in TRAITS])
    mean_vec = V.mean(axis=0)
    mean_unit = mean_vec / np.linalg.norm(mean_vec)

    residuals = {}
    for trait in TRAITS:
        proj = np.dot(raw_vectors[trait], mean_unit) * mean_unit
        residuals[trait] = raw_vectors[trait] - proj

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_id": args.model_id,
        "layer": layer,
        "alpha": args.alpha,
        "questions": QUESTIONS,
        "conditions": {},
    }

    # Conditions: baseline (alpha=0), each original trait, each residual
    conditions = [("baseline", None, 0.0)]
    for trait in TRAITS:
        conditions.append((f"original_{trait}", raw_vectors[trait], args.alpha))
        conditions.append((f"residual_{trait}", residuals[trait], args.alpha))
    conditions.append(("shared_only", mean_vec, args.alpha))

    # We only need to load the model once (use the first trait's model as the base)
    model = RIASECPersonaModel.load_or_create(
        target_model_id=args.model_id, trait=TRAITS[0], layer=layer, device=args.device
    )
    model.model.eval()

    for cond_name, vector, alpha in conditions:
        logger.info("Condition: %s (alpha=%.1f)", cond_name, alpha)

        if vector is not None:
            model.response_persona_vector = torch.tensor(
                vector, dtype=model.response_persona_vector.dtype
            )
            model._persona_base = None

        cond_responses = []
        for q_idx, question in enumerate(QUESTIONS):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the question thoughtfully.",
                },
                {"role": "user", "content": question},
            ]

            with model._steering_delta(alpha):
                response = generate_response(model, model.tokenizer, messages, model.device)

            cond_responses.append(
                {
                    "question_idx": q_idx,
                    "question": question,
                    "response": response,
                }
            )
            logger.info("  Q%d: %s...", q_idx, response[:80])

        results["conditions"][cond_name] = cond_responses

    safe_model = args.model_id.replace("/", "__")
    out_path = output_dir / f"{safe_model}_qualitative_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_path)

    model.close()
    del model
    gc.collect()

    # Print sample comparison
    print(f"\n{'=' * 70}")
    print(f"SAMPLE RESPONSES (Q0: '{QUESTIONS[0]}')")
    print(f"{'=' * 70}")
    for cond_name in [
        "baseline",
        "residual_artistic",
        "residual_conventional",
        "residual_investigative",
    ]:
        if cond_name in results["conditions"]:
            resp = results["conditions"][cond_name][0]["response"]
            print(f"\n--- {cond_name} ---")
            print(resp[:300])


if __name__ == "__main__":
    main()
