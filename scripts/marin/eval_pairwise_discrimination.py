#!/usr/bin/env python
"""
Pairwise forced-choice evaluation of RIASEC persona vectors.

Instead of asking "Do you like X?" (YES/NO), which the shared "agree"
direction overwhelms, this test asks "Are you more X or Y?" for each
of the 15 trait pairs. The shared direction cancels out in this
comparison, so only the trait-specific residual content matters.

This should directly capture the geometric specificity that the
single-trait logprob evaluation misses.
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="pairwise-discrimination")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Short trait descriptions for prompts
TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def pairwise_logprob(model, tokenizer, device, trait_a, trait_b):
    """Compare logprob of trait_a vs trait_b descriptions."""
    desc_a = TRAIT_DESCRIPTIONS[trait_a]
    desc_b = TRAIT_DESCRIPTIONS[trait_b]

    messages = [
        {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
        {"role": "user", "content": (
            f"Which describes you better?\n"
            f"A) I am {desc_a}\n"
            f"B) I am {desc_b}\n"
            f"Answer:"
        )},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get logprobs for A and B
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)

    a_logprob = log_probs[a_ids[0]].item()
    b_logprob = log_probs[b_ids[0]].item()

    return a_logprob - b_logprob  # Positive means prefers trait_a


def eval_pairwise_matrix(model, tokenizer, device, blocks, inject_layer, vector, alpha):
    """Evaluate all 15 pairs under steering with the given vector."""
    vec_t = torch.tensor(vector, dtype=torch.float16).unsqueeze(0).to(device)
    delta = alpha * vec_t

    def make_hook(d):
        def hook_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += d
                return (hs,) + out[1:]
            out[:, -1, :] += d
            return out
        return hook_fn

    hook_handle = blocks[inject_layer].register_forward_hook(make_hook(delta))
    try:
        results = {}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                gap = pairwise_logprob(model, tokenizer, device, trait_a, trait_b)
                results[f"{trait_a}-{trait_b}"] = gap
    finally:
        hook_handle.remove()

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    logger.info("Model: %s, mid=%d", args.model_id, mid_layer)

    safe_model = args.model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load all-layers vectors
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Compute residual vectors at mid layer
    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    full_vectors = {t: all_layer_vectors[t][mid_layer + 1] for t in TRAITS}
    residual_vectors = {}
    for t in TRAITS:
        vec = full_vectors[t]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()
    device = args.device
    blocks = get_decoder_blocks(model)

    # Baseline: no steering
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device, trait_a, trait_b)
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"PAIRWISE FORCED-CHOICE EVALUATION: {args.model_id}")
    print(f"{'='*70}")

    print(f"\nBaseline (no steering):")
    for pair, gap in sorted(baseline.items()):
        t1, t2 = pair.split("-")
        winner = t1 if gap > 0 else t2
        print(f"  {pair:>25}: {gap:+.3f} → prefers {winner}")

    # Evaluate with full and residual vectors
    all_results = {"baseline": baseline}

    for vec_type, vectors in [("full", full_vectors), ("residual", residual_vectors)]:
        logger.info("Evaluating %s vectors...", vec_type)
        steer_results = {}

        print(f"\n--- Steering with {vec_type} vectors (alpha={args.alpha}) ---")

        correct = 0
        total = 0

        for steer_trait in TRAITS:
            pairs = eval_pairwise_matrix(
                model, tokenizer, device, blocks, mid_layer,
                vectors[steer_trait], args.alpha
            )
            steer_results[steer_trait] = pairs

            # For each pair involving the steered trait, check if it wins
            trait_correct = 0
            trait_total = 0
            print(f"\n  Steer: {steer_trait}")
            for pair, gap in sorted(pairs.items()):
                t1, t2 = pair.split("-")
                if steer_trait not in (t1, t2):
                    continue

                # The steered trait should be preferred
                if steer_trait == t1:
                    is_correct = gap > 0
                else:
                    is_correct = gap < 0

                marker = "✓" if is_correct else "✗"
                winner = t1 if gap > 0 else t2
                print(f"    {pair:>25}: {gap:+.3f} → {winner} {marker}")

                trait_correct += int(is_correct)
                trait_total += 1
                correct += int(is_correct)
                total += 1

            print(f"    Score: {trait_correct}/{trait_total}")

        all_results[vec_type] = steer_results
        print(f"\n  Overall: {correct}/{total} correct ({correct/max(total,1)*100:.0f}%)")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for vec_type in ["full", "residual"]:
        correct = 0
        total = 0
        for steer_trait in TRAITS:
            for pair, gap in all_results[vec_type][steer_trait].items():
                t1, t2 = pair.split("-")
                if steer_trait not in (t1, t2):
                    continue
                if steer_trait == t1:
                    is_correct = gap > 0
                else:
                    is_correct = gap < 0
                correct += int(is_correct)
                total += 1

        print(f"  {vec_type:>10}: {correct}/{total} ({correct/max(total,1)*100:.0f}%)")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairwise_discrimination_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
