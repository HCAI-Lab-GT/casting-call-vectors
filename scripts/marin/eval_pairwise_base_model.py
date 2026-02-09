#!/usr/bin/env python
"""
Pairwise forced-choice evaluation adapted for BASE models (no chat template).
Uses completion-style prompts instead of chat templates.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="pairwise-base")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

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


def pairwise_logprob_base(model, tokenizer, device, trait_a, trait_b):
    """Compare logprob of A vs B for base model using completion-style prompt."""
    desc_a = TRAIT_DESCRIPTIONS[trait_a]
    desc_b = TRAIT_DESCRIPTIONS[trait_b]

    prompt = (
        f"Which describes you better?\n"
        f"A) I am {desc_a}\n"
        f"B) I am {desc_b}\n"
        f"Answer:"
    )

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Try multiple token variants for A and B
    a_candidates = ["A", " A", "a", " a"]
    b_candidates = ["B", " B", "b", " b"]

    a_logprob = -float("inf")
    b_logprob = -float("inf")

    for text in a_candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            a_logprob = max(a_logprob, log_probs[ids[0]].item())

    for text in b_candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            b_logprob = max(b_logprob, log_probs[ids[0]].item())

    return a_logprob - b_logprob


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="marin-community/marin-32b-base")
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    logger.info("Model: %s, mid=%d", args.model_id, mid_layer)

    safe_model = args.model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load vectors
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Compute residual vectors
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
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Get device of mid-layer block
    for param in blocks[mid_layer].parameters():
        block_device = param.device
        break

    # Baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_base(model, tokenizer, block_device, trait_a, trait_b)
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"PAIRWISE BASE MODEL: {args.model_id}")
    print(f"{'='*70}")

    print(f"\nBaseline (no steering):")
    for pair, gap in sorted(baseline.items()):
        t1, t2 = pair.split("-")
        winner = t1 if gap > 0 else t2
        print(f"  {pair:>25}: {gap:+.3f} → prefers {winner}")

    all_results = {"baseline": baseline}

    # Sweep alphas for residual vectors
    alphas = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    print(f"\n--- Alpha sweep (residual vectors) ---")
    print(f"  {'Alpha':>6} {'Delta%':>8} {'MeanΔ':>8}")
    print(f"  {'-'*26}")

    best_alpha = None
    best_acc = 0

    for alpha in alphas:
        correct_delta = 0
        total = 0
        deltas = []

        for steer_trait in TRAITS:
            vec = residual_vectors[steer_trait]
            vec_t = torch.tensor(vec, dtype=torch.float16).to(block_device)
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

            hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta))
            try:
                for i, trait_a in enumerate(TRAITS):
                    for j, trait_b in enumerate(TRAITS):
                        if i >= j:
                            continue
                        if steer_trait not in (trait_a, trait_b):
                            continue
                        gap = pairwise_logprob_base(model, tokenizer, block_device, trait_a, trait_b)
                        base_gap = baseline[f"{trait_a}-{trait_b}"]
                        if steer_trait == trait_a:
                            d = gap - base_gap
                        else:
                            d = base_gap - gap
                        correct_delta += int(d > 0)
                        total += 1
                        deltas.append(d)
            finally:
                hook_handle.remove()

        acc = correct_delta / total
        print(f"  {alpha:>6.1f} {acc:>7.0%}  {np.mean(deltas):>+7.3f}")

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

        all_results[f"residual_alpha_{alpha}"] = {
            "delta_accuracy": acc,
            "mean_delta": float(np.mean(deltas)),
            "correct": correct_delta,
            "total": total,
        }

    # Also do full vectors at best alpha
    print(f"\n--- Full vectors at alpha={best_alpha} ---")
    correct_delta = 0
    total = 0
    deltas = []

    for steer_trait in TRAITS:
        vec = full_vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).to(block_device)
        delta = best_alpha * vec_t

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta))
        try:
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    if steer_trait not in (trait_a, trait_b):
                        continue
                    gap = pairwise_logprob_base(model, tokenizer, block_device, trait_a, trait_b)
                    base_gap = baseline[f"{trait_a}-{trait_b}"]
                    if steer_trait == trait_a:
                        d = gap - base_gap
                    else:
                        d = base_gap - gap
                    correct_delta += int(d > 0)
                    total += 1
                    deltas.append(d)
        finally:
            hook_handle.remove()

    acc = correct_delta / total
    print(f"  Delta: {acc:.0%}, MeanΔ: {np.mean(deltas):+.3f}")
    all_results[f"full_alpha_{best_alpha}"] = {
        "delta_accuracy": acc,
        "mean_delta": float(np.mean(deltas)),
    }

    print(f"\n  Best residual alpha: {best_alpha} ({best_acc:.0%})")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairwise_base_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
