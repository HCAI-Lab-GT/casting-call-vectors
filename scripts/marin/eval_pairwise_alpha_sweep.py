#!/usr/bin/env python
"""
Sweep alpha for pairwise discrimination to find optimal steering strength.
Also tests: does higher alpha improve specificity or just strength?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="pairwise-alpha-sweep")

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


def pairwise_logprob(model, tokenizer, device, trait_a, trait_b):
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
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    return log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()


def eval_at_alpha(model, tokenizer, device, blocks, inject_layer, vectors, alpha, baseline):
    """Evaluate pairwise discrimination at a given alpha, return delta accuracy."""
    correct_delta = 0
    total = 0
    deltas = []

    for steer_trait in TRAITS:
        vec = vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
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
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    if steer_trait not in (trait_a, trait_b):
                        continue

                    gap = pairwise_logprob(model, tokenizer, device, trait_a, trait_b)
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

    return {
        "delta_accuracy": correct_delta / total,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "correct": correct_delta,
        "total": total,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="marin-community/marin-8b-instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()
    device = args.device
    blocks = get_decoder_blocks(model)

    # Compute baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device, trait_a, trait_b)
            baseline[f"{trait_a}-{trait_b}"] = gap

    # Sweep alphas
    alphas = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    results = {"baseline": baseline, "sweeps": {}}

    print(f"\n{'='*70}")
    print(f"ALPHA SWEEP - PAIRWISE DISCRIMINATION: {args.model_id}")
    print(f"{'='*70}")

    for vec_type, vectors in [("full", full_vectors), ("residual", residual_vectors)]:
        print(f"\n--- {vec_type.upper()} vectors ---")
        print(f"  {'Alpha':>6} {'Delta%':>8} {'MeanΔ':>8} {'MedianΔ':>8}")
        print(f"  {'-'*34}")
        results["sweeps"][vec_type] = {}

        for alpha in alphas:
            logger.info("Alpha=%.1f, %s", alpha, vec_type)
            r = eval_at_alpha(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline)
            results["sweeps"][vec_type][str(alpha)] = r
            print(f"  {alpha:>6.1f} {r['delta_accuracy']:>7.0%}  {r['mean_delta']:>+7.3f}  {r['median_delta']:>+7.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairwise_alpha_sweep_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
