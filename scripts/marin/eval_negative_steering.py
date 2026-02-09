#!/usr/bin/env python
"""
Test negative steering: Does steering with -α suppress a trait?
If positive alpha makes model more artistic, does negative alpha make it LESS artistic?

Uses pairwise forced-choice at optimal config (completion prompt, residual vectors).
Tests SmolLM3-3B (best model: 100% at positive) to see if negative also works.

Conditions:
- Baseline (α=0)
- Positive steering (α=+2)
- Negative steering (α=-2)

For each steered trait, we check:
- Positive: shift gap in FAVOR of steered trait (delta > 0)
- Negative: shift gap AGAINST steered trait (delta < 0)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="negative-steering")

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


def pairwise_logprob_completion(model, tokenizer, device, trait_a, trait_b):
    desc_a = TRAIT_DESCRIPTIONS[trait_a]
    desc_b = TRAIT_DESCRIPTIONS[trait_b]
    prompt = f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    a_candidates = ["A", " A", "a", " a"]
    b_candidates = ["B", " B", "b", " b"]
    a_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in a_candidates if tokenizer.encode(t, add_special_tokens=False))
    b_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in b_candidates if tokenizer.encode(t, add_special_tokens=False))
    return a_lp - b_lp


def eval_at_alpha(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    """Evaluate delta accuracy. For negative alpha, we expect deltas < 0."""
    deltas = []
    pair_detail = {}

    for steer_trait in TRAITS:
        vec = vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))
        try:
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    if steer_trait not in (trait_a, trait_b):
                        continue
                    gap = pairwise_logprob_completion(model, tokenizer, device, trait_a, trait_b)
                    base_gap = baseline[f"{trait_a}-{trait_b}"]
                    if steer_trait == trait_a:
                        d = gap - base_gap  # Positive = steered trait more preferred
                    else:
                        d = base_gap - gap  # Positive = steered trait more preferred
                    deltas.append(d)
                    pair_detail[f"steer_{steer_trait}_{trait_a}-{trait_b}"] = float(d)
        finally:
            hook_handle.remove()

    # For positive alpha: correct if delta > 0
    # For negative alpha: correct if delta < 0
    if alpha > 0:
        correct = sum(1 for d in deltas if d > 0)
    else:
        correct = sum(1 for d in deltas if d < 0)

    return {
        "alpha": float(alpha),
        "delta_accuracy": correct / len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "correct": correct,
        "total": len(deltas),
        "pair_detail": pair_detail,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

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

    residual_vectors = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    # Load model
    logger.info("Loading model: %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Compute baseline using completion-style prompts
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_completion(model, tokenizer, args.device, trait_a, trait_b)
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"NEGATIVE STEERING TEST: {args.model_id}")
    print(f"{'='*70}")

    results = {"baseline": baseline, "model": args.model_id}

    # Test positive and negative alphas
    alphas = [0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0]

    print(f"\n  {'Alpha':>6} {'Direction':>10} {'Acc':>6} {'MeanΔ':>8}")
    print(f"  {'-'*34}")

    for alpha in sorted(alphas, key=lambda x: -x):
        r = eval_at_alpha(model, tokenizer, args.device, blocks, mid_layer,
                         residual_vectors, alpha, baseline)
        direction = "Enhance" if alpha > 0 else "Suppress"
        results[f"alpha_{alpha}"] = r
        print(f"  {alpha:>+6.1f} {direction:>10} {r['delta_accuracy']:>5.0%} {r['mean_delta']:>+7.3f}")

    # Check linearity: positive and negative deltas should be antisymmetric
    print(f"\n{'='*70}")
    print(f"LINEARITY CHECK: α=+2 vs α=-2")
    print(f"{'='*70}")

    pos = results.get("alpha_2.0", {}).get("pair_detail", {})
    neg = results.get("alpha_-2.0", {}).get("pair_detail", {})

    if pos and neg:
        pos_vals = []
        neg_vals = []
        for key in sorted(pos.keys()):
            if key in neg:
                p = pos[key]
                n = neg[key]
                pos_vals.append(p)
                neg_vals.append(n)

        correlation = np.corrcoef(pos_vals, neg_vals)[0, 1]
        # Perfect antisymmetry: pos_vals = -neg_vals, so correlation should be -1
        ratio = np.mean([-n / p if abs(p) > 0.001 else 0 for p, n in zip(pos_vals, neg_vals) if abs(p) > 0.001])
        print(f"  Correlation(+2, -2): r = {correlation:.3f} (expected: -1.000)")
        print(f"  Mean ratio(-2/+2):   {ratio:.3f} (expected: +1.000 for perfect linearity)")
        results["linearity"] = {
            "correlation": float(correlation),
            "ratio": float(ratio),
        }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"negative_steering_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
