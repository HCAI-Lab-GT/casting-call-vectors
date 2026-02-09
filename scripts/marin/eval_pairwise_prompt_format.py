#!/usr/bin/env python
"""
Controlled experiment: Test instruct models with COMPLETION-STYLE prompts
(instead of chat templates) in pairwise discrimination.

This disentangles whether the base model advantage comes from:
(a) Prompt format (completion vs chat template), or
(b) Model properties (no refusal behavior, lower shared fraction)

We test Marin 8B instruct and SmolLM3-3B instruct with both prompt styles.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="pairwise-prompt-format")

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


def pairwise_logprob_chat(model, tokenizer, device, trait_a, trait_b):
    """Original chat-template style evaluation."""
    desc_a = TRAIT_DESCRIPTIONS[trait_a]
    desc_b = TRAIT_DESCRIPTIONS[trait_b]

    messages = [
        {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
        {"role": "user", "content": f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

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


def pairwise_logprob_completion(model, tokenizer, device, trait_a, trait_b):
    """Completion-style evaluation (same as base model eval)."""
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


def eval_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline, eval_fn):
    """Evaluate delta accuracy for a given set of vectors and eval function."""
    correct_delta = 0
    total = 0
    deltas = []

    for steer_trait in TRAITS:
        vec = vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).to(device)
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
                    gap = eval_fn(model, tokenizer, device, trait_a, trait_b)
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
        "delta_accuracy": correct_delta / total if total > 0 else 0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0,
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

    alphas = [1.0, 2.0, 3.0, 5.0]

    print(f"\n{'='*70}")
    print(f"PROMPT FORMAT COMPARISON: {args.model_id}")
    print(f"{'='*70}")

    results = {}

    # --- Chat-template baseline ---
    logger.info("Computing chat-template baseline...")
    baseline_chat = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, args.device, trait_a, trait_b)
            baseline_chat[f"{trait_a}-{trait_b}"] = gap

    results["baseline_chat"] = baseline_chat

    # --- Completion-style baseline ---
    logger.info("Computing completion-style baseline...")
    baseline_completion = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_completion(model, tokenizer, args.device, trait_a, trait_b)
            baseline_completion[f"{trait_a}-{trait_b}"] = gap

    results["baseline_completion"] = baseline_completion

    print(f"\nBaseline comparison:")
    print(f"  {'Pair':>30} {'Chat':>8} {'Completion':>10} {'Diff':>8}")
    print(f"  {'-'*60}")
    for pair in sorted(baseline_chat.keys()):
        chat_gap = baseline_chat[pair]
        comp_gap = baseline_completion[pair]
        print(f"  {pair:>30} {chat_gap:>+8.3f} {comp_gap:>+10.3f} {comp_gap - chat_gap:>+8.3f}")

    # --- Sweep alphas with both prompt styles ---
    print(f"\n{'='*70}")
    print(f"Alpha sweep: residual vectors, chat-template vs completion")
    print(f"{'='*70}")

    print(f"\n  {'Alpha':>6} {'Chat Δ%':>8} {'Comp Δ%':>8} {'Diff':>8}")
    print(f"  {'-'*34}")

    for alpha in alphas:
        r_chat = eval_discrimination(
            model, tokenizer, args.device, blocks, mid_layer,
            residual_vectors, alpha, baseline_chat, pairwise_logprob_chat,
        )
        r_comp = eval_discrimination(
            model, tokenizer, args.device, blocks, mid_layer,
            residual_vectors, alpha, baseline_completion, pairwise_logprob_completion,
        )

        results[f"residual_chat_alpha_{alpha}"] = r_chat
        results[f"residual_completion_alpha_{alpha}"] = r_comp

        diff = r_comp["delta_accuracy"] - r_chat["delta_accuracy"]
        print(f"  {alpha:>6.1f} {r_chat['delta_accuracy']:>7.0%} {r_comp['delta_accuracy']:>7.0%} {diff:>+7.0%}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairwise_prompt_format_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
