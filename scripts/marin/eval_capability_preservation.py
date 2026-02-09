#!/usr/bin/env python
"""
Capability preservation under personality steering.

Critical practical question: does steering toward a personality trait
degrade the model's general capabilities?

Tests:
1. Factual QA: Can the steered model still answer basic knowledge questions?
2. Reasoning: Can it do simple math/logic?
3. Instruction following: Does it still follow formatting requests?
4. Perplexity: What's the perplexity cost of steering on general text?

Tested at alpha = 0.5, 1.0, 2.0, 3.0 for each of the 6 RIASEC traits.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="capability")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Simple factual QA test set
FACTUAL_QA = [
    {
        "question": "What is the capital of France?",
        "answer_tokens": ["Paris"],
    },
    {
        "question": "What planet is closest to the Sun?",
        "answer_tokens": ["Mercury"],
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer_tokens": ["Shakespeare", "William"],
    },
    {
        "question": "What is the chemical symbol for water?",
        "answer_tokens": ["H2O", "H₂O"],
    },
    {
        "question": "How many continents are there?",
        "answer_tokens": ["seven", "7"],
    },
]

# Simple reasoning
REASONING = [
    {
        "question": "What is 17 + 28?",
        "answer_tokens": ["45"],
    },
    {
        "question": "If a shirt costs $20 and is 50% off, how much does it cost?",
        "answer_tokens": ["10", "$10"],
    },
    {
        "question": "What comes next in the pattern: 2, 4, 8, 16, ?",
        "answer_tokens": ["32"],
    },
]

# Perplexity test sentences (general text)
PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the old oak tree.",
    "Machine learning algorithms have transformed how we process large datasets.",
    "The restaurant served an excellent pasta with fresh tomatoes and basil.",
    "Democracy requires active participation from citizens in the political process.",
    "The sunset painted the sky in shades of orange, pink, and purple.",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_residual_vectors(model_id, riasec_dir):
    safe_model = model_id.replace("/", "__")
    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs
    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer


def compute_answer_logprob(model, tokenizer, device, question, answer_tokens):
    """Check if any expected answer token appears with high probability."""
    messages = [
        {"role": "system", "content": "Answer briefly and directly."},
        {"role": "user", "content": question},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get top-k tokens
    top_k_values, top_k_indices = torch.topk(log_probs, k=50)
    top_tokens = [tokenizer.decode([idx.item()]).strip().lower() for idx in top_k_indices]

    # Check if any answer token is in top-k
    found_in_top_k = False
    best_rank = 999
    best_logprob = -999.0
    for ans in answer_tokens:
        ans_lower = ans.lower()
        # Check exact match in top-k decoded tokens
        for rank, tok in enumerate(top_tokens):
            if ans_lower in tok or tok in ans_lower:
                found_in_top_k = True
                if rank < best_rank:
                    best_rank = rank
                    best_logprob = top_k_values[rank].item()
                break

        # Also try encoding the answer and checking its logprob directly
        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
        if len(ans_ids) > 0:
            lp = log_probs[ans_ids[0]].item()
            if lp > best_logprob:
                best_logprob = lp
                found_in_top_k = True

    return found_in_top_k, best_rank, best_logprob


def compute_perplexity(model, tokenizer, device, text):
    """Compute perplexity of text under the model."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, :-1, :]
    targets = input_ids[0, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return torch.exp(-token_log_probs.mean()).item()


def steered_qa_test(model, tokenizer, device, blocks, mid_layer, vec, alpha, qa_items):
    """Run QA test under steering."""
    if vec is not None:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
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

        hook = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))
    else:
        hook = None

    try:
        results = []
        for item in qa_items:
            found, rank, lp = compute_answer_logprob(
                model, tokenizer, device, item["question"], item["answer_tokens"])
            results.append({
                "question": item["question"],
                "found": found,
                "rank": rank,
                "logprob": lp,
            })
    finally:
        if hook:
            hook.remove()

    return results


def steered_perplexity_test(model, tokenizer, device, blocks, mid_layer, vec, alpha, texts):
    """Compute perplexity under steering."""
    if vec is not None:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
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

        hook = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))
    else:
        hook = None

    try:
        ppls = []
        for text in texts:
            ppl = compute_perplexity(model, tokenizer, device, text)
            ppls.append(ppl)
    finally:
        if hook:
            hook.remove()

    return ppls


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    alphas = [0.5, 1.0, 2.0, 3.0]

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)

    # Load model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    print(f"\n{'='*70}")
    print(f"CAPABILITY PRESERVATION UNDER PERSONALITY STEERING")
    print(f"Target: Marin 8B")
    print(f"{'='*70}")

    results = {}

    # Baseline (no steering)
    logger.info("Testing baseline (no steering)...")
    base_qa = steered_qa_test(model, tokenizer, device, blocks, mid_layer, None, 0, FACTUAL_QA)
    base_reasoning = steered_qa_test(model, tokenizer, device, blocks, mid_layer, None, 0, REASONING)
    base_ppl = steered_perplexity_test(model, tokenizer, device, blocks, mid_layer, None, 0, PERPLEXITY_TEXTS)

    base_qa_acc = sum(1 for r in base_qa if r["found"]) / len(base_qa)
    base_reason_acc = sum(1 for r in base_reasoning if r["found"]) / len(base_reasoning)
    base_mean_ppl = np.mean(base_ppl)

    print(f"\n  BASELINE (no steering):")
    print(f"    Factual QA: {base_qa_acc:.0%} ({sum(1 for r in base_qa if r['found'])}/{len(base_qa)})")
    print(f"    Reasoning: {base_reason_acc:.0%} ({sum(1 for r in base_reasoning if r['found'])}/{len(base_reasoning)})")
    print(f"    Mean perplexity: {base_mean_ppl:.1f}")

    results["baseline"] = {
        "qa_accuracy": float(base_qa_acc),
        "reasoning_accuracy": float(base_reason_acc),
        "mean_perplexity": float(base_mean_ppl),
        "per_text_ppl": [float(p) for p in base_ppl],
    }

    # Test each trait at each alpha
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*70}")

        alpha_results = {}

        for trait in TRAITS:
            logger.info(f"Testing {trait} at α={alpha}...")

            vec = residual[trait].astype(np.float32)
            qa = steered_qa_test(model, tokenizer, device, blocks, mid_layer, vec, alpha, FACTUAL_QA)
            reasoning = steered_qa_test(model, tokenizer, device, blocks, mid_layer, vec, alpha, REASONING)
            ppl = steered_perplexity_test(model, tokenizer, device, blocks, mid_layer, vec, alpha, PERPLEXITY_TEXTS)

            qa_acc = sum(1 for r in qa if r["found"]) / len(qa)
            reason_acc = sum(1 for r in reasoning if r["found"]) / len(reasoning)
            mean_ppl = np.mean(ppl)
            ppl_ratio = mean_ppl / base_mean_ppl

            print(f"  {trait:>15}: QA={qa_acc:.0%}, Reasoning={reason_acc:.0%}, "
                  f"PPL={mean_ppl:.1f} ({ppl_ratio:.2f}× baseline)")

            alpha_results[trait] = {
                "qa_accuracy": float(qa_acc),
                "reasoning_accuracy": float(reason_acc),
                "mean_perplexity": float(mean_ppl),
                "perplexity_ratio": float(ppl_ratio),
                "per_text_ppl": [float(p) for p in ppl],
            }

        results[str(alpha)] = alpha_results

        # Summary for this alpha
        mean_qa = np.mean([alpha_results[t]["qa_accuracy"] for t in TRAITS])
        mean_reason = np.mean([alpha_results[t]["reasoning_accuracy"] for t in TRAITS])
        mean_ppl_ratio = np.mean([alpha_results[t]["perplexity_ratio"] for t in TRAITS])
        print(f"\n  α={alpha} MEAN: QA={mean_qa:.0%}, Reasoning={mean_reason:.0%}, PPL ratio={mean_ppl_ratio:.2f}×")

    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'':>15}  {'Baseline':>8}", end="")
    for alpha in alphas:
        print(f"  {'α='+str(alpha):>8}", end="")
    print()

    for metric_name, metric_key in [("Factual QA", "qa_accuracy"), ("Reasoning", "reasoning_accuracy")]:
        row = f"  {metric_name:>15}  {results['baseline'][metric_key]:>7.0%}"
        for alpha in alphas:
            mean_val = np.mean([results[str(alpha)][t][metric_key] for t in TRAITS])
            row += f"  {mean_val:>7.0%}"
        print(row)

    row = f"  {'PPL ratio':>15}  {'1.00':>8}"
    for alpha in alphas:
        mean_ratio = np.mean([results[str(alpha)][t]["perplexity_ratio"] for t in TRAITS])
        row += f"  {mean_ratio:>7.2f}×"
    print(row)

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "capability_preservation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
