#!/usr/bin/env python
"""
Generalization test: Does pairwise discrimination hold with NOVEL trait descriptions?

Existing test uses: "I am creative and artistic" vs "I am organized and conventional"
These descriptions overlap with the survey items used to extract vectors.

This test uses alternative descriptions to check generalization:
Set A: Original descriptions (from prior experiments)
Set B: Activity-based descriptions (focus on behaviors)
Set C: Value-based descriptions (focus on what matters)

If all three sets show high discrimination, the vectors encode genuine
personality constructs, not just the specific words from the survey.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="pairwise-generalization")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Set A: Original descriptions
DESC_ORIGINAL = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Set B: Activity-based descriptions
DESC_ACTIVITY = {
    "artistic": "someone who spends time painting, writing poetry, and attending art galleries",
    "conventional": "someone who keeps detailed records, follows schedules, and maintains orderly systems",
    "enterprising": "someone who starts businesses, persuades others, and takes on leadership roles",
    "investigative": "someone who reads research papers, solves puzzles, and conducts experiments",
    "realistic": "someone who fixes things with their hands, builds furniture, and works outdoors",
    "social": "someone who volunteers, counsels friends, and organizes community events",
}

# Set C: Value-based descriptions
DESC_VALUES = {
    "artistic": "someone who values self-expression and beauty above efficiency and order",
    "conventional": "someone who values stability, clear rules, and doing things the proper way",
    "enterprising": "someone who values influence, competition, and financial success",
    "investigative": "someone who values understanding how things work and discovering new knowledge",
    "realistic": "someone who values getting tangible results through direct physical effort",
    "social": "someone who values helping others grow and creating harmony in relationships",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def pairwise_logprob(model, tokenizer, device, desc_a, desc_b):
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


def eval_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline, descriptions):
    """Evaluate delta accuracy with a given description set."""
    correct = 0
    total = 0
    deltas = []

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
                    gap = pairwise_logprob(model, tokenizer, device,
                                          descriptions[trait_a], descriptions[trait_b])
                    base_gap = baseline[f"{trait_a}-{trait_b}"]
                    if steer_trait == trait_a:
                        d = gap - base_gap
                    else:
                        d = base_gap - gap
                    correct += int(d > 0)
                    total += 1
                    deltas.append(d)
        finally:
            hook_handle.remove()

    return {
        "delta_accuracy": correct / total if total else 0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0,
        "correct": correct,
        "total": total,
    }


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"

    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual_vectors = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    desc_sets = [
        ("Original", DESC_ORIGINAL),
        ("Activity-based", DESC_ACTIVITY),
        ("Value-based", DESC_VALUES),
    ]

    alphas = [1.0, 2.0, 3.0]

    results = {}

    print(f"\n{'='*70}")
    print(f"GENERALIZATION TEST: Novel Trait Descriptions")
    print(f"Model: {model_id}")
    print(f"{'='*70}")

    for desc_name, descriptions in desc_sets:
        print(f"\n--- {desc_name} descriptions ---")

        # Compute baseline for this description set
        baseline = {}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                gap = pairwise_logprob(model, tokenizer, device,
                                     descriptions[trait_a], descriptions[trait_b])
                baseline[f"{trait_a}-{trait_b}"] = gap

        results[f"baseline_{desc_name}"] = baseline

        print(f"  {'Alpha':>6} {'Delta%':>8} {'MeanΔ':>8}")
        print(f"  {'-'*26}")

        for alpha in alphas:
            r = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                   residual_vectors, alpha, baseline, descriptions)
            results[f"{desc_name}_alpha_{alpha}"] = r
            print(f"  {alpha:>6.1f} {r['delta_accuracy']:>7.0%} {r['mean_delta']:>+7.3f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Description Set':>16}", end="")
    for alpha in alphas:
        print(f"  α={alpha:.0f}", end="")
    print()
    print(f"  {'-'*16}", end="")
    for _ in alphas:
        print(f"  ----", end="")
    print()

    for desc_name, _ in desc_sets:
        print(f"  {desc_name:>16}", end="")
        for alpha in alphas:
            r = results.get(f"{desc_name}_alpha_{alpha}", {})
            acc = r.get("delta_accuracy", 0)
            print(f"  {acc:>3.0%}", end="")
        print()

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairwise_generalization_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
