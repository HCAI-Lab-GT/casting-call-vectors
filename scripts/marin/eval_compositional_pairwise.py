#!/usr/bin/env python
"""
Compositional pairwise steering: Does steering with Trait_A + Trait_B
make both traits preferred over non-steered traits?

Tests on SmolLM3-3B (100% single-trait discrimination with completion prompts).
Uses α=1 per component (total magnitude ≈ 2α for each component).

For each adjacent pair on Holland hexagon (6 pairs), steers with both traits
and checks: does the model prefer each component over non-component traits?
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="compositional-pairwise")

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


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 1.0  # Per-component alpha

    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    safe_model = model_id.replace("/", "__")
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
    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Compute baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_completion(model, tokenizer, device, trait_a, trait_b)
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"COMPOSITIONAL PAIRWISE STEERING: {model_id}")
    print(f"Alpha per component: {alpha}")
    print(f"{'='*70}")

    results = {"baseline": baseline, "alpha": alpha}

    # Test all 15 pairs of traits
    all_pairs = list(combinations(TRAITS, 2))

    total_correct = 0
    total_tests = 0
    pair_results = {}

    for comp_a, comp_b in all_pairs:
        # Create compositional vector
        combined_vec = alpha * residual_vectors[comp_a] + alpha * residual_vectors[comp_b]
        vec_t = torch.tensor(combined_vec, dtype=torch.float16).unsqueeze(0).to(device)

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(vec_t))
        try:
            # Test: does comp_a beat non-components? Does comp_b beat non-components?
            comp_set = {comp_a, comp_b}
            non_components = [t for t in TRAITS if t not in comp_set]

            comp_correct = 0
            comp_total = 0
            detail = {}

            for comp in [comp_a, comp_b]:
                for non_comp in non_components:
                    # Get pair key (alphabetical order)
                    if comp < non_comp:
                        pair_key = f"{comp}-{non_comp}"
                        gap = pairwise_logprob_completion(model, tokenizer, device, comp, non_comp)
                        base_gap = baseline[pair_key]
                        delta = gap - base_gap  # Positive = steered component more preferred
                    else:
                        pair_key = f"{non_comp}-{comp}"
                        gap = pairwise_logprob_completion(model, tokenizer, device, non_comp, comp)
                        base_gap = baseline[pair_key]
                        delta = base_gap - gap  # Positive = steered component more preferred

                    correct = delta > 0
                    comp_correct += int(correct)
                    comp_total += 1
                    total_correct += int(correct)
                    total_tests += 1
                    detail[f"{comp}_vs_{non_comp}"] = {
                        "delta": float(delta),
                        "correct": correct,
                    }

            # Also test: what happens between the two components?
            if comp_a < comp_b:
                pair_key = f"{comp_a}-{comp_b}"
                gap = pairwise_logprob_completion(model, tokenizer, device, comp_a, comp_b)
                base_gap = baseline[pair_key]
                internal_delta = gap - base_gap
            else:
                pair_key = f"{comp_b}-{comp_a}"
                gap = pairwise_logprob_completion(model, tokenizer, device, comp_b, comp_a)
                base_gap = baseline[pair_key]
                internal_delta = base_gap - gap

            acc = comp_correct / comp_total
            pair_results[f"{comp_a}+{comp_b}"] = {
                "accuracy": acc,
                "correct": comp_correct,
                "total": comp_total,
                "internal_shift": float(internal_delta),
                "detail": detail,
            }

            print(f"\n  {comp_a}+{comp_b}: {acc:.0%} ({comp_correct}/{comp_total}) "
                  f"[internal shift: {internal_delta:+.3f}]")
            for key, info in sorted(detail.items()):
                mark = "✓" if info["correct"] else "✗"
                print(f"    {key:>30}: {info['delta']:>+.3f} {mark}")
        finally:
            hook_handle.remove()

    overall_acc = total_correct / total_tests
    print(f"\n{'='*70}")
    print(f"OVERALL: {overall_acc:.0%} ({total_correct}/{total_tests})")
    print(f"{'='*70}")

    results["pair_results"] = pair_results
    results["overall"] = {"accuracy": overall_acc, "correct": total_correct, "total": total_tests}

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compositional_pairwise_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
