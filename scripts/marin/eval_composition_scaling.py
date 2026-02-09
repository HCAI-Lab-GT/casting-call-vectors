#!/usr/bin/env python
"""
Composition scaling: How does discrimination accuracy scale with the number
of simultaneously composed traits?

Tests 1 through 5 component compositions at alpha=1 per component.
Maps the full scaling curve.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="composition-scaling")

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


def eval_composition(model, tokenizer, device, blocks, mid_layer,
                     residual_vectors, alpha, baseline, comp_traits):
    """Evaluate composition of comp_traits. Check each component beats each non-component."""
    comp_set = set(comp_traits)
    non_comps = [t for t in TRAITS if t not in comp_set]

    if not non_comps:
        return {"accuracy": 1.0, "correct": 0, "total": 0}  # All 6 traits = trivial

    combined = sum(alpha * residual_vectors[t] for t in comp_traits)
    vec_t = torch.tensor(combined, dtype=torch.float16).unsqueeze(0).to(device)

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

    correct = 0
    total = 0
    try:
        for comp in comp_traits:
            for non_comp in non_comps:
                if comp < non_comp:
                    pair_key = f"{comp}-{non_comp}"
                    gap = pairwise_logprob(model, tokenizer, device,
                                          TRAIT_DESCRIPTIONS[comp], TRAIT_DESCRIPTIONS[non_comp])
                    base_gap = baseline[pair_key]
                    delta = gap - base_gap
                else:
                    pair_key = f"{non_comp}-{comp}"
                    gap = pairwise_logprob(model, tokenizer, device,
                                          TRAIT_DESCRIPTIONS[non_comp], TRAIT_DESCRIPTIONS[comp])
                    base_gap = baseline[pair_key]
                    delta = base_gap - gap

                correct += int(delta > 0)
                total += 1
    finally:
        hook_handle.remove()

    return {
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
    }


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 1.0

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

    # Baseline
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device,
                                  TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    results = {"baseline": baseline, "alpha": alpha, "scaling": {}}

    print(f"\n{'='*70}")
    print(f"COMPOSITION SCALING: 1 to 5 components")
    print(f"Model: {model_id}, Alpha per component: {alpha}")
    print(f"{'='*70}")

    for k in range(1, 6):
        all_combos = list(combinations(TRAITS, k))
        n_combos = len(all_combos)
        tests_per = k * (6 - k)  # components × non-components

        total_correct = 0
        total_tests = 0
        perfect_count = 0

        combo_results = {}
        for combo in all_combos:
            r = eval_composition(model, tokenizer, device, blocks, mid_layer,
                                residual_vectors, alpha, baseline, combo)
            total_correct += r["correct"]
            total_tests += r["total"]
            if r["accuracy"] >= 1.0:
                perfect_count += 1
            combo_results["+".join(combo)] = r

        overall_acc = total_correct / total_tests if total_tests else 0

        print(f"\n  k={k}: {n_combos} combos × {tests_per} tests/combo = {total_tests} total")
        print(f"    Overall: {overall_acc:.1%} ({total_correct}/{total_tests})")
        print(f"    Perfect combos: {perfect_count}/{n_combos}")

        results["scaling"][str(k)] = {
            "n_combos": n_combos,
            "tests_per_combo": tests_per,
            "total_tests": total_tests,
            "total_correct": total_correct,
            "overall_accuracy": float(overall_acc),
            "perfect_combos": perfect_count,
            "combos": combo_results,
        }

    # Summary table
    print(f"\n{'='*70}")
    print(f"COMPOSITION SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'k':>3}  {'Combos':>7}  {'Tests':>6}  {'Accuracy':>9}  {'Perfect':>8}")
    print(f"  {'-'*38}")
    for k in range(1, 6):
        d = results["scaling"][str(k)]
        print(f"  {k:>3}  {d['n_combos']:>7}  {d['total_tests']:>6}"
              f"  {d['overall_accuracy']:>8.1%}  {d['perfect_combos']:>3}/{d['n_combos']}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"composition_scaling_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
