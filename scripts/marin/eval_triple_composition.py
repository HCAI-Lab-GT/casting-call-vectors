#!/usr/bin/env python
"""
Triple-trait compositional steering.

If dual-trait works at 96% (α=1 per component), does triple-trait work?
Steers with 3 trait vectors simultaneously (α=1 each), checks if all 3
components are preferred over the 3 non-components.

Tests all 20 possible 3-trait combinations from 6 RIASEC traits.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="triple-composition")

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


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 1.0  # Per component

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

    # Compute baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device,
                                  TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"TRIPLE-TRAIT COMPOSITIONAL STEERING")
    print(f"Model: {model_id}, Alpha per component: {alpha}")
    print(f"{'='*70}")

    all_triples = list(combinations(TRAITS, 3))
    results = {"baseline": baseline, "alpha": alpha, "triples": {}}

    total_correct = 0
    total_tests = 0

    for triple in all_triples:
        comp_set = set(triple)
        non_comps = [t for t in TRAITS if t not in comp_set]

        # Create combined vector
        combined = sum(alpha * residual_vectors[t] for t in triple)
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
        tests = 0
        detail = {}

        try:
            # Test: does each component beat each non-component?
            for comp in triple:
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

                    is_correct = delta > 0
                    correct += int(is_correct)
                    tests += 1
                    total_correct += int(is_correct)
                    total_tests += 1
                    detail[f"{comp}_vs_{non_comp}"] = {
                        "delta": float(delta),
                        "correct": is_correct,
                    }
        finally:
            hook_handle.remove()

        acc = correct / tests
        key = "+".join(triple)
        results["triples"][key] = {
            "accuracy": float(acc),
            "correct": correct,
            "total": tests,
            "detail": detail,
        }

        mark = "***" if acc == 1.0 else ""
        print(f"  {key:>35}: {acc:.0%} ({correct}/{tests}) {mark}")

    overall = total_correct / total_tests

    print(f"\n{'='*70}")
    print(f"OVERALL: {overall:.0%} ({total_correct}/{total_tests})")
    print(f"{'='*70}")

    # Compare with dual-trait
    perfect_triples = sum(1 for d in results["triples"].values() if d["accuracy"] >= 1.0)
    print(f"  Perfect triples (100%): {perfect_triples}/{len(all_triples)}")

    # Per-component accuracy
    print(f"\n--- Per-trait accuracy as component ---")
    for trait in TRAITS:
        trait_correct = 0
        trait_total = 0
        for key, data in results["triples"].items():
            if trait in key.split("+"):
                for det_key, det_val in data["detail"].items():
                    if det_key.startswith(f"{trait}_vs_"):
                        trait_correct += int(det_val["correct"])
                        trait_total += 1
        if trait_total > 0:
            print(f"  {trait:>14}: {trait_correct}/{trait_total} ({trait_correct/trait_total:.0%})")

    results["overall"] = {"accuracy": float(overall), "correct": total_correct, "total": total_tests}

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"triple_composition_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
