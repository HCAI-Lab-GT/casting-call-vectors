#!/usr/bin/env python
"""
Cross-model compositional steering via zero-calibration.

Compose two personality traits in SOURCE model's standardized 5D space,
transfer the composed vector to TARGET model, and evaluate whether BOTH
traits are expressed.

This tests whether the zero-calibration transfer preserves not just
individual traits but ALGEBRAIC COMBINATIONS of traits.

For each of the 15 trait pairs:
1. Compute v_compose = v_A + v_B in source's standardized 5D
2. Transfer to target via canonical sign convention
3. Steer target model with the composed vector
4. Measure: does the steered model prefer BOTH A and B over other traits?

Success criterion: for a composed A+B vector, the steered model should
prefer A over non-A non-B traits, AND B over non-A non-B traits.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="xmodel-composition")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

SOURCE_MODEL = ("SmolLM3", "HuggingFaceTB/SmolLM3-3B")
TARGET_MODEL = ("Marin-8B", "marin-community/marin-8b-instruct")


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


def get_5d_coords_and_basis(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d


def canonical_sign_convention(coords_5d):
    signs = np.ones(5)
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1
    for pc in range(1, 5):
        loadings = {t: coords_5d[t][pc] for t in TRAITS}
        max_trait = max(loadings, key=lambda t: abs(loadings[t]))
        if loadings[max_trait] > 0:
            signs[pc] = -1
    return signs


def standardize_coords(coords_5d, basis_5d):
    signs = canonical_sign_convention(coords_5d)
    std_coords = {t: signs * coords_5d[t] for t in TRAITS}
    std_basis = np.diag(signs) @ basis_5d
    return std_coords, std_basis


def pairwise_logprob_chat(model, tokenizer, device, desc_a, desc_b):
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
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    return log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()


def eval_composition(model, tokenizer, device, blocks, mid_layer, steer_vec, alpha,
                     trait_a, trait_b, baseline):
    """Evaluate a composed A+B vector: does it prefer both A and B?

    Success: steered model prefers A over non-A-non-B traits,
             AND prefers B over non-A-non-B traits.
    """
    vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
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
        # For each component trait, check: does the composed vector
        # prefer that component over the non-component traits?
        a_correct = 0
        a_total = 0
        b_correct = 0
        b_total = 0

        others = [t for t in TRAITS if t not in (trait_a, trait_b)]

        for other in others:
            # Does it prefer A over other?
            i_a = TRAITS.index(trait_a)
            i_o = TRAITS.index(other)
            if i_a < i_o:
                key = f"{trait_a}-{other}"
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                           TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[other])
                d = gap - baseline[key]
                a_correct += int(d > 0)
            else:
                key = f"{other}-{trait_a}"
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                           TRAIT_DESCRIPTIONS[other], TRAIT_DESCRIPTIONS[trait_a])
                d = baseline[key] - gap
                a_correct += int(d > 0)
            a_total += 1

            # Does it prefer B over other?
            i_b = TRAITS.index(trait_b)
            if i_b < i_o:
                key = f"{trait_b}-{other}"
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                           TRAIT_DESCRIPTIONS[trait_b], TRAIT_DESCRIPTIONS[other])
                d = gap - baseline[key]
                b_correct += int(d > 0)
            else:
                key = f"{other}-{trait_b}"
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                           TRAIT_DESCRIPTIONS[other], TRAIT_DESCRIPTIONS[trait_b])
                d = baseline[key] - gap
                b_correct += int(d > 0)
            b_total += 1

    finally:
        hook_handle.remove()

    a_acc = a_correct / a_total if a_total else 0
    b_acc = b_correct / b_total if b_total else 0
    both_acc = (a_correct + b_correct) / (a_total + b_total)

    return a_acc, b_acc, both_acc


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    source_name, source_id = SOURCE_MODEL
    target_name, target_id = TARGET_MODEL

    # Load source vectors
    logger.info(f"Loading source ({source_name}) vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    source_coords, source_basis = get_5d_coords_and_basis(source_residual)
    source_std, source_std_basis = standardize_coords(source_coords, source_basis)

    # Load target vectors (for scaling and self-comparison)
    logger.info(f"Loading target ({target_name}) vectors...")
    target_residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    target_coords, target_basis = get_5d_coords_and_basis(target_residual)
    target_std, target_std_basis = standardize_coords(target_coords, target_basis)

    # Scale
    source_norms = np.mean([np.linalg.norm(source_std[t]) for t in TRAITS])
    target_norms = np.mean([np.linalg.norm(target_std[t]) for t in TRAITS])
    scale = target_norms / source_norms

    # Build composed vectors for all 15 pairs
    logger.info("Building composed vectors...")
    composed_vectors = {}
    for trait_a, trait_b in combinations(TRAITS, 2):
        # Compose in source's standardized 5D
        composed_5d = scale * (source_std[trait_a] + source_std[trait_b])
        # Transfer to target full-dim
        composed_full = (target_std_basis.T @ composed_5d).astype(np.float32)
        composed_vectors[(trait_a, trait_b)] = composed_full

    # Also build self-composed for comparison
    self_composed = {}
    for trait_a, trait_b in combinations(TRAITS, 2):
        composed_5d = target_std[trait_a] + target_std[trait_b]
        composed_full = (target_std_basis.T @ composed_5d).astype(np.float32)
        self_composed[(trait_a, trait_b)] = composed_full

    # Load target model
    logger.info(f"Loading {target_name}...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL COMPOSITIONAL STEERING VIA ZERO-CALIBRATION")
    print(f"Source: {source_name}, Target: {target_name}")
    print(f"{'='*70}")

    results = {}
    cross_total = 0
    cross_correct = 0
    self_total = 0
    self_correct = 0

    print(f"\n  {'Pair':>25}  {'Cross A':>7}  {'Cross B':>7}  {'Cross Both':>10}  "
          f"{'Self A':>6}  {'Self B':>6}  {'Self Both':>9}")
    print(f"  {'-'*80}")

    for trait_a, trait_b in combinations(TRAITS, 2):
        # Cross-model composition
        logger.info(f"Testing {trait_a}+{trait_b} (cross-model)...")
        c_a, c_b, c_both = eval_composition(
            model, tokenizer, device, blocks, mid_layer,
            composed_vectors[(trait_a, trait_b)], alpha, trait_a, trait_b, baseline)

        # Self composition
        logger.info(f"Testing {trait_a}+{trait_b} (self)...")
        s_a, s_b, s_both = eval_composition(
            model, tokenizer, device, blocks, mid_layer,
            self_composed[(trait_a, trait_b)], alpha, trait_a, trait_b, baseline)

        pair_name = f"{trait_a[:4]}+{trait_b[:4]}"
        print(f"  {pair_name:>25}  {c_a:>6.0%}  {c_b:>6.0%}  {c_both:>9.0%}  "
              f"{s_a:>5.0%}  {s_b:>5.0%}  {s_both:>8.0%}")

        results[f"{trait_a}+{trait_b}"] = {
            "cross": {"a_acc": float(c_a), "b_acc": float(c_b), "both_acc": float(c_both)},
            "self": {"a_acc": float(s_a), "b_acc": float(s_b), "both_acc": float(s_both)},
        }

        cross_correct += int(c_both >= 0.5) * 8  # count of correct comparisons
        cross_total += 8
        self_correct += int(s_both >= 0.5) * 8
        self_total += 8

    # Summary
    cross_pair_success = sum(1 for v in results.values() if v["cross"]["both_acc"] >= 0.5)
    self_pair_success = sum(1 for v in results.values() if v["self"]["both_acc"] >= 0.5)
    total_pairs = len(results)

    cross_mean = np.mean([v["cross"]["both_acc"] for v in results.values()])
    self_mean = np.mean([v["self"]["both_acc"] for v in results.values()])

    print(f"\n--- Summary ---")
    print(f"  Cross-model ({source_name}→{target_name}): {cross_pair_success}/{total_pairs} pairs succeed, mean={cross_mean:.0%}")
    print(f"  Self ({target_name}→{target_name}):         {self_pair_success}/{total_pairs} pairs succeed, mean={self_mean:.0%}")

    results["summary"] = {
        "cross_pair_success": cross_pair_success,
        "self_pair_success": self_pair_success,
        "total_pairs": total_pairs,
        "cross_mean_accuracy": float(cross_mean),
        "self_mean_accuracy": float(self_mean),
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_model_composition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
