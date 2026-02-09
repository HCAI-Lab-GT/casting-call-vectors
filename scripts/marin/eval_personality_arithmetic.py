#!/usr/bin/env python
"""
Personality vector arithmetic: Do 5D operations produce predictable behavioral effects?

Tests:
1. Subtraction: trait_A - trait_B → should boost A and suppress B
2. Averaging: mean(A,B,C) → should boost all three equally
3. Holland opposite cancellation: A + opposite(A) → should be near-zero
4. Double subtraction: A - B - C → should boost A, suppress both B and C
5. Differential: (A - centroid) → should isolate A's unique signature

This tests the ALGEBRAIC structure of the 5D personality subspace.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="arithmetic")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Holland hexagonal adjacencies (circular order: R-I-A-S-E-C)
HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
HOLLAND_OPPOSITES = {
    "realistic": "social",
    "investigative": "enterprising",
    "artistic": "conventional",
    "social": "realistic",
    "enterprising": "investigative",
    "conventional": "artistic",
}


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


def reconstruct_from_5d(coords_5d, basis_5d):
    return (basis_5d.T @ coords_5d).astype(np.float32)


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


def measure_trait_profile(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline):
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

    hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))
    try:
        trait_deltas = {t: 0.0 for t in TRAITS}
        trait_counts = {t: 0 for t in TRAITS}

        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                           TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                base_gap = baseline[f"{trait_a}-{trait_b}"]
                shift = gap - base_gap
                trait_deltas[trait_a] += shift
                trait_counts[trait_a] += 1
                trait_deltas[trait_b] -= shift
                trait_counts[trait_b] += 1

        for t in TRAITS:
            if trait_counts[t] > 0:
                trait_deltas[t] /= trait_counts[t]
    finally:
        hook_handle.remove()

    return trait_deltas


def generate_steered(model, tokenizer, device, blocks, mid_layer, steer_vec, alpha,
                     prompt, max_new_tokens=80):
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
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        hook_handle.remove()

    return generated.strip()


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    coords_5d, basis_5d = get_5d_coords_and_basis(residual)

    # Load model
    logger.info("Loading Marin 8B...")
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

    gen_prompt = "In my free time, I love to"

    # Centroid of all traits
    centroid = np.mean([coords_5d[t] for t in TRAITS], axis=0)

    print(f"\n{'='*70}")
    print(f"PERSONALITY VECTOR ARITHMETIC IN 5D")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    results = {}

    # ===== TEST 1: Subtraction (A - B) =====
    print(f"\n{'='*70}")
    print(f"TEST 1: SUBTRACTION (A - B)")
    print(f"{'='*70}")

    subtraction_tests = [
        ("artistic", "conventional"),   # Holland opposites
        ("investigative", "enterprising"),  # Holland opposites
        ("realistic", "social"),        # Holland opposites
        ("artistic", "realistic"),      # Holland alternates
        ("social", "conventional"),     # Holland alternates
    ]

    sub_results = {}
    for trait_a, trait_b in subtraction_tests:
        logger.info(f"Testing {trait_a} - {trait_b}...")
        diff_coords = coords_5d[trait_a] - coords_5d[trait_b]
        diff_vec = reconstruct_from_5d(diff_coords, basis_5d)

        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, diff_vec, alpha, baseline)
        gen = generate_steered(
            model, tokenizer, device, blocks, mid_layer, diff_vec, alpha * 3, gen_prompt)

        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        a_delta = profile[trait_a]
        b_delta = profile[trait_b]

        print(f"\n  {trait_a} - {trait_b}:")
        print(f"    {trait_a} delta: {a_delta:+.3f} (should be positive)")
        print(f"    {trait_b} delta: {b_delta:+.3f} (should be negative)")
        print(f"    Correct direction: {'YES' if a_delta > 0 and b_delta < 0 else 'NO'}")
        print(f"    Profile: " + "  ".join(f"{t[:4]}={d:+.2f}" for t, d in sorted_prof))
        print(f"    Generated: {gen[:120]}...")

        sub_results[f"{trait_a}-{trait_b}"] = {
            "a_delta": float(a_delta),
            "b_delta": float(b_delta),
            "correct_direction": bool(a_delta > 0 and b_delta < 0),
            "profile": {t: float(d) for t, d in profile.items()},
            "generation": gen,
            "vec_norm": float(np.linalg.norm(diff_vec)),
        }

    results["subtraction"] = sub_results

    # ===== TEST 2: Holland Opposite Cancellation =====
    print(f"\n{'='*70}")
    print(f"TEST 2: HOLLAND OPPOSITE CANCELLATION (A + opposite(A))")
    print(f"{'='*70}")

    cancel_results = {}
    for trait in ["artistic", "investigative", "realistic"]:
        opp = HOLLAND_OPPOSITES[trait]
        logger.info(f"Testing {trait} + {opp} (Holland opposites)...")
        sum_coords = coords_5d[trait] + coords_5d[opp]
        sum_vec = reconstruct_from_5d(sum_coords, basis_5d)

        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, sum_vec, alpha, baseline)

        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        max_abs_delta = max(abs(d) for d in profile.values())

        print(f"\n  {trait} + {opp} (opposites):")
        print(f"    Max |delta|: {max_abs_delta:.3f} (should be small)")
        print(f"    Sum norm: {np.linalg.norm(sum_vec):.1f} (vs individual ~50)")
        print(f"    Profile: " + "  ".join(f"{t[:4]}={d:+.2f}" for t, d in sorted_prof))

        cancel_results[f"{trait}+{opp}"] = {
            "max_abs_delta": float(max_abs_delta),
            "sum_norm": float(np.linalg.norm(sum_vec)),
            "profile": {t: float(d) for t, d in profile.items()},
        }

    results["cancellation"] = cancel_results

    # ===== TEST 3: Deviation from centroid =====
    print(f"\n{'='*70}")
    print(f"TEST 3: DEVIATION FROM CENTROID (trait - mean)")
    print(f"{'='*70}")

    dev_results = {}
    for trait in TRAITS:
        logger.info(f"Testing {trait} - centroid...")
        dev_coords = coords_5d[trait] - centroid
        dev_vec = reconstruct_from_5d(dev_coords, basis_5d)

        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, dev_vec, alpha, baseline)

        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        top_trait = sorted_prof[0][0]
        target_delta = profile[trait]

        print(f"\n  {trait} - centroid:")
        print(f"    Target delta: {target_delta:+.3f}")
        print(f"    Top trait: {top_trait} {'(correct!)' if top_trait == trait else '(WRONG)'}")
        print(f"    Profile: " + "  ".join(f"{t[:4]}={d:+.2f}" for t, d in sorted_prof))

        dev_results[trait] = {
            "target_delta": float(target_delta),
            "top_trait": top_trait,
            "correct": top_trait == trait,
            "profile": {t: float(d) for t, d in profile.items()},
        }

    results["centroid_deviation"] = dev_results

    # ===== TEST 4: Triple average =====
    print(f"\n{'='*70}")
    print(f"TEST 4: TRIPLE AVERAGES")
    print(f"{'='*70}")

    triple_tests = [
        ("artistic", "investigative", "social"),     # Diverse
        ("realistic", "conventional", "enterprising"),  # Complementary triad
        ("artistic", "social", "enterprising"),       # People-oriented
    ]

    triple_results = {}
    for t1, t2, t3 in triple_tests:
        logger.info(f"Testing mean({t1}, {t2}, {t3})...")
        avg_coords = (coords_5d[t1] + coords_5d[t2] + coords_5d[t3]) / 3.0
        avg_vec = reconstruct_from_5d(avg_coords, basis_5d)

        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, avg_vec, alpha, baseline)
        gen = generate_steered(
            model, tokenizer, device, blocks, mid_layer, avg_vec, alpha * 3, gen_prompt)

        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        targets_in_top3 = sum(1 for t, _ in sorted_prof[:3] if t in (t1, t2, t3))

        print(f"\n  mean({t1[:4]}, {t2[:4]}, {t3[:4]}):")
        print(f"    Targets in top 3: {targets_in_top3}/3")
        print(f"    Target deltas: {t1[:4]}={profile[t1]:+.3f}, {t2[:4]}={profile[t2]:+.3f}, {t3[:4]}={profile[t3]:+.3f}")
        print(f"    Profile: " + "  ".join(f"{t[:4]}={d:+.2f}" for t, d in sorted_prof))
        print(f"    Generated: {gen[:120]}...")

        key = f"{t1}+{t2}+{t3}"
        triple_results[key] = {
            "targets_in_top3": targets_in_top3,
            "target_deltas": {t1: float(profile[t1]), t2: float(profile[t2]), t3: float(profile[t3])},
            "profile": {t: float(d) for t, d in profile.items()},
            "generation": gen,
        }

    results["triple_average"] = triple_results

    # ===== TEST 5: Double subtraction (A - B - C) =====
    print(f"\n{'='*70}")
    print(f"TEST 5: DOUBLE SUBTRACTION (A - B - C)")
    print(f"{'='*70}")

    double_sub_tests = [
        ("artistic", "conventional", "realistic"),   # Boost artistic, suppress opposite + alternate
        ("social", "realistic", "investigative"),     # Boost social, suppress opposite + adjacent
    ]

    dsub_results = {}
    for t_boost, t_sup1, t_sup2 in double_sub_tests:
        logger.info(f"Testing {t_boost} - {t_sup1} - {t_sup2}...")
        dsub_coords = coords_5d[t_boost] - coords_5d[t_sup1] - coords_5d[t_sup2]
        dsub_vec = reconstruct_from_5d(dsub_coords, basis_5d)

        profile = measure_trait_profile(
            model, tokenizer, device, blocks, mid_layer, dsub_vec, alpha, baseline)
        gen = generate_steered(
            model, tokenizer, device, blocks, mid_layer, dsub_vec, alpha * 2, gen_prompt)

        sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
        boost_delta = profile[t_boost]
        sup1_delta = profile[t_sup1]
        sup2_delta = profile[t_sup2]

        print(f"\n  {t_boost} - {t_sup1} - {t_sup2}:")
        print(f"    Boost ({t_boost}): {boost_delta:+.3f} (should be positive)")
        print(f"    Suppress ({t_sup1}): {sup1_delta:+.3f} (should be negative)")
        print(f"    Suppress ({t_sup2}): {sup2_delta:+.3f} (should be negative)")
        print(f"    All correct: {'YES' if boost_delta > 0 and sup1_delta < 0 and sup2_delta < 0 else 'NO'}")
        print(f"    Profile: " + "  ".join(f"{t[:4]}={d:+.2f}" for t, d in sorted_prof))
        print(f"    Generated: {gen[:120]}...")

        key = f"{t_boost}-{t_sup1}-{t_sup2}"
        dsub_results[key] = {
            "boost_delta": float(boost_delta),
            "suppress1_delta": float(sup1_delta),
            "suppress2_delta": float(sup2_delta),
            "all_correct": bool(boost_delta > 0 and sup1_delta < 0 and sup2_delta < 0),
            "profile": {t: float(d) for t, d in profile.items()},
            "generation": gen,
        }

    results["double_subtraction"] = dsub_results

    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    # Subtraction accuracy
    sub_correct = sum(1 for r in sub_results.values() if r["correct_direction"])
    print(f"\n  Subtraction (A-B): {sub_correct}/{len(sub_results)} correct direction")

    # Cancellation
    mean_cancel = np.mean([r["max_abs_delta"] for r in cancel_results.values()])
    print(f"  Holland cancellation: mean max|delta| = {mean_cancel:.3f}")

    # Centroid deviation
    dev_correct = sum(1 for r in dev_results.values() if r["correct"])
    print(f"  Centroid deviation: {dev_correct}/{len(dev_results)} correct top trait")

    # Triple average
    mean_top3 = np.mean([r["targets_in_top3"] for r in triple_results.values()])
    print(f"  Triple average: mean targets in top 3 = {mean_top3:.1f}/3.0")

    # Double subtraction
    dsub_correct = sum(1 for r in dsub_results.values() if r["all_correct"])
    print(f"  Double subtraction: {dsub_correct}/{len(dsub_results)} all correct directions")

    results["summary"] = {
        "subtraction_correct": f"{sub_correct}/{len(sub_results)}",
        "mean_cancellation_delta": float(mean_cancel),
        "centroid_deviation_correct": f"{dev_correct}/{len(dev_results)}",
        "mean_triple_top3": float(mean_top3),
        "double_subtraction_correct": f"{dsub_correct}/{len(dsub_results)}",
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_arithmetic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
