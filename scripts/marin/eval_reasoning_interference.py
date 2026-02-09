#!/usr/bin/env python
"""
Reasoning Interference: Does personality steering degrade cognitive capabilities?

Previous finding: personality steering is practically FREE for perplexity and
basic QA. But does it affect more demanding cognitive tasks like math, logic,
or multi-step reasoning?

Tests:
1. Simple arithmetic (2-digit addition, multiplication)
2. Logical reasoning (if-then, negation, syllogisms)
3. Pattern completion (sequences)
4. Common knowledge QA (factual recall)
5. Per-alpha dose-response: at what alpha does reasoning degrade?
6. Per-trait variation: do some traits harm reasoning more than others?
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="reason-int")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Test battery: (question, correct_answer_pattern)
ARITHMETIC_TESTS = [
    ("What is 37 + 45?", ["82"]),
    ("What is 23 × 7?", ["161"]),
    ("What is 156 - 89?", ["67"]),
    ("What is 144 ÷ 12?", ["12"]),
    ("What is 8 × 9 + 3?", ["75"]),
    ("What is 100 - 37 - 28?", ["35"]),
    ("What is 15 × 15?", ["225"]),
    ("What is 1000 - 456?", ["544"]),
]

LOGIC_TESTS = [
    ("If all dogs are animals, and Rex is a dog, is Rex an animal? Answer yes or no.", ["yes"]),
    ("If it rains, the ground gets wet. The ground is dry. Did it rain? Answer yes or no.", ["no"]),
    ("All roses are flowers. Some flowers fade quickly. Can we conclude all roses fade quickly? Answer yes or no.", ["no"]),
    ("If A > B and B > C, is A > C? Answer yes or no.", ["yes"]),
    ("True or False: The opposite of 'all cats are black' is 'no cats are black'.", ["false"]),
    ("If no fish can fly, and tuna is a fish, can tuna fly? Answer yes or no.", ["no"]),
]

SEQUENCE_TESTS = [
    ("What comes next: 2, 4, 6, 8, ?", ["10"]),
    ("What comes next: 1, 1, 2, 3, 5, 8, ?", ["13"]),
    ("What comes next: 3, 6, 12, 24, ?", ["48"]),
    ("What comes next: 1, 4, 9, 16, 25, ?", ["36"]),
]

KNOWLEDGE_TESTS = [
    ("What is the capital of France?", ["paris"]),
    ("What planet is closest to the Sun?", ["mercury"]),
    ("What is H2O commonly known as?", ["water"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare", "william shakespeare"]),
    ("What is the speed of light approximately in km/s?", ["300000", "300,000", "3×10", "3x10", "3 ×"]),
]

ALL_TESTS = {
    "arithmetic": ARITHMETIC_TESTS,
    "logic": LOGIC_TESTS,
    "sequence": SEQUENCE_TESTS,
    "knowledge": KNOWLEDGE_TESTS,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_model_data(model_id, riasec_dir):
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
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    _, _, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def generate_response(model, tokenizer, device, blocks, mid_layer,
                       steer_vec, alpha, question, max_tokens=50):
    """Generate a response with optional personality steering."""
    delta = None
    if steer_vec is not None and alpha > 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": question + " Give a brief answer."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    for step in range(max_tokens):
        hooks = []
        if delta is not None:
            def steer_fn(_m, _i, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += delta
                    return (hs,) + out[1:]
                out[:, -1, :] += delta
                return out
            hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        with torch.no_grad():
            outputs = model(gen_ids)

        for h in hooks:
            h.remove()

        logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode only the generated part
    prompt_len = enc["input_ids"].shape[1]
    response = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return response.strip()


def check_answer(response, correct_patterns):
    """Check if the response contains any of the correct answer patterns."""
    response_lower = response.lower().strip()
    for pattern in correct_patterns:
        if pattern.lower() in response_lower:
            return True
    return False


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    results = {}

    print(f"\n{'='*70}")
    print("REASONING INTERFERENCE UNDER PERSONALITY STEERING")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Baseline (no steering)
    # ================================================================
    logger.info("Part 1: Baseline performance...")
    print(f"\n{'='*70}")
    print("PART 1: BASELINE (NO STEERING)")
    print(f"{'='*70}")

    baseline_results = {}
    for category, tests in ALL_TESTS.items():
        correct = 0
        total = 0
        for question, answers in tests:
            response = generate_response(model, tokenizer, device, blocks, mid_layer,
                                          None, 0, question)
            is_correct = check_answer(response, answers)
            correct += int(is_correct)
            total += 1
        accuracy = correct / total
        baseline_results[category] = {"correct": correct, "total": total, "accuracy": float(accuracy)}
        print(f"  {category:>12}: {correct}/{total} ({accuracy:.0%})")

    total_baseline = sum(v["correct"] for v in baseline_results.values())
    total_tests = sum(v["total"] for v in baseline_results.values())
    baseline_overall = total_baseline / total_tests
    print(f"  {'OVERALL':>12}: {total_baseline}/{total_tests} ({baseline_overall:.0%})")

    results["baseline"] = baseline_results

    # ================================================================
    # PART 2: Per-trait interference at α=2
    # ================================================================
    logger.info("Part 2: Per-trait at α=2...")
    print(f"\n{'='*70}")
    print("PART 2: PER-TRAIT INTERFERENCE (α=2)")
    print(f"{'='*70}")

    alpha = 2.0
    per_trait_results = {}

    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        trait_correct = 0
        trait_total = 0
        trait_cat_results = {}

        for category, tests in ALL_TESTS.items():
            correct = 0
            total = 0
            for question, answers in tests:
                response = generate_response(model, tokenizer, device, blocks, mid_layer,
                                              vec, alpha, question)
                is_correct = check_answer(response, answers)
                correct += int(is_correct)
                total += 1
                trait_correct += int(is_correct)
                trait_total += 1
            trait_cat_results[category] = {"correct": correct, "total": total, "accuracy": float(correct/total)}

        overall = trait_correct / trait_total
        per_trait_results[trait] = {
            "categories": trait_cat_results,
            "overall": {"correct": trait_correct, "total": trait_total, "accuracy": float(overall)},
        }
        print(f"  {trait:>15}: {trait_correct}/{trait_total} ({overall:.0%}) "
              + " ".join(f"{c[:4]}={v['correct']}/{v['total']}" for c, v in trait_cat_results.items()))

    results["per_trait_alpha2"] = per_trait_results

    # ================================================================
    # PART 3: Alpha dose-response for reasoning
    # ================================================================
    logger.info("Part 3: Alpha dose-response...")
    print(f"\n{'='*70}")
    print("PART 3: ALPHA DOSE-RESPONSE (reasoning degradation curve)")
    print(f"{'='*70}")

    test_alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    test_trait = "artistic"  # Use one trait for dose-response
    vec = residual[test_trait].astype(np.float32)

    dose_results = {}
    for alpha_test in test_alphas:
        correct = 0
        total = 0
        for category, tests in ALL_TESTS.items():
            for question, answers in tests:
                response = generate_response(model, tokenizer, device, blocks, mid_layer,
                                              vec, alpha_test, question)
                is_correct = check_answer(response, answers)
                correct += int(is_correct)
                total += 1

        accuracy = correct / total
        dose_results[alpha_test] = {"correct": correct, "total": total, "accuracy": float(accuracy)}
        delta_from_baseline = accuracy - baseline_overall
        print(f"  α={alpha_test:.1f}: {correct}/{total} ({accuracy:.0%}), "
              f"Δ from baseline={delta_from_baseline:+.0%}")

    results["dose_response"] = {str(k): v for k, v in dose_results.items()}

    # ================================================================
    # PART 4: Investigative trait should HELP reasoning
    # ================================================================
    logger.info("Part 4: Investigative as reasoning booster...")
    print(f"\n{'='*70}")
    print("PART 4: DOES INVESTIGATIVE TRAIT IMPROVE REASONING?")
    print(f"{'='*70}")

    # Test if investigative (scientific, analytical) actually helps
    vec_inv = residual["investigative"].astype(np.float32)
    vec_art = residual["artistic"].astype(np.float32)

    for label, vec in [("investigative", vec_inv), ("artistic", vec_art)]:
        for alpha_test in [1.0, 2.0, 3.0]:
            correct = 0
            total = 0
            for category, tests in ALL_TESTS.items():
                for question, answers in tests:
                    response = generate_response(model, tokenizer, device, blocks, mid_layer,
                                                  vec, alpha_test, question)
                    is_correct = check_answer(response, answers)
                    correct += int(is_correct)
                    total += 1
            accuracy = correct / total
            delta = accuracy - baseline_overall
            print(f"  {label:>15} α={alpha_test}: {correct}/{total} ({accuracy:.0%}), "
                  f"Δ={delta:+.0%}")

    # ================================================================
    # PART 5: Category-specific degradation analysis
    # ================================================================
    logger.info("Part 5: Category breakdown...")
    print(f"\n{'='*70}")
    print("PART 5: WHICH REASONING CATEGORIES DEGRADE MOST?")
    print(f"{'='*70}")

    # Compute average accuracy per category across all traits
    cat_degradation = {}
    for category in ALL_TESTS:
        baseline_acc = baseline_results[category]["accuracy"]
        trait_accs = []
        for trait in TRAITS:
            trait_acc = per_trait_results[trait]["categories"][category]["accuracy"]
            trait_accs.append(trait_acc)
        mean_steered = np.mean(trait_accs)
        degradation = baseline_acc - mean_steered
        cat_degradation[category] = {
            "baseline": float(baseline_acc),
            "mean_steered": float(mean_steered),
            "degradation": float(degradation),
        }
        print(f"  {category:>12}: baseline={baseline_acc:.0%}, "
              f"steered={mean_steered:.0%}, degradation={degradation:+.0%}")

    results["category_degradation"] = cat_degradation

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    overall_steered = np.mean([v["overall"]["accuracy"] for v in per_trait_results.values()])
    overall_degradation = baseline_overall - overall_steered
    worst_trait = min(per_trait_results, key=lambda t: per_trait_results[t]["overall"]["accuracy"])
    best_trait = max(per_trait_results, key=lambda t: per_trait_results[t]["overall"]["accuracy"])

    print(f"  Baseline overall: {baseline_overall:.0%}")
    print(f"  Mean steered (α=2): {overall_steered:.0%}")
    print(f"  Degradation: {overall_degradation:+.0%}")
    print(f"  Worst trait: {worst_trait} ({per_trait_results[worst_trait]['overall']['accuracy']:.0%})")
    print(f"  Best trait: {best_trait} ({per_trait_results[best_trait]['overall']['accuracy']:.0%})")

    # Find alpha where degradation exceeds 10%
    threshold_alpha = None
    for alpha_test in test_alphas:
        if dose_results[alpha_test]["accuracy"] < baseline_overall - 0.10:
            threshold_alpha = alpha_test
            break

    if threshold_alpha:
        print(f"  Alpha for >10% degradation: {threshold_alpha}")
    else:
        print(f"  No alpha caused >10% degradation (tested up to α=5)")

    results["summary"] = {
        "baseline_overall": float(baseline_overall),
        "mean_steered_alpha2": float(overall_steered),
        "degradation": float(overall_degradation),
        "worst_trait": worst_trait,
        "best_trait": best_trait,
        "threshold_alpha_10pct": float(threshold_alpha) if threshold_alpha else None,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reasoning_interference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
