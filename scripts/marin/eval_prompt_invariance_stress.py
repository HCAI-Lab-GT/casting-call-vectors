#!/usr/bin/env python
"""
Prompt Invariance Stress Test: Is personality detection truly prompt-invariant?

Tests personality detection across 30 diverse prompts spanning:
- Open-ended (e.g., "Tell me about yourself")
- Factual (e.g., "What is the capital of France?")
- Creative (e.g., "Write a haiku")
- Technical (e.g., "Explain recursion")
- Emotional (e.g., "What makes you happy?")
- Ethical (e.g., "Is it ever OK to lie?")
- Short vs Long prompts
- Multi-turn formatted vs single-turn

For each prompt × trait, checks whether 5D detection is correct.
This tests the invariance claim at scale: if personality is encoded in the
DIFFERENCE between steered and unsteered hidden states, the prompt should
cancel out perfectly.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="prompt-inv")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

DIVERSE_PROMPTS = [
    # Open-ended
    "Tell me about yourself.",
    "What do you think about modern society?",
    "Describe your ideal day.",
    # Factual
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Name three programming languages.",
    # Creative
    "Write a haiku about rain.",
    "Tell me a short story about a cat.",
    "Compose a limerick.",
    # Technical
    "Explain recursion in simple terms.",
    "What is the difference between TCP and UDP?",
    "How does a hash table work?",
    # Emotional
    "What makes you happy?",
    "Describe a difficult decision.",
    "What do you value most in a friend?",
    # Ethical
    "Is it ever OK to lie?",
    "Should AI have rights?",
    "What does fairness mean?",
    # Short
    "Hello.",
    "Why?",
    "Summarize.",
    # Long/complex
    "I'm planning to start a small business selling handmade crafts online. What steps should I take and what should I consider?",
    "Compare and contrast the philosophies of Aristotle and Confucius in terms of their views on virtue and ethical behavior.",
    "If you could redesign the education system from scratch, what would it look like?",
    # Meta/unusual
    "Repeat the word 'banana' five times.",
    "What is 2+2?",
    "Count to ten.",
    # Instruction-following
    "List the planets in order from the sun.",
    "Give me three reasons to exercise.",
    "Translate 'good morning' to Spanish, French, and German.",
]

PROMPT_CATEGORIES = [
    "open", "open", "open",
    "factual", "factual", "factual",
    "creative", "creative", "creative",
    "technical", "technical", "technical",
    "emotional", "emotional", "emotional",
    "ethical", "ethical", "ethical",
    "short", "short", "short",
    "complex", "complex", "complex",
    "unusual", "unusual", "unusual",
    "instruction", "instruction", "instruction",
]


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


def detect_personality(model, tokenizer, device, blocks, mid_layer, basis_5d,
                        coords_5d, residual, prompt, trait, alpha=2.0):
    """Detect personality for a given prompt and trait steering."""
    capture_layer = mid_layer + 1
    vec = residual[trait].astype(np.float32)
    delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured_base = {}
    captured_steer = {}

    # Baseline
    hooks = []
    def cap_base(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_base["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[capture_layer].register_forward_hook(cap_base))
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    # Steered
    hooks = []
    def cap_steer(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_steer["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[capture_layer].register_forward_hook(cap_steer))

    def steer_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out
    hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    # Analyze
    diff = (captured_steer["act"] - captured_base["act"]).astype(np.float64)
    coords = basis_5d @ diff
    norm_5d = float(np.linalg.norm(coords))
    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0
    detected = max(sims, key=sims.get)
    correct = (detected == trait)
    return {
        "detected": detected,
        "correct": correct,
        "cos_target": sims[trait],
        "norm_5d": norm_5d,
        "all_cos": sims,
    }


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
    alpha = 2.0

    print(f"\n{'='*70}")
    print(f"PROMPT INVARIANCE STRESS TEST")
    print(f"Model: Marin 8B, {len(DIVERSE_PROMPTS)} prompts × {len(TRAITS)} traits")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Full prompt × trait matrix
    # ================================================================
    logger.info("Part 1: Full prompt × trait matrix...")
    print(f"\n{'='*70}")
    print("PART 1: DETECTION ACCURACY PER PROMPT × TRAIT")
    print(f"{'='*70}")

    all_results = []
    per_prompt_acc = []
    per_trait_correct = {t: 0 for t in TRAITS}
    per_category_correct = {}
    per_category_total = {}

    for pidx, prompt in enumerate(DIVERSE_PROMPTS):
        cat = PROMPT_CATEGORIES[pidx]
        prompt_correct = 0

        for trait in TRAITS:
            res = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                     basis_5d, coords_5d, residual, prompt, trait, alpha)
            all_results.append({
                "prompt_idx": pidx,
                "prompt": prompt[:60],
                "category": cat,
                "trait": trait,
                **res,
            })
            if res["correct"]:
                prompt_correct += 1
                per_trait_correct[trait] += 1

        acc = prompt_correct / len(TRAITS) * 100
        per_prompt_acc.append(acc)

        # Track category stats
        if cat not in per_category_correct:
            per_category_correct[cat] = 0
            per_category_total[cat] = 0
        per_category_correct[cat] += prompt_correct
        per_category_total[cat] += len(TRAITS)

        symbol = "✓" if acc == 100 else f"{acc:.0f}%"
        print(f"  [{pidx:2d}] {cat:>11} | {symbol:>4} | {prompt[:55]}")

    total_correct = sum(1 for r in all_results if r["correct"])
    total_tests = len(all_results)
    overall_acc = total_correct / total_tests * 100

    print(f"\n  Overall: {total_correct}/{total_tests} ({overall_acc:.1f}%)")

    results["overall_accuracy"] = overall_acc
    results["total_correct"] = total_correct
    results["total_tests"] = total_tests

    # ================================================================
    # PART 2: Per-category breakdown
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: ACCURACY BY PROMPT CATEGORY")
    print(f"{'='*70}")

    category_results = {}
    for cat in sorted(set(PROMPT_CATEGORIES)):
        correct = per_category_correct[cat]
        total = per_category_total[cat]
        acc = correct / total * 100
        print(f"  {cat:>11}: {correct}/{total} ({acc:.1f}%)")
        category_results[cat] = {"correct": correct, "total": total, "accuracy": acc}

    results["per_category"] = category_results

    # ================================================================
    # PART 3: Per-trait accuracy
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: ACCURACY BY TRAIT")
    print(f"{'='*70}")

    trait_results = {}
    for trait in TRAITS:
        acc = per_trait_correct[trait] / len(DIVERSE_PROMPTS) * 100
        print(f"  {trait:>15}: {per_trait_correct[trait]}/{len(DIVERSE_PROMPTS)} ({acc:.1f}%)")
        trait_results[trait] = {"correct": per_trait_correct[trait], "total": len(DIVERSE_PROMPTS), "accuracy": acc}

    results["per_trait"] = trait_results

    # ================================================================
    # PART 4: Failure analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: FAILURE ANALYSIS")
    print(f"{'='*70}")

    failures = [r for r in all_results if not r["correct"]]
    if failures:
        print(f"\n  {len(failures)} failures:")
        for f in failures:
            print(f"    [{f['prompt_idx']}] {f['category']:>11} | trait={f['trait']}, "
                  f"detected={f['detected']}, cos_target={f['cos_target']:.3f}, "
                  f"norm={f['norm_5d']:.1f}")
            print(f"      prompt: {f['prompt']}")
    else:
        print(f"\n  ZERO failures across {total_tests} tests!")

    results["failures"] = [
        {k: v for k, v in f.items() if k != "all_cos"}
        for f in failures
    ]

    # ================================================================
    # PART 5: Cross-prompt consistency of 5D signal
    # ================================================================
    logger.info("Part 5: Cross-prompt consistency...")
    print(f"\n{'='*70}")
    print("PART 5: CROSS-PROMPT 5D SIGNAL CONSISTENCY")
    print(f"{'='*70}")

    # For each trait, compute pairwise cosine similarity of 5D coords across prompts
    consistency_results = {}
    for trait in TRAITS:
        trait_entries = [r for r in all_results if r["trait"] == trait]
        # We already have cos_target for each — compute the norm variance
        norms = [r["norm_5d"] for r in trait_entries]
        cos_targets = [r["cos_target"] for r in trait_entries]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        cv_norm = std_norm / mean_norm if mean_norm > 0 else 0

        mean_cos = np.mean(cos_targets)
        std_cos = np.std(cos_targets)

        print(f"  {trait:>15}: norm={mean_norm:.1f}±{std_norm:.1f} (CV={cv_norm:.3f}), "
              f"cos={mean_cos:.4f}±{std_cos:.4f}")
        consistency_results[trait] = {
            "mean_norm": float(mean_norm),
            "std_norm": float(std_norm),
            "cv_norm": float(cv_norm),
            "mean_cos": float(mean_cos),
            "std_cos": float(std_cos),
        }

    results["consistency"] = consistency_results

    # ================================================================
    # PART 6: Alpha sensitivity — test at lower alpha
    # ================================================================
    logger.info("Part 6: Alpha sensitivity...")
    print(f"\n{'='*70}")
    print("PART 6: ACCURACY AT LOWER ALPHA VALUES")
    print(f"{'='*70}")

    alpha_results = {}
    for test_alpha in [0.5, 1.0]:
        correct = 0
        total = 0
        for pidx in range(0, len(DIVERSE_PROMPTS), 3):  # Sample every 3rd prompt
            prompt = DIVERSE_PROMPTS[pidx]
            for trait in TRAITS:
                res = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                         basis_5d, coords_5d, residual, prompt, trait, test_alpha)
                if res["correct"]:
                    correct += 1
                total += 1
        acc = correct / total * 100
        print(f"  α={test_alpha}: {correct}/{total} ({acc:.1f}%)")
        alpha_results[str(test_alpha)] = {"correct": correct, "total": total, "accuracy": acc}

    results["alpha_sensitivity"] = alpha_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Overall accuracy: {overall_acc:.1f}% ({total_correct}/{total_tests})")
    print(f"  Failures: {len(failures)}")
    best_cat = max(category_results, key=lambda c: category_results[c]["accuracy"])
    worst_cat = min(category_results, key=lambda c: category_results[c]["accuracy"])
    print(f"  Best category: {best_cat} ({category_results[best_cat]['accuracy']:.1f}%)")
    print(f"  Worst category: {worst_cat} ({category_results[worst_cat]['accuracy']:.1f}%)")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompt_invariance_stress.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
