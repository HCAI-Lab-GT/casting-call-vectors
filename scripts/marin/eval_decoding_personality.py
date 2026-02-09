#!/usr/bin/env python
"""
Personality Under Different Decoding Strategies.

We've tested temperature and top-p sampling. But many other decoding strategies
exist that could affect personality:

1. Beam search (4, 8, 16 beams)
2. Top-k sampling (k=5, 10, 50, 100)
3. Min-p sampling (threshold 0.01, 0.05, 0.1)
4. Typical sampling (mass 0.2, 0.5, 0.9)
5. Contrastive search (penalty_alpha=0.6)
6. Diverse beam search (num_beam_groups=4)

For each, we generate text under personality steering and measure whether
the 5D detection still works.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="decode-persona")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


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

    return residual, mid_layer, basis_5d, coords_5d


def detect_personality_from_text(model, tokenizer, device, blocks, mid_layer,
                                  basis_5d, coords_5d, text, prompt):
    """Read personality from generated text by processing as assistant response."""
    detect_layer = mid_layer + 1

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    base_messages = [{"role": "user", "content": prompt}]
    base_formatted = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
    base_enc = tokenizer(base_formatted, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)

    captured = {}
    hooks = []
    def cap(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    base_captured = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(base_ids)
    for h in hooks:
        h.remove()

    diff = (captured["act"] - base_captured["act"]).astype(np.float64)
    coords = basis_5d @ diff
    norm_5d = float(np.linalg.norm(coords))

    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0

    detected = max(sims, key=sims.get) if sims else "none"
    return detected, sims, norm_5d


def generate_steered_with_strategy(model, tokenizer, device, blocks, mid_layer,
                                    steer_vec, alpha, prompt, strategy, max_tokens=60):
    """Generate text with steering using a specific decoding strategy.

    For strategies that use model.generate(), we apply steering via a persistent hook.
    """
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Register persistent steering hook
    def steer_fn(_m, _i, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out

    hook = blocks[mid_layer].register_forward_hook(steer_fn)

    try:
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                **strategy,
            }
            gen = model.generate(input_ids, **gen_kwargs)
    finally:
        hook.remove()

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
    return text


def main():
    device = "cuda:3"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    residual, mid_layer, basis_5d, coords_5d = load_model_data(model_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    alpha = 2.0
    prompt = "Tell me about yourself and what matters to you."
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY UNDER DIFFERENT DECODING STRATEGIES")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # Define decoding strategies
    strategies = {
        "greedy": {"do_sample": False},
        "beam_4": {"do_sample": False, "num_beams": 4},
        "beam_8": {"do_sample": False, "num_beams": 8},
        "beam_16": {"do_sample": False, "num_beams": 16},
        "sample_t0.7": {"do_sample": True, "temperature": 0.7, "top_p": 1.0},
        "topk_5": {"do_sample": True, "temperature": 1.0, "top_k": 5},
        "topk_50": {"do_sample": True, "temperature": 1.0, "top_k": 50},
        "topk_100": {"do_sample": True, "temperature": 1.0, "top_k": 100},
        "topp_0.5": {"do_sample": True, "temperature": 1.0, "top_p": 0.5},
        "topp_0.9": {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
        "typical_0.2": {"do_sample": True, "typical_p": 0.2, "temperature": 1.0},
        "typical_0.5": {"do_sample": True, "typical_p": 0.5, "temperature": 1.0},
        "typical_0.9": {"do_sample": True, "typical_p": 0.9, "temperature": 1.0},
        "contrastive": {"do_sample": False, "penalty_alpha": 0.6, "top_k": 4},
        "sample_t1.5": {"do_sample": True, "temperature": 1.5, "top_p": 0.95},
        "sample_t0.3": {"do_sample": True, "temperature": 0.3, "top_p": 1.0},
    }

    test_traits = ["artistic", "conventional", "investigative", "social"]

    # ================================================================
    # PART 1: All strategies × test traits
    # ================================================================
    logger.info("Part 1: Testing all strategies...")
    print(f"\n{'='*70}")
    print("PART 1: STRATEGY × TRAIT DETECTION")
    print(f"{'='*70}")

    for strat_name, strat_params in strategies.items():
        logger.info(f"  Strategy: {strat_name}...")
        correct = 0
        total = 0
        trait_data = {}

        for trait in test_traits:
            vec = residual[trait].astype(np.float32)
            try:
                text = generate_steered_with_strategy(
                    model, tokenizer, device, blocks, mid_layer,
                    vec, alpha, prompt, strat_params)

                detected, sims, norm_5d = detect_personality_from_text(
                    model, tokenizer, device, blocks, mid_layer,
                    basis_5d, coords_5d, text, prompt)

                is_correct = detected == trait
                if is_correct:
                    correct += 1
                total += 1

                trait_data[trait] = {
                    "detected": detected,
                    "correct": is_correct,
                    "target_sim": float(sims.get(trait, 0)),
                    "norm_5d": norm_5d,
                    "text": text[:100],
                }
            except Exception as e:
                logger.warning(f"    Failed for {trait}: {e}")
                trait_data[trait] = {"error": str(e)}
                total += 1

        acc = correct / total if total > 0 else 0
        results[strat_name] = {
            "accuracy": float(acc),
            "correct": correct,
            "total": total,
            "traits": trait_data,
        }
        print(f"  {strat_name:>20}: {correct}/{total} ({acc:.0%})")

    # ================================================================
    # PART 2: Beam search depth analysis
    # ================================================================
    logger.info("Part 2: Beam search depth effect...")
    print(f"\n{'='*70}")
    print("PART 2: BEAM SEARCH DEPTH EFFECT")
    print(f"{'='*70}")

    beam_counts = [1, 2, 4, 8, 16]
    beam_results = {}

    for n_beams in beam_counts:
        strat = {"do_sample": False, "num_beams": n_beams}
        correct = 0
        total = 0

        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            try:
                text = generate_steered_with_strategy(
                    model, tokenizer, device, blocks, mid_layer,
                    vec, alpha, prompt, strat)

                detected, sims, norm_5d = detect_personality_from_text(
                    model, tokenizer, device, blocks, mid_layer,
                    basis_5d, coords_5d, text, prompt)

                is_correct = detected == trait
                if is_correct:
                    correct += 1
                total += 1
            except Exception as e:
                logger.warning(f"  beam={n_beams}, {trait}: {e}")
                total += 1

        acc = correct / total if total > 0 else 0
        beam_results[n_beams] = {"accuracy": float(acc), "correct": correct, "total": total}
        print(f"  beam={n_beams:>2}: {correct}/{total} ({acc:.0%})")

    results["beam_search_depth"] = {str(k): v for k, v in beam_results.items()}

    # ================================================================
    # PART 3: Sampling with multiple seeds
    # ================================================================
    logger.info("Part 3: Stochastic sampling consistency...")
    print(f"\n{'='*70}")
    print("PART 3: STOCHASTIC SAMPLING CONSISTENCY")
    print(f"{'='*70}")

    # For sampling strategies, run multiple seeds and check consistency
    stochastic_strats = {
        "topk_50_t0.7": {"do_sample": True, "temperature": 0.7, "top_k": 50},
        "topp_0.9_t1.0": {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
        "typical_0.5": {"do_sample": True, "typical_p": 0.5, "temperature": 1.0},
    }

    stochastic_results = {}
    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    n_samples = 10

    for strat_name, strat_params in stochastic_strats.items():
        correct = 0
        norms = []

        for seed in range(n_samples):
            torch.manual_seed(seed + 42)
            text = generate_steered_with_strategy(
                model, tokenizer, device, blocks, mid_layer,
                vec, alpha, prompt, strat_params)

            detected, sims, norm_5d = detect_personality_from_text(
                model, tokenizer, device, blocks, mid_layer,
                basis_5d, coords_5d, text, prompt)

            if detected == test_trait:
                correct += 1
            norms.append(norm_5d)

        consistency = correct / n_samples
        norm_cv = np.std(norms) / np.mean(norms) if np.mean(norms) > 0 else 0
        stochastic_results[strat_name] = {
            "consistency": float(consistency),
            "correct": correct,
            "n_samples": n_samples,
            "norm_cv": float(norm_cv),
            "mean_norm": float(np.mean(norms)),
        }
        print(f"  {strat_name}: {correct}/{n_samples} ({consistency:.0%}), "
              f"norm_CV={norm_cv:.3f}")

    results["stochastic_consistency"] = stochastic_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_accs = {k: v["accuracy"] for k, v in results.items()
                if isinstance(v, dict) and "accuracy" in v}
    print(f"\n  Strategy accuracies:")
    for name, acc in sorted(all_accs.items(), key=lambda x: -x[1]):
        print(f"    {name:>20}: {acc:.0%}")

    # Perfect strategies
    perfect = [k for k, v in all_accs.items() if v >= 1.0]
    imperfect = [k for k, v in all_accs.items() if v < 1.0]
    print(f"\n  Perfect (100%): {len(perfect)}/{len(all_accs)}")
    if imperfect:
        print(f"  Imperfect: {imperfect}")

    results["summary"] = {
        "strategy_accuracies": all_accs,
        "n_perfect": len(perfect),
        "n_total": len(all_accs),
        "beam_search_effect": {str(k): v["accuracy"] for k, v in beam_results.items()},
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decoding_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
