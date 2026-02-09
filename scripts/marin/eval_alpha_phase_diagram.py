#!/usr/bin/env python
"""
Alpha Phase Diagram: Map the complete alpha landscape.

Prior experiments tested alpha at a few discrete points (0.5, 1, 2, 3, 5).
This experiment maps the COMPLETE alpha landscape from 0.001 to 100 with
fine-grained resolution to find:

1. Minimum detectable alpha (sensitivity threshold)
2. Phase transitions in detection accuracy
3. Coherence collapse point (perplexity explosion)
4. The Goldilocks zone (high detection + low perplexity)
5. Asymptotic behavior at extreme alphas

Also tests NEGATIVE alpha (anti-personality) behavior.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="alpha-phase")

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


def measure_at_alpha(model, tokenizer, device, blocks, mid_layer, basis_5d,
                     coords_5d, steer_vec, alpha, prompt, max_tokens=40):
    """Measure detection, norm, perplexity, and coherence at a specific alpha."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    # Generate with steering, collect per-token logits for perplexity
    all_logprobs = []
    all_coords = []

    for step in range(max_tokens):
        captured = {}
        hooks = []

        def steer_fn(_m, _i, out, _delta=delta):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += _delta
                return (hs,) + out[1:]
            out[:, -1, :] += _delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        def cap_fn(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

        with torch.no_grad():
            outputs = model(gen_ids)
        for h in hooks:
            h.remove()

        # Logprobs of generated token
        logits = outputs.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=0)
        next_token = torch.argmax(logits).item()
        logprob = float(torch.log(probs[next_token] + 1e-10))
        all_logprobs.append(logprob)

        # 5D coordinates
        if "act" in captured:
            coords = basis_5d @ captured["act"].astype(np.float64)
            all_coords.append(coords)

        next_id = torch.tensor([[next_token]], device=device)
        gen_ids = torch.cat([gen_ids, next_id], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    # Get baseline for detection
    base_msgs = [{"role": "user", "content": prompt}]
    base_fmt = tokenizer.apply_chat_template(base_msgs, tokenize=False, add_generation_prompt=True)
    base_enc = tokenizer(base_fmt, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)

    base_cap = {}
    hooks = []
    def cb(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cb))
    with torch.no_grad():
        model(base_ids)
    for h in hooks:
        h.remove()

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)

    # Mean 5D coordinates
    if all_coords:
        mean_coords = np.mean(all_coords, axis=0)
    else:
        mean_coords = np.zeros(5)

    # Subtract baseline from mean coords
    base_diff = (all_coords[0] if all_coords else np.zeros(5))  # Approximate
    norm_5d = float(np.linalg.norm(mean_coords))

    # Detect trait
    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(mean_coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0
    detected = max(sims, key=sims.get) if sims else "none"

    # Perplexity
    mean_logprob = np.mean(all_logprobs) if all_logprobs else -10
    ppl = float(np.exp(-mean_logprob))

    # Entropy of token distribution
    mean_entropy = float(-mean_logprob)  # Approximation: cross-entropy ≈ -logprob

    # Unique tokens ratio (diversity)
    gen_tokens = gen_ids[0, prompt_len:].tolist()
    unique_ratio = len(set(gen_tokens)) / max(len(gen_tokens), 1)

    return {
        "detected": detected,
        "similarities": sims,
        "norm_5d": norm_5d,
        "perplexity": ppl,
        "mean_logprob": float(mean_logprob),
        "unique_token_ratio": float(unique_ratio),
        "n_tokens": len(all_logprobs),
        "text": text[:200],
    }


def main():
    device = "cuda:2"
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

    prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("ALPHA PHASE DIAGRAM")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Fine-grained positive alpha sweep
    # ================================================================
    logger.info("Part 1: Positive alpha sweep...")
    print(f"\n{'='*70}")
    print("PART 1: POSITIVE ALPHA SWEEP (0.001 to 100)")
    print(f"{'='*70}")

    alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5,
              0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0,
              30.0, 50.0, 75.0, 100.0]

    test_traits = ["artistic", "conventional", "investigative"]
    alpha_results = {}

    for alpha in alphas:
        trait_results = {}
        correct = 0
        total = 0

        for trait in test_traits:
            vec = residual[trait].astype(np.float32)
            result = measure_at_alpha(model, tokenizer, device, blocks, mid_layer,
                                       basis_5d, coords_5d, vec, alpha, prompt)
            is_correct = result["detected"] == trait
            if is_correct:
                correct += 1
            total += 1

            trait_results[trait] = {
                "detected": result["detected"],
                "correct": is_correct,
                "target_sim": float(result["similarities"].get(trait, 0)),
                "norm_5d": result["norm_5d"],
                "perplexity": result["perplexity"],
                "mean_logprob": result["mean_logprob"],
                "unique_ratio": result["unique_token_ratio"],
                "n_tokens": result["n_tokens"],
                "text": result["text"][:100],
            }

        acc = correct / total
        mean_ppl = np.mean([v["perplexity"] for v in trait_results.values()])
        mean_norm = np.mean([v["norm_5d"] for v in trait_results.values()])

        alpha_results[alpha] = {
            "accuracy": float(acc),
            "mean_perplexity": float(mean_ppl),
            "mean_norm_5d": float(mean_norm),
            "traits": trait_results,
        }

        print(f"  α={alpha:>7.3f}: acc={acc:.0%}, ppl={mean_ppl:>8.1f}, "
              f"norm={mean_norm:>7.1f}")

    results["positive_alpha"] = {str(k): v for k, v in alpha_results.items()}

    # ================================================================
    # PART 2: Negative alpha (anti-personality)
    # ================================================================
    logger.info("Part 2: Negative alpha sweep...")
    print(f"\n{'='*70}")
    print("PART 2: NEGATIVE ALPHA (ANTI-PERSONALITY)")
    print(f"{'='*70}")

    neg_alphas = [-0.5, -1.0, -2.0, -3.0, -5.0, -10.0]
    neg_results = {}

    for alpha in neg_alphas:
        trait_results = {}
        for trait in test_traits:
            vec = residual[trait].astype(np.float32)
            result = measure_at_alpha(model, tokenizer, device, blocks, mid_layer,
                                       basis_5d, coords_5d, vec, alpha, prompt)

            trait_results[trait] = {
                "detected": result["detected"],
                "target_sim": float(result["similarities"].get(trait, 0)),
                "norm_5d": result["norm_5d"],
                "perplexity": result["perplexity"],
                "text": result["text"][:100],
            }

        mean_ppl = np.mean([v["perplexity"] for v in trait_results.values()])
        neg_results[alpha] = {
            "mean_perplexity": float(mean_ppl),
            "traits": trait_results,
        }

        print(f"  α={alpha:>7.1f}: ppl={mean_ppl:>8.1f}")
        for t, d in trait_results.items():
            print(f"    {t}: detected={d['detected']}, sim={d['target_sim']:+.3f}")

    results["negative_alpha"] = {str(k): v for k, v in neg_results.items()}

    # ================================================================
    # PART 3: All 6 traits at key alphas
    # ================================================================
    logger.info("Part 3: All traits at key alphas...")
    print(f"\n{'='*70}")
    print("PART 3: ALL 6 TRAITS AT KEY ALPHAS")
    print(f"{'='*70}")

    key_alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    full_results = {}

    for alpha in key_alphas:
        correct = 0
        total = 0
        trait_data = {}

        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            result = measure_at_alpha(model, tokenizer, device, blocks, mid_layer,
                                       basis_5d, coords_5d, vec, alpha, prompt)
            is_correct = result["detected"] == trait
            if is_correct:
                correct += 1
            total += 1
            trait_data[trait] = {
                "detected": result["detected"],
                "correct": is_correct,
                "norm_5d": result["norm_5d"],
                "perplexity": result["perplexity"],
            }

        acc = correct / total
        mean_ppl = np.mean([v["perplexity"] for v in trait_data.values()])
        full_results[alpha] = {
            "accuracy": float(acc),
            "mean_perplexity": float(mean_ppl),
            "traits": trait_data,
        }
        print(f"  α={alpha:>7.2f}: {correct}/6 ({acc:.0%}), ppl={mean_ppl:.1f}")

    results["all_traits"] = {str(k): v for k, v in full_results.items()}

    # ================================================================
    # PART 4: Perplexity vs baseline ratio
    # ================================================================
    logger.info("Part 4: Baseline perplexity...")
    print(f"\n{'='*70}")
    print("PART 4: PERPLEXITY RATIO (STEERED/BASELINE)")
    print(f"{'='*70}")

    # Baseline perplexity (no steering)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=40, do_sample=False)
    prompt_len = enc["input_ids"].shape[1]

    # Compute baseline PPL
    with torch.no_grad():
        outputs = model(gen)
    logits = outputs.logits[0, prompt_len-1:-1, :].float()
    targets = gen[0, prompt_len:]
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs[range(len(targets)), targets]
    baseline_ppl = float(torch.exp(-token_logprobs.mean()))

    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    ppl_ratios = {}
    for alpha_str, data in results["positive_alpha"].items():
        ratio = data["mean_perplexity"] / baseline_ppl if baseline_ppl > 0 else 0
        ppl_ratios[alpha_str] = float(ratio)
        if float(alpha_str) in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
            print(f"  α={alpha_str:>7}: PPL ratio = {ratio:.2f}×")

    results["ppl_ratios"] = ppl_ratios
    results["baseline_ppl"] = float(baseline_ppl)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Find phase boundaries
    # 1. Detection threshold (first alpha with >50% accuracy)
    sorted_alphas = sorted(results["positive_alpha"].items(), key=lambda x: float(x[0]))
    detect_threshold = None
    for alpha_str, data in sorted_alphas:
        if data["accuracy"] > 0.5:
            detect_threshold = float(alpha_str)
            break

    # 2. Perfect detection range
    perfect_start = None
    perfect_end = None
    for alpha_str, data in sorted_alphas:
        if data["accuracy"] >= 1.0:
            if perfect_start is None:
                perfect_start = float(alpha_str)
            perfect_end = float(alpha_str)

    # 3. Coherence collapse (PPL > 100)
    collapse_alpha = None
    for alpha_str, data in sorted_alphas:
        if data["mean_perplexity"] > 100:
            collapse_alpha = float(alpha_str)
            break

    # 4. PPL doubling point
    doubling_alpha = None
    for alpha_str, ratio in ppl_ratios.items():
        if ratio > 2.0:
            doubling_alpha = float(alpha_str)
            break

    print(f"  Detection threshold (>50%): α={detect_threshold}")
    print(f"  Perfect detection range: α={perfect_start}–{perfect_end}")
    print(f"  PPL doubling point: α={doubling_alpha}")
    print(f"  Coherence collapse (PPL>100): α={collapse_alpha}")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    results["summary"] = {
        "detection_threshold": detect_threshold,
        "perfect_detection_start": perfect_start,
        "perfect_detection_end": perfect_end,
        "ppl_doubling_alpha": doubling_alpha,
        "coherence_collapse_alpha": collapse_alpha,
        "baseline_ppl": float(baseline_ppl),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "alpha_phase_diagram.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
