#!/usr/bin/env python
"""
Personality Signal Dynamics During Autoregressive Generation.

Tracks the 5D personality signal at each generated token during text
generation. Reveals:
1. Does the personality signal grow, decay, or oscillate during generation?
2. Is the signal stable across different generated tokens?
3. Does the generated text reinforce or attenuate the steered personality?
4. Is there a "personality onset" during generation (analogous to layer onset)?

Uses hook-based monitoring during model.generate() to capture activations
at the detection layer for EACH generated token.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="gen-dynamics")

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
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def generate_with_tracking(model, tokenizer, device, blocks, mid_layer,
                            basis_5d, coords_5d, user_prompt,
                            steer_vec=None, alpha=0.0, system_prompt=None,
                            max_tokens=100):
    """
    Generate text while tracking 5D personality signal at each token.
    Returns: generated text, per-token 5D data, token strings.
    """
    capture_layer = mid_layer + 1

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Storage for per-token activations
    token_activations = []
    generation_step = [0]  # mutable counter

    # Capture hook at detection layer
    def capture_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        # During generation, only the last token's activation is new
        act = hs[0, -1, :].detach().cpu().numpy().copy()
        token_activations.append(act)
        return out

    cap_handle = blocks[capture_layer].register_forward_hook(capture_hook)

    # Steering hook
    steer_handle = None
    if steer_vec is not None and alpha != 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        steer_handle = blocks[mid_layer].register_forward_hook(steer_fn)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy for determinism
                temperature=1.0,
            )
    finally:
        cap_handle.remove()
        if steer_handle:
            steer_handle.remove()

    # Decode
    generated_ids = output_ids[0, input_ids.shape[1]:]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # With KV caching, model.generate() does:
    # 1. ONE prefill forward pass (all prompt tokens) → 1 activation captured
    # 2. One forward pass per generated token → 1 activation each
    # So token_activations[0] = prefill (prompt's last position)
    # and token_activations[1:] = per-generated-token activations
    prefill_act = token_activations[0] if token_activations else None
    gen_activations = token_activations[1:] if len(token_activations) > 1 else []

    return {
        "text": generated_text,
        "tokens": generated_tokens,
        "activations": gen_activations,
        "prefill_act": prefill_act,
        "prompt_len": input_ids.shape[1],
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
    gen_prompt = "Tell me about your interests and what you enjoy doing."

    print(f"\n{'='*70}")
    print("PERSONALITY SIGNAL DYNAMICS DURING GENERATION")
    print(f"Model: Marin 8B, Detection: L{mid_layer+1}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Baseline generation (no steering)
    # ================================================================
    logger.info("Part 1: Baseline generation...")
    print(f"\n{'='*70}")
    print("PART 1: BASELINE GENERATION (no steering)")
    print(f"{'='*70}")

    baseline_gen = generate_with_tracking(
        model, tokenizer, device, blocks, mid_layer,
        basis_5d, coords_5d, gen_prompt, max_tokens=80)

    print(f"\n  Generated ({len(baseline_gen['tokens'])} tokens):")
    print(f"  {baseline_gen['text'][:200]}...")

    # Analyze per-token personality
    gen_acts = baseline_gen["activations"]  # Already sliced to generation tokens only
    gen_tokens = baseline_gen["tokens"]

    # Use the prefill activation (prompt's last position) as baseline
    baseline_act = baseline_gen["prefill_act"]

    baseline_data = []
    print(f"\n  {'Step':>5} {'Token':>20} {'5D Norm':>10} {'Detected':>15} {'Cos':>8}")

    for i, (act, tok) in enumerate(zip(gen_acts, gen_tokens)):
        diff = (act - baseline_act).astype(np.float64)
        coords = basis_5d @ diff
        norm_5d = float(np.linalg.norm(coords))
        full_norm = float(np.linalg.norm(diff))
        capture = norm_5d / full_norm if full_norm > 1e-6 else 0

        sims = {}
        for t in TRAITS:
            if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                sims[t] = float(np.dot(coords, coords_5d[t]) / (
                    norm_5d * np.linalg.norm(coords_5d[t])))
            else:
                sims[t] = 0
        best = max(sims, key=sims.get)

        if i < 30 or i % 10 == 0:
            print(f"  {i:>5} {tok[:20]:>20} {norm_5d:>10.2f} {best:>15} {sims.get(best, 0):>8.3f}")

        baseline_data.append({
            "step": i,
            "token": tok,
            "5d_norm": norm_5d,
            "full_norm": full_norm,
            "capture_ratio": capture,
            "detected_trait": best,
            "max_cosine": float(sims.get(best, 0)),
        })

    results["baseline_generation"] = baseline_data

    # ================================================================
    # PART 2: Steered generation — track personality during generation
    # ================================================================
    logger.info("Part 2: Steered generation tracking...")
    print(f"\n{'='*70}")
    print("PART 2: STEERED GENERATION — PERSONALITY SIGNAL PER TOKEN")
    print(f"{'='*70}")

    steered_results = {}

    for test_trait in ["artistic", "investigative", "social"]:
        vec = residual[test_trait].astype(np.float32)
        alpha = 2.0
        logger.info(f"  {test_trait} α={alpha}...")

        steered_gen = generate_with_tracking(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, gen_prompt,
            steer_vec=vec, alpha=alpha, max_tokens=80)

        print(f"\n  {test_trait} α={alpha}:")
        print(f"  Generated: {steered_gen['text'][:200]}...")

        gen_acts_s = steered_gen["activations"]  # Already generation-only
        gen_tokens_s = steered_gen["tokens"]

        # Use same baseline activation
        trait_data = []
        print(f"\n  {'Step':>5} {'Token':>20} {'5D Norm':>10} {'Detected':>15} {'Target Cos':>10}")

        for i, (act, tok) in enumerate(zip(gen_acts_s, gen_tokens_s)):
            diff = (act - baseline_act).astype(np.float64)
            coords = basis_5d @ diff
            norm_5d = float(np.linalg.norm(coords))

            sims = {}
            for t in TRAITS:
                if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                    sims[t] = float(np.dot(coords, coords_5d[t]) / (
                        norm_5d * np.linalg.norm(coords_5d[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)

            if i < 20 or i % 10 == 0:
                print(f"  {i:>5} {tok[:20]:>20} {norm_5d:>10.2f} {best:>15} {sims.get(test_trait, 0):>10.3f}")

            trait_data.append({
                "step": i,
                "token": tok,
                "5d_norm": norm_5d,
                "detected_trait": best,
                "target_cosine": float(sims.get(test_trait, 0)),
                "max_cosine": float(sims.get(best, 0)),
                "correct": best == test_trait,
            })

        steered_results[test_trait] = {
            "text": steered_gen["text"],
            "per_token": trait_data,
            "mean_5d_norm": float(np.mean([d["5d_norm"] for d in trait_data])),
            "mean_target_cos": float(np.mean([d["target_cosine"] for d in trait_data])),
            "correct_fraction": float(np.mean([d["correct"] for d in trait_data])),
        }

        # Summary stats
        norms = [d["5d_norm"] for d in trait_data]
        cos_vals = [d["target_cosine"] for d in trait_data]
        correct_frac = np.mean([d["correct"] for d in trait_data])

        print(f"\n  Summary: mean 5D norm={np.mean(norms):.2f} (std={np.std(norms):.2f})")
        print(f"  Mean target cos={np.mean(cos_vals):.3f}, correct={correct_frac:.1%}")

        # Trend analysis
        if len(norms) > 10:
            first_quarter = np.mean(norms[:len(norms)//4])
            last_quarter = np.mean(norms[-len(norms)//4:])
            trend = (last_quarter - first_quarter) / first_quarter if first_quarter > 0 else 0
            print(f"  Trend: first quarter={first_quarter:.2f}, last quarter={last_quarter:.2f} "
                  f"({trend:+.1%} change)")

    results["steered_generation"] = steered_results

    # ================================================================
    # PART 3: Alpha sweep during generation
    # ================================================================
    logger.info("Part 3: Alpha sweep generation dynamics...")
    print(f"\n{'='*70}")
    print("PART 3: ALPHA SWEEP — HOW DOES ALPHA AFFECT GENERATION DYNAMICS?")
    print(f"{'='*70}")

    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    alpha_results = {}

    for alpha in alphas:
        logger.info(f"  artistic α={alpha}...")

        gen = generate_with_tracking(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, gen_prompt,
            steer_vec=vec, alpha=alpha, max_tokens=60)

        gen_acts_a = gen["activations"]  # Already generation-only
        gen_tokens_a = gen["tokens"]

        per_tok = []
        for i, (act, tok) in enumerate(zip(gen_acts_a, gen_tokens_a)):
            diff = (act - baseline_act).astype(np.float64)
            coords = basis_5d @ diff
            norm_5d = float(np.linalg.norm(coords))
            sims = {}
            for t in TRAITS:
                if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                    sims[t] = float(np.dot(coords, coords_5d[t]) / (
                        norm_5d * np.linalg.norm(coords_5d[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)
            per_tok.append({
                "step": i,
                "5d_norm": norm_5d,
                "detected_trait": best,
                "target_cosine": float(sims.get(test_trait, 0)),
                "correct": best == test_trait,
            })

        mean_norm = float(np.mean([d["5d_norm"] for d in per_tok]))
        mean_cos = float(np.mean([d["target_cosine"] for d in per_tok]))
        correct = float(np.mean([d["correct"] for d in per_tok]))

        print(f"  α={alpha:>4.1f}: mean_5d_norm={mean_norm:>7.2f}, "
              f"mean_cos={mean_cos:>6.3f}, correct={correct:>5.1%}")
        print(f"          text: {gen['text'][:100]}...")

        alpha_results[f"alpha_{alpha}"] = {
            "text": gen["text"],
            "mean_5d_norm": mean_norm,
            "mean_target_cosine": mean_cos,
            "correct_fraction": correct,
            "num_tokens": len(per_tok),
        }

    results["alpha_sweep"] = alpha_results

    # ================================================================
    # PART 4: System prompt generation dynamics
    # ================================================================
    logger.info("Part 4: System prompt generation dynamics...")
    print(f"\n{'='*70}")
    print("PART 4: SYSTEM PROMPT — PERSONALITY DURING GENERATION")
    print(f"{'='*70}")

    SYSTEM_PROMPTS = {
        "artistic": (
            "You are a deeply creative and artistic individual. You value self-expression, "
            "beauty, and originality above all else."
        ),
        "social": (
            "You are a deeply caring and social individual. You value helping others and "
            "building supportive communities."
        ),
    }

    sysp_results = {}

    for sp_trait, sys_prompt in SYSTEM_PROMPTS.items():
        logger.info(f"  {sp_trait} system prompt...")

        gen = generate_with_tracking(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, gen_prompt,
            system_prompt=sys_prompt, max_tokens=80)

        gen_acts_sp = gen["activations"]  # Already generation-only
        gen_tokens_sp = gen["tokens"]

        per_tok = []
        for i, (act, tok) in enumerate(zip(gen_acts_sp, gen_tokens_sp)):
            diff = (act - baseline_act).astype(np.float64)
            coords = basis_5d @ diff
            norm_5d = float(np.linalg.norm(coords))
            sims = {}
            for t in TRAITS:
                if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                    sims[t] = float(np.dot(coords, coords_5d[t]) / (
                        norm_5d * np.linalg.norm(coords_5d[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)
            per_tok.append({
                "step": i,
                "token": tok,
                "5d_norm": norm_5d,
                "detected_trait": best,
                "target_cosine": float(sims.get(sp_trait, 0)),
                "correct": best == sp_trait,
            })

        mean_norm = float(np.mean([d["5d_norm"] for d in per_tok]))
        mean_cos = float(np.mean([d["target_cosine"] for d in per_tok]))
        correct = float(np.mean([d["correct"] for d in per_tok]))

        print(f"\n  {sp_trait}: mean_norm={mean_norm:.2f}, cos={mean_cos:.3f}, correct={correct:.1%}")
        print(f"  Generated: {gen['text'][:200]}...")

        # Trend
        norms = [d["5d_norm"] for d in per_tok]
        if len(norms) > 10:
            first_q = np.mean(norms[:len(norms)//4])
            last_q = np.mean(norms[-len(norms)//4:])
            print(f"  Trend: {first_q:.2f} → {last_q:.2f}")

        sysp_results[sp_trait] = {
            "text": gen["text"],
            "per_token": per_tok,
            "mean_5d_norm": mean_norm,
            "mean_target_cosine": mean_cos,
            "correct_fraction": correct,
        }

    results["sysprompt_generation"] = sysp_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Steered generation personality detection:")
    for trait, data in steered_results.items():
        print(f"  {trait}: correct={data['correct_fraction']:.1%}, "
              f"mean_cos={data['mean_target_cos']:.3f}, "
              f"mean_5d={data['mean_5d_norm']:.2f}")

    print(f"\n  System prompt generation personality detection:")
    for trait, data in sysp_results.items():
        print(f"  {trait}: correct={data['correct_fraction']:.1%}, "
              f"mean_cos={data['mean_target_cosine']:.3f}, "
              f"mean_5d={data['mean_5d_norm']:.2f}")

    print(f"\n  Alpha-dependent generation dynamics (artistic):")
    for alpha_key, data in alpha_results.items():
        print(f"  {alpha_key}: mean_5d={data['mean_5d_norm']:.2f}, "
              f"correct={data['correct_fraction']:.1%}")

    results["summary"] = {
        "model": model_id,
        "gen_prompt": gen_prompt,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation_dynamics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
