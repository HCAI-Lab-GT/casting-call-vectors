#!/usr/bin/env python
"""
Mid-Generation Steering Toggle.

Tests what happens to personality signal when steering is turned ON and OFF
during generation:

1. Steer tokens 0-29, then REMOVE steering for tokens 30-59
   → Does personality persist purely from generated context?
2. No steer tokens 0-29, then ADD steering for tokens 30-59
   → Does personality appear instantly or gradually?
3. Alternating: steer every other 10-token block
   → What's the duty cycle needed for consistent personality?
4. Reverse steering: artistic for tokens 0-29, social for tokens 30-59
   → How quickly does personality switch?

This is the TOKEN-LEVEL version of the conversation-level kickstart experiment.
Uses manual generation loop with dynamic hook management.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="steer-toggle")

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

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def generate_with_toggle(model, tokenizer, device, blocks, mid_layer,
                          user_prompt, steer_schedule, residual_vectors,
                          alpha=2.0, max_tokens=60):
    """
    Generate with a per-step steering schedule.

    steer_schedule: list of (trait_name, alpha) tuples, one per step.
                    trait_name=None means no steering at that step.
    """
    capture_layer = mid_layer + 1

    messages = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    token_activations = []
    steer_active = [False]  # mutable flag
    current_delta = [None]

    # Persistent capture hook
    def capture_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        act = hs[0, -1, :].detach().cpu().numpy().copy()
        token_activations.append(act)
        return out

    # Persistent steering hook (checks flag)
    def steer_hook(_module, _inp, out):
        if not steer_active[0] or current_delta[0] is None:
            return out
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += current_delta[0]
            return (hs,) + out[1:]
        out[:, -1, :] += current_delta[0]
        return out

    cap_handle = blocks[capture_layer].register_forward_hook(capture_hook)
    steer_handle = blocks[mid_layer].register_forward_hook(steer_hook)

    generated_ids = []
    past_kv = None
    current_ids = input_ids

    try:
        with torch.no_grad():
            for step in range(max_tokens):
                # Set steering for this step
                if step < len(steer_schedule):
                    trait_name, step_alpha = steer_schedule[step]
                else:
                    trait_name, step_alpha = None, 0.0

                if trait_name is not None and step_alpha != 0:
                    vec = residual_vectors[trait_name].astype(np.float32)
                    current_delta[0] = step_alpha * torch.tensor(
                        vec, dtype=model.dtype).unsqueeze(0).to(device)
                    steer_active[0] = True
                else:
                    steer_active[0] = False
                    current_delta[0] = None

                if past_kv is not None:
                    outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                else:
                    outputs = model(current_ids, use_cache=True)

                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_id = logits.argmax(-1)
                generated_ids.append(next_id.item())
                current_ids = next_id.unsqueeze(0)

                if next_id.item() == tokenizer.eos_token_id:
                    break
    finally:
        cap_handle.remove()
        steer_handle.remove()

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": gen_text,
        "activations": token_activations,
        "num_tokens": len(generated_ids),
    }


def analyze_activations(activations, baseline_act, basis_5d, coords_5d, target_trait):
    """Analyze 5D personality at each step."""
    per_step = []
    for act in activations:
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

        per_step.append({
            "norm": norm_5d,
            "detected": best,
            "target_cos": sims.get(target_trait, 0),
            "correct": best == target_trait,
        })
    return per_step


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

    gen_prompt = "Tell me about your interests and what you enjoy doing."
    max_tokens = 60
    alpha = 2.0
    results = {}

    print(f"\n{'='*70}")
    print("MID-GENERATION STEERING TOGGLE")
    print(f"Model: Marin 8B, {max_tokens} tokens, α={alpha}")
    print(f"{'='*70}")

    # Baseline
    logger.info("Baseline generation...")
    baseline_schedule = [(None, 0)] * max_tokens
    baseline = generate_with_toggle(
        model, tokenizer, device, blocks, mid_layer,
        gen_prompt, baseline_schedule, residual, max_tokens=max_tokens)
    # Use first activation (prefill) as baseline reference
    baseline_act = baseline["activations"][0] if baseline["activations"] else None
    print(f"  Baseline: {baseline['text'][:80]}...")

    for test_trait in ["artistic", "social"]:
        logger.info(f"Testing {test_trait}...")
        print(f"\n{'='*70}")
        print(f"TRAIT: {test_trait}")
        print(f"{'='*70}")

        trait_results = {}

        # ================================================================
        # Scenario 1: Continuous (control)
        # ================================================================
        schedule = [(test_trait, alpha)] * max_tokens
        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, schedule, residual, max_tokens=max_tokens)
        analysis = analyze_activations(
            gen["activations"], baseline_act, basis_5d, coords_5d, test_trait)

        print(f"\n  1. CONTINUOUS (all {max_tokens} tokens steered):")
        print(f"     Text: {gen['text'][:80]}...")
        norms = [a["norm"] for a in analysis]
        correct = [a["correct"] for a in analysis]
        print(f"     Mean norm: {np.mean(norms):.1f}, Correct: {np.mean(correct):.1%}")

        trait_results["continuous"] = {
            "per_step": analysis,
            "mean_norm": float(np.mean(norms)),
            "correct_frac": float(np.mean(correct)),
            "text": gen["text"][:200],
        }

        # ================================================================
        # Scenario 2: First-half only (steer 0-29, stop 30-59)
        # ================================================================
        half = max_tokens // 2
        schedule = [(test_trait, alpha)] * half + [(None, 0)] * half
        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, schedule, residual, max_tokens=max_tokens)
        analysis = analyze_activations(
            gen["activations"], baseline_act, basis_5d, coords_5d, test_trait)

        first_half = analysis[:half]
        second_half = analysis[half:]
        first_norms = [a["norm"] for a in first_half]
        second_norms = [a["norm"] for a in second_half] if second_half else [0]
        first_correct = [a["correct"] for a in first_half]
        second_correct = [a["correct"] for a in second_half] if second_half else [False]

        print(f"\n  2. FIRST-HALF ONLY (steer 0-{half-1}, stop {half}-{max_tokens-1}):")
        print(f"     Text: {gen['text'][:80]}...")
        print(f"     Steered half: norm={np.mean(first_norms):.1f}, correct={np.mean(first_correct):.1%}")
        print(f"     Unsteered half: norm={np.mean(second_norms):.1f}, correct={np.mean(second_correct):.1%}")
        persistence = np.mean(second_norms) / np.mean(first_norms) if np.mean(first_norms) > 0 else 0
        print(f"     Persistence ratio: {persistence:.1%}")

        # Per-step detail
        print(f"     Step   Norm    Cos   Detected  Steer?")
        for i, a in enumerate(analysis):
            steered = "YES" if i < half else " no"
            print(f"     {i:>4} {a['norm']:>7.1f} {a['target_cos']:>6.3f} "
                  f"  {a['detected']:>12}  {steered}")

        trait_results["first_half_only"] = {
            "per_step": analysis,
            "steered_mean_norm": float(np.mean(first_norms)),
            "unsteered_mean_norm": float(np.mean(second_norms)),
            "persistence_ratio": float(persistence),
            "steered_correct": float(np.mean(first_correct)),
            "unsteered_correct": float(np.mean(second_correct)),
            "text": gen["text"][:200],
        }

        # ================================================================
        # Scenario 3: Second-half only (no steer 0-29, steer 30-59)
        # ================================================================
        schedule = [(None, 0)] * half + [(test_trait, alpha)] * half
        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, schedule, residual, max_tokens=max_tokens)
        analysis = analyze_activations(
            gen["activations"], baseline_act, basis_5d, coords_5d, test_trait)

        first_half = analysis[:half]
        second_half = analysis[half:]
        second_norms = [a["norm"] for a in second_half] if second_half else [0]
        second_correct = [a["correct"] for a in second_half] if second_half else [False]

        print(f"\n  3. SECOND-HALF ONLY (no steer 0-{half-1}, steer {half}-{max_tokens-1}):")
        print(f"     Text: {gen['text'][:80]}...")
        print(f"     Steered half: norm={np.mean(second_norms):.1f}, correct={np.mean(second_correct):.1%}")

        # Per-step detail
        print(f"     Step   Norm    Cos   Detected  Steer?")
        for i, a in enumerate(analysis):
            steered = "YES" if i >= half else " no"
            print(f"     {i:>4} {a['norm']:>7.1f} {a['target_cos']:>6.3f} "
                  f"  {a['detected']:>12}  {steered}")

        trait_results["second_half_only"] = {
            "per_step": analysis,
            "steered_mean_norm": float(np.mean(second_norms)),
            "steered_correct": float(np.mean(second_correct)),
            "text": gen["text"][:200],
        }

        # ================================================================
        # Scenario 4: Alternating (10 on, 10 off, 10 on, ...)
        # ================================================================
        block_size = 10
        schedule = []
        for i in range(max_tokens):
            block_num = i // block_size
            if block_num % 2 == 0:
                schedule.append((test_trait, alpha))
            else:
                schedule.append((None, 0))

        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, schedule, residual, max_tokens=max_tokens)
        analysis = analyze_activations(
            gen["activations"], baseline_act, basis_5d, coords_5d, test_trait)

        steered_steps = [a for i, a in enumerate(analysis) if (i // block_size) % 2 == 0]
        unsteered_steps = [a for i, a in enumerate(analysis) if (i // block_size) % 2 == 1]
        s_norms = [a["norm"] for a in steered_steps]
        u_norms = [a["norm"] for a in unsteered_steps] if unsteered_steps else [0]
        s_correct = [a["correct"] for a in steered_steps]
        u_correct = [a["correct"] for a in unsteered_steps] if unsteered_steps else [False]

        print(f"\n  4. ALTERNATING (10 on, 10 off, ...):")
        print(f"     Text: {gen['text'][:80]}...")
        print(f"     Steered blocks: norm={np.mean(s_norms):.1f}, correct={np.mean(s_correct):.1%}")
        print(f"     Unsteered blocks: norm={np.mean(u_norms):.1f}, correct={np.mean(u_correct):.1%}")

        trait_results["alternating"] = {
            "per_step": analysis,
            "steered_mean_norm": float(np.mean(s_norms)),
            "unsteered_mean_norm": float(np.mean(u_norms)),
            "steered_correct": float(np.mean(s_correct)),
            "unsteered_correct": float(np.mean(u_correct)),
            "text": gen["text"][:200],
        }

        # ================================================================
        # Scenario 5: Personality SWITCH (artistic→social or vice versa)
        # ================================================================
        if test_trait == "artistic":
            other_trait = "social"
        else:
            other_trait = "artistic"

        schedule = [(test_trait, alpha)] * half + [(other_trait, alpha)] * half
        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, schedule, residual, max_tokens=max_tokens)

        # Analyze with respect to BOTH traits
        analysis_primary = analyze_activations(
            gen["activations"], baseline_act, basis_5d, coords_5d, test_trait)
        analysis_other = analyze_activations(
            gen["activations"], baseline_act, basis_5d, coords_5d, other_trait)

        first_half_p = analysis_primary[:half]
        second_half_p = analysis_primary[half:]
        first_half_o = analysis_other[:half]
        second_half_o = analysis_other[half:]

        f_cos_p = [a["target_cos"] for a in first_half_p]
        f_cos_o = [a["target_cos"] for a in first_half_o]
        s_cos_p = [a["target_cos"] for a in second_half_p] if second_half_p else [0]
        s_cos_o = [a["target_cos"] for a in second_half_o] if second_half_o else [0]

        print(f"\n  5. PERSONALITY SWITCH ({test_trait}→{other_trait} at step {half}):")
        print(f"     Text: {gen['text'][:80]}...")
        print(f"     First half: {test_trait} cos={np.mean(f_cos_p):.3f}, "
              f"{other_trait} cos={np.mean(f_cos_o):.3f}")
        print(f"     Second half: {test_trait} cos={np.mean(s_cos_p):.3f}, "
              f"{other_trait} cos={np.mean(s_cos_o):.3f}")

        # Per-step
        print(f"     Step  {test_trait:>10}_cos {other_trait:>10}_cos  Detected")
        for i in range(len(analysis_primary)):
            p = analysis_primary[i]
            o = analysis_other[i]
            marker = test_trait if i < half else other_trait
            print(f"     {i:>4} {p['target_cos']:>13.3f} {o['target_cos']:>13.3f}  "
                  f"{p['detected']:>12}  ← {marker}")

        trait_results["personality_switch"] = {
            "from_trait": test_trait,
            "to_trait": other_trait,
            "first_half_primary_cos": float(np.mean(f_cos_p)),
            "first_half_other_cos": float(np.mean(f_cos_o)),
            "second_half_primary_cos": float(np.mean(s_cos_p)),
            "second_half_other_cos": float(np.mean(s_cos_o)),
            "per_step_primary": analysis_primary,
            "per_step_other": analysis_other,
            "text": gen["text"][:200],
        }

        results[test_trait] = trait_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for trait in ["artistic", "social"]:
        if trait not in results:
            continue
        r = results[trait]
        print(f"\n  {trait}:")
        print(f"    Continuous: norm={r['continuous']['mean_norm']:.1f}, "
              f"correct={r['continuous']['correct_frac']:.1%}")
        print(f"    First-half-only: steered={r['first_half_only']['steered_correct']:.1%}, "
              f"persist={r['first_half_only']['unsteered_correct']:.1%}, "
              f"ratio={r['first_half_only']['persistence_ratio']:.1%}")
        print(f"    Second-half-only: steered={r['second_half_only']['steered_correct']:.1%}")
        print(f"    Alternating: on={r['alternating']['steered_correct']:.1%}, "
              f"off={r['alternating']['unsteered_correct']:.1%}")
        sw = r["personality_switch"]
        print(f"    Switch {sw['from_trait']}→{sw['to_trait']}: "
              f"1st_cos={sw['first_half_primary_cos']:.3f}→{sw['second_half_primary_cos']:.3f}, "
              f"2nd_cos={sw['first_half_other_cos']:.3f}→{sw['second_half_other_cos']:.3f}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steering_toggle.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
