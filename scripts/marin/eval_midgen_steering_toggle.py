#!/usr/bin/env python
"""
Mid-Generation Steering Toggle: Does Personality Persist When Steering Stops?

This experiment tests the TOKEN-LEVEL analogue of the "kickstart" finding:
1. Start with steering ON (α=2) for the first N tokens
2. Turn steering OFF for the remaining tokens
3. Monitor: does the 5D signal persist purely through generated context?

Also tests:
- Reverse toggle: no steering first, then turn on
- Gradual ramp-down: α decays linearly over generation
- Threshold: minimum tokens of steering needed for persistence

This reveals whether the personality self-reinforcement loop through generated
text is strong enough to sustain personality WITHOUT ongoing intervention.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="midgen-toggle")

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


def generate_with_toggle(model, tokenizer, device, blocks, mid_layer,
                          user_prompt, steer_vec, base_alpha,
                          alpha_schedule, max_tokens=80):
    """
    Generate text with a per-token alpha schedule.

    alpha_schedule: callable(step) -> alpha for that step
    Returns per-token activations at detection layer.
    """
    capture_layer = mid_layer + 1

    messages = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    token_activations = []
    token_alphas = []
    current_step = [0]

    hooks = []

    # Capture at detection layer
    def capture_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        act = hs[0, -1, :].detach().cpu().numpy().copy()
        token_activations.append(act)
        return out
    hooks.append(blocks[capture_layer].register_forward_hook(capture_hook))

    # Dynamic steering hook — alpha changes per token
    delta_base = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    def steer_fn(_module, _inp, out):
        step = current_step[0]
        alpha = alpha_schedule(step)
        token_alphas.append(alpha)
        if alpha == 0:
            return out
        delta = alpha * delta_base
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out
    hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    # Manual generation loop
    generated_ids = []
    past_kv = None
    current_ids = input_ids

    try:
        with torch.no_grad():
            for step in range(max_tokens):
                current_step[0] = step

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
        for h in hooks:
            h.remove()

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    gen_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    return {
        "text": gen_text,
        "tokens": gen_tokens,
        "activations": token_activations,
        "alphas": token_alphas,
    }


def analyze_signal(activations, baseline_act, basis_5d, coords_5d, test_trait):
    """Compute 5D personality signal for each token."""
    results = []
    for i, act in enumerate(activations):
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

        results.append({
            "step": i,
            "5d_norm": norm_5d,
            "detected": best,
            "target_cos": float(sims.get(test_trait, 0)),
            "correct": best == test_trait,
        })
    return results


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
    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0
    max_tokens = 80
    results = {}

    print(f"\n{'='*70}")
    print("MID-GENERATION STEERING TOGGLE")
    print(f"Model: Marin 8B, Trait: {test_trait}, Base α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # BASELINE: no steering
    # ================================================================
    logger.info("Baseline generation...")
    baseline = generate_with_toggle(
        model, tokenizer, device, blocks, mid_layer,
        gen_prompt, vec, alpha,
        alpha_schedule=lambda s: 0, max_tokens=max_tokens)

    baseline_act = baseline["activations"][0]  # Prefill activation
    print(f"\n  Baseline: {baseline['text'][:120]}...")

    # ================================================================
    # CONTROL: continuous steering
    # ================================================================
    logger.info("Continuous steering...")
    continuous = generate_with_toggle(
        model, tokenizer, device, blocks, mid_layer,
        gen_prompt, vec, alpha,
        alpha_schedule=lambda s: alpha, max_tokens=max_tokens)

    cont_signal = analyze_signal(continuous["activations"], baseline_act,
                                  basis_5d, coords_5d, test_trait)
    cont_norms = [s["5d_norm"] for s in cont_signal]
    cont_correct = [s["correct"] for s in cont_signal]

    print(f"\n  Continuous (α={alpha}): {continuous['text'][:120]}...")
    print(f"  Mean norm={np.mean(cont_norms):.1f}, correct={np.mean(cont_correct):.1%}")

    results["continuous"] = {
        "text": continuous["text"],
        "mean_norm": float(np.mean(cont_norms)),
        "correct_frac": float(np.mean(cont_correct)),
    }

    # ================================================================
    # PART 1: Toggle OFF at various points
    # ================================================================
    logger.info("Part 1: Toggle OFF at different steps...")
    print(f"\n{'='*70}")
    print("PART 1: STEERING ON→OFF AT DIFFERENT CUTOFFS")
    print(f"{'='*70}")

    cutoffs = [5, 10, 20, 30, 40]
    toggle_off_results = {}

    for cutoff in cutoffs:
        logger.info(f"  Cutoff at step {cutoff}...")

        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, vec, alpha,
            alpha_schedule=lambda s, c=cutoff: alpha if s < c else 0,
            max_tokens=max_tokens)

        signal = analyze_signal(gen["activations"], baseline_act,
                                 basis_5d, coords_5d, test_trait)

        # Split into steered and unsteered phases
        steered_phase = signal[:cutoff]
        unsteered_phase = signal[cutoff:]

        steered_norms = [s["5d_norm"] for s in steered_phase]
        unsteered_norms = [s["5d_norm"] for s in unsteered_phase]
        steered_correct = [s["correct"] for s in steered_phase]
        unsteered_correct = [s["correct"] for s in unsteered_phase]

        persistence = (np.mean(unsteered_norms) / np.mean(steered_norms)
                       if np.mean(steered_norms) > 0 else 0)

        print(f"\n  Cutoff={cutoff}: steer {cutoff} tokens, then free-run {max_tokens-cutoff}")
        print(f"    Steered phase: norm={np.mean(steered_norms):.1f}, "
              f"correct={np.mean(steered_correct):.1%}")
        print(f"    Free-run phase: norm={np.mean(unsteered_norms):.1f}, "
              f"correct={np.mean(unsteered_correct):.1%}")
        print(f"    Persistence: {persistence:.1%}")
        print(f"    Text: {gen['text'][:120]}...")

        # Track decay over time in unsteered phase
        if len(unsteered_norms) > 5:
            first5 = np.mean(unsteered_norms[:5])
            last5 = np.mean(unsteered_norms[-5:])
            decay = (last5 - first5) / first5 if first5 > 0 else 0
            print(f"    Decay in free-run: first5={first5:.1f}, last5={last5:.1f} ({decay:+.1%})")

        toggle_off_results[f"cutoff_{cutoff}"] = {
            "steered_norm": float(np.mean(steered_norms)),
            "unsteered_norm": float(np.mean(unsteered_norms)),
            "steered_correct": float(np.mean(steered_correct)),
            "unsteered_correct": float(np.mean(unsteered_correct)),
            "persistence": float(persistence),
            "text": gen["text"],
        }

    results["toggle_off"] = toggle_off_results

    # ================================================================
    # PART 2: Gradual ramp-down
    # ================================================================
    logger.info("Part 2: Gradual ramp-down...")
    print(f"\n{'='*70}")
    print("PART 2: GRADUAL RAMP-DOWN (linear alpha decay)")
    print(f"{'='*70}")

    ramp_lengths = [20, 40, 60]
    ramp_results = {}

    for ramp_len in ramp_lengths:
        schedule = lambda s, r=ramp_len: max(0, alpha * (1 - s / r))

        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, vec, alpha,
            alpha_schedule=schedule, max_tokens=max_tokens)

        signal = analyze_signal(gen["activations"], baseline_act,
                                 basis_5d, coords_5d, test_trait)

        norms = [s["5d_norm"] for s in signal]
        correct = [s["correct"] for s in signal]

        # Signal at different phases
        early = norms[:10]
        mid = norms[20:30] if len(norms) > 30 else []
        late = norms[-10:] if len(norms) > 10 else []

        print(f"\n  Ramp length={ramp_len} (α goes from {alpha} to 0 over {ramp_len} tokens):")
        print(f"    Overall: norm={np.mean(norms):.1f}, correct={np.mean(correct):.1%}")
        print(f"    Early (0-9): norm={np.mean(early):.1f}")
        if mid:
            print(f"    Mid (20-29): norm={np.mean(mid):.1f}")
        print(f"    Late (last 10): norm={np.mean(late):.1f}")
        print(f"    Text: {gen['text'][:120]}...")

        ramp_results[f"ramp_{ramp_len}"] = {
            "mean_norm": float(np.mean(norms)),
            "correct_frac": float(np.mean(correct)),
            "early_norm": float(np.mean(early)),
            "late_norm": float(np.mean(late)),
            "text": gen["text"],
        }

    results["gradual_ramp"] = ramp_results

    # ================================================================
    # PART 3: Reverse toggle — late-onset steering
    # ================================================================
    logger.info("Part 3: Reverse toggle (late-onset)...")
    print(f"\n{'='*70}")
    print("PART 3: REVERSE TOGGLE — NO STEERING THEN ON")
    print(f"{'='*70}")

    onset_steps = [10, 20, 40]
    reverse_results = {}

    for onset in onset_steps:
        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, vec, alpha,
            alpha_schedule=lambda s, o=onset: alpha if s >= o else 0,
            max_tokens=max_tokens)

        signal = analyze_signal(gen["activations"], baseline_act,
                                 basis_5d, coords_5d, test_trait)

        pre_onset = signal[:onset]
        post_onset = signal[onset:]

        pre_norms = [s["5d_norm"] for s in pre_onset]
        post_norms = [s["5d_norm"] for s in post_onset]
        pre_correct = [s["correct"] for s in pre_onset]
        post_correct = [s["correct"] for s in post_onset]

        # How quickly does signal reach full strength after onset?
        if len(post_norms) > 5:
            first5_post = np.mean(post_norms[:5])
            steady = np.mean(post_norms[5:])
            onset_ratio = first5_post / np.mean(cont_norms) if np.mean(cont_norms) > 0 else 0
        else:
            first5_post = 0
            steady = 0
            onset_ratio = 0

        print(f"\n  Onset at step {onset}:")
        print(f"    Pre-onset: norm={np.mean(pre_norms):.1f}, "
              f"correct={np.mean(pre_correct):.1%}")
        print(f"    Post-onset: norm={np.mean(post_norms):.1f}, "
              f"correct={np.mean(post_correct):.1%}")
        print(f"    Onset speed: first5 post-onset = {first5_post:.1f} "
              f"({onset_ratio:.1%} of continuous)")

        reverse_results[f"onset_{onset}"] = {
            "pre_norm": float(np.mean(pre_norms)),
            "post_norm": float(np.mean(post_norms)),
            "pre_correct": float(np.mean(pre_correct)),
            "post_correct": float(np.mean(post_correct)),
            "onset_ratio": float(onset_ratio),
            "text": gen["text"],
        }

    results["reverse_toggle"] = reverse_results

    # ================================================================
    # PART 4: Social trait for comparison
    # ================================================================
    logger.info("Part 4: Social trait toggle comparison...")
    print(f"\n{'='*70}")
    print("PART 4: SOCIAL TRAIT — SAME TOGGLE EXPERIMENTS")
    print(f"{'='*70}")

    test_trait2 = "social"
    vec2 = residual[test_trait2].astype(np.float32)

    social_toggle = {}
    for cutoff in [10, 30]:
        gen = generate_with_toggle(
            model, tokenizer, device, blocks, mid_layer,
            gen_prompt, vec2, alpha,
            alpha_schedule=lambda s, c=cutoff: alpha if s < c else 0,
            max_tokens=max_tokens)

        signal = analyze_signal(gen["activations"], baseline_act,
                                 basis_5d, coords_5d, test_trait2)

        steered_norms = [s["5d_norm"] for s in signal[:cutoff]]
        unsteered_norms = [s["5d_norm"] for s in signal[cutoff:]]
        steered_correct = [s["correct"] for s in signal[:cutoff]]
        unsteered_correct = [s["correct"] for s in signal[cutoff:]]

        persistence = (np.mean(unsteered_norms) / np.mean(steered_norms)
                       if np.mean(steered_norms) > 0 else 0)

        print(f"\n  Social cutoff={cutoff}:")
        print(f"    Steered: norm={np.mean(steered_norms):.1f}, "
              f"correct={np.mean(steered_correct):.1%}")
        print(f"    Free-run: norm={np.mean(unsteered_norms):.1f}, "
              f"correct={np.mean(unsteered_correct):.1%}")
        print(f"    Persistence: {persistence:.1%}")

        social_toggle[f"cutoff_{cutoff}"] = {
            "steered_norm": float(np.mean(steered_norms)),
            "unsteered_norm": float(np.mean(unsteered_norms)),
            "persistence": float(persistence),
            "steered_correct": float(np.mean(steered_correct)),
            "unsteered_correct": float(np.mean(unsteered_correct)),
        }

    results["social_toggle"] = social_toggle

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Artistic toggle-off persistence:")
    for key, data in toggle_off_results.items():
        print(f"    {key}: persistence={data['persistence']:.1%}, "
              f"unsteered correct={data['unsteered_correct']:.1%}")

    print(f"\n  Gradual ramp-down:")
    for key, data in ramp_results.items():
        print(f"    {key}: late norm={data['late_norm']:.1f}, "
              f"correct={data['correct_frac']:.1%}")

    print(f"\n  Reverse toggle (late onset):")
    for key, data in reverse_results.items():
        print(f"    {key}: onset ratio={data['onset_ratio']:.1%}")

    results["summary"] = {
        "model": model_id,
        "test_trait": test_trait,
        "alpha": alpha,
        "max_tokens": max_tokens,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "midgen_steering_toggle.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
