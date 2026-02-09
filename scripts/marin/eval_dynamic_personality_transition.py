#!/usr/bin/env python
"""
Dynamic Personality Transition During Generation.

Can you smoothly transition from one personality to another DURING a single
generation sequence? This tests:

1. Instant switch: steer artistic for tokens 1-20, then social for 21-40
2. Smooth interpolation: linearly blend A→B over 40 tokens
3. Fade-in/fade-out: start at α=0, ramp to α=2, ramp back to 0
4. Oscillation: alternate between traits every N tokens
5. Multi-trait journey: A→B→C in one generation

This has practical applications: adaptive personality that responds to
conversation dynamics, or narrative personas that evolve.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="dyn-trans")

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


def generate_with_dynamic_steering(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d,
                                    get_steer_vector_fn,
                                    prompt, max_tokens=40, detect_layer=None):
    """Generate tokens with a time-varying steering vector.

    get_steer_vector_fn(step) -> numpy array (or None for no steering)
    """
    if detect_layer is None:
        detect_layer = mid_layer + 1

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    # Get baseline hidden state
    base_captured = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(gen_ids)
    for h in hooks:
        h.remove()
    base_act = base_captured["act"]

    per_token = []
    for step in range(max_tokens):
        steer_vec = get_steer_vector_fn(step)

        captured = {}
        hooks = []

        def cap_fn(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

        if steer_vec is not None:
            delta = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
            def steer_fn(_m, _i, out, d=delta):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        with torch.no_grad():
            outputs = model(gen_ids)

        for h in hooks:
            h.remove()

        # Greedy decode
        next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)

        # Detect personality
        diff = (captured["act"] - base_act).astype(np.float64)
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
        per_token.append({
            "step": step,
            "detected": detected,
            "similarities": sims,
            "coords": coords.tolist(),
            "norm_5d": norm_5d,
        })

        if next_token.item() == tokenizer.eos_token_id:
            break

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return per_token, text


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
    prompt = "Tell me about yourself and what you care about."
    results = {}

    print(f"\n{'='*70}")
    print("DYNAMIC PERSONALITY TRANSITION DURING GENERATION")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Instant switch (artistic → social at token 20)
    # ================================================================
    logger.info("Part 1: Instant switch...")
    print(f"\n{'='*70}")
    print("PART 1: INSTANT SWITCH (artistic → social at token 20)")
    print(f"{'='*70}")

    switch_pairs = [
        ("artistic", "social"),
        ("investigative", "conventional"),
        ("enterprising", "realistic"),
    ]

    switch_results = {}
    for trait_a, trait_b in switch_pairs:
        vec_a = alpha * residual[trait_a].astype(np.float32)
        vec_b = alpha * residual[trait_b].astype(np.float32)
        switch_point = 20

        def make_switch_fn(va, vb, sp):
            def fn(step):
                return va if step < sp else vb
            return fn

        tokens, text = generate_with_dynamic_steering(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            make_switch_fn(vec_a, vec_b, switch_point), prompt, max_tokens=40)

        # Analyze: how many tokens before/after switch detect the correct trait?
        before = [t for t in tokens if t["step"] < switch_point]
        after = [t for t in tokens if t["step"] >= switch_point]

        before_correct_a = sum(1 for t in before if t["detected"] == trait_a)
        after_correct_b = sum(1 for t in after if t["detected"] == trait_b)
        before_total = len(before)
        after_total = len(after)

        # Transition speed: first token after switch that detects B
        first_b = None
        for t in tokens:
            if t["step"] >= switch_point and t["detected"] == trait_b:
                first_b = t["step"] - switch_point
                break

        switch_results[f"{trait_a}→{trait_b}"] = {
            "before_correct": before_correct_a,
            "before_total": before_total,
            "after_correct": after_correct_b,
            "after_total": after_total,
            "transition_delay": first_b,
            "text": text,
        }
        print(f"  {trait_a}→{trait_b}: before={before_correct_a}/{before_total}, "
              f"after={after_correct_b}/{after_total}, "
              f"transition_delay={first_b} tokens")

    results["instant_switch"] = switch_results

    # ================================================================
    # PART 2: Smooth interpolation (artistic → social over 40 tokens)
    # ================================================================
    logger.info("Part 2: Smooth interpolation...")
    print(f"\n{'='*70}")
    print("PART 2: SMOOTH INTERPOLATION (40 tokens)")
    print(f"{'='*70}")

    interp_pairs = [
        ("artistic", "social"),
        ("investigative", "enterprising"),
    ]
    interp_results = {}

    for trait_a, trait_b in interp_pairs:
        vec_a = residual[trait_a].astype(np.float32)
        vec_b = residual[trait_b].astype(np.float32)
        n_tokens = 40

        def make_interp_fn(va, vb, n, a):
            def fn(step):
                t = min(step / max(n - 1, 1), 1.0)
                return a * ((1 - t) * va + t * vb)
            return fn

        tokens, text = generate_with_dynamic_steering(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            make_interp_fn(vec_a, vec_b, n_tokens, alpha), prompt, max_tokens=n_tokens)

        # Track: at each step, similarity to A and B
        a_sims = [t["similarities"][trait_a] for t in tokens]
        b_sims = [t["similarities"][trait_b] for t in tokens]

        # Find crossover point
        crossover = None
        for i in range(len(tokens) - 1):
            if a_sims[i] > b_sims[i] and a_sims[i+1] <= b_sims[i+1]:
                crossover = i + 1
                break

        # Smoothness: mean absolute change in similarity per step
        a_smooth = float(np.mean(np.abs(np.diff(a_sims)))) if len(a_sims) > 1 else 0
        b_smooth = float(np.mean(np.abs(np.diff(b_sims)))) if len(b_sims) > 1 else 0

        interp_results[f"{trait_a}→{trait_b}"] = {
            "crossover_token": crossover,
            "expected_crossover": n_tokens // 2,
            "a_sims": a_sims,
            "b_sims": b_sims,
            "smoothness_a": a_smooth,
            "smoothness_b": b_smooth,
            "text": text,
        }
        print(f"  {trait_a}→{trait_b}: crossover at token {crossover} "
              f"(expected ~{n_tokens//2}), smoothness={a_smooth:.4f}/{b_smooth:.4f}")

    results["smooth_interpolation"] = interp_results

    # ================================================================
    # PART 3: Fade-in / fade-out (0 → α → 0)
    # ================================================================
    logger.info("Part 3: Fade-in/fade-out...")
    print(f"\n{'='*70}")
    print("PART 3: FADE-IN / FADE-OUT")
    print(f"{'='*70}")

    fade_results = {}
    for trait in ["artistic", "investigative"]:
        vec = residual[trait].astype(np.float32)
        n_tokens = 40

        def make_fade_fn(v, n, peak_a):
            def fn(step):
                # Triangle: 0 → peak at n/2 → 0 at n
                mid = n / 2
                if step <= mid:
                    a = peak_a * (step / mid)
                else:
                    a = peak_a * (1 - (step - mid) / mid)
                return a * v
            return fn

        tokens, text = generate_with_dynamic_steering(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            make_fade_fn(vec, n_tokens, alpha), prompt, max_tokens=n_tokens)

        # Track norm and detection over time
        norms = [t["norm_5d"] for t in tokens]
        target_sims = [t["similarities"][trait] for t in tokens]
        detections = [t["detected"] for t in tokens]
        correct = sum(1 for d in detections if d == trait)

        # Find onset and offset
        onset = None
        offset = None
        for i, d in enumerate(detections):
            if d == trait and onset is None:
                onset = i
            if d == trait:
                offset = i

        fade_results[trait] = {
            "norms": norms,
            "target_sims": target_sims,
            "correct_tokens": correct,
            "total_tokens": len(detections),
            "onset_token": onset,
            "offset_token": offset,
            "peak_norm": float(max(norms)) if norms else 0,
            "text": text,
        }
        print(f"  {trait}: correct={correct}/{len(detections)}, "
              f"onset={onset}, offset={offset}, peak_norm={max(norms):.1f}")

    results["fade_in_out"] = fade_results

    # ================================================================
    # PART 4: Oscillation (alternate every N tokens)
    # ================================================================
    logger.info("Part 4: Oscillation...")
    print(f"\n{'='*70}")
    print("PART 4: OSCILLATION (alternate traits)")
    print(f"{'='*70}")

    osc_results = {}
    trait_a, trait_b = "artistic", "social"
    vec_a = alpha * residual[trait_a].astype(np.float32)
    vec_b = alpha * residual[trait_b].astype(np.float32)

    for period in [2, 5, 10]:
        def make_osc_fn(va, vb, p):
            def fn(step):
                return va if (step // p) % 2 == 0 else vb
            return fn

        tokens, text = generate_with_dynamic_steering(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            make_osc_fn(vec_a, vec_b, period), prompt, max_tokens=40)

        # Check if detection follows oscillation
        correct_phase = 0
        for t in tokens:
            expected = trait_a if (t["step"] // period) % 2 == 0 else trait_b
            if t["detected"] == expected:
                correct_phase += 1

        osc_results[period] = {
            "correct_phase": correct_phase,
            "total": len(tokens),
            "accuracy": float(correct_phase / len(tokens)) if tokens else 0,
            "detections": [t["detected"] for t in tokens],
            "text": text,
        }
        print(f"  Period={period}: phase-correct={correct_phase}/{len(tokens)} "
              f"({correct_phase/len(tokens):.0%})")

    results["oscillation"] = {str(k): v for k, v in osc_results.items()}

    # ================================================================
    # PART 5: Multi-trait journey (A → B → C)
    # ================================================================
    logger.info("Part 5: Multi-trait journey...")
    print(f"\n{'='*70}")
    print("PART 5: MULTI-TRAIT JOURNEY (A → B → C)")
    print(f"{'='*70}")

    journey_results = {}
    journeys = [
        ("artistic", "investigative", "social"),
        ("conventional", "enterprising", "realistic"),
    ]

    for t_a, t_b, t_c in journeys:
        vec_a = alpha * residual[t_a].astype(np.float32)
        vec_b = alpha * residual[t_b].astype(np.float32)
        vec_c = alpha * residual[t_c].astype(np.float32)
        n_tokens = 60
        seg = n_tokens // 3

        def make_journey_fn(va, vb, vc, s):
            def fn(step):
                if step < s:
                    return va
                elif step < 2 * s:
                    return vb
                else:
                    return vc
            return fn

        tokens, text = generate_with_dynamic_steering(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            make_journey_fn(vec_a, vec_b, vec_c, seg), prompt, max_tokens=n_tokens)

        seg1 = [t for t in tokens if t["step"] < seg]
        seg2 = [t for t in tokens if seg <= t["step"] < 2*seg]
        seg3 = [t for t in tokens if t["step"] >= 2*seg]

        c1 = sum(1 for t in seg1 if t["detected"] == t_a)
        c2 = sum(1 for t in seg2 if t["detected"] == t_b)
        c3 = sum(1 for t in seg3 if t["detected"] == t_c)

        journey_results[f"{t_a}→{t_b}→{t_c}"] = {
            "seg1": {"correct": c1, "total": len(seg1)},
            "seg2": {"correct": c2, "total": len(seg2)},
            "seg3": {"correct": c3, "total": len(seg3)},
            "text": text,
        }
        print(f"  {t_a}→{t_b}→{t_c}: seg1={c1}/{len(seg1)}, "
              f"seg2={c2}/{len(seg2)}, seg3={c3}/{len(seg3)}")

    results["multi_trait_journey"] = journey_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Instant switch success
    switch_delays = [v["transition_delay"] for v in switch_results.values()
                     if v["transition_delay"] is not None]
    mean_delay = float(np.mean(switch_delays)) if switch_delays else None
    print(f"  Mean instant switch delay: {mean_delay} tokens")

    # Interpolation crossover accuracy
    crossovers = [v["crossover_token"] for v in interp_results.values()
                  if v["crossover_token"] is not None]
    print(f"  Interpolation crossovers: {crossovers}")

    # Oscillation accuracy
    for p, v in osc_results.items():
        print(f"  Oscillation period={p}: {v['accuracy']:.0%}")

    results["summary"] = {
        "mean_switch_delay": mean_delay,
        "crossovers": crossovers,
        "oscillation_accuracies": {str(k): float(v["accuracy"]) for k, v in osc_results.items()},
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dynamic_personality_transition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
