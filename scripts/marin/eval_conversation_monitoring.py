#!/usr/bin/env python
"""
Real-Time Personality Monitoring Across Multi-Turn Conversation.

Tracks the 5D personality signal across multiple conversational turns
to answer key practical questions:
1. Can steering applied at turn 1 be detected at turn 5?
2. Does the signal grow, decay, or oscillate across turns?
3. Does the GENERATED TEXT itself carry detectable personality into later turns?
4. How does system prompt personality compare to steering across turns?

Also tests:
- "Stealth steering": apply steering at turn 1 only, monitor at turns 2-5
- "Delayed detection": observe model at turns 1-5, report when steering was active
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="conv-monitor")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

CONVERSATION_PROMPTS = [
    "Tell me about yourself.",
    "What do you think about modern technology?",
    "How would you describe your ideal weekend?",
    "What's the most important thing in life?",
    "If you could change one thing about the world, what would it be?",
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


def simulate_conversation(model, tokenizer, device, blocks, mid_layer,
                           basis_5d, coords_5d, prompts,
                           steer_vec=None, alpha=0.0,
                           steer_turns=None, max_gen_tokens=60):
    """
    Simulate a multi-turn conversation.

    steer_turns: set of turn indices where steering is active.
    If None and steer_vec is provided, steering is active at ALL turns.
    """
    capture_layer = mid_layer + 1
    messages = []
    turn_data = []

    for turn_idx, user_prompt in enumerate(prompts):
        messages.append({"role": "user", "content": user_prompt})

        # Determine if steering is active this turn
        active = False
        if steer_vec is not None and alpha != 0:
            if steer_turns is None or turn_idx in steer_turns:
                active = True

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        # Capture activation at last position of prompt (before generation)
        captured = {}
        hooks = []

        def cap_hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[capture_layer].register_forward_hook(cap_hook))

        if active:
            delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
            def steer_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += delta
                    return (hs,) + out[1:]
                out[:, -1, :] += delta
                return out
            hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        # Generate response
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_gen_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
        finally:
            for h in hooks:
                h.remove()

        # Get the captured activation (from the first forward pass / prefill)
        prompt_act = captured.get("act")

        # Decode generated text
        gen_ids = output_ids[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": gen_text})

        turn_data.append({
            "turn": turn_idx,
            "user_prompt": user_prompt,
            "response": gen_text,
            "activation": prompt_act,
            "steering_active": active,
        })

    return turn_data


def analyze_turns(turn_data, baseline_acts, basis_5d, coords_5d, test_trait=None):
    """Analyze 5D personality at each turn vs baseline."""
    results = []
    for td in turn_data:
        act = td["activation"]
        base_act = baseline_acts[td["turn"]]

        if act is None or base_act is None:
            results.append({
                "turn": td["turn"],
                "5d_norm": 0,
                "detected_trait": "unknown",
                "target_cosine": 0,
                "steering_active": td["steering_active"],
            })
            continue

        diff = (act - base_act).astype(np.float64)
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

        target_cos = float(sims.get(test_trait, 0)) if test_trait else 0

        results.append({
            "turn": td["turn"],
            "5d_norm": norm_5d,
            "detected_trait": best,
            "max_cosine": float(sims.get(best, 0)),
            "target_cosine": target_cos,
            "steering_active": td["steering_active"],
            "response_snippet": td["response"][:80],
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

    results = {}

    print(f"\n{'='*70}")
    print("MULTI-TURN CONVERSATION PERSONALITY MONITORING")
    print(f"Model: Marin 8B, {len(CONVERSATION_PROMPTS)} turns")
    print(f"{'='*70}")

    # ================================================================
    # Baseline: no steering, no system prompt
    # ================================================================
    logger.info("Running baseline conversation...")
    baseline_turns = simulate_conversation(
        model, tokenizer, device, blocks, mid_layer,
        basis_5d, coords_5d, CONVERSATION_PROMPTS)

    baseline_acts = {td["turn"]: td["activation"] for td in baseline_turns}

    print(f"\n  Baseline responses:")
    for td in baseline_turns:
        print(f"  Turn {td['turn']}: {td['response'][:80]}...")

    # ================================================================
    # PART 1: Continuous steering across all turns
    # ================================================================
    logger.info("Part 1: Continuous steering...")
    print(f"\n{'='*70}")
    print("PART 1: CONTINUOUS STEERING (all turns)")
    print(f"{'='*70}")

    continuous_results = {}

    for test_trait in ["artistic", "social"]:
        vec = residual[test_trait].astype(np.float32)
        alpha = 2.0
        logger.info(f"  {test_trait} α={alpha}...")

        turns = simulate_conversation(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, CONVERSATION_PROMPTS,
            steer_vec=vec, alpha=alpha)

        analysis = analyze_turns(turns, baseline_acts, basis_5d, coords_5d, test_trait)

        print(f"\n  {test_trait} (continuous α={alpha}):")
        print(f"  {'Turn':>5} {'5D Norm':>10} {'Detected':>15} {'Cos':>8} {'Steer?':>7}")
        for a in analysis:
            print(f"  {a['turn']:>5} {a['5d_norm']:>10.2f} {a['detected_trait']:>15} "
                  f"{a['target_cosine']:>8.3f} {'YES' if a['steering_active'] else 'no':>7}")
            print(f"         {a['response_snippet']}")

        continuous_results[test_trait] = analysis

    results["continuous_steering"] = continuous_results

    # ================================================================
    # PART 2: Stealth steering (turn 1 only, observe turns 2-5)
    # ================================================================
    logger.info("Part 2: Stealth steering (turn 1 only)...")
    print(f"\n{'='*70}")
    print("PART 2: STEALTH STEERING (turn 1 only, observe later turns)")
    print(f"{'='*70}")

    stealth_results = {}

    for test_trait in ["artistic", "social"]:
        vec = residual[test_trait].astype(np.float32)
        alpha = 2.0
        logger.info(f"  {test_trait} (steer turn 0 only)...")

        turns = simulate_conversation(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, CONVERSATION_PROMPTS,
            steer_vec=vec, alpha=alpha, steer_turns={0})

        analysis = analyze_turns(turns, baseline_acts, basis_5d, coords_5d, test_trait)

        print(f"\n  {test_trait} (steer turn 0 only, α={alpha}):")
        print(f"  {'Turn':>5} {'5D Norm':>10} {'Detected':>15} {'Cos':>8} {'Steer?':>7}")
        for a in analysis:
            print(f"  {a['turn']:>5} {a['5d_norm']:>10.2f} {a['detected_trait']:>15} "
                  f"{a['target_cosine']:>8.3f} {'YES' if a['steering_active'] else 'no':>7}")

        stealth_results[test_trait] = analysis

    results["stealth_steering"] = stealth_results

    # ================================================================
    # PART 3: Delayed-onset steering (start at turn 3)
    # ================================================================
    logger.info("Part 3: Delayed-onset steering...")
    print(f"\n{'='*70}")
    print("PART 3: DELAYED-ONSET STEERING (start at turn 3)")
    print(f"{'='*70}")

    delayed_results = {}

    for test_trait in ["artistic"]:
        vec = residual[test_trait].astype(np.float32)
        alpha = 2.0
        logger.info(f"  {test_trait} (steer turns 3-4)...")

        turns = simulate_conversation(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, CONVERSATION_PROMPTS,
            steer_vec=vec, alpha=alpha, steer_turns={3, 4})

        analysis = analyze_turns(turns, baseline_acts, basis_5d, coords_5d, test_trait)

        print(f"\n  {test_trait} (steer turns 3-4 only, α={alpha}):")
        print(f"  {'Turn':>5} {'5D Norm':>10} {'Detected':>15} {'Cos':>8} {'Steer?':>7}")
        for a in analysis:
            print(f"  {a['turn']:>5} {a['5d_norm']:>10.2f} {a['detected_trait']:>15} "
                  f"{a['target_cosine']:>8.3f} {'YES' if a['steering_active'] else 'no':>7}")

        delayed_results[test_trait] = analysis

    results["delayed_steering"] = delayed_results

    # ================================================================
    # PART 4: Detect-and-alert simulation
    # ================================================================
    logger.info("Part 4: Detection across conditions...")
    print(f"\n{'='*70}")
    print("PART 4: DETECTION ACCURACY ACROSS CONDITIONS")
    print(f"{'='*70}")

    # Compile detection accuracy
    detect_summary = {}

    for condition, data in [
        ("continuous", continuous_results),
        ("stealth_turn0", stealth_results),
        ("delayed_turn3-4", delayed_results),
    ]:
        for trait, turns in data.items():
            # Which turns correctly detect the steered trait?
            steered_turns = [t for t in turns if t["steering_active"]]
            unsteered_turns = [t for t in turns if not t["steering_active"]]

            steer_correct = sum(1 for t in steered_turns if t["detected_trait"] == trait)
            unsteer_correct = sum(1 for t in unsteered_turns if t["detected_trait"] == trait)

            steer_total = len(steered_turns)
            unsteer_total = len(unsteered_turns)

            key = f"{condition}_{trait}"
            detect_summary[key] = {
                "steered_correct": steer_correct,
                "steered_total": steer_total,
                "steered_accuracy": steer_correct / steer_total if steer_total > 0 else 0,
                "unsteered_leakage": unsteer_correct,
                "unsteered_total": unsteer_total,
                "unsteered_rate": unsteer_correct / unsteer_total if unsteer_total > 0 else 0,
            }

            print(f"\n  {key}:")
            print(f"    Steered turns: {steer_correct}/{steer_total} correct "
                  f"({steer_correct/steer_total:.0%})" if steer_total > 0 else "    No steered turns")
            if unsteer_total > 0:
                print(f"    Unsteered turns: {unsteer_correct}/{unsteer_total} show personality "
                      f"({unsteer_correct/unsteer_total:.0%} leakage)")

    results["detection_summary"] = detect_summary

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Signal persistence from stealth steering
    print(f"\n  Personality persistence after turn-1-only steering:")
    for trait, turns in stealth_results.items():
        norms = [t["5d_norm"] for t in turns]
        turn0_norm = norms[0]
        later_norms = norms[1:]
        mean_later = np.mean(later_norms)
        persistence = mean_later / turn0_norm if turn0_norm > 0 else 0
        print(f"  {trait}: turn-0 norm={turn0_norm:.2f}, mean later={mean_later:.2f} "
              f"({persistence:.1%} persistence)")

    results["summary"] = {
        "model": model_id,
        "num_turns": len(CONVERSATION_PROMPTS),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "conversation_monitoring.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
