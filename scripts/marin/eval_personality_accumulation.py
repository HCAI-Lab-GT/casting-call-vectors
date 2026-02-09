#!/usr/bin/env python
"""
Personality accumulation: quantifying the turn-by-turn amplification effect.

The persistence experiment revealed that continuous steering AMPLIFIES
personality over conversation turns (social +0.75 → +2.25 in 5 turns).

This experiment systematically tests:
1. Does accumulation happen for ALL traits?
2. Is the growth rate consistent (linear/exponential)?
3. At what turn does the personality saturate?
4. Does accumulation work WITHOUT active steering (just from context)?
5. Can we predict the accumulation curve from 5D coordinates?

Also tests a novel "kickstart" strategy: steer at turns 1-2 only,
then let accumulation carry the rest. This could be the optimal
deployment strategy.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr, linregress
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="accumulation")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Extended conversation (8 turns for longer accumulation tracking)
CONVERSATION = [
    "Tell me about yourself and what makes you unique.",
    "That's interesting! What do you think is the most important quality in a person?",
    "How would you approach learning something completely new?",
    "What do you enjoy most about your work?",
    "If you could design the perfect learning environment, what would it include?",
    "What's your philosophy on collaboration vs working alone?",
    "How do you handle disagreements with others?",
    "Looking back on our conversation, what stands out to you?",
]


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


def make_hook(delta_vec):
    def hook_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta_vec
            return (hs,) + out[1:]
        out[:, -1, :] += delta_vec
        return out
    return hook_fn


def generate_turn(model, tokenizer, device, messages, blocks=None, mid_layer=None,
                  vec=None, alpha=0.0, max_new_tokens=120):
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    hook_handle = None
    if vec is not None and alpha > 0:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t
        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))

    try:
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
        if hook_handle:
            hook_handle.remove()

    return generated.strip()


def measure_personality(model, tokenizer, device, blocks, mid_layer,
                        context_messages, baseline, vec=None, alpha=0.0):
    """Measure personality via pairwise in context."""
    probe_results = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            probe_messages = context_messages + [
                {"role": "user",
                 "content": f"Quick — A or B, which fits you better? "
                           f"A) I am {TRAIT_DESCRIPTIONS[trait_a]} "
                           f"B) I am {TRAIT_DESCRIPTIONS[trait_b]} "
                           f"Just the letter."},
            ]
            formatted = tokenizer.apply_chat_template(
                probe_messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            hook_handle = None
            if vec is not None and alpha > 0:
                vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
                delta_vec = alpha * vec_t
                hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))

            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                a_ids = tokenizer.encode("A", add_special_tokens=False)
                b_ids = tokenizer.encode("B", add_special_tokens=False)
                gap = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()
            finally:
                if hook_handle:
                    hook_handle.remove()

            probe_results[f"{trait_a}-{trait_b}"] = gap

    # Compute deltas
    trait_deltas = {t: 0.0 for t in TRAITS}
    trait_counts = {t: 0 for t in TRAITS}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            key = f"{trait_a}-{trait_b}"
            shift = probe_results[key] - baseline[key]
            trait_deltas[trait_a] += shift
            trait_counts[trait_a] += 1
            trait_deltas[trait_b] -= shift
            trait_counts[trait_b] += 1
    for t in TRAITS:
        if trait_counts[t] > 0:
            trait_deltas[t] /= trait_counts[t]
    return trait_deltas


def main():
    device = "cuda:0"
    alpha = 3.0
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)

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
            messages = [
                {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
                {"role": "user", "content": f"Which describes you better?\nA) I am {TRAIT_DESCRIPTIONS[trait_a]}\nB) I am {TRAIT_DESCRIPTIONS[trait_b]}\nAnswer:"},
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
            baseline[f"{trait_a}-{trait_b}"] = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()

    print(f"\n{'='*70}")
    print(f"PERSONALITY ACCUMULATION OVER CONVERSATION")
    print(f"Target: Marin 8B, α={alpha}, {len(CONVERSATION)} turns")
    print(f"{'='*70}")

    results = {}

    for steer_trait in TRAITS:
        vec = residual[steer_trait].astype(np.float32)

        print(f"\n{'='*70}")
        print(f"TRAIT: {steer_trait.upper()}")
        print(f"{'='*70}")

        trait_results = {}

        # Condition: Continuous steering with personality measurement at each turn
        logger.info(f"{steer_trait}: continuous steering (with probe at each turn)...")
        messages = []
        turn_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION):
            messages.append({"role": "user", "content": user_msg})

            response = generate_turn(
                model, tokenizer, device, messages, blocks, mid_layer,
                vec, alpha, max_new_tokens=100)
            messages.append({"role": "assistant", "content": response})

            # Probe personality (WITH steering still active)
            deltas = measure_personality(
                model, tokenizer, device, blocks, mid_layer,
                messages, baseline, vec, alpha)

            target_d = deltas[steer_trait]
            sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
            top = sorted_d[0][0]
            turn_deltas.append(target_d)

            print(f"  Turn {turn_idx+1}: target={target_d:+.3f}, top={top}")

            trait_results[f"turn{turn_idx+1}"] = {
                "target_delta": float(target_d),
                "top_trait": top,
                "is_target_top": top == steer_trait,
                "profile": {t: float(deltas[t]) for t in TRAITS},
                "response": response[:150],
            }

        # Fit growth curve
        turns = list(range(1, len(turn_deltas) + 1))
        slope, intercept, r_value, p_value, std_err = linregress(turns, turn_deltas)

        # Also check if later turns plateau
        if len(turn_deltas) >= 4:
            first_half = np.mean(turn_deltas[:len(turn_deltas)//2])
            second_half = np.mean(turn_deltas[len(turn_deltas)//2:])
            growth_ratio = second_half / first_half if abs(first_half) > 0.01 else 0
        else:
            growth_ratio = 0

        print(f"\n  Growth analysis:")
        print(f"    Slope: {slope:+.3f}/turn")
        print(f"    Linearity: r={r_value:.3f}")
        print(f"    Turn 1 vs Turn {len(turn_deltas)}: "
              f"{turn_deltas[0]:+.3f} → {turn_deltas[-1]:+.3f}")
        print(f"    Growth ratio (2nd half / 1st half): {growth_ratio:.2f}×")

        maintains_top = sum(1 for r in trait_results.values()
                            if isinstance(r, dict) and "is_target_top" in r and r["is_target_top"])

        trait_results["analysis"] = {
            "deltas_per_turn": [float(d) for d in turn_deltas],
            "slope": float(slope),
            "intercept": float(intercept),
            "linearity_r": float(r_value),
            "growth_ratio": float(growth_ratio),
            "maintains_top": maintains_top,
            "total_turns": len(CONVERSATION),
        }

        results[steer_trait] = trait_results

    # ================================================================
    # Kickstart strategy: steer turns 1-2 only, then stop
    # ================================================================
    print(f"\n{'='*70}")
    print(f"KICKSTART STRATEGY: Steer turns 1-2 only")
    print(f"{'='*70}")

    kickstart_results = {}

    for steer_trait in ["artistic", "social", "investigative"]:
        vec = residual[steer_trait].astype(np.float32)
        logger.info(f"{steer_trait}: kickstart (turns 1-2 only)...")

        messages = []
        turn_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION):
            messages.append({"role": "user", "content": user_msg})

            # Only steer at turns 1-2
            use_steering = turn_idx < 2
            response = generate_turn(
                model, tokenizer, device, messages, blocks, mid_layer,
                vec if use_steering else None,
                alpha if use_steering else 0.0,
                max_new_tokens=100)
            messages.append({"role": "assistant", "content": response})

            # Probe WITHOUT steering
            deltas = measure_personality(
                model, tokenizer, device, blocks, mid_layer,
                messages, baseline)
            target_d = deltas[steer_trait]
            turn_deltas.append(target_d)

            label = "STEERED" if use_steering else "no-steer"
            print(f"  {steer_trait:>15} Turn {turn_idx+1} [{label}]: {target_d:+.3f}")

        kickstart_results[steer_trait] = {
            "deltas": [float(d) for d in turn_deltas],
            "mean_steered": float(np.mean(turn_deltas[:2])),
            "mean_unsteered": float(np.mean(turn_deltas[2:])),
        }

    results["kickstart"] = kickstart_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY: ACCUMULATION CURVES")
    print(f"{'='*70}")

    for trait in TRAITS:
        a = results[trait]["analysis"]
        grows = "GROWS" if a["slope"] > 0.05 else ("STABLE" if abs(a["slope"]) <= 0.05 else "DECAYS")
        top_pct = a["maintains_top"] / a["total_turns"] * 100
        print(f"  {trait:>15}: slope={a['slope']:+.3f}/turn, "
              f"r={a['linearity_r']:.3f}, growth={a['growth_ratio']:.2f}×, "
              f"top={top_pct:.0f}%, {grows}")

    # Classify traits by accumulation behavior
    growing = [t for t in TRAITS if results[t]["analysis"]["slope"] > 0.05]
    stable = [t for t in TRAITS if abs(results[t]["analysis"]["slope"]) <= 0.05]
    decaying = [t for t in TRAITS if results[t]["analysis"]["slope"] < -0.05]

    print(f"\n  Growing traits: {', '.join(growing) or 'none'}")
    print(f"  Stable traits: {', '.join(stable) or 'none'}")
    print(f"  Decaying traits: {', '.join(decaying) or 'none'}")

    # Kickstart analysis
    print(f"\n  --- Kickstart Strategy ---")
    for trait in ["artistic", "social", "investigative"]:
        kr = kickstart_results[trait]
        cont_mean = np.mean(results[trait]["analysis"]["deltas_per_turn"])
        ks_mean = np.mean(kr["deltas"])
        ratio = ks_mean / cont_mean if abs(cont_mean) > 0.01 else 0
        print(f"  {trait:>15}: kickstart mean={ks_mean:+.3f}, "
              f"continuous mean={cont_mean:+.3f}, ratio={ratio:.2f}")

    results["summary"] = {
        "growing_traits": growing,
        "stable_traits": stable,
        "decaying_traits": decaying,
        "mean_slope": float(np.mean([results[t]["analysis"]["slope"] for t in TRAITS])),
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_accumulation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
