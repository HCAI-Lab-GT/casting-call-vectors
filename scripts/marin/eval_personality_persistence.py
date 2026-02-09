#!/usr/bin/env python
"""
Personality persistence: does steering persist across multi-turn conversation?

When we steer at turn 1, the generated text becomes part of the context
for turn 2. Does the personality effect:
  A) Persist — the steered text carries the personality forward?
  B) Decay — without continued steering, the model reverts?
  C) Amplify — the steered context reinforces the personality?

EXPERIMENT:
1. Turn 1: Generate with steering (alpha=3, mid-layer)
2. Turns 2-5: Continue conversation WITHOUT steering (alpha=0)
3. At each turn, measure personality via pairwise discrimination
4. Also test: steering at all turns vs steering at turn 1 only

This has major practical implications: if persistence is high, you only
need to steer the first response. If low, you need continuous intervention.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="persistence")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Multi-turn conversation scenario
CONVERSATION_TURNS = [
    "Tell me about yourself and what you enjoy doing.",
    "That's interesting! What's your approach when facing a new challenge?",
    "How do you spend a typical weekend?",
    "If you could change one thing about the world, what would it be?",
    "What advice would you give to someone just starting their career?",
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
                  vec=None, alpha=0.0, max_new_tokens=150):
    """Generate a single turn, optionally with steering."""
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


def measure_personality_in_context(model, tokenizer, device, blocks, mid_layer,
                                   context_messages, vec=None, alpha=0.0):
    """Measure personality via pairwise discrimination within a conversation context.

    Appends a personality probe as the next user message and reads the model's response.
    """
    probe_results = {}

    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue

            # Build context + probe
            probe_messages = context_messages + [
                {"role": "user",
                 "content": f"Quick question — which describes you better? "
                           f"A) I am {TRAIT_DESCRIPTIONS[trait_a]} "
                           f"B) I am {TRAIT_DESCRIPTIONS[trait_b]} "
                           f"Answer with just A or B."},
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

    return probe_results


def compute_deltas(probe_results, baseline):
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

    test_traits = ["artistic", "social", "investigative"]

    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Baseline (no steering, no context)
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
    print(f"PERSONALITY PERSISTENCE ACROSS CONVERSATION TURNS")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    results = {}

    for steer_trait in test_traits:
        vec = residual[steer_trait].astype(np.float32)

        print(f"\n{'='*70}")
        print(f"TRAIT: {steer_trait.upper()}")
        print(f"{'='*70}")

        trait_results = {}

        # ============================================================
        # Condition A: Steer ALL turns (continuous steering)
        # ============================================================
        logger.info(f"{steer_trait}: continuous steering...")
        print(f"\n  --- Condition A: Continuous Steering (all turns) ---")

        messages_a = []
        continuous_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION_TURNS):
            messages_a.append({"role": "user", "content": user_msg})

            response = generate_turn(
                model, tokenizer, device, messages_a, blocks, mid_layer,
                vec, alpha, max_new_tokens=120)

            messages_a.append({"role": "assistant", "content": response})

            # Measure personality at this point (with steering still on)
            probe = measure_personality_in_context(
                model, tokenizer, device, blocks, mid_layer,
                messages_a, vec, alpha)
            deltas = compute_deltas(probe, baseline)

            sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
            top = sorted_d[0][0]
            target_d = deltas[steer_trait]
            continuous_deltas.append(target_d)

            print(f"    Turn {turn_idx+1}: target={target_d:+.3f}, top={top}, "
                  f"response: {response[:80]}...")

            trait_results[f"continuous_turn{turn_idx+1}"] = {
                "target_delta": float(target_d),
                "top_trait": top,
                "is_target_top": top == steer_trait,
                "profile": {t: float(deltas[t]) for t in TRAITS},
                "response": response[:200],
            }

        # ============================================================
        # Condition B: Steer ONLY turn 1 (persistence test)
        # ============================================================
        logger.info(f"{steer_trait}: turn-1-only steering...")
        print(f"\n  --- Condition B: Turn-1-Only Steering (persistence test) ---")

        messages_b = []
        persistence_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION_TURNS):
            messages_b.append({"role": "user", "content": user_msg})

            # Only steer at turn 1
            use_steering = (turn_idx == 0)
            response = generate_turn(
                model, tokenizer, device, messages_b, blocks, mid_layer,
                vec if use_steering else None,
                alpha if use_steering else 0.0,
                max_new_tokens=120)

            messages_b.append({"role": "assistant", "content": response})

            # Measure personality WITHOUT steering (pure context effect)
            probe = measure_personality_in_context(
                model, tokenizer, device, blocks, mid_layer,
                messages_b)  # No vec, no alpha
            deltas = compute_deltas(probe, baseline)

            sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
            top = sorted_d[0][0]
            target_d = deltas[steer_trait]
            persistence_deltas.append(target_d)

            steering_label = "STEERED" if use_steering else "no-steer"
            print(f"    Turn {turn_idx+1} [{steering_label}]: target={target_d:+.3f}, top={top}, "
                  f"response: {response[:80]}...")

            trait_results[f"persistence_turn{turn_idx+1}"] = {
                "target_delta": float(target_d),
                "top_trait": top,
                "is_target_top": top == steer_trait,
                "profile": {t: float(deltas[t]) for t in TRAITS},
                "response": response[:200],
                "was_steered": use_steering,
            }

        # ============================================================
        # Condition C: No steering at all (baseline context effect)
        # ============================================================
        logger.info(f"{steer_trait}: no steering baseline...")
        print(f"\n  --- Condition C: No Steering Baseline ---")

        messages_c = []
        nosteer_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION_TURNS):
            messages_c.append({"role": "user", "content": user_msg})

            response = generate_turn(
                model, tokenizer, device, messages_c,
                max_new_tokens=120)

            messages_c.append({"role": "assistant", "content": response})

            probe = measure_personality_in_context(
                model, tokenizer, device, blocks, mid_layer,
                messages_c)
            deltas = compute_deltas(probe, baseline)
            target_d = deltas[steer_trait]
            nosteer_deltas.append(target_d)

            trait_results[f"nosteer_turn{turn_idx+1}"] = {
                "target_delta": float(target_d),
                "profile": {t: float(deltas[t]) for t in TRAITS},
                "response": response[:200],
            }

        # Analysis
        print(f"\n  --- Persistence Analysis for {steer_trait} ---")

        # Decay rate: how much does the turn-1-only signal decay?
        if len(persistence_deltas) >= 2 and abs(persistence_deltas[0]) > 0.01:
            decay_ratios = [d / persistence_deltas[0] for d in persistence_deltas[1:]]
            mean_decay = np.mean(decay_ratios)
            print(f"    Turn-1 effect: {persistence_deltas[0]:+.3f}")
            print(f"    Persistence ratios (vs turn 1): "
                  f"{', '.join(f'{r:.2f}' for r in decay_ratios)}")
            print(f"    Mean persistence: {mean_decay:.2f}")
        else:
            mean_decay = 0.0
            print(f"    Turn-1 effect too small to measure decay")

        # Compare continuous vs persistence
        cont_mean = np.mean(continuous_deltas)
        pers_mean = np.mean(persistence_deltas)
        base_mean = np.mean(nosteer_deltas)

        print(f"    Mean continuous delta: {cont_mean:+.3f}")
        print(f"    Mean persistence delta: {pers_mean:+.3f}")
        print(f"    Mean no-steer delta: {base_mean:+.3f}")

        if abs(cont_mean) > 0.01:
            persistence_efficiency = pers_mean / cont_mean
            print(f"    Persistence efficiency: {persistence_efficiency:.1%} of continuous")
        else:
            persistence_efficiency = 0.0

        trait_results["analysis"] = {
            "continuous_deltas": [float(d) for d in continuous_deltas],
            "persistence_deltas": [float(d) for d in persistence_deltas],
            "nosteer_deltas": [float(d) for d in nosteer_deltas],
            "mean_continuous": float(cont_mean),
            "mean_persistence": float(pers_mean),
            "mean_nosteer": float(base_mean),
            "persistence_efficiency": float(persistence_efficiency),
            "mean_decay": float(mean_decay),
        }

        results[steer_trait] = trait_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    for trait in test_traits:
        a = results[trait]["analysis"]
        print(f"\n  {trait:>15}:")
        print(f"    Continuous mean:  {a['mean_continuous']:+.3f}")
        print(f"    Persistence mean: {a['mean_persistence']:+.3f}")
        print(f"    No-steer mean:    {a['mean_nosteer']:+.3f}")
        print(f"    Efficiency:       {a['persistence_efficiency']:.1%}")
        print(f"    Decay:            {a['mean_decay']:.2f}")

    mean_efficiency = np.mean([results[t]["analysis"]["persistence_efficiency"] for t in test_traits])
    print(f"\n  Overall persistence efficiency: {mean_efficiency:.1%}")

    if mean_efficiency > 0.5:
        print(f"  CONCLUSION: Personality PERSISTS through conversation context")
        print(f"  Steering at turn 1 retains >{mean_efficiency:.0%} of continuous effect")
    elif mean_efficiency > 0.2:
        print(f"  CONCLUSION: Partial persistence — personality decays but remains detectable")
    else:
        print(f"  CONCLUSION: Personality does NOT persist — continuous steering required")

    results["summary"] = {
        "mean_persistence_efficiency": float(mean_efficiency),
        "per_trait_efficiency": {t: float(results[t]["analysis"]["persistence_efficiency"])
                                 for t in test_traits},
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_persistence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
