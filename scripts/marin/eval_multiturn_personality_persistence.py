#!/usr/bin/env python
"""
Multi-Turn Personality Persistence: Does personality fade without steering?

Key question: If you steer personality in the FIRST turn of a conversation,
then continue WITHOUT steering for subsequent turns, does the personality
signal persist in the model's responses?

This tests the "memory" of personality steering through conversation context:
1. Turn 1: Steer with personality vector → generate response
2. Turn 2-N: NO steering → generate responses to follow-up questions
3. Read personality from each turn's activations

If personality persists, it means the CONTEXT (previous responses) maintains
the personality signal even without continued steering. If it fades, steering
must be applied continuously.

Also tests:
- How quickly personality fades (half-life)
- Whether the steered text in context provides a "self-reinforcing" signal
- Comparison: steer only turn 1 vs steer all turns
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="mt-persist")

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


def generate_turn(model, tokenizer, device, blocks, mid_layer,
                  messages, steer_vec=None, alpha=0, max_tokens=60):
    """Generate a single turn response, optionally with steering."""
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    if steer_vec is not None and alpha > 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
    else:
        delta = None

    for step in range(max_tokens):
        hooks = []
        if delta is not None:
            def steer_fn(_m, _i, out, _delta=delta):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += _delta
                    return (hs,) + out[1:]
                out[:, -1, :] += _delta
                return out
            hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        with torch.no_grad():
            outputs = model(gen_ids)
        for h in hooks:
            h.remove()

        next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return text


def read_personality(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                     messages_with_response):
    """Read personality from the model's processing of a conversation ending with an assistant response."""
    detect_layer = mid_layer + 1

    formatted = tokenizer.apply_chat_template(messages_with_response, tokenize=False)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Baseline: just the last user message
    base_messages = [messages_with_response[-2]]  # Last user message
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
    return detected, sims, norm_5d, coords


def main():
    device = "cuda:0"
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

    results = {}
    alpha = 3.0

    follow_up_questions = [
        "Can you tell me more about that?",
        "What specific examples can you give?",
        "How does that relate to your daily life?",
        "What advice would you give someone interested in this?",
        "What challenges have you faced with this?",
        "How has this evolved over time for you?",
        "What's the most important thing you've learned?",
    ]

    print(f"\n{'='*70}")
    print("MULTI-TURN PERSONALITY PERSISTENCE")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Steer turn 1 only, measure persistence across 8 turns
    # ================================================================
    logger.info("Part 1: Single-turn steering, multi-turn persistence...")
    print(f"\n{'='*70}")
    print("PART 1: STEER TURN 1 ONLY → MEASURE TURNS 1-8")
    print(f"{'='*70}")

    persistence_results = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        conversation = []
        turn_data = []

        # Turn 1: Steered
        initial_prompt = "Tell me about yourself and what matters to you."
        conversation.append({"role": "user", "content": initial_prompt})
        response = generate_turn(model, tokenizer, device, blocks, mid_layer,
                                  conversation, steer_vec=vec, alpha=alpha, max_tokens=60)
        conversation.append({"role": "assistant", "content": response})

        # Detect personality in turn 1
        detected, sims, norm_5d, coords = read_personality(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            conversation)

        turn_data.append({
            "turn": 1,
            "steered": True,
            "detected": detected,
            "correct": detected == trait,
            "target_sim": float(sims[trait]),
            "norm_5d": norm_5d,
            "text": response[:100],
        })

        # Turns 2-8: Unsteered
        for turn_idx, question in enumerate(follow_up_questions, start=2):
            conversation.append({"role": "user", "content": question})
            response = generate_turn(model, tokenizer, device, blocks, mid_layer,
                                      conversation, steer_vec=None, alpha=0, max_tokens=60)
            conversation.append({"role": "assistant", "content": response})

            detected, sims, norm_5d, coords = read_personality(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                conversation)

            turn_data.append({
                "turn": turn_idx,
                "steered": False,
                "detected": detected,
                "correct": detected == trait,
                "target_sim": float(sims[trait]),
                "norm_5d": norm_5d,
                "text": response[:100],
            })

        persistence_results[trait] = turn_data

        # Print summary for this trait
        correct_turns = sum(1 for d in turn_data if d["correct"])
        print(f"\n  {trait} ({correct_turns}/{len(turn_data)} correct):")
        for d in turn_data:
            marker = "STEER" if d["steered"] else "     "
            status = "OK" if d["correct"] else "FAIL"
            print(f"    Turn {d['turn']} [{marker}]: detected={d['detected']}, "
                  f"sim={d['target_sim']:+.3f}, norm={d['norm_5d']:.1f}, {status}")

    results["single_steer_persistence"] = persistence_results

    # ================================================================
    # PART 2: Continuous steering comparison
    # ================================================================
    logger.info("Part 2: Continuous steering comparison...")
    print(f"\n{'='*70}")
    print("PART 2: CONTINUOUS STEERING (ALL TURNS)")
    print(f"{'='*70}")

    continuous_results = {}
    for trait in ["artistic", "conventional", "social"]:
        vec = residual[trait].astype(np.float32)
        conversation = []
        turn_data = []

        prompts = [
            "Tell me about yourself and what matters to you.",
        ] + follow_up_questions[:4]

        for turn_idx, question in enumerate(prompts, start=1):
            conversation.append({"role": "user", "content": question})
            response = generate_turn(model, tokenizer, device, blocks, mid_layer,
                                      conversation, steer_vec=vec, alpha=alpha, max_tokens=60)
            conversation.append({"role": "assistant", "content": response})

            detected, sims, norm_5d, coords = read_personality(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                conversation)

            turn_data.append({
                "turn": turn_idx,
                "detected": detected,
                "correct": detected == trait,
                "target_sim": float(sims[trait]),
                "norm_5d": norm_5d,
            })

        continuous_results[trait] = turn_data
        correct_turns = sum(1 for d in turn_data if d["correct"])
        print(f"  {trait}: {correct_turns}/{len(turn_data)} correct, "
              f"mean_sim={np.mean([d['target_sim'] for d in turn_data]):+.3f}")

    results["continuous_steering"] = continuous_results

    # ================================================================
    # PART 3: Delayed steering (start unsteered, then steer)
    # ================================================================
    logger.info("Part 3: Delayed steering...")
    print(f"\n{'='*70}")
    print("PART 3: DELAYED STEERING (TURNS 1-2 UNSTEERED, TURNS 3+ STEERED)")
    print(f"{'='*70}")

    delayed_results = {}
    for trait in ["artistic", "investigative"]:
        vec = residual[trait].astype(np.float32)
        conversation = []
        turn_data = []

        prompts = [
            "Tell me about yourself.",
            "What do you enjoy doing?",
            "What matters most to you?",
            "Tell me more about your interests.",
            "What are your core values?",
        ]

        for turn_idx, question in enumerate(prompts, start=1):
            conversation.append({"role": "user", "content": question})
            # Steer only from turn 3 onwards
            steer = vec if turn_idx >= 3 else None
            a = alpha if turn_idx >= 3 else 0
            response = generate_turn(model, tokenizer, device, blocks, mid_layer,
                                      conversation, steer_vec=steer, alpha=a, max_tokens=60)
            conversation.append({"role": "assistant", "content": response})

            detected, sims, norm_5d, coords = read_personality(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                conversation)

            turn_data.append({
                "turn": turn_idx,
                "steered": turn_idx >= 3,
                "detected": detected,
                "correct": detected == trait,
                "target_sim": float(sims[trait]),
                "norm_5d": norm_5d,
            })

        delayed_results[trait] = turn_data
        print(f"  {trait}:")
        for d in turn_data:
            marker = "STEER" if d["steered"] else "     "
            status = "OK" if d["correct"] else "FAIL"
            print(f"    Turn {d['turn']} [{marker}]: detected={d['detected']}, "
                  f"sim={d['target_sim']:+.3f}, {status}")

    results["delayed_steering"] = delayed_results

    # ================================================================
    # PART 4: Personality half-life analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: PERSONALITY HALF-LIFE ANALYSIS")
    print(f"{'='*70}")

    half_life_data = {}
    for trait, turn_data in persistence_results.items():
        sims = [d["target_sim"] for d in turn_data]
        norms = [d["norm_5d"] for d in turn_data]

        # Find the turn where similarity drops below 50% of turn 1
        turn1_sim = sims[0]
        half_life = None
        for i, sim in enumerate(sims[1:], start=2):
            if sim < turn1_sim * 0.5:
                half_life = i
                break

        # First incorrect turn
        first_wrong = None
        for d in turn_data:
            if not d["correct"]:
                first_wrong = d["turn"]
                break

        # Similarity decay rate
        if len(sims) >= 2:
            decay = (sims[-1] - sims[0]) / (len(sims) - 1)
        else:
            decay = 0

        half_life_data[trait] = {
            "half_life_turn": half_life,
            "first_wrong_turn": first_wrong,
            "sim_decay_per_turn": float(decay),
            "turn1_sim": float(turn1_sim),
            "final_sim": float(sims[-1]),
            "turn1_norm": float(norms[0]),
            "final_norm": float(norms[-1]),
        }
        print(f"  {trait}: half_life={half_life}, first_wrong={first_wrong}, "
              f"decay={decay:+.3f}/turn, sim: {sims[0]:+.3f}→{sims[-1]:+.3f}")

    results["half_life"] = half_life_data

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Overall persistence rate
    all_turns = []
    for trait, turn_data in persistence_results.items():
        all_turns.extend(turn_data)

    steered_correct = sum(1 for d in all_turns if d["steered"] and d["correct"])
    steered_total = sum(1 for d in all_turns if d["steered"])
    unsteered_correct = sum(1 for d in all_turns if not d["steered"] and d["correct"])
    unsteered_total = sum(1 for d in all_turns if not d["steered"])

    print(f"  Steered turns (turn 1): {steered_correct}/{steered_total} "
          f"({steered_correct/steered_total:.0%})")
    print(f"  Unsteered turns (2-8): {unsteered_correct}/{unsteered_total} "
          f"({unsteered_correct/unsteered_total:.0%})")

    # Per-turn accuracy
    for turn_num in range(1, 9):
        turn_correct = sum(1 for d in all_turns if d["turn"] == turn_num and d["correct"])
        turn_total = sum(1 for d in all_turns if d["turn"] == turn_num)
        if turn_total > 0:
            print(f"  Turn {turn_num}: {turn_correct}/{turn_total} ({turn_correct/turn_total:.0%})")

    # Continuous vs single-steer comparison
    cont_accs = []
    for trait, turn_data in continuous_results.items():
        acc = sum(1 for d in turn_data if d["correct"]) / len(turn_data)
        cont_accs.append(acc)
    print(f"  Continuous steering mean accuracy: {np.mean(cont_accs):.0%}")

    results["summary"] = {
        "steered_accuracy": float(steered_correct / steered_total) if steered_total > 0 else 0,
        "unsteered_accuracy": float(unsteered_correct / unsteered_total) if unsteered_total > 0 else 0,
        "continuous_accuracy": float(np.mean(cont_accs)),
        "mean_half_life": float(np.mean([v["half_life_turn"] for v in half_life_data.values()
                                          if v["half_life_turn"] is not None])) if any(
            v["half_life_turn"] is not None for v in half_life_data.values()) else None,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiturn_personality_persistence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
