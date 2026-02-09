#!/usr/bin/env python
"""
Per-Position Personality Analysis.

All prior experiments captured personality signal at the LAST token position only.
This experiment examines where personality lives across ALL token positions.

Questions:
1. Is personality concentrated at the last position, or distributed?
2. For system prompts, do the system prompt TOKENS carry the personality?
3. For activation steering (injected at last position), does it propagate to other positions?
4. Which positions are most informative for personality detection?

Captures activations at every token position at mid_layer+1, projects onto 5D basis.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="per-position")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

PERSONALITY_SYSTEM_PROMPTS = {
    "artistic": (
        "You are a deeply creative and artistic individual. You value self-expression, "
        "beauty, and originality above all else. You see the world through an aesthetic lens "
        "and are drawn to art, music, writing, and creative endeavors."
    ),
    "investigative": (
        "You are an analytical and scientific individual. You are driven by curiosity and "
        "the desire to understand how things work. You value knowledge, logic, and rational "
        "thinking. You prefer working independently on challenging puzzles."
    ),
    "social": (
        "You are a deeply caring and social individual. You value helping others, building "
        "relationships, and creating supportive communities. You believe in cooperation, "
        "empathy, and making the world better through human connection."
    ),
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


def capture_all_positions(model, tokenizer, device, blocks, layer_idx,
                           user_prompt, system_prompt=None,
                           steer_vec=None, alpha=0.0, steer_layer=None):
    """Capture activations at ALL token positions at a given layer."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Get token strings for position labeling
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    captured = {}
    hooks = []

    def cap_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["all"] = hs[0, :, :].detach().cpu().numpy().copy()  # [seq_len, hidden]
        return out
    hooks.append(blocks[layer_idx].register_forward_hook(cap_hook))

    if steer_vec is not None and alpha != 0 and steer_layer is not None:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[steer_layer].register_forward_hook(steer_fn))

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured["all"], tokens, formatted


def segment_positions(tokens, formatted, tokenizer):
    """Identify which positions correspond to system prompt, user message, etc."""
    # Simple heuristic: find the boundaries from chat template
    segments = {}
    n = len(tokens)

    # Try to identify system/user/assistant boundaries
    text = formatted
    sys_end = text.find("user")  # rough heuristic

    # For Llama-style models, look for special tokens
    segment_labels = []
    current_segment = "prefix"
    for i, tok in enumerate(tokens):
        tok_str = str(tok).lower()
        if "system" in tok_str:
            current_segment = "system"
        elif "user" in tok_str:
            current_segment = "user"
        elif "assistant" in tok_str:
            current_segment = "assistant"
        segment_labels.append(current_segment)

    return segment_labels


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
    capture_layer = mid_layer + 1

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("PER-POSITION PERSONALITY ANALYSIS")
    print(f"Model: Marin 8B, Layer: L{capture_layer}")
    print(f"{'='*70}")

    # ================================================================
    # Capture baseline (no system prompt, no steering)
    # ================================================================
    logger.info("Capturing baseline at all positions...")
    base_acts, base_tokens, base_fmt = capture_all_positions(
        model, tokenizer, device, blocks, capture_layer, detect_prompt)
    n_base = base_acts.shape[0]
    print(f"\n  Baseline: {n_base} tokens")

    # ================================================================
    # PART 1: Activation steering — per-position personality signal
    # ================================================================
    logger.info("Part 1: Activation steering per-position analysis...")
    print(f"\n{'='*70}")
    print("PART 1: ACTIVATION STEERING — PERSONALITY AT EACH POSITION")
    print(f"(artistic α=2, injected at L{mid_layer} last position)")
    print(f"{'='*70}")

    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0

    steer_acts, steer_tokens, steer_fmt = capture_all_positions(
        model, tokenizer, device, blocks, capture_layer, detect_prompt,
        steer_vec=vec, alpha=alpha, steer_layer=mid_layer)

    # Same prompt, so same tokenization
    assert steer_acts.shape[0] == base_acts.shape[0], \
        f"Token count mismatch: {steer_acts.shape[0]} vs {base_acts.shape[0]}"

    steer_pos_data = []
    print(f"\n  {'Pos':>5} {'Token':>20} {'Full Δ':>10} {'5D Δ':>10} {'Capture':>10} {'Detected':>15}")

    for pos in range(n_base):
        diff = (steer_acts[pos] - base_acts[pos]).astype(np.float64)
        full_norm = float(np.linalg.norm(diff))
        coords = basis_5d @ diff
        norm_5d = float(np.linalg.norm(coords))
        capture = norm_5d / full_norm if full_norm > 1e-6 else 0

        # Detect trait
        sims = {}
        for t in TRAITS:
            if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                sims[t] = float(np.dot(coords, coords_5d[t]) / (
                    norm_5d * np.linalg.norm(coords_5d[t])))
            else:
                sims[t] = 0
        best = max(sims, key=sims.get)

        tok_str = base_tokens[pos][:20]
        print(f"  {pos:>5} {tok_str:>20} {full_norm:>10.2f} {norm_5d:>10.2f} {capture:>10.3f} {best:>15}")

        steer_pos_data.append({
            "position": pos,
            "token": base_tokens[pos],
            "full_norm": full_norm,
            "5d_norm": norm_5d,
            "capture_ratio": capture,
            "detected_trait": best,
            "target_cosine": float(sims.get(test_trait, 0)),
        })

    results["steering_per_position"] = steer_pos_data

    # Summary
    last_pos = steer_pos_data[-1]
    other_pos = steer_pos_data[:-1]
    last_5d = last_pos["5d_norm"]
    mean_other_5d = np.mean([d["5d_norm"] for d in other_pos]) if other_pos else 0
    print(f"\n  Last position 5D norm: {last_5d:.2f}")
    print(f"  Mean other positions:  {mean_other_5d:.2f}")
    print(f"  Last/other ratio:      {last_5d/mean_other_5d:.1f}×" if mean_other_5d > 0 else "  Mean other = 0")

    # ================================================================
    # PART 2: System prompt — per-position personality signal
    # ================================================================
    logger.info("Part 2: System prompt per-position analysis...")
    print(f"\n{'='*70}")
    print("PART 2: SYSTEM PROMPT — PERSONALITY AT EACH POSITION")
    print(f"{'='*70}")

    sysp_pos_data = {}

    for sp_trait, sys_prompt in PERSONALITY_SYSTEM_PROMPTS.items():
        logger.info(f"  {sp_trait}...")

        # Need baseline WITH same tokenization structure
        # Capture with system prompt
        sp_acts, sp_tokens, sp_fmt = capture_all_positions(
            model, tokenizer, device, blocks, capture_layer, detect_prompt,
            system_prompt=sys_prompt)

        # Capture baseline with NEUTRAL system prompt (same length)
        neutral_prompt = "You are a helpful assistant."
        neutral_acts, neutral_tokens, neutral_fmt = capture_all_positions(
            model, tokenizer, device, blocks, capture_layer, detect_prompt,
            system_prompt=neutral_prompt)

        # These may have different lengths, so compute aligned diff
        n_sp = sp_acts.shape[0]
        n_neutral = neutral_acts.shape[0]

        # Segment the positions
        sp_segments = segment_positions(sp_tokens, sp_fmt, tokenizer)

        trait_data = []
        print(f"\n  {sp_trait} ({n_sp} tokens vs neutral {n_neutral} tokens):")

        # Analyze the system prompt tokens vs user tokens vs assistant tokens
        segment_stats = {}

        # For comparison, align from the end (user + assistant tokens should match)
        # But system prompt tokens differ. We'll analyze the full sequence.
        print(f"  {'Pos':>5} {'Segment':>10} {'Token':>20} {'Act Norm':>10}")

        # Just show per-position activation norms (not diff, since lengths differ)
        for pos in range(min(n_sp, 60)):  # Limit output
            tok_str = sp_tokens[pos][:20]
            act_norm = float(np.linalg.norm(sp_acts[pos]))
            seg = sp_segments[pos] if pos < len(sp_segments) else "unknown"

            if pos % 5 == 0 or pos < 10 or pos >= n_sp - 5:
                print(f"  {pos:>5} {seg:>10} {tok_str:>20} {act_norm:>10.2f}")

            if seg not in segment_stats:
                segment_stats[seg] = {"norms": [], "positions": []}
            segment_stats[seg]["norms"].append(act_norm)
            segment_stats[seg]["positions"].append(pos)

        # Now compare: system prompt positions vs user positions
        # Use the LAST few tokens (user prompt + assistant) which should be comparable
        # across different system prompts

        # Compare trailing positions with no-system-prompt baseline
        # Use as many trailing positions as the baseline has
        if n_sp >= n_base:
            # The last n_base tokens should roughly correspond
            trailing_sp = sp_acts[-n_base:]
            trailing_diff = (trailing_sp - base_acts).astype(np.float64)

            per_pos = []
            for pos in range(n_base):
                diff = trailing_diff[pos]
                full_norm = float(np.linalg.norm(diff))
                coords = basis_5d @ diff
                norm_5d = float(np.linalg.norm(coords))
                capture = norm_5d / full_norm if full_norm > 1e-6 else 0

                sims = {}
                for t in TRAITS:
                    if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                        sims[t] = float(np.dot(coords, coords_5d[t]) / (
                            norm_5d * np.linalg.norm(coords_5d[t])))
                    else:
                        sims[t] = 0
                best = max(sims, key=sims.get)

                per_pos.append({
                    "position": pos,
                    "offset_from_end": n_base - 1 - pos,
                    "token": base_tokens[pos],
                    "full_norm": full_norm,
                    "5d_norm": norm_5d,
                    "capture_ratio": capture,
                    "detected_trait": best,
                    "target_cosine": float(sims.get(sp_trait, 0)),
                })

            trait_data = per_pos

            # Print trailing analysis
            print(f"\n  Trailing {n_base} positions (vs no-system-prompt baseline):")
            print(f"  {'Pos':>5} {'Token':>20} {'Full Δ':>10} {'5D Δ':>10} {'Capture':>10} {'Detected':>15}")
            for d in per_pos:
                print(f"  {d['position']:>5} {d['token'][:20]:>20} {d['full_norm']:>10.2f} "
                      f"{d['5d_norm']:>10.2f} {d['capture_ratio']:>10.3f} {d['detected_trait']:>15}")

            # Summary by position
            norms_5d = [d["5d_norm"] for d in per_pos]
            last_5d = norms_5d[-1]
            mean_other = np.mean(norms_5d[:-1]) if len(norms_5d) > 1 else 0
            print(f"\n  Last position 5D: {last_5d:.2f}, mean others: {mean_other:.2f}")
            if mean_other > 0:
                print(f"  Last/other ratio: {last_5d/mean_other:.1f}×")

        sysp_pos_data[sp_trait] = trait_data

    results["sysprompt_per_position"] = sysp_pos_data

    # ================================================================
    # PART 3: Position-specific detection accuracy
    # ================================================================
    logger.info("Part 3: Position-specific detection accuracy...")
    print(f"\n{'='*70}")
    print("PART 3: WHICH POSITION IS BEST FOR PERSONALITY DETECTION?")
    print(f"{'='*70}")

    # For activation steering, test detection at each position
    print(f"\n  Activation steering (artistic α=2):")
    correct_at_pos = []
    for d in steer_pos_data:
        correct_at_pos.append(1 if d["detected_trait"] == test_trait else 0)

    correct_positions = [i for i, c in enumerate(correct_at_pos) if c == 1]
    print(f"  Correct at {len(correct_positions)}/{n_base} positions: {correct_positions}")

    # For system prompts
    for sp_trait, trait_data in sysp_pos_data.items():
        if trait_data:
            correct = [1 if d["detected_trait"] == sp_trait else 0 for d in trait_data]
            correct_pos = [d["position"] for d, c in zip(trait_data, correct) if c == 1]
            print(f"  {sp_trait}: correct at {len(correct_pos)}/{len(trait_data)} positions: {correct_pos}")

    results["detection_accuracy"] = {
        "steering": {
            "correct_positions": correct_positions,
            "total_positions": n_base,
            "accuracy": len(correct_positions) / n_base,
        }
    }
    for sp_trait, trait_data in sysp_pos_data.items():
        if trait_data:
            correct = [1 if d["detected_trait"] == sp_trait else 0 for d in trait_data]
            results["detection_accuracy"][sp_trait] = {
                "correct_positions": sum(correct),
                "total_positions": len(trait_data),
                "accuracy": sum(correct) / len(trait_data) if trait_data else 0,
            }

    # ================================================================
    # PART 4: Multi-position aggregation for detection
    # ================================================================
    logger.info("Part 4: Multi-position aggregation...")
    print(f"\n{'='*70}")
    print("PART 4: DOES AGGREGATING ACROSS POSITIONS IMPROVE DETECTION?")
    print(f"{'='*70}")

    # For steering: aggregate 5D coords across all positions
    all_steer_coords = []
    for d in steer_pos_data:
        diff = (steer_acts[d["position"]] - base_acts[d["position"]]).astype(np.float64)
        coords = basis_5d @ diff
        all_steer_coords.append(coords)

    # Mean across all positions
    mean_coords = np.mean(all_steer_coords, axis=0)
    mean_norm = float(np.linalg.norm(mean_coords))
    sims = {}
    for t in TRAITS:
        if mean_norm > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(mean_coords, coords_5d[t]) / (
                mean_norm * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0
    mean_detected = max(sims, key=sims.get)

    # Last-only
    last_coords = all_steer_coords[-1]
    last_norm = float(np.linalg.norm(last_coords))

    print(f"\n  Steering detection comparison:")
    print(f"    Last-only: 5D norm={last_norm:.2f}, detected={steer_pos_data[-1]['detected_trait']}")
    print(f"    All-mean:  5D norm={mean_norm:.2f}, detected={mean_detected}")
    print(f"    Last-only cos({test_trait})={steer_pos_data[-1]['target_cosine']:.3f}")
    print(f"    All-mean cos({test_trait})={sims.get(test_trait, 0):.3f}")

    results["aggregation"] = {
        "steering_last_norm": float(last_norm),
        "steering_mean_norm": float(mean_norm),
        "steering_last_detected": steer_pos_data[-1]["detected_trait"],
        "steering_mean_detected": mean_detected,
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Activation steering (last-position injection):")
    last_5d = steer_pos_data[-1]["5d_norm"]
    other_5ds = [d["5d_norm"] for d in steer_pos_data[:-1]]
    mean_other = np.mean(other_5ds) if other_5ds else 0
    print(f"    Last position 5D norm: {last_5d:.2f}")
    print(f"    Mean other positions:  {mean_other:.2f}")
    if mean_other > 0:
        print(f"    Concentration ratio:   {last_5d/mean_other:.1f}×")

    for sp_trait, trait_data in sysp_pos_data.items():
        if trait_data:
            norms = [d["5d_norm"] for d in trait_data]
            last_5d_sp = norms[-1]
            mean_other_sp = np.mean(norms[:-1]) if len(norms) > 1 else 0
            print(f"\n  System prompt ({sp_trait}):")
            print(f"    Last position 5D norm: {last_5d_sp:.2f}")
            print(f"    Mean other positions:  {mean_other_sp:.2f}")
            if mean_other_sp > 0:
                print(f"    Concentration ratio:   {last_5d_sp/mean_other_sp:.1f}×")

    results["summary"] = {
        "model": model_id,
        "capture_layer": capture_layer,
        "baseline_tokens": n_base,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_position_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
