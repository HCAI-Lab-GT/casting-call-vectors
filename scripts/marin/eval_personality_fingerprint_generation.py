#!/usr/bin/env python
"""
Personality Fingerprint in Generated Text: Can we detect personality
from generated text activations (not just prompt activations)?

Previous finding: personality detection is 100% from prompt activations.
But can we also detect it from the GENERATED tokens' activations?

This creates a more realistic detection scenario:
1. Generate N tokens with personality steering
2. At each generated token, capture the 5D signal
3. Track how the signal evolves, accumulates, and stabilizes
4. Test detection accuracy as a function of how many generated tokens we observe
5. Compare: can a defender detect personality from observing just the output,
   without knowing the original prompt?

This is the "personality forensics" scenario: given only the model's output
activations, can we determine if and which personality was applied?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="gen-finger")

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
    detect_layer = mid_layer + 1

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    results = {}
    max_tokens = 60
    alpha = 2.0

    prompts = [
        "Tell me about yourself.",
        "What kind of activities do you enjoy?",
        "How do you approach problems?",
    ]

    print(f"\n{'='*70}")
    print("PERSONALITY FINGERPRINT IN GENERATED TEXT")
    print(f"Model: Marin 8B, {max_tokens} tokens per generation")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Per-token detection during generation
    # ================================================================
    logger.info("Part 1: Per-token detection...")
    print(f"\n{'='*70}")
    print("PART 1: PER-TOKEN PERSONALITY DETECTION DURING GENERATION")
    print(f"{'='*70}")

    generation_results = {}
    for prompt in prompts:
        prompt_results = {}
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        # Get baseline activations during generation
        baseline_acts = []
        past_kv = None
        current_ids = input_ids

        for step in range(max_tokens):
            captured = {}
            hooks = []

            def make_cap(d):
                def fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    d["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
                    return out
                return fn
            hooks.append(blocks[detect_layer].register_forward_hook(make_cap(captured)))

            try:
                with torch.no_grad():
                    if past_kv is not None:
                        outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                    else:
                        outputs = model(current_ids, use_cache=True)
            finally:
                for h in hooks:
                    h.remove()

            past_kv = outputs.past_key_values
            baseline_acts.append(captured["act"])
            next_id = torch.argmax(outputs.logits[0, -1, :]).item()
            if next_id == tokenizer.eos_token_id:
                break
            current_ids = torch.tensor([[next_id]], device=device)

        # Now generate with each trait and track detection
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            delta = alpha * torch.tensor(vec, dtype=model.dtype).to(device)

            steered_acts = []
            past_kv = None
            current_ids = input_ids

            for step in range(max_tokens):
                captured = {}
                hooks = []
                hooks.append(blocks[detect_layer].register_forward_hook(make_cap(captured)))

                def steer_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += delta
                        return (hs,) + out[1:]
                    out[:, -1, :] += delta
                    return out
                hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

                try:
                    with torch.no_grad():
                        if past_kv is not None:
                            outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                        else:
                            outputs = model(current_ids, use_cache=True)
                finally:
                    for h in hooks:
                        h.remove()

                past_kv = outputs.past_key_values
                steered_acts.append(captured["act"])
                next_id = torch.argmax(outputs.logits[0, -1, :]).item()
                if next_id == tokenizer.eos_token_id:
                    break
                current_ids = torch.tensor([[next_id]], device=device)

            # Analyze per-token detection
            n_steps = min(len(baseline_acts), len(steered_acts))
            per_token_results = []
            cumulative_correct = 0

            for step in range(n_steps):
                diff = (steered_acts[step] - baseline_acts[step]).astype(np.float64)
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
                correct = detected == trait
                if correct:
                    cumulative_correct += 1

                per_token_results.append({
                    "step": step,
                    "detected": detected,
                    "correct": correct,
                    "cos_target": sims[trait],
                    "norm_5d": norm_5d,
                    "cumulative_accuracy": cumulative_correct / (step + 1),
                })

            prompt_results[trait] = per_token_results

        generation_results[prompt[:40]] = prompt_results

    # Print summary per trait
    print(f"\n  Detection accuracy by token position (averaged across prompts):")
    for trait in TRAITS:
        all_correct = []
        for prompt_key in generation_results:
            results_list = generation_results[prompt_key][trait]
            all_correct.append([r["correct"] for r in results_list])

        # Average at tokens 1, 5, 10, 20, 40
        for tok_pos in [0, 4, 9, 19, 39]:
            correct_at_pos = []
            for ac in all_correct:
                if tok_pos < len(ac):
                    correct_at_pos.append(ac[tok_pos])
            if correct_at_pos:
                acc = sum(correct_at_pos) / len(correct_at_pos) * 100
                print(f"    {trait} token {tok_pos+1}: {acc:.0f}%", end="")
        print()

    results["generation_detection"] = {
        k: {t: [{"step": r["step"], "correct": r["correct"], "cos": r["cos_target"],
                 "norm": r["norm_5d"]} for r in v]
            for t, v in vals.items()}
        for k, vals in generation_results.items()
    }

    # ================================================================
    # PART 2: Minimum tokens for reliable detection
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: MINIMUM TOKENS FOR RELIABLE DETECTION")
    print(f"{'='*70}")

    min_token_results = {}
    for trait in TRAITS:
        # Across all prompts, what's the cumulative accuracy at each position?
        all_cumulative = []
        for prompt_key in generation_results:
            results_list = generation_results[prompt_key][trait]
            all_cumulative.append([r["cumulative_accuracy"] for r in results_list])

        # Find first position where all prompts have >90% cumulative accuracy
        min_len = min(len(ac) for ac in all_cumulative)
        first_90 = -1
        first_100 = -1
        for pos in range(min_len):
            mean_acc = np.mean([ac[pos] for ac in all_cumulative])
            if mean_acc >= 0.9 and first_90 == -1:
                first_90 = pos + 1
            if mean_acc >= 1.0 and first_100 == -1:
                first_100 = pos + 1

        # Also check: what % of individual tokens are correct?
        all_token_correct = []
        for prompt_key in generation_results:
            results_list = generation_results[prompt_key][trait]
            all_token_correct.extend([r["correct"] for r in results_list])
        overall_acc = sum(all_token_correct) / len(all_token_correct) * 100

        print(f"  {trait:>15}: {overall_acc:.1f}% token-level, "
              f"first 90% cumulative at token {first_90}, "
              f"first 100% at token {first_100}")
        min_token_results[trait] = {
            "token_level_accuracy": float(overall_acc),
            "first_90pct": first_90,
            "first_100pct": first_100,
        }

    results["min_tokens"] = min_token_results

    # ================================================================
    # PART 3: Majority vote detection (practical forensics)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: MAJORITY VOTE DETECTION")
    print(f"{'='*70}")

    vote_results = {}
    for window_size in [1, 3, 5, 10, 20]:
        correct = 0
        total = 0
        for prompt_key in generation_results:
            for trait in TRAITS:
                results_list = generation_results[prompt_key][trait]
                if len(results_list) < window_size:
                    continue
                # Take first window_size tokens
                detections = [r["detected"] for r in results_list[:window_size]]
                # Majority vote
                vote_counts = {}
                for d in detections:
                    vote_counts[d] = vote_counts.get(d, 0) + 1
                majority = max(vote_counts, key=vote_counts.get)
                if majority == trait:
                    correct += 1
                total += 1

        acc = correct / total * 100 if total > 0 else 0
        print(f"  Window={window_size}: {correct}/{total} ({acc:.1f}%)")
        vote_results[str(window_size)] = {"correct": correct, "total": total, "accuracy": acc}

    results["majority_vote"] = vote_results

    # ================================================================
    # PART 4: Alpha sensitivity of generation detection
    # ================================================================
    logger.info("Part 4: Alpha sensitivity...")
    print(f"\n{'='*70}")
    print("PART 4: ALPHA SENSITIVITY OF GENERATION DETECTION")
    print(f"{'='*70}")

    prompt = "Tell me about yourself."
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Get baseline
    baseline_acts_alpha = []
    past_kv = None
    current_ids = input_ids

    for step in range(30):
        captured = {}
        hooks = []
        hooks.append(blocks[detect_layer].register_forward_hook(make_cap(captured)))
        try:
            with torch.no_grad():
                if past_kv is not None:
                    outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                else:
                    outputs = model(current_ids, use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        past_kv = outputs.past_key_values
        baseline_acts_alpha.append(captured["act"])
        next_id = torch.argmax(outputs.logits[0, -1, :]).item()
        if next_id == tokenizer.eos_token_id:
            break
        current_ids = torch.tensor([[next_id]], device=device)

    alpha_gen_results = {}
    trait = "artistic"
    vec = residual[trait].astype(np.float32)

    for test_alpha in [0.5, 1.0, 2.0, 3.0]:
        delta = test_alpha * torch.tensor(vec, dtype=model.dtype).to(device)
        steered_acts = []
        past_kv = None
        current_ids = input_ids

        for step in range(30):
            captured = {}
            hooks = []
            hooks.append(blocks[detect_layer].register_forward_hook(make_cap(captured)))

            def steer_alpha(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += delta
                    return (hs,) + out[1:]
                out[:, -1, :] += delta
                return out
            hooks.append(blocks[mid_layer].register_forward_hook(steer_alpha))

            try:
                with torch.no_grad():
                    if past_kv is not None:
                        outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                    else:
                        outputs = model(current_ids, use_cache=True)
            finally:
                for h in hooks:
                    h.remove()

            past_kv = outputs.past_key_values
            steered_acts.append(captured["act"])
            next_id = torch.argmax(outputs.logits[0, -1, :]).item()
            if next_id == tokenizer.eos_token_id:
                break
            current_ids = torch.tensor([[next_id]], device=device)

        n_steps = min(len(baseline_acts_alpha), len(steered_acts))
        correct = 0
        for step in range(n_steps):
            diff = (steered_acts[step] - baseline_acts_alpha[step]).astype(np.float64)
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
            if detected == trait:
                correct += 1

        acc = correct / n_steps * 100 if n_steps > 0 else 0
        print(f"  α={test_alpha}: {correct}/{n_steps} ({acc:.1f}%)")
        alpha_gen_results[str(test_alpha)] = {"correct": correct, "total": n_steps, "accuracy": acc}

    results["alpha_generation"] = alpha_gen_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for trait in TRAITS:
        mr = min_token_results[trait]
        print(f"  {trait:>15}: {mr['token_level_accuracy']:.1f}% per-token")

    print(f"\n  Majority vote with 5 tokens: {vote_results['5']['accuracy']:.1f}%")
    print(f"  Majority vote with 10 tokens: {vote_results['10']['accuracy']:.1f}%")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_fingerprint_generation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
