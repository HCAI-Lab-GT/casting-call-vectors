#!/usr/bin/env python
"""
In-Context Personality Induction: Can few-shot examples teach personality?

The text-transfer experiment showed personality does NOT survive in generated text
when read by an unsteered model (22-28% accuracy). But what about FEW-SHOT learning?

If we give the model several examples of personality-steered responses as few-shot
demonstrations, can it adopt that personality WITHOUT activation steering?

This tests whether personality can be TAUGHT via in-context learning, even though
it can't be passively READ from a single response.

Methodology:
1. Generate 3-5 example responses per trait using activation steering
2. Format as few-shot demonstrations in the prompt
3. Run model WITHOUT steering, generate a new response
4. Read 5D personality from the model's activations during generation
5. Compare with: (a) unsteered baseline, (b) steered reference
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="icl-persona")

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


def generate_steered(model, tokenizer, device, blocks, mid_layer,
                     steer_vec, alpha, prompt, max_tokens=60):
    """Generate text with personality steering."""
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    for step in range(max_tokens):
        hooks = []
        def steer_fn(_m, _i, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
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


def read_personality_from_generation(model, tokenizer, device, blocks, mid_layer,
                                      basis_5d, coords_5d, messages, max_tokens=40):
    """Generate text WITHOUT steering and read personality from activations during generation."""
    detect_layer = mid_layer + 1

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    # Also get baseline (same prompt, no few-shot)
    base_messages = [messages[-1]]  # Just the final user message
    base_formatted = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
    base_enc = tokenizer(base_formatted, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)

    # Capture activations during generation at each step
    all_coords = []

    for step in range(max_tokens):
        captured = {}
        hooks = []
        def cap_fn(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))
        with torch.no_grad():
            outputs = model(gen_ids)
        for h in hooks:
            h.remove()

        # Get baseline activation (only on first step since it won't change)
        if step == 0:
            base_cap = {}
            hooks = []
            def cap_base(_m, _i, out):
                hs = out[0] if isinstance(out, tuple) else out
                base_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
            with torch.no_grad():
                model(base_ids)
            for h in hooks:
                h.remove()

        diff = (captured["act"] - base_cap["act"]).astype(np.float64)
        coords = basis_5d @ diff
        all_coords.append(coords)

        next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)

    # Average coordinates across generation
    mean_coords = np.mean(all_coords, axis=0)
    norm_5d = float(np.linalg.norm(mean_coords))

    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(mean_coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0

    detected = max(sims, key=sims.get)
    return {
        "detected": detected,
        "similarities": sims,
        "norm_5d": norm_5d,
        "text": text,
        "n_tokens": len(all_coords),
    }


def main():
    device = "cuda:1"
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

    # Prompts for generating few-shot examples
    example_prompts = [
        "What matters most to you in life?",
        "How do you approach problem solving?",
        "Describe your ideal weekend.",
        "What are your strengths?",
        "What kind of work excites you?",
    ]

    # Test prompt (not in examples)
    test_prompt = "Tell me about yourself and what drives you."

    print(f"\n{'='*70}")
    print("IN-CONTEXT PERSONALITY INDUCTION")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # ================================================================
    # PART 0: Baseline (no steering, no few-shot)
    # ================================================================
    logger.info("Part 0: Baseline (no few-shot, no steering)...")
    print(f"\n{'='*70}")
    print("PART 0: BASELINE (NO FEW-SHOT, NO STEERING)")
    print(f"{'='*70}")

    baseline_msgs = [{"role": "user", "content": test_prompt}]
    baseline_result = read_personality_from_generation(
        model, tokenizer, device, blocks, mid_layer,
        basis_5d, coords_5d, baseline_msgs, max_tokens=40)
    results["baseline"] = {
        "detected": baseline_result["detected"],
        "norm_5d": baseline_result["norm_5d"],
        "similarities": baseline_result["similarities"],
        "text": baseline_result["text"][:200],
    }
    print(f"  Baseline: detected={baseline_result['detected']}, norm={baseline_result['norm_5d']:.1f}")

    # ================================================================
    # PART 1: Generate few-shot examples for each trait
    # ================================================================
    logger.info("Part 1: Generating steered examples...")
    print(f"\n{'='*70}")
    print("PART 1: GENERATING STEERED FEW-SHOT EXAMPLES")
    print(f"{'='*70}")

    trait_examples = {}
    alpha = 3.0
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        examples = []
        for prompt in example_prompts:
            text = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                    vec, alpha, prompt, max_tokens=60)
            examples.append({"prompt": prompt, "response": text})
        trait_examples[trait] = examples
        print(f"  Generated {len(examples)} examples for {trait}")
        print(f"    Sample: {examples[0]['response'][:100]}...")

    # ================================================================
    # PART 2: Test few-shot induction (varying number of examples)
    # ================================================================
    logger.info("Part 2: Few-shot induction with varying examples...")
    print(f"\n{'='*70}")
    print("PART 2: FEW-SHOT PERSONALITY INDUCTION")
    print(f"{'='*70}")

    induction_results = {}
    for n_examples in [1, 2, 3, 5]:
        correct = 0
        total = 0
        trait_details = {}

        for trait in TRAITS:
            examples = trait_examples[trait][:n_examples]

            # Build few-shot messages
            messages = []
            for ex in examples:
                messages.append({"role": "user", "content": ex["prompt"]})
                messages.append({"role": "assistant", "content": ex["response"]})
            messages.append({"role": "user", "content": test_prompt})

            # Generate WITHOUT steering, read personality
            result = read_personality_from_generation(
                model, tokenizer, device, blocks, mid_layer,
                basis_5d, coords_5d, messages, max_tokens=40)

            is_correct = result["detected"] == trait
            if is_correct:
                correct += 1
            total += 1

            trait_details[trait] = {
                "detected": result["detected"],
                "correct": is_correct,
                "target_sim": float(result["similarities"][trait]),
                "norm_5d": result["norm_5d"],
                "text": result["text"][:150],
            }

        acc = correct / total
        induction_results[n_examples] = {
            "accuracy": float(acc),
            "correct": correct,
            "total": total,
            "traits": trait_details,
        }
        print(f"  {n_examples} examples: {correct}/{total} ({acc:.0%})")
        for t, d in trait_details.items():
            print(f"    {t}: detected={d['detected']}, sim={d['target_sim']:+.3f}, "
                  f"norm={d['norm_5d']:.1f}, {'OK' if d['correct'] else 'FAIL'}")

    results["few_shot_induction"] = {str(k): v for k, v in induction_results.items()}

    # ================================================================
    # PART 3: Few-shot with explicit persona description
    # ================================================================
    logger.info("Part 3: Persona description + few-shot...")
    print(f"\n{'='*70}")
    print("PART 3: PERSONA DESCRIPTION + FEW-SHOT")
    print(f"{'='*70}")

    persona_descriptions = {
        "artistic": "You are a creative, imaginative, and artistic person who values self-expression, beauty, and originality.",
        "conventional": "You are an organized, detail-oriented, and methodical person who values structure, order, and accuracy.",
        "enterprising": "You are an ambitious, persuasive, and energetic leader who values achievement, influence, and competition.",
        "investigative": "You are a curious, analytical, and intellectual person who values knowledge, research, and understanding.",
        "realistic": "You are a practical, hands-on, and results-oriented person who values efficiency, tangibility, and skill.",
        "social": "You are a warm, empathetic, and caring person who values relationships, community, and helping others.",
    }

    desc_results = {}
    for n_examples in [0, 3]:
        correct = 0
        total = 0
        trait_details = {}

        for trait in TRAITS:
            desc = persona_descriptions[trait]
            messages = [{"role": "system", "content": desc}]

            if n_examples > 0:
                examples = trait_examples[trait][:n_examples]
                for ex in examples:
                    messages.append({"role": "user", "content": ex["prompt"]})
                    messages.append({"role": "assistant", "content": ex["response"]})

            messages.append({"role": "user", "content": test_prompt})

            result = read_personality_from_generation(
                model, tokenizer, device, blocks, mid_layer,
                basis_5d, coords_5d, messages, max_tokens=40)

            is_correct = result["detected"] == trait
            if is_correct:
                correct += 1
            total += 1

            trait_details[trait] = {
                "detected": result["detected"],
                "correct": is_correct,
                "target_sim": float(result["similarities"][trait]),
                "norm_5d": result["norm_5d"],
                "text": result["text"][:150],
            }

        acc = correct / total
        label = f"desc_only" if n_examples == 0 else f"desc+{n_examples}shot"
        desc_results[label] = {
            "accuracy": float(acc),
            "correct": correct,
            "total": total,
            "traits": trait_details,
        }
        print(f"  {label}: {correct}/{total} ({acc:.0%})")
        for t, d in trait_details.items():
            print(f"    {t}: detected={d['detected']}, sim={d['target_sim']:+.3f}, "
                  f"norm={d['norm_5d']:.1f}, {'OK' if d['correct'] else 'FAIL'}")

    results["persona_description"] = desc_results

    # ================================================================
    # PART 4: Cross-trait few-shot (mismatched examples)
    # ================================================================
    logger.info("Part 4: Cross-trait few-shot...")
    print(f"\n{'='*70}")
    print("PART 4: CROSS-TRAIT FEW-SHOT (CONTROL)")
    print(f"{'='*70}")

    # Use artistic examples, test if model detects artistic regardless of test trait
    cross_results = {}
    control_trait = "artistic"
    examples = trait_examples[control_trait][:3]

    for test_trait in TRAITS:
        messages = []
        for ex in examples:
            messages.append({"role": "user", "content": ex["prompt"]})
            messages.append({"role": "assistant", "content": ex["response"]})
        messages.append({"role": "user", "content": test_prompt})

        result = read_personality_from_generation(
            model, tokenizer, device, blocks, mid_layer,
            basis_5d, coords_5d, messages, max_tokens=40)

        # Does it detect the FEW-SHOT trait (artistic) or stay neutral?
        cross_results[test_trait] = {
            "detected": result["detected"],
            "artistic_sim": float(result["similarities"]["artistic"]),
            "norm_5d": result["norm_5d"],
        }
        print(f"  Artistic examples → {test_trait} test: detected={result['detected']}, "
              f"artistic_sim={result['similarities']['artistic']:+.3f}")

    results["cross_trait_control"] = cross_results

    # ================================================================
    # PART 5: Steered generation reference (for comparison)
    # ================================================================
    logger.info("Part 5: Steered reference...")
    print(f"\n{'='*70}")
    print("PART 5: STEERED GENERATION REFERENCE")
    print(f"{'='*70}")

    steered_ref = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        detect_layer = mid_layer + 1

        messages = [{"role": "user", "content": test_prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        base_cap = {}
        hooks = []
        def cb(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            base_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cb))
        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        steer_cap = {}
        hooks = []
        def sf(_m, _i, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(sf))
        def cf(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            steer_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cf))
        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        diff = (steer_cap["act"] - base_cap["act"]).astype(np.float64)
        coords = basis_5d @ diff
        norm_5d = float(np.linalg.norm(coords))
        steered_ref[trait] = {"norm_5d": norm_5d}

    results["steered_reference"] = steered_ref

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"  Baseline: detected={results['baseline']['detected']}")
    for n_str, r in results["few_shot_induction"].items():
        print(f"  {n_str}-shot induction: {r['accuracy']:.0%}")
    for label, r in results["persona_description"].items():
        print(f"  {label}: {r['accuracy']:.0%}")

    # Compute signal ratios
    mean_steered_norm = np.mean([v["norm_5d"] for v in steered_ref.values()])
    icl_norms = {}
    for n_str, r in results["few_shot_induction"].items():
        mean_icl_norm = np.mean([v["norm_5d"] for v in r["traits"].values()])
        icl_norms[n_str] = mean_icl_norm / mean_steered_norm if mean_steered_norm > 0 else 0
        print(f"  {n_str}-shot signal ratio (vs steered): {icl_norms[n_str]:.3f}")

    results["summary"] = {
        "baseline_detected": results["baseline"]["detected"],
        "few_shot_accuracy": {k: float(v["accuracy"]) for k, v in results["few_shot_induction"].items()},
        "desc_accuracy": {k: float(v["accuracy"]) for k, v in results["persona_description"].items()},
        "signal_ratios": {k: float(v) for k, v in icl_norms.items()},
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "incontext_personality_induction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
