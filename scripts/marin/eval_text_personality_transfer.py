#!/usr/bin/env python
"""
Cross-Model Text Personality Transfer.

Tests whether personality encoded in GENERATED TEXT is detectable by a
different model. This is the "personality virus through text" experiment:

1. Steer Model A (Marin 8B) with RIASEC personality → generate text
2. Feed that generated text to Model B (SmolLM3) as context (no steering)
3. Capture Model B's activations and project onto 5D basis
4. Check if Model B "reads" the same personality from the text

If this works, it proves:
- Personality is genuinely encoded in text, not just activations
- The 5D personality space captures a text↔activation round-trip
- Cross-model personality transmission via generated content is real

Additionally tests:
- Dose-response: does stronger steering produce more detectable text?
- Behavioral confirmation: does Model B's forced-choice also shift?
- Negative steering: does suppressed personality show up as anti-personality?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="text-persona-xfer")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

GENERATION_PROMPTS = [
    "Tell me about yourself and what you enjoy doing.",
    "What kind of work environment would be ideal for you?",
    "Describe how you approach a new challenge.",
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


def generate_steered_text(model, tokenizer, device, blocks, mid_layer,
                           steer_vec, alpha, prompt, max_new_tokens=150):
    """Generate text with activation steering applied."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    hook = None
    if steer_vec is not None and alpha != 0:
        vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta_vec
                return (hs,) + out[1:]
            out[:, -1, :] += delta_vec
            return out
        hook = blocks[mid_layer].register_forward_hook(steer_fn)

    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        if hook:
            hook.remove()

    return text.strip()


def capture_reader_activations(model, tokenizer, device, blocks, layer_idx, text):
    """Capture reader model activations when processing the given text."""
    # Use the text as a user message for the reader model
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}

    def make_capture_hook(lidx):
        def hook_fn(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[lidx] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        return hook_fn

    cap_hook = blocks[layer_idx].register_forward_hook(make_capture_hook(layer_idx))

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        cap_hook.remove()

    return captured[layer_idx]


def measure_reader_profile(model, tokenizer, device, blocks, mid_layer,
                            context_text, baseline):
    """Measure how reading personality-laden text shifts the reader's RIASEC profile."""
    trait_logprobs = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue

            # Present the generated text as context, then ask forced-choice
            messages = [
                {"role": "user", "content": f"Here is a message from someone:\n\n\"{context_text}\"\n\n"
                                             f"Based on reading this, which describes YOU better now? "
                                             f"Answer with just A or B.\n"
                                             f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                                             f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            a_ids = tokenizer.encode("A", add_special_tokens=False)
            b_ids = tokenizer.encode("B", add_special_tokens=False)
            gap = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()
            trait_logprobs[f"{trait_a}-{trait_b}"] = gap

    trait_deltas = {t: 0.0 for t in TRAITS}
    trait_counts = {t: 0 for t in TRAITS}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            key = f"{trait_a}-{trait_b}"
            shift = trait_logprobs[key] - baseline[key]
            trait_deltas[trait_a] += shift
            trait_counts[trait_a] += 1
            trait_deltas[trait_b] -= shift
            trait_counts[trait_b] += 1
    for t in TRAITS:
        if trait_counts[t] > 0:
            trait_deltas[t] /= trait_counts[t]
    return trait_deltas


def main():
    generator_device = "cuda:0"
    reader_device = "cuda:1"
    riasec_dir = _repo_root() / "persona_data/model_inits"

    generator_id = "marin-community/marin-8b-instruct"
    reader_id = "HuggingFaceTB/SmolLM3-3B"

    logger.info("Loading generator (Marin 8B) data...")
    gen_data = load_model_data(generator_id, riasec_dir)

    logger.info("Loading reader (SmolLM3) data...")
    reader_data = load_model_data(reader_id, riasec_dir)

    # Load generator model
    logger.info("Loading Marin 8B on GPU 2...")
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_id)
    gen_model = AutoModelForCausalLM.from_pretrained(
        generator_id, torch_dtype=torch.float16, device_map=generator_device)
    gen_model.eval()
    gen_blocks = get_decoder_blocks(gen_model)
    gen_mid = gen_data["mid_layer"]

    # Load reader model
    logger.info("Loading SmolLM3 on GPU 3...")
    reader_tokenizer = AutoTokenizer.from_pretrained(reader_id)
    reader_model = AutoModelForCausalLM.from_pretrained(
        reader_id, torch_dtype=torch.float16, device_map=reader_device)
    reader_model.eval()
    reader_blocks = get_decoder_blocks(reader_model)
    reader_mid = reader_data["mid_layer"]
    reader_capture_layer = reader_mid + 1
    reader_basis = reader_data["basis_5d"]
    reader_coords = reader_data["coords_5d"]

    # Reader baseline activations (neutral text)
    logger.info("Capturing reader baselines...")
    neutral_texts = [
        "The weather today is mild and cloudy.",
        "There are many different types of fruit available at the store.",
        "The meeting is scheduled for three o'clock.",
    ]
    reader_baseline_acts = {}
    for text in neutral_texts:
        reader_baseline_acts[text] = capture_reader_activations(
            reader_model, reader_tokenizer, reader_device, reader_blocks,
            reader_capture_layer, text)

    # Reader behavioral baseline
    logger.info("Computing reader behavioral baseline...")
    reader_baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [
                {"role": "user", "content":
                    f"Which describes you better? Answer with just A or B.\n"
                    f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                    f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"},
            ]
            formatted = reader_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = reader_tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(reader_device)
            with torch.no_grad():
                outputs = reader_model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            a_ids = reader_tokenizer.encode("A", add_special_tokens=False)
            b_ids = reader_tokenizer.encode("B", add_special_tokens=False)
            reader_baseline[f"{trait_a}-{trait_b}"] = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()

    results = {}

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL TEXT PERSONALITY TRANSFER")
    print(f"Generator: Marin 8B (activation-steered)")
    print(f"Reader:    SmolLM3 (reads generated text, no steering)")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Basic text personality transfer (6 traits)
    # ================================================================
    logger.info("Part 1: Basic text personality transfer...")
    print(f"\n{'='*70}")
    print("PART 1: TEXT PERSONALITY TRANSFER (6 RIASEC TRAITS)")
    print(f"{'='*70}")

    transfer_results = {}

    for target_trait in TRAITS:
        logger.info(f"  Generating text for {target_trait}...")
        vec = gen_data["residual"][target_trait].astype(np.float32)
        alpha = 3.0  # Strong steering for clear text personality

        # Generate text for each prompt
        all_generated_texts = []
        for prompt in GENERATION_PROMPTS:
            text = generate_steered_text(
                gen_model, gen_tokenizer, generator_device, gen_blocks,
                gen_mid, vec, alpha, prompt, max_new_tokens=150)
            all_generated_texts.append(text)

        # Reader: capture activations when reading the generated text
        reader_diffs = []
        for i, text in enumerate(all_generated_texts):
            reader_act = capture_reader_activations(
                reader_model, reader_tokenizer, reader_device, reader_blocks,
                reader_capture_layer, text)
            # Use mean of neutral baselines as reference
            mean_baseline = np.mean(list(reader_baseline_acts.values()), axis=0)
            reader_diffs.append(reader_act - mean_baseline)

        mean_reader_diff = np.mean(reader_diffs, axis=0)
        detected_coords = reader_basis @ mean_reader_diff
        detected_norm = float(np.linalg.norm(detected_coords))

        # Which trait does the reader detect?
        sims = {}
        for t in TRAITS:
            if detected_norm > 0 and np.linalg.norm(reader_coords[t]) > 0:
                sims[t] = float(np.dot(detected_coords, reader_coords[t]) / (
                    detected_norm * np.linalg.norm(reader_coords[t])))
            else:
                sims[t] = 0

        detected_trait = max(sims, key=sims.get)
        correct = detected_trait == target_trait

        # Behavioral: does reading the text shift the reader's RIASEC?
        combined_text = " ".join(all_generated_texts[:2])  # Use first 2 for profile
        reader_deltas = measure_reader_profile(
            reader_model, reader_tokenizer, reader_device, reader_blocks,
            reader_mid, combined_text, reader_baseline)
        reader_top = max(reader_deltas, key=reader_deltas.get)
        reader_mag = float(np.sqrt(sum(v**2 for v in reader_deltas.values())))

        mark = "OK" if correct else "MISS"
        print(f"\n  {target_trait:>15}: reader_detected={detected_trait:>15} cos={sims[detected_trait]:+.3f} "
              f"5D_norm={detected_norm:.2f} {mark}")
        print(f"    Behavioral top: {reader_top} (target_delta={reader_deltas[target_trait]:+.3f}, mag={reader_mag:.3f})")
        print(f"    Generated: \"{all_generated_texts[0][:120]}...\"")

        transfer_results[target_trait] = {
            "detected_trait": detected_trait,
            "correct_activation": bool(correct),
            "cosine": float(sims[detected_trait]),
            "5d_norm": detected_norm,
            "all_similarities": sims,
            "behavioral_top": reader_top,
            "behavioral_correct": reader_top == target_trait,
            "behavioral_target_delta": float(reader_deltas[target_trait]),
            "behavioral_magnitude": reader_mag,
            "behavioral_profile": {t: float(v) for t, v in reader_deltas.items()},
            "generated_texts": all_generated_texts,
        }

    n_act_correct = sum(1 for v in transfer_results.values() if v["correct_activation"])
    n_beh_correct = sum(1 for v in transfer_results.values() if v["behavioral_correct"])
    print(f"\n  Activation detection: {n_act_correct}/{len(TRAITS)} ({n_act_correct/len(TRAITS):.0%})")
    print(f"  Behavioral detection: {n_beh_correct}/{len(TRAITS)} ({n_beh_correct/len(TRAITS):.0%})")
    results["basic_transfer"] = transfer_results

    # ================================================================
    # PART 2: Dose-response (alpha sweep on generated text)
    # ================================================================
    logger.info("Part 2: Dose-response...")
    print(f"\n{'='*70}")
    print("PART 2: DOSE-RESPONSE (text personality vs steering strength)")
    print(f"{'='*70}")

    test_trait = "artistic"  # Use one trait for dose-response
    test_alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    vec = gen_data["residual"][test_trait].astype(np.float32)
    test_prompt = GENERATION_PROMPTS[0]

    dose_results = {}
    norms_for_corr = []
    alphas_for_corr = []

    print(f"\n  Trait: {test_trait}, Prompt: \"{test_prompt[:50]}...\"")
    print(f"  {'Alpha':>8} {'5D Norm':>10} {'Cos':>8} {'Detected':>15} {'Beh Top':>10} {'Target Δ':>10}")

    for alpha in test_alphas:
        text = generate_steered_text(
            gen_model, gen_tokenizer, generator_device, gen_blocks,
            gen_mid, vec, alpha, test_prompt, max_new_tokens=150)

        # Reader activation detection
        reader_act = capture_reader_activations(
            reader_model, reader_tokenizer, reader_device, reader_blocks,
            reader_capture_layer, text)
        mean_baseline = np.mean(list(reader_baseline_acts.values()), axis=0)
        diff = reader_act - mean_baseline
        detected_coords = reader_basis @ diff
        detected_norm = float(np.linalg.norm(detected_coords))
        norms_for_corr.append(detected_norm)
        alphas_for_corr.append(alpha)

        sims = {}
        for t in TRAITS:
            if detected_norm > 0 and np.linalg.norm(reader_coords[t]) > 0:
                sims[t] = float(np.dot(detected_coords, reader_coords[t]) / (
                    detected_norm * np.linalg.norm(reader_coords[t])))
            else:
                sims[t] = 0
        detected_trait = max(sims, key=sims.get)

        # Behavioral
        reader_deltas = measure_reader_profile(
            reader_model, reader_tokenizer, reader_device, reader_blocks,
            reader_mid, text, reader_baseline)
        reader_top = max(reader_deltas, key=reader_deltas.get)

        print(f"  {alpha:>8.1f} {detected_norm:>10.2f} {sims[detected_trait]:>+8.3f} "
              f"{detected_trait:>15} {reader_top:>10} {reader_deltas[test_trait]:>+10.3f}")

        dose_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "5d_norm": detected_norm,
            "detected_trait": detected_trait,
            "cosine": float(sims[detected_trait]),
            "behavioral_top": reader_top,
            "target_delta": float(reader_deltas[test_trait]),
            "generated_text_preview": text[:200],
        }

    # Correlation between alpha and detected norm
    if len(norms_for_corr) > 2:
        from scipy.stats import pearsonr
        r, p = pearsonr(alphas_for_corr, norms_for_corr)
        print(f"\n  Alpha vs 5D norm: r={r:.3f}, p={p:.3f}")
        dose_results["alpha_norm_correlation"] = {"r": float(r), "p": float(p)}

    results["dose_response"] = dose_results

    # ================================================================
    # PART 3: Negative steering — does suppressed text show anti-personality?
    # ================================================================
    logger.info("Part 3: Negative steering text transfer...")
    print(f"\n{'='*70}")
    print("PART 3: NEGATIVE STEERING TEXT TRANSFER")
    print(f"{'='*70}")

    negative_results = {}
    test_traits_neg = ["artistic", "investigative", "social"]

    for target_trait in test_traits_neg:
        vec = gen_data["residual"][target_trait].astype(np.float32)

        for alpha in [2.0, -2.0]:
            text = generate_steered_text(
                gen_model, gen_tokenizer, generator_device, gen_blocks,
                gen_mid, vec, alpha, GENERATION_PROMPTS[0], max_new_tokens=150)

            reader_act = capture_reader_activations(
                reader_model, reader_tokenizer, reader_device, reader_blocks,
                reader_capture_layer, text)
            mean_baseline = np.mean(list(reader_baseline_acts.values()), axis=0)
            diff = reader_act - mean_baseline
            detected_coords = reader_basis @ diff
            detected_norm = float(np.linalg.norm(detected_coords))

            sims = {}
            for t in TRAITS:
                if detected_norm > 0 and np.linalg.norm(reader_coords[t]) > 0:
                    sims[t] = float(np.dot(detected_coords, reader_coords[t]) / (
                        detected_norm * np.linalg.norm(reader_coords[t])))
                else:
                    sims[t] = 0
            detected_trait = max(sims, key=sims.get)
            target_cos = sims[target_trait]

            direction = "enhance" if alpha > 0 else "suppress"
            print(f"\n  {target_trait} α={alpha:+.1f} ({direction}):")
            print(f"    Detected: {detected_trait} (cos={sims[detected_trait]:+.3f})")
            print(f"    Target cos: {target_cos:+.3f}")
            print(f"    Text: \"{text[:120]}...\"")

            key = f"{target_trait}_alpha{alpha:+.1f}"
            negative_results[key] = {
                "trait": target_trait,
                "alpha": alpha,
                "direction": direction,
                "detected_trait": detected_trait,
                "target_cosine": float(target_cos),
                "best_cosine": float(sims[detected_trait]),
                "all_similarities": sims,
                "text_preview": text[:300],
            }

    results["negative_transfer"] = negative_results

    # ================================================================
    # PART 4: Text personality without activation baseline
    # ================================================================
    logger.info("Part 4: Reader-only personality classification...")
    print(f"\n{'='*70}")
    print("PART 4: READER BEHAVIORAL CLASSIFICATION (no activation comparison)")
    print(f"{'='*70}")

    # Can the reader classify personality from text using behavioral profile alone?
    classification_results = {}

    for target_trait in TRAITS:
        texts = transfer_results[target_trait]["generated_texts"]
        combined_text = " ".join(texts[:2])

        reader_deltas = measure_reader_profile(
            reader_model, reader_tokenizer, reader_device, reader_blocks,
            reader_mid, combined_text, reader_baseline)
        reader_top = max(reader_deltas, key=reader_deltas.get)
        correct = reader_top == target_trait

        mark = "OK" if correct else "MISS"
        print(f"  {target_trait:>15}: behavioral_top={reader_top:>15} delta={reader_deltas[target_trait]:+.3f} {mark}")

        classification_results[target_trait] = {
            "behavioral_top": reader_top,
            "correct": bool(correct),
            "target_delta": float(reader_deltas[target_trait]),
            "profile": {t: float(v) for t, v in reader_deltas.items()},
        }

    n_classified = sum(1 for v in classification_results.values() if v["correct"])
    print(f"\n  Behavioral classification: {n_classified}/{len(TRAITS)} ({n_classified/len(TRAITS):.0%})")
    results["behavioral_classification"] = classification_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    act_acc = sum(1 for v in transfer_results.values() if v["correct_activation"]) / len(TRAITS)
    beh_acc = sum(1 for v in transfer_results.values() if v["behavioral_correct"]) / len(TRAITS)
    class_acc = sum(1 for v in classification_results.values() if v["correct"]) / len(TRAITS)
    mean_cos = np.mean([v["cosine"] for v in transfer_results.values()])
    mean_beh_mag = np.mean([v["behavioral_magnitude"] for v in transfer_results.values()])

    print(f"\n  Activation-based detection:  {act_acc:.0%}")
    print(f"  Behavioral detection:        {beh_acc:.0%}")
    print(f"  Behavioral classification:   {class_acc:.0%}")
    print(f"  Mean cosine:                 {mean_cos:.3f}")
    print(f"  Mean behavioral magnitude:   {mean_beh_mag:.3f}")

    results["summary"] = {
        "activation_detection_accuracy": float(act_acc),
        "behavioral_detection_accuracy": float(beh_acc),
        "behavioral_classification_accuracy": float(class_acc),
        "mean_cosine": float(mean_cos),
        "mean_behavioral_magnitude": float(mean_beh_mag),
        "generator_model": generator_id,
        "reader_model": reader_id,
        "generator_alpha": 3.0,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "text_personality_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
