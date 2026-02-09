#!/usr/bin/env python
"""
System prompt vs activation steering: complementary or competitive?

In deployment, personality is often set via system prompts ("You are a creative
artist..."). Does activation steering:
  A) Override the system prompt? (steering dominates)
  B) Complement it? (additive effects)
  C) Conflict when mismatched? (system prompt resists)

Tests:
1. Matching: artistic system prompt + artistic steering → synergy?
2. Conflicting: artistic system prompt + conventional steering → which wins?
3. Neutral: no system prompt + steering → baseline for comparison
4. System-only: artistic system prompt + no steering → how strong is prompt alone?

This has major practical implications: if system prompts and steering stack,
we can achieve stronger personality effects. If they conflict, we need to
understand the interaction.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="sys-prompt")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# System prompts that explicitly set personality
SYSTEM_PROMPTS = {
    "artistic": "You are a deeply creative, artistic soul. You express yourself through art, "
                "imagination, and unconventional thinking. You value beauty, originality, and "
                "self-expression above all else.",
    "conventional": "You are a highly organized, methodical person. You value structure, rules, "
                    "efficiency, and tradition. You prefer clear procedures and predictable outcomes.",
    "investigative": "You are a deeply analytical, scientific thinker. You love research, data, "
                     "logical reasoning, and understanding how things work. You value knowledge above all.",
    "social": "You are a warm, empathetic helper. You deeply care about others' well-being, "
              "enjoy counseling and teaching, and prioritize human connection above everything.",
    "neutral": "You are a helpful AI assistant.",
}


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


def measure_profile_with_system(model, tokenizer, device, blocks, mid_layer,
                                 vec, alpha, system_prompt):
    """Measure personality with a specific system prompt and optional steering."""
    hook_handle = None
    if vec is not None and alpha > 0:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))

    try:
        trait_logprobs = {}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Which describes you better? Answer with just A or B.\n"
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
    finally:
        if hook_handle:
            hook_handle.remove()

    return trait_logprobs


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
    alpha = 1.0
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

    # Baseline: neutral system prompt, no steering
    logger.info("Computing baseline...")
    baseline_raw = measure_profile_with_system(
        model, tokenizer, device, blocks, mid_layer,
        None, 0, SYSTEM_PROMPTS["neutral"])

    print(f"\n{'='*70}")
    print(f"SYSTEM PROMPT × ACTIVATION STEERING INTERACTION")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    results = {}

    test_traits = ["artistic", "conventional", "investigative", "social"]

    for sys_trait in test_traits:
        print(f"\n{'='*70}")
        print(f"SYSTEM PROMPT: {sys_trait.upper()}")
        print(f"{'='*70}")

        trait_results = {}

        # 1. System prompt ONLY (no steering)
        logger.info(f"Testing system prompt only: {sys_trait}...")
        raw = measure_profile_with_system(
            model, tokenizer, device, blocks, mid_layer,
            None, 0, SYSTEM_PROMPTS[sys_trait])
        deltas = compute_deltas(raw, baseline_raw)
        sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
        print(f"\n  System-only: top={sorted_d[0][0]}, "
              f"target={deltas[sys_trait]:+.3f}, "
              f"profile: {' '.join(f'{t[:4]}={d:+.3f}' for t, d in sorted_d)}")
        trait_results["system_only"] = {
            "target_delta": float(deltas[sys_trait]),
            "top_trait": sorted_d[0][0],
            "is_target_top": sorted_d[0][0] == sys_trait,
            "profile": {t: float(deltas[t]) for t in TRAITS},
        }

        # 2. Steering ONLY (neutral system prompt)
        logger.info(f"Testing steering only: {sys_trait}...")
        vec = residual[sys_trait].astype(np.float32)
        raw = measure_profile_with_system(
            model, tokenizer, device, blocks, mid_layer,
            vec, alpha, SYSTEM_PROMPTS["neutral"])
        deltas = compute_deltas(raw, baseline_raw)
        sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
        print(f"  Steer-only:  top={sorted_d[0][0]}, "
              f"target={deltas[sys_trait]:+.3f}, "
              f"profile: {' '.join(f'{t[:4]}={d:+.3f}' for t, d in sorted_d)}")
        trait_results["steer_only"] = {
            "target_delta": float(deltas[sys_trait]),
            "top_trait": sorted_d[0][0],
            "is_target_top": sorted_d[0][0] == sys_trait,
            "profile": {t: float(deltas[t]) for t in TRAITS},
        }

        # 3. MATCHING: system prompt + matching steering
        logger.info(f"Testing matching: sys={sys_trait} + steer={sys_trait}...")
        raw = measure_profile_with_system(
            model, tokenizer, device, blocks, mid_layer,
            vec, alpha, SYSTEM_PROMPTS[sys_trait])
        deltas = compute_deltas(raw, baseline_raw)
        sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
        print(f"  MATCHING:    top={sorted_d[0][0]}, "
              f"target={deltas[sys_trait]:+.3f}, "
              f"profile: {' '.join(f'{t[:4]}={d:+.3f}' for t, d in sorted_d)}")
        trait_results["matching"] = {
            "target_delta": float(deltas[sys_trait]),
            "top_trait": sorted_d[0][0],
            "is_target_top": sorted_d[0][0] == sys_trait,
            "profile": {t: float(deltas[t]) for t in TRAITS},
        }

        # 4. CONFLICTING: system prompt + opposite steering
        # Use Holland opposite
        HOLLAND = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
        h_idx = HOLLAND.index(sys_trait)
        opposite = HOLLAND[(h_idx + 3) % 6]
        opp_vec = residual[opposite].astype(np.float32)

        logger.info(f"Testing conflicting: sys={sys_trait} + steer={opposite}...")
        raw = measure_profile_with_system(
            model, tokenizer, device, blocks, mid_layer,
            opp_vec, alpha, SYSTEM_PROMPTS[sys_trait])
        deltas = compute_deltas(raw, baseline_raw)
        sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
        print(f"  CONFLICT:    top={sorted_d[0][0]} (sys={sys_trait}, steer={opposite}), "
              f"sys_target={deltas[sys_trait]:+.3f}, steer_target={deltas[opposite]:+.3f}")
        trait_results["conflicting"] = {
            "system_trait": sys_trait,
            "steer_trait": opposite,
            "system_delta": float(deltas[sys_trait]),
            "steer_delta": float(deltas[opposite]),
            "top_trait": sorted_d[0][0],
            "winner": "system" if deltas[sys_trait] > deltas[opposite] else "steering",
            "profile": {t: float(deltas[t]) for t in TRAITS},
        }

        # Analysis
        sys_effect = trait_results["system_only"]["target_delta"]
        steer_effect = trait_results["steer_only"]["target_delta"]
        combined_effect = trait_results["matching"]["target_delta"]
        predicted_additive = sys_effect + steer_effect

        synergy = combined_effect / predicted_additive if abs(predicted_additive) > 0.01 else 0

        print(f"\n  Analysis:")
        print(f"    System effect:     {sys_effect:+.3f}")
        print(f"    Steer effect:      {steer_effect:+.3f}")
        print(f"    Combined effect:   {combined_effect:+.3f}")
        print(f"    Predicted (sum):   {predicted_additive:+.3f}")
        print(f"    Synergy ratio:     {synergy:.2f}× (1.0 = perfectly additive)")
        print(f"    Conflict winner:   {trait_results['conflicting']['winner']}")

        trait_results["analysis"] = {
            "system_effect": float(sys_effect),
            "steer_effect": float(steer_effect),
            "combined_effect": float(combined_effect),
            "predicted_additive": float(predicted_additive),
            "synergy_ratio": float(synergy),
            "conflict_winner": trait_results["conflicting"]["winner"],
        }

        results[sys_trait] = trait_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Trait':>15} {'System':>8} {'Steer':>8} {'Combined':>10} {'Predicted':>10} {'Synergy':>8} {'Conflict':>10}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for trait in test_traits:
        a = results[trait]["analysis"]
        print(f"  {trait:>15} {a['system_effect']:>+8.3f} {a['steer_effect']:>+8.3f} "
              f"{a['combined_effect']:>+10.3f} {a['predicted_additive']:>+10.3f} "
              f"{a['synergy_ratio']:>8.2f} {a['conflict_winner']:>10}")

    mean_synergy = np.mean([results[t]["analysis"]["synergy_ratio"] for t in test_traits])
    steer_wins = sum(1 for t in test_traits if results[t]["analysis"]["conflict_winner"] == "steering")

    print(f"\n  Mean synergy ratio: {mean_synergy:.2f}")
    print(f"  Steering wins conflicts: {steer_wins}/{len(test_traits)}")

    if mean_synergy > 1.1:
        print(f"  CONCLUSION: System prompt and steering are SYNERGISTIC (>1.1×)")
    elif mean_synergy > 0.9:
        print(f"  CONCLUSION: System prompt and steering are ADDITIVE (~1.0×)")
    elif mean_synergy > 0.5:
        print(f"  CONCLUSION: System prompt and steering are SUB-ADDITIVE (0.5-0.9×)")
    else:
        print(f"  CONCLUSION: System prompt and steering INTERFERE (<0.5×)")

    results["summary"] = {
        "mean_synergy": float(mean_synergy),
        "steer_wins_conflicts": steer_wins,
        "total_conflicts": len(test_traits),
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "system_prompt_interaction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
