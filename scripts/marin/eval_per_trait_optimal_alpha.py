#!/usr/bin/env python
"""
Per-trait optimal alpha: fine-grained dose-response to find linearity limits.

We know that:
- Artistic works well up to α=3+ (strong PC1 dominance)
- Social breaks at α=2+ (PC1 contamination at high amplification)
- At α=1, all traits work perfectly (97% pairwise)

This experiment finds the EXACT alpha where each trait's linearity breaks
by testing a fine grid of alpha values (0.25, 0.5, 0.75, 1.0, 1.5, 2.0,
2.5, 3.0, 4.0, 5.0) for all 6 traits.

For each alpha, measures:
1. Is the target trait still #1?
2. What's the target delta?
3. What's the margin over #2?

Output: per-trait alpha curves and optimal operating ranges.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="optimal-alpha")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
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


def pairwise_logprob_chat(model, tokenizer, device, desc_a, desc_b):
    messages = [
        {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
        {"role": "user", "content": f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"},
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
    return log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()


def measure_profile(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline):
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
        trait_deltas = {t: 0.0 for t in TRAITS}
        trait_counts = {t: 0 for t in TRAITS}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                            TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                base_gap = baseline[f"{trait_a}-{trait_b}"]
                shift = gap - base_gap
                trait_deltas[trait_a] += shift
                trait_counts[trait_a] += 1
                trait_deltas[trait_b] -= shift
                trait_counts[trait_b] += 1
        for t in TRAITS:
            if trait_counts[t] > 0:
                trait_deltas[t] /= trait_counts[t]
    finally:
        hook_handle.remove()
    return trait_deltas


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    alpha_grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

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
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                        TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"PER-TRAIT OPTIMAL ALPHA")
    print(f"Target: Marin 8B")
    print(f"Alpha grid: {alpha_grid}")
    print(f"{'='*70}")

    results = {}

    for steer_trait in TRAITS:
        vec = residual[steer_trait].astype(np.float32)
        trait_results = {}

        print(f"\n{'='*70}")
        print(f"TRAIT: {steer_trait.upper()}")
        print(f"{'='*70}")

        for alpha in alpha_grid:
            logger.info(f"{steer_trait} at α={alpha}...")
            profile = measure_profile(
                model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline)

            sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
            top = sorted_prof[0][0]
            top_delta = sorted_prof[0][1]
            target_delta = profile[steer_trait]
            margin = target_delta - sorted_prof[1][1] if top == steer_trait else target_delta - top_delta

            print(f"  α={alpha:>4}: target={target_delta:+.3f}, top={top}, "
                  f"margin={margin:+.3f}, "
                  f"profile: {' '.join(f'{t[:4]}={d:+.2f}' for t, d in sorted_prof[:3])}")

            trait_results[str(alpha)] = {
                "alpha": alpha,
                "target_delta": float(target_delta),
                "top_trait": top,
                "is_target_top": top == steer_trait,
                "margin": float(margin),
                "profile": {t: float(profile[t]) for t in TRAITS},
            }

        results[steer_trait] = trait_results

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS: Per-Trait Optimal Alpha")
    print(f"{'='*70}")

    for steer_trait in TRAITS:
        # Find max alpha where target is still #1
        max_good_alpha = 0
        max_delta = 0
        max_delta_alpha = 0

        for alpha in alpha_grid:
            data = results[steer_trait][str(alpha)]
            if data["is_target_top"]:
                max_good_alpha = alpha
            if data["target_delta"] > max_delta:
                max_delta = data["target_delta"]
                max_delta_alpha = alpha

        # Find the "linearity limit" — alpha where delta/alpha starts decreasing
        efficiency = []
        for alpha in alpha_grid:
            data = results[steer_trait][str(alpha)]
            eff = data["target_delta"] / alpha if alpha > 0 else 0
            efficiency.append(eff)

        # Peak efficiency alpha
        peak_eff_idx = np.argmax(efficiency)
        peak_eff_alpha = alpha_grid[peak_eff_idx]

        print(f"\n  {steer_trait:>15}:")
        print(f"    Max alpha as #1:   {max_good_alpha}")
        print(f"    Peak delta alpha:  {max_delta_alpha} (delta={max_delta:+.3f})")
        print(f"    Peak efficiency α: {peak_eff_alpha} (delta/α={efficiency[peak_eff_idx]:.3f})")
        print(f"    Efficiency curve:  " + " ".join(f"{e:.2f}" for e in efficiency))

        results[steer_trait]["analysis"] = {
            "max_alpha_as_top": max_good_alpha,
            "peak_delta_alpha": max_delta_alpha,
            "peak_delta": float(max_delta),
            "peak_efficiency_alpha": peak_eff_alpha,
            "peak_efficiency": float(efficiency[peak_eff_idx]),
            "efficiency_curve": [float(e) for e in efficiency],
        }

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: OPTIMAL OPERATING RANGES")
    print(f"{'='*70}")
    print(f"  {'Trait':>15} {'Max α (top)':>12} {'Peak Δ α':>10} {'Peak eff α':>12} {'Peak eff':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for trait in TRAITS:
        a = results[trait]["analysis"]
        print(f"  {trait:>15} {a['max_alpha_as_top']:>12.1f} "
              f"{a['peak_delta_alpha']:>10.1f} "
              f"{a['peak_efficiency_alpha']:>12.1f} "
              f"{a['peak_efficiency']:>10.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_trait_optimal_alpha.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
