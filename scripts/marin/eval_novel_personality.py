#!/usr/bin/env python
"""
Novel personality creation from arbitrary 5D coordinates.

Tests whether the 5D personality subspace supports continuous navigation
to points that don't correspond to any of the 6 RIASEC types.

Creates:
1. Midpoints between trait pairs (e.g., Artistic+Investigative centroid)
2. Anti-trait directions (opposite of each trait in 5D)
3. Random points on the 5D unit sphere
4. The origin (zero personality)

For each synthetic personality point:
- Reconstruct in full model dimension space
- Steer Marin 8B with the synthetic vector
- Measure logprob profile across all 6 traits
- Generate text to qualitatively assess the persona

This tests whether the personality space is genuinely continuous
and navigable, not just a discrete set of 6 fixed points.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="novel-personality")

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


def get_5d_coords_and_basis(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d


def reconstruct_from_5d(coords_5d, basis_5d, mean_norm):
    """Reconstruct a full-dimensional vector from 5D coordinates."""
    vec = basis_5d.T @ coords_5d
    # Scale to match mean trait vector norm
    current_norm = np.linalg.norm(vec)
    if current_norm > 0:
        vec = vec * (mean_norm / current_norm)
    return vec.astype(np.float32)


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


def measure_trait_profile(model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline):
    """Measure how steering with vec affects preference for each trait.

    Returns a dict of trait -> mean delta (how much this steering boosts that trait).
    """
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
                shift = gap - base_gap  # positive = more toward A, negative = more toward B
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


def generate_steered(model, tokenizer, device, blocks, mid_layer, steer_vec, alpha,
                     prompt, max_new_tokens=100):
    vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
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
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

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
        hook_handle.remove()

    return generated.strip()


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    target_id = "marin-community/marin-8b-instruct"

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    coords_5d, basis_5d = get_5d_coords_and_basis(residual)

    # Mean norm for scaling novel vectors
    mean_norm = np.mean([np.linalg.norm(residual[t]) for t in TRAITS])
    logger.info(f"Mean residual norm: {mean_norm:.3f}")

    # Create novel personality points
    novel_personalities = {}

    # 1. Named trait vectors (reference)
    for t in TRAITS:
        novel_personalities[f"trait_{t}"] = {
            "coords": coords_5d[t],
            "label": f"Pure {t}",
            "type": "reference",
        }

    # 2. Midpoints between select trait pairs
    interesting_pairs = [
        ("artistic", "investigative", "Creative Scientist"),
        ("social", "enterprising", "Social Leader"),
        ("realistic", "investigative", "Hands-on Researcher"),
        ("artistic", "social", "Community Artist"),
        ("conventional", "enterprising", "Business Manager"),
        ("artistic", "conventional", "Artistic+Conventional (Holland opposites)"),
    ]
    for t1, t2, label in interesting_pairs:
        mid = (coords_5d[t1] + coords_5d[t2]) / 2
        novel_personalities[f"mid_{t1}_{t2}"] = {
            "coords": mid,
            "label": label,
            "type": "midpoint",
        }

    # 3. Triple-trait centroids
    triple_combos = [
        (["artistic", "investigative", "social"], "Creative Academic Mentor"),
        (["realistic", "conventional", "enterprising"], "Industrial Manager"),
        (["artistic", "social", "enterprising"], "Creative Entrepreneur"),
    ]
    for traits, label in triple_combos:
        centroid = np.mean([coords_5d[t] for t in traits], axis=0)
        novel_personalities[f"triple_{'_'.join(traits)}"] = {
            "coords": centroid,
            "label": label,
            "type": "triple_centroid",
        }

    # 4. Anti-trait directions (negative of each trait)
    for t in TRAITS:
        novel_personalities[f"anti_{t}"] = {
            "coords": -coords_5d[t],
            "label": f"Anti-{t}",
            "type": "anti_trait",
        }

    # 5. Random unit vectors in 5D (for null comparison)
    rng = np.random.RandomState(42)
    for i in range(3):
        rand_dir = rng.randn(5)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        # Scale to match typical trait coordinate norm
        typical_norm = np.mean([np.linalg.norm(coords_5d[t]) for t in TRAITS])
        rand_dir = rand_dir * typical_norm
        novel_personalities[f"random_{i}"] = {
            "coords": rand_dir,
            "label": f"Random direction {i+1}",
            "type": "random",
        }

    # Load model
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

    # Test each novel personality
    gen_prompt = "In my free time, I love to"

    print(f"\n{'='*70}")
    print(f"NOVEL PERSONALITY CREATION FROM ARBITRARY 5D COORDINATES")
    print(f"{'='*70}")

    results = {}

    # Group by type
    for ptype in ["reference", "midpoint", "triple_centroid", "anti_trait", "random"]:
        entries = {k: v for k, v in novel_personalities.items() if v["type"] == ptype}
        if not entries:
            continue

        print(f"\n{'='*70}")
        print(f"TYPE: {ptype.upper()}")
        print(f"{'='*70}")

        for name, info in entries.items():
            logger.info(f"Testing {name} ({info['label']})...")

            # Reconstruct full-dim vector from 5D coords
            vec = reconstruct_from_5d(info["coords"], basis_5d, mean_norm)

            # Measure trait profile
            profile = measure_trait_profile(
                model, tokenizer, device, blocks, mid_layer, vec, alpha, baseline)

            # Identify top-2 traits
            sorted_traits = sorted(profile.items(), key=lambda x: -x[1])
            top_trait = sorted_traits[0][0]
            top2 = [t for t, _ in sorted_traits[:2]]

            # Generate text
            gen_text = generate_steered(
                model, tokenizer, device, blocks, mid_layer, vec, alpha * 3, gen_prompt)

            print(f"\n  {name}: {info['label']}")
            print(f"    Profile: " + "  ".join(f"{t[:4]}={d:+.2f}" for t, d in sorted_traits))
            print(f"    Top trait: {top_trait} ({profile[top_trait]:+.2f})")
            print(f"    Generated: {gen_text[:120]}...")

            # Check for midpoints: does the profile show BOTH constituent traits boosted?
            constituents = []
            if info["type"] == "midpoint":
                t1, t2 = name.replace("mid_", "").split("_", 1)
                # Handle the Holland opposites case
                for t in TRAITS:
                    if t1 in name and t2 in name:
                        break
                constituents = [t1, t2]
                both_boosted = all(profile[t] > 0 for t in constituents)
                print(f"    Constituent boost: {t1}={profile[t1]:+.2f}, {t2}={profile[t2]:+.2f}"
                      f"  {'BOTH BOOSTED' if both_boosted else 'one suppressed'}")

            results[name] = {
                "label": info["label"],
                "type": info["type"],
                "coords_5d": info["coords"].tolist(),
                "profile": {t: float(d) for t, d in profile.items()},
                "top_trait": top_trait,
                "generation": gen_text,
            }

    # Analysis: Do midpoint personas create genuine blends?
    print(f"\n{'='*70}")
    print(f"ANALYSIS: MIDPOINT BLENDING QUALITY")
    print(f"{'='*70}")

    midpoint_entries = {k: v for k, v in results.items() if v["type"] == "midpoint"}
    blend_successes = 0
    blend_total = 0
    for name, result in midpoint_entries.items():
        parts = name.replace("mid_", "").split("_", 1)
        if len(parts) == 2:
            t1, t2 = parts
            if t1 in TRAITS and t2 in TRAITS:
                blend_total += 1
                p1 = result["profile"][t1]
                p2 = result["profile"][t2]
                # Success if both traits are in top 3
                sorted_profile = sorted(result["profile"].items(), key=lambda x: -x[1])
                top3 = [t for t, _ in sorted_profile[:3]]
                success = t1 in top3 and t2 in top3
                if success:
                    blend_successes += 1
                print(f"  {result['label']}: {t1}={p1:+.2f}, {t2}={p2:+.2f}, "
                      f"both in top 3: {'YES' if success else 'NO'}")

    if blend_total > 0:
        print(f"\n  Blend success rate: {blend_successes}/{blend_total} ({blend_successes/blend_total:.0%})")

    # Analysis: Anti-traits
    print(f"\n{'='*70}")
    print(f"ANALYSIS: ANTI-TRAIT DIRECTION VALIDITY")
    print(f"{'='*70}")

    anti_entries = {k: v for k, v in results.items() if v["type"] == "anti_trait"}
    anti_successes = 0
    anti_total = 0
    for name, result in anti_entries.items():
        trait = name.replace("anti_", "")
        if trait in TRAITS:
            anti_total += 1
            # The anti-trait should SUPPRESS the original trait (negative profile value)
            suppressed = result["profile"][trait] < 0
            sorted_profile = sorted(result["profile"].items(), key=lambda x: -x[1])
            # And boost the opposite trait
            top_boosted = sorted_profile[0][0]
            if suppressed:
                anti_successes += 1
            print(f"  Anti-{trait}: {trait}={result['profile'][trait]:+.2f} "
                  f"({'SUPPRESSED' if suppressed else 'NOT suppressed'}), "
                  f"top boosted: {top_boosted}")

    if anti_total > 0:
        print(f"\n  Anti-trait suppression rate: {anti_successes}/{anti_total} ({anti_successes/anti_total:.0%})")

    # Summary
    results["summary"] = {
        "total_novel_personalities": len(novel_personalities),
        "blend_success_rate": float(blend_successes / blend_total) if blend_total > 0 else 0,
        "anti_suppression_rate": float(anti_successes / anti_total) if anti_total > 0 else 0,
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "novel_personality_creation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
