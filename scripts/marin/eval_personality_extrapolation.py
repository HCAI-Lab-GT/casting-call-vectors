#!/usr/bin/env python
"""
Personality extrapolation: Can we amplify trait coordinates in 5D?

The 6 RIASEC vectors sit on a near-regular simplex in 5D.
What happens if we steer with coordinates BEYOND the simplex?

Tests:
1. Amplified traits: 1×, 1.5×, 2×, 3× of each trait's 5D coordinates
2. Does the trait profile get proportionally stronger?
3. At what amplification does the model break?
4. Does generation text become more extreme?

This tests the linearity and boundary of the personality subspace.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="extrapolation")

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


def reconstruct_from_5d(coords_5d, basis_5d):
    return (basis_5d.T @ coords_5d).astype(np.float32)


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


def generate_steered(model, tokenizer, device, blocks, mid_layer, steer_vec, alpha,
                     prompt, max_new_tokens=80):
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
    alpha = 1.0  # Fixed alpha — we scale the 5D coordinates instead
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    # Amplification factors
    amplifications = [0.5, 1.0, 1.5, 2.0, 3.0]

    # Test on 3 representative traits
    test_traits = ["artistic", "investigative", "social"]

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    coords_5d, basis_5d = get_5d_coords_and_basis(residual)

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

    gen_prompt = "In my free time, I love to"

    print(f"\n{'='*70}")
    print(f"PERSONALITY EXTRAPOLATION (Amplified 5D Coordinates)")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    results = {}

    for trait in test_traits:
        print(f"\n{'='*70}")
        print(f"TRAIT: {trait.upper()}")
        print(f"  Original 5D coords: [{', '.join(f'{c:+.2f}' for c in coords_5d[trait])}]")
        print(f"  Original norm: {np.linalg.norm(coords_5d[trait]):.2f}")
        print(f"{'='*70}")

        trait_results = {}

        for amp in amplifications:
            logger.info(f"Testing {trait} at {amp}× amplification...")

            # Amplify 5D coordinates
            amp_coords = amp * coords_5d[trait]
            # Reconstruct full-dim vector
            amp_vec = reconstruct_from_5d(amp_coords, basis_5d)

            # Measure trait profile
            profile = measure_trait_profile(
                model, tokenizer, device, blocks, mid_layer, amp_vec, alpha, baseline)

            # Generate text
            gen = generate_steered(
                model, tokenizer, device, blocks, mid_layer, amp_vec, alpha * 3, gen_prompt)

            sorted_prof = sorted(profile.items(), key=lambda x: -x[1])
            top_trait = sorted_prof[0][0]
            target_delta = profile[trait]

            print(f"\n  {amp}× amplification (norm={np.linalg.norm(amp_vec):.1f}):")
            print(f"    Target trait delta: {target_delta:+.3f}")
            print(f"    Top trait: {top_trait} ({profile[top_trait]:+.3f})")
            print(f"    Profile: " + "  ".join(f"{t[:4]}={d:+.2f}" for t, d in sorted_prof))
            print(f"    Generated: {gen[:120]}...")

            trait_results[str(amp)] = {
                "amplification": amp,
                "vec_norm": float(np.linalg.norm(amp_vec)),
                "profile": {t: float(d) for t, d in profile.items()},
                "target_delta": float(target_delta),
                "top_trait": top_trait,
                "is_target_top": top_trait == trait,
                "generation": gen,
            }

        results[trait] = trait_results

    # Analysis: linearity of amplification
    print(f"\n{'='*70}")
    print(f"LINEARITY ANALYSIS")
    print(f"{'='*70}")

    for trait in test_traits:
        print(f"\n  {trait.upper()}:")
        deltas = []
        amps = []
        for amp in amplifications:
            r = results[trait][str(amp)]
            deltas.append(r["target_delta"])
            amps.append(amp)
            is_top = "✓" if r["is_target_top"] else "✗"
            print(f"    {amp}×: delta={r['target_delta']:+.3f}, top={r['top_trait']} {is_top}")

        # Check linearity
        from scipy.stats import pearsonr
        r, p = pearsonr(amps, deltas)
        print(f"    Linearity: r={r:.3f} (p={p:.3f})")

        # Ratio test: is 2× delta ≈ 2 * 1× delta?
        if str(1.0) in results[trait] and str(2.0) in results[trait]:
            d1 = results[trait]["1.0"]["target_delta"]
            d2 = results[trait]["2.0"]["target_delta"]
            if abs(d1) > 0.001:
                ratio = d2 / d1
                print(f"    2× / 1× ratio: {ratio:.2f} (perfect linearity = 2.00)")

        results[trait]["linearity"] = {"r": float(r), "p": float(p)}

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    for trait in test_traits:
        r_vals = [results[trait][str(amp)] for amp in amplifications]
        maintain_top = sum(1 for r in r_vals if r["is_target_top"])
        print(f"\n  {trait}:")
        print(f"    Target stays on top: {maintain_top}/{len(amplifications)} amplifications")
        print(f"    Linearity: r = {results[trait]['linearity']['r']:.3f}")
        deltas = [r["target_delta"] for r in r_vals]
        print(f"    Delta range: {min(deltas):+.3f} to {max(deltas):+.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_extrapolation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
