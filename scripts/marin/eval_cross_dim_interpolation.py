#!/usr/bin/env python
"""
Cross-dim interpolation: Does smooth personality blending survive
cross-dimensional transfer?

Tests: In SmolLM3's 5D space, linearly interpolate between two traits.
Transfer each interpolated vector to Marin 8B's 4096d space.
Evaluate whether the preference shift on Marin is monotonic.

This combines two established findings:
1. Smooth interpolation works within a model
2. Cross-dim transfer preserves discrimination

If cross-dim interpolation is also smooth, it proves the 5D bridge
preserves continuous structure, not just discrete trait identities.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-dim-interp")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Holland hex distances: 1=adjacent, 2=alternate, 3=opposite
PAIRS_TO_TEST = [
    ("artistic", "social", 1),       # Adjacent
    ("artistic", "investigative", 2), # Alternate
    ("artistic", "conventional", 3),  # Opposite
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
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer


def get_5d_coordinates(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d


def fit_procrustes(source_5d, target_5d):
    S = np.stack([source_5d[t] for t in TRAITS])
    T = np.stack([target_5d[t] for t in TRAITS])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))
    return R, scale


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


def main():
    source_id = "HuggingFaceTB/SmolLM3-3B"
    target_id = "marin-community/marin-8b-instruct"
    device = "cuda:0"
    alpha = 1.0
    t_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load vectors
    logger.info("Loading vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    target_residual, target_mid = load_residual_vectors(target_id, riasec_dir)

    source_5d, _ = get_5d_coordinates(source_residual)
    target_5d, target_basis = get_5d_coordinates(target_residual)

    R, scale = fit_procrustes(source_5d, target_5d)

    # Load target model
    logger.info("Loading target model: %s", target_id)
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    target_dim = target_residual[TRAITS[0]].shape[0]

    # Baseline
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"CROSS-DIM INTERPOLATION")
    print(f"Source: {source_id} (5D) → Target: {target_id} ({target_dim}d)")
    print(f"{'='*70}")

    results = {"source": source_id, "target": target_id, "alpha": alpha, "pairs": {}}

    for trait_a, trait_b, holland_dist in PAIRS_TO_TEST:
        print(f"\n--- {trait_a} ↔ {trait_b} (Holland distance={holland_dist}) ---")
        print(f"  t=0: pure {trait_a}, t=1: pure {trait_b}")

        pair_key = f"{trait_a}-{trait_b}"

        # The target pair: which describes you better, A or B?
        desc_a = TRAIT_DESCRIPTIONS[trait_a]
        desc_b = TRAIT_DESCRIPTIONS[trait_b]

        gaps = []
        for t in t_values:
            # Interpolate in SmolLM3's 5D space
            interp_5d = (1 - t) * source_5d[trait_a] + t * source_5d[trait_b]

            # Transfer to Marin's 4096d space
            aligned_5d = scale * (R @ interp_5d)
            vec_full = (target_basis.T @ aligned_5d).astype(np.float32)

            # Steer and evaluate
            vec_t = torch.tensor(vec_full, dtype=torch.float16).unsqueeze(0).to(device)
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

            hook_handle = blocks[target_mid].register_forward_hook(make_hook(delta_vec))
            try:
                gap = pairwise_logprob_chat(model, tokenizer, device, desc_a, desc_b)
            finally:
                hook_handle.remove()

            # Positive gap = prefers A (trait_a), negative = prefers B (trait_b)
            gaps.append(gap)
            preference = "A" if gap > 0 else "B"
            direction = f"→{trait_a}" if gap > 0 else f"→{trait_b}"
            print(f"  t={t:.1f}: gap={gap:+.3f} ({direction})")

        # Analyze monotonicity
        is_monotonic = all(gaps[i] >= gaps[i+1] for i in range(len(gaps)-1))
        # Spearman correlation with t
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(t_values, gaps)

        # Find crossover point
        crossover = None
        for i in range(len(gaps)-1):
            if gaps[i] > 0 and gaps[i+1] <= 0:
                # Linear interpolation to find exact crossover
                crossover = t_values[i] + (t_values[i+1] - t_values[i]) * gaps[i] / (gaps[i] - gaps[i+1])
                break

        print(f"\n  Monotonic: {'YES' if is_monotonic else 'NO'}")
        print(f"  Spearman ρ: {rho:.3f} (p={p_val:.4f})")
        if crossover is not None:
            print(f"  Crossover: t≈{crossover:.2f}")

        results["pairs"][pair_key] = {
            "trait_a": trait_a,
            "trait_b": trait_b,
            "holland_distance": holland_dist,
            "t_values": t_values,
            "gaps": [float(g) for g in gaps],
            "is_monotonic": bool(is_monotonic),
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "crossover": float(crossover) if crossover is not None else None,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Cross-dim interpolation preserves continuous structure?")
    print(f"{'='*70}")
    n_monotonic = sum(1 for p in results["pairs"].values() if p["is_monotonic"])
    print(f"  Monotonic pairs: {n_monotonic}/{len(results['pairs'])}")
    for pair_key, data in results["pairs"].items():
        print(f"  {pair_key:>30}: ρ={data['spearman_rho']:.3f}, "
              f"monotonic={'YES' if data['is_monotonic'] else 'NO'}, "
              f"crossover={'t=' + str(round(data['crossover'], 2)) if data['crossover'] is not None else 'N/A'}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_dim_interpolation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
