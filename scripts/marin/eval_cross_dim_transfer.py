#!/usr/bin/env python
"""
Cross-DIMENSIONAL steering transfer: Can vectors from a 2048-dim model
steer a 4096-dim model?

Pipeline:
1. Extract 5D personality coordinates from SmolLM3 (2048d)
2. Extract 5D personality coordinates from Marin 8B (4096d)
3. Procrustes-align in 5D
4. For each SmolLM3 trait vector: project to 5D, align, reconstruct in 4096d
5. Inject into Marin 8B and test discrimination

Also tests: Marin 8B self-vectors as positive control.

This is the hardest transfer test: different architecture, different size,
different hidden dimension.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-dim-transfer")

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

    # Compute residual vectors
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
    """Project residual vectors into 5D using PCA of the 6 residuals."""
    V = np.stack([residual_vectors[t] for t in TRAITS])
    # SVD to get 5D basis
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    # Use top 5 components (6th should be ~0)
    basis_5d = Vt[:5]  # (5, hidden_dim)

    # Project each vector to 5D coordinates
    coords_5d = {}
    for t in TRAITS:
        coords_5d[t] = basis_5d @ residual_vectors[t]  # (5,)

    return coords_5d, basis_5d


def procrustes_5d(source_coords, target_coords):
    """Procrustes alignment in 5D: find rotation R such that R @ source ≈ target."""
    S = np.stack([source_coords[t] for t in TRAITS])  # (6, 5)
    T = np.stack([target_coords[t] for t in TRAITS])  # (6, 5)

    # Normalize
    S_normed = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_normed = T / np.linalg.norm(T, axis=1, keepdims=True)

    # Find rotation: M = S^T @ T, then SVD
    M = S_normed.T @ T_normed  # (5, 5)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt  # (5, 5) rotation matrix

    # Compute scale factor
    source_norms = np.array([np.linalg.norm(source_coords[t]) for t in TRAITS])
    target_norms = np.array([np.linalg.norm(target_coords[t]) for t in TRAITS])
    scale = np.mean(target_norms) / np.mean(source_norms)

    return R, scale


def pairwise_logprob_chat(model, tokenizer, device, desc_a, desc_b):
    messages = [
        {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
        {"role": "user", "content": (
            f"Which describes you better?\n"
            f"A) I am {desc_a}\n"
            f"B) I am {desc_b}\n"
            f"Answer:"
        )},
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


def eval_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    correct = 0
    total = 0
    deltas = []

    for steer_trait in TRAITS:
        vec = vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
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
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    if steer_trait not in (trait_a, trait_b):
                        continue
                    gap = pairwise_logprob_chat(model, tokenizer, device,
                                               TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                    base_gap = baseline[f"{trait_a}-{trait_b}"]
                    if steer_trait == trait_a:
                        d = gap - base_gap
                    else:
                        d = base_gap - gap
                    correct += int(d > 0)
                    total += 1
                    deltas.append(d)
        finally:
            hook_handle.remove()

    return {
        "delta_accuracy": correct / total if total else 0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0,
        "correct": correct,
        "total": total,
    }


def main():
    source_id = "HuggingFaceTB/SmolLM3-3B"    # 2048-dim
    target_id = "marin-community/marin-8b-instruct"  # 4096-dim
    device = "cuda:0"
    alpha = 1.0

    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load residual vectors for both models
    logger.info("Loading vectors for %s", source_id)
    source_residual, source_mid = load_residual_vectors(source_id, riasec_dir)

    logger.info("Loading vectors for %s", target_id)
    target_residual, target_mid = load_residual_vectors(target_id, riasec_dir)

    source_dim = source_residual[TRAITS[0]].shape[0]
    target_dim = target_residual[TRAITS[0]].shape[0]
    logger.info("Source dim: %d, Target dim: %d", source_dim, target_dim)

    # Get 5D coordinates for both
    source_5d, source_basis = get_5d_coordinates(source_residual)
    target_5d, target_basis = get_5d_coordinates(target_residual)

    # Procrustes alignment in 5D
    R, scale = procrustes_5d(source_5d, target_5d)

    # Validate alignment quality (LOO test)
    loo_correct = 0
    loo_cosines = []
    for held_out in TRAITS:
        # Fit on 5 traits
        train_traits = [t for t in TRAITS if t != held_out]
        train_source = {t: source_5d[t] for t in train_traits}
        train_target = {t: target_5d[t] for t in train_traits}

        S = np.stack([train_source[t] for t in train_traits])
        T = np.stack([train_target[t] for t in train_traits])
        S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
        T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
        M = S_n.T @ T_n
        U, _, Vt = np.linalg.svd(M)
        R_loo = U @ Vt
        s_loo = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))

        # Predict held-out
        pred_5d = s_loo * (R_loo @ source_5d[held_out])
        actual_5d = target_5d[held_out]

        cos = np.dot(pred_5d, actual_5d) / (np.linalg.norm(pred_5d) * np.linalg.norm(actual_5d))
        loo_cosines.append(cos)

        # Check: is predicted nearest to actual?
        best_match = max(TRAITS, key=lambda t: np.dot(pred_5d, target_5d[t]) /
                        (np.linalg.norm(pred_5d) * np.linalg.norm(target_5d[t])))
        loo_correct += int(best_match == held_out)

    logger.info("LOO in 5D: %d/6, min cosine: %.4f", loo_correct, min(loo_cosines))

    # Build transferred vectors: source → 5D → Procrustes → 5D → target full dim
    transferred_vectors = {}
    for t in TRAITS:
        # Source residual → 5D coordinates
        source_coord = source_5d[t]
        # Procrustes rotate and scale
        aligned_coord = scale * (R @ source_coord)
        # Reconstruct in target's full dimension using target basis
        transferred_vectors[t] = (target_basis.T @ aligned_coord).astype(np.float32)

    # Verify transferred vectors are in target dimension
    for t in TRAITS:
        assert transferred_vectors[t].shape[0] == target_dim, \
            f"Expected {target_dim}, got {transferred_vectors[t].shape[0]}"

    # Validate: cosine between transferred and actual target vectors
    print(f"\n{'='*70}")
    print(f"CROSS-DIMENSIONAL TRANSFER: {source_id} → {target_id}")
    print(f"Source dim: {source_dim}, Target dim: {target_dim}")
    print(f"{'='*70}")

    print(f"\n--- Alignment quality ---")
    print(f"  LOO prediction: {loo_correct}/6 correct")
    print(f"  LOO cosines: {', '.join(f'{c:.3f}' for c in loo_cosines)}")

    print(f"\n--- Transferred vs actual target vectors ---")
    for t in TRAITS:
        cos = np.dot(transferred_vectors[t], target_residual[t]) / \
              (np.linalg.norm(transferred_vectors[t]) * np.linalg.norm(target_residual[t]))
        print(f"  {t:>14}: cosine = {cos:.4f}")

    # Load target model for steering evaluation
    logger.info("Loading target model: %s", target_id)
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
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

    # Test conditions
    results = {"baseline": baseline, "alpha": alpha}

    print(f"\n--- Steering evaluation ---")

    # Condition 1: Self vectors (positive control)
    r_self = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                target_residual, alpha, baseline)
    print(f"  Self (Marin 8B):           {r_self['delta_accuracy']:.0%} ({r_self['correct']}/{r_self['total']}), "
          f"mean_delta={r_self['mean_delta']:+.4f}")
    results["self"] = r_self

    # Condition 2: Cross-dim transferred vectors
    r_transfer = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                    transferred_vectors, alpha, baseline)
    print(f"  Cross-dim (SmolLM→Marin):  {r_transfer['delta_accuracy']:.0%} ({r_transfer['correct']}/{r_transfer['total']}), "
          f"mean_delta={r_transfer['mean_delta']:+.4f}")
    results["cross_dim_transfer"] = r_transfer

    # Condition 3: Random vectors (control)
    np.random.seed(42)
    target_norms = {t: np.linalg.norm(target_residual[t]) for t in TRAITS}
    random_vecs = {}
    for t in TRAITS:
        rv = np.random.randn(target_dim).astype(np.float32)
        rv = rv / np.linalg.norm(rv) * target_norms[t]
        random_vecs[t] = rv

    r_random = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                  random_vecs, alpha, baseline)
    print(f"  Random (matched norm):     {r_random['delta_accuracy']:.0%} ({r_random['correct']}/{r_random['total']}), "
          f"mean_delta={r_random['mean_delta']:+.4f}")
    results["random"] = r_random

    # Also test alpha=2
    print(f"\n--- At alpha=2 ---")
    for label, vecs in [("Self", target_residual), ("Cross-dim", transferred_vectors)]:
        r = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                               vecs, 2.0, baseline)
        print(f"  {label:>20}: {r['delta_accuracy']:.0%} ({r['correct']}/{r['total']}), "
              f"mean_delta={r['mean_delta']:+.4f}")
        results[f"{label.lower().replace('-', '_')}_alpha2"] = r

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Self (Marin 8B, 4096d):        {r_self['delta_accuracy']:.0%}")
    print(f"  Cross-dim (SmolLM→Marin):       {r_transfer['delta_accuracy']:.0%}")
    print(f"  Random:                          {r_random['delta_accuracy']:.0%}")
    print(f"  Source dim: {source_dim}, Target dim: {target_dim}")
    print(f"  Personality subspace: 5D (shared between both)")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_dim_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
