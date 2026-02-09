#!/usr/bin/env python
"""
Cross-trait interference under transfer.

When steering toward trait X, we affect all 6 traits. The interference
pattern (which other traits go up/down) reflects the RIASEC hexagonal structure.

Question: does this interference pattern TRANSFER across models?
If SmolLM3 shows artistic→social suppression, does Marin 8B show the same
pattern when using SmolLM3's transferred vectors?

This tests whether the BEHAVIORAL structure (not just the geometry) transfers.

Compares:
1. Marin 8B self-steering interference matrix
2. Marin 8B with SmolLM3-transferred vectors interference matrix
3. Correlation between the two matrices
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="transfer-interference")

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


def canonical_sign_convention(coords_5d):
    signs = np.ones(5)
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1
    for pc in range(1, 5):
        loadings = {t: coords_5d[t][pc] for t in TRAITS}
        max_trait = max(loadings, key=lambda t: abs(loadings[t]))
        if loadings[max_trait] > 0:
            signs[pc] = -1
    return signs


def build_transferred_vectors(source_coords, source_basis, target_coords, target_basis, sign_vector):
    corrected_coords = {t: sign_vector * source_coords[t] for t in TRAITS}
    source_norms = np.mean([np.linalg.norm(corrected_coords[t]) for t in TRAITS])
    target_norms = np.mean([np.linalg.norm(target_coords[t]) for t in TRAITS])
    scale = target_norms / source_norms
    transferred = {}
    for t in TRAITS:
        target_coord = scale * corrected_coords[t]
        transferred[t] = (target_basis.T @ target_coord).astype(np.float32)
    return transferred


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


def compute_interference_matrix(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    """6×6 interference matrix: [steer_trait][measured_trait] = delta."""
    matrix = np.zeros((6, 6))

    for si, steer_trait in enumerate(TRAITS):
        vec = vectors[steer_trait]
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
                    matrix[si, TRAITS.index(t)] = trait_deltas[t]
        finally:
            hook_handle.remove()

    return matrix


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    source_id = "HuggingFaceTB/SmolLM3-3B"
    target_id = "marin-community/marin-8b-instruct"

    # Load vectors
    logger.info("Loading vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    source_coords, source_basis = get_5d_coords_and_basis(source_residual)
    target_residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    target_coords, target_basis = get_5d_coords_and_basis(target_residual)

    # Transfer
    source_canon = canonical_sign_convention(source_coords)
    target_canon = canonical_sign_convention(target_coords)
    correct_signs = target_canon * source_canon
    transferred = build_transferred_vectors(
        source_coords, source_basis, target_coords, target_basis, correct_signs)

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

    print(f"\n{'='*70}")
    print(f"TRANSFER INTERFERENCE MATRIX COMPARISON")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"Source: SmolLM3-3B")
    print(f"{'='*70}")

    # Self-steering interference matrix
    logger.info("Computing self-steering interference matrix...")
    self_matrix = compute_interference_matrix(
        model, tokenizer, device, blocks, mid_layer, target_residual, alpha, baseline)

    print(f"\n--- Self-Steering Interference Matrix ---")
    print(f"  {'':>12}", end="")
    for t in TRAITS:
        print(f"  {t[:4]:>6}", end="")
    print()
    for i, t in enumerate(TRAITS):
        print(f"  →{t:>10}", end="")
        for j in range(6):
            print(f"  {self_matrix[i,j]:>+6.2f}", end="")
        print()

    # Transfer interference matrix
    logger.info("Computing transfer interference matrix...")
    transfer_matrix = compute_interference_matrix(
        model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)

    print(f"\n--- Transfer (SmolLM3→Marin) Interference Matrix ---")
    print(f"  {'':>12}", end="")
    for t in TRAITS:
        print(f"  {t[:4]:>6}", end="")
    print()
    for i, t in enumerate(TRAITS):
        print(f"  →{t:>10}", end="")
        for j in range(6):
            print(f"  {transfer_matrix[i,j]:>+6.2f}", end="")
        print()

    # Compare matrices
    print(f"\n--- Matrix Comparison ---")

    # Flatten and correlate
    self_flat = self_matrix.flatten()
    transfer_flat = transfer_matrix.flatten()
    from scipy.stats import pearsonr, spearmanr
    r, p_r = pearsonr(self_flat, transfer_flat)
    rho, p_rho = spearmanr(self_flat, transfer_flat)
    print(f"  Pearson r: {r:.3f} (p={p_r:.6f})")
    print(f"  Spearman ρ: {rho:.3f} (p={p_rho:.6f})")

    # Off-diagonal only (more informative — diagonal is trivially positive)
    mask = ~np.eye(6, dtype=bool)
    self_off = self_matrix[mask]
    transfer_off = transfer_matrix[mask]
    r_off, p_off = pearsonr(self_off, transfer_off)
    rho_off, p_rho_off = spearmanr(self_off, transfer_off)
    print(f"\n  Off-diagonal only:")
    print(f"  Pearson r: {r_off:.3f} (p={p_off:.6f})")
    print(f"  Spearman ρ: {rho_off:.3f} (p={p_rho_off:.6f})")

    # Per-trait interference preservation
    print(f"\n--- Per-Trait Interference Preservation ---")
    for i, steer_trait in enumerate(TRAITS):
        self_profile = self_matrix[i]
        transfer_profile = transfer_matrix[i]
        r_t, _ = pearsonr(self_profile, transfer_profile)
        # Check if same top/bottom
        self_top = TRAITS[np.argmax(self_profile)]
        transfer_top = TRAITS[np.argmax(transfer_profile)]
        self_bot = TRAITS[np.argmin(self_profile)]
        transfer_bot = TRAITS[np.argmin(transfer_profile)]
        print(f"  →{steer_trait:>15}: r={r_t:.3f}, top={self_top}→{transfer_top} {'✓' if self_top == transfer_top else '✗'}, "
              f"bot={self_bot}→{transfer_bot} {'✓' if self_bot == transfer_bot else '✗'}")

    # Holland hexagonal structure in both matrices
    print(f"\n--- Holland Structure in Interference ---")
    HOLLAND = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]

    for label, mat in [("Self", self_matrix), ("Transfer", transfer_matrix)]:
        adj_sum = 0
        alt_sum = 0
        opp_sum = 0
        n_adj = n_alt = n_opp = 0
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                hi = HOLLAND.index(TRAITS[i])
                hj = HOLLAND.index(TRAITS[j])
                dist = min(abs(hi - hj), 6 - abs(hi - hj))
                val = mat[i, j]
                if dist == 1:
                    adj_sum += val
                    n_adj += 1
                elif dist == 2:
                    alt_sum += val
                    n_alt += 1
                elif dist == 3:
                    opp_sum += val
                    n_opp += 1

        print(f"  {label:>8}: adj={adj_sum/n_adj:+.3f}, alt={alt_sum/n_alt:+.3f}, opp={opp_sum/n_opp:+.3f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Full matrix correlation: r={r:.3f}")
    print(f"  Off-diagonal correlation: r={r_off:.3f}")
    if r_off > 0.8:
        print(f"  CONCLUSION: Interference pattern TRANSFERS strongly (r > 0.8)")
    elif r_off > 0.5:
        print(f"  CONCLUSION: Interference pattern partially transfers (0.5 < r < 0.8)")
    else:
        print(f"  CONCLUSION: Interference pattern does NOT transfer well (r < 0.5)")

    # Save
    results = {
        "self_matrix": self_matrix.tolist(),
        "transfer_matrix": transfer_matrix.tolist(),
        "full_pearson_r": float(r),
        "full_pearson_p": float(p_r),
        "offdiag_pearson_r": float(r_off),
        "offdiag_pearson_p": float(p_off),
        "offdiag_spearman_rho": float(rho_off),
    }
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer_interference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
