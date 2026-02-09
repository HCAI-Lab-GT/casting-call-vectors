#!/usr/bin/env python
"""
Orthogonal steering control: Personality is in 5D. What about the other d-5 dims?

Constructs steering vectors that are:
1. IN the 5D personality subspace (positive control — should work)
2. ORTHOGONAL to the 5D personality subspace (negative control — should NOT work)
3. Random (baseline)

If personality is truly concentrated in 5 dimensions, then:
- In-subspace steering → high discrimination
- Orthogonal steering → zero discrimination (same as unsteered)
- Random → ~50% (chance)

This is the strongest possible demonstration that personality is a low-rank phenomenon.
Tests on both SmolLM3-3B (completion format) and Marin-8B (chat format).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="orthogonal-control")

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
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer, shared_dir


def get_5d_basis(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]  # (5, hidden_dim)
    return basis_5d


def construct_orthogonal_vectors(residual_vectors, basis_5d, shared_dir, n_trials=5):
    """Construct vectors orthogonal to the personality subspace."""
    hidden_dim = residual_vectors[TRAITS[0]].shape[0]

    # The personality subspace is spanned by: shared_dir (1d) + basis_5d (5d) = 6d total
    # We want vectors orthogonal to ALL of these
    personality_basis = np.vstack([shared_dir.reshape(1, -1), basis_5d])  # (6, hidden_dim)

    # Use Gram-Schmidt / QR to get orthogonal complement
    Q, _ = np.linalg.qr(personality_basis.T)  # (hidden_dim, 6)
    personality_projector = Q @ Q.T  # Projects onto personality subspace

    orthogonal_vecs = {}
    np.random.seed(42)

    for trial in range(n_trials):
        trial_vecs = {}
        for t in TRAITS:
            # Generate random vector
            rv = np.random.randn(hidden_dim).astype(np.float64)
            # Remove personality subspace component
            personality_component = personality_projector @ rv
            ortho = rv - personality_component
            # Match norm of the real personality vector
            target_norm = np.linalg.norm(residual_vectors[t])
            ortho = ortho / np.linalg.norm(ortho) * target_norm

            # Verify orthogonality
            for basis_vec in personality_basis:
                dot = abs(np.dot(ortho, basis_vec / np.linalg.norm(basis_vec)))
                assert dot < 1e-6, f"Not orthogonal: dot={dot}"

            trial_vecs[t] = ortho.astype(np.float32)
        orthogonal_vecs[trial] = trial_vecs

    return orthogonal_vecs


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
    target_id = "marin-community/marin-8b-instruct"
    device = "cuda:0"
    alpha = 1.0
    n_trials = 5

    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer, shared_dir = load_residual_vectors(target_id, riasec_dir)
    basis_5d = get_5d_basis(residual)
    hidden_dim = residual[TRAITS[0]].shape[0]

    # Construct orthogonal vectors
    logger.info("Constructing orthogonal vectors (%d trials)...", n_trials)
    ortho_trials = construct_orthogonal_vectors(residual, basis_5d, shared_dir, n_trials)

    # Also construct in-subspace control: project residual onto 5D then reconstruct
    in_subspace = {}
    for t in TRAITS:
        coords = basis_5d @ residual[t]
        reconstructed = basis_5d.T @ coords
        in_subspace[t] = reconstructed.astype(np.float32)

    # Verify: in-subspace vectors should be nearly identical to residuals
    print(f"\n--- In-subspace reconstruction quality ---")
    for t in TRAITS:
        cos = np.dot(in_subspace[t], residual[t]) / \
              (np.linalg.norm(in_subspace[t]) * np.linalg.norm(residual[t]))
        norm_ratio = np.linalg.norm(in_subspace[t]) / np.linalg.norm(residual[t])
        print(f"  {t:>14}: cosine={cos:.6f}, norm_ratio={norm_ratio:.6f}")

    # Load model
    logger.info("Loading model: %s", target_id)
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
    print(f"ORTHOGONAL CONTROL: Is personality concentrated in 5D?")
    print(f"Model: {target_id} ({hidden_dim}d)")
    print(f"{'='*70}")

    # Test 1: Real residual vectors (positive control)
    r_real = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                residual, alpha, baseline)
    print(f"\n  Real residual vectors:      {r_real['delta_accuracy']:.0%} ({r_real['correct']}/{r_real['total']})")

    # Test 2: In-subspace (reconstructed from 5D)
    r_in_sub = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                   in_subspace, alpha, baseline)
    print(f"  In-subspace (5D→{hidden_dim}d):     {r_in_sub['delta_accuracy']:.0%} ({r_in_sub['correct']}/{r_in_sub['total']})")

    # Test 3: Orthogonal vectors (should be ~50%)
    ortho_accs = []
    for trial_idx in range(n_trials):
        r_ortho = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                      ortho_trials[trial_idx], alpha, baseline)
        ortho_accs.append(r_ortho['delta_accuracy'])
        print(f"  Orthogonal trial {trial_idx+1}:         {r_ortho['delta_accuracy']:.0%} ({r_ortho['correct']}/{r_ortho['total']})")

    mean_ortho = np.mean(ortho_accs)
    std_ortho = np.std(ortho_accs)

    # Test 4: Random (baseline)
    np.random.seed(99)
    random_vecs = {}
    for t in TRAITS:
        rv = np.random.randn(hidden_dim).astype(np.float32)
        rv = rv / np.linalg.norm(rv) * np.linalg.norm(residual[t])
        random_vecs[t] = rv
    r_random = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                   random_vecs, alpha, baseline)
    print(f"  Random:                     {r_random['delta_accuracy']:.0%} ({r_random['correct']}/{r_random['total']})")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Real personality vectors: {r_real['delta_accuracy']:.0%}")
    print(f"  5D-reconstructed:        {r_in_sub['delta_accuracy']:.0%}")
    print(f"  Orthogonal to 5D:        {mean_ortho:.0%} ± {std_ortho:.0%} (n={n_trials})")
    print(f"  Random:                  {r_random['delta_accuracy']:.0%}")
    print(f"\n  Personality subspace: 5 / {hidden_dim} dimensions ({5/hidden_dim*100:.2f}%)")

    if r_in_sub['delta_accuracy'] >= r_real['delta_accuracy'] - 0.05:
        print(f"  5D-reconstructed matches real → personality IS in 5D")
    if mean_ortho < 0.55:
        print(f"  Orthogonal ≈ chance → remaining {hidden_dim-5}d carries NO personality")

    results = {
        "model": target_id,
        "hidden_dim": hidden_dim,
        "alpha": alpha,
        "real": r_real,
        "in_subspace": r_in_sub,
        "orthogonal_accs": [float(a) for a in ortho_accs],
        "orthogonal_mean": float(mean_ortho),
        "orthogonal_std": float(std_ortho),
        "random": r_random,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "orthogonal_control.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
