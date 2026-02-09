#!/usr/bin/env python
"""
Reverse cross-dimensional transfer: Marin 8B (4096d) → SmolLM3 (2048d).

This is the harder direction — compressing from higher to lower dimension.
If both directions work, the 5D personality bridge is truly bidirectional.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-dim-reverse")

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
    return residual, mid_layer


def get_5d_coordinates(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {}
    for t in TRAITS:
        coords_5d[t] = basis_5d @ residual_vectors[t]
    return coords_5d, basis_5d


def pairwise_logprob(model, tokenizer, device, desc_a, desc_b):
    prompt = f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    a_candidates = ["A", " A", "a", " a"]
    b_candidates = ["B", " B", "b", " b"]
    a_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in a_candidates if tokenizer.encode(t, add_special_tokens=False))
    b_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in b_candidates if tokenizer.encode(t, add_special_tokens=False))
    return a_lp - b_lp


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
                    gap = pairwise_logprob(model, tokenizer, device,
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
    source_id = "marin-community/marin-8b-instruct"  # 4096-dim
    target_id = "HuggingFaceTB/SmolLM3-3B"             # 2048-dim
    device = "cuda:0"
    alpha = 2.0  # SmolLM3 optimal alpha

    riasec_dir = _repo_root() / "persona_data/model_inits"

    logger.info("Loading vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    target_residual, target_mid = load_residual_vectors(target_id, riasec_dir)

    source_dim = source_residual[TRAITS[0]].shape[0]
    target_dim = target_residual[TRAITS[0]].shape[0]

    source_5d, _ = get_5d_coordinates(source_residual)
    target_5d, target_basis = get_5d_coordinates(target_residual)

    # Procrustes in 5D: source → target
    S = np.stack([source_5d[t] for t in TRAITS])
    T = np.stack([target_5d[t] for t in TRAITS])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))

    # Transfer: source → 5D → Procrustes → target 5D → target full dim
    transferred = {}
    for t in TRAITS:
        aligned = scale * (R @ source_5d[t])
        transferred[t] = (target_basis.T @ aligned).astype(np.float32)

    # Validate cosines
    print(f"\n{'='*70}")
    print(f"REVERSE CROSS-DIM TRANSFER: {source_id} → {target_id}")
    print(f"Source dim: {source_dim} → Target dim: {target_dim}")
    print(f"{'='*70}")

    print(f"\n--- Transferred vs actual target vectors ---")
    for t in TRAITS:
        cos = np.dot(transferred[t], target_residual[t]) / \
              (np.linalg.norm(transferred[t]) * np.linalg.norm(target_residual[t]))
        print(f"  {t:>14}: cosine = {cos:.4f}")

    # Load target model
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
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device,
                                  TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    # Test
    print(f"\n--- Steering evaluation (α={alpha}) ---")

    r_self = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                target_residual, alpha, baseline)
    print(f"  Self (SmolLM3):             {r_self['delta_accuracy']:.0%} ({r_self['correct']}/{r_self['total']})")

    r_transfer = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                    transferred, alpha, baseline)
    print(f"  Cross-dim (Marin→SmolLM3):  {r_transfer['delta_accuracy']:.0%} ({r_transfer['correct']}/{r_transfer['total']})")

    print(f"\n{'='*70}")
    print(f"BIDIRECTIONAL CROSS-DIM SUMMARY")
    print(f"{'='*70}")
    print(f"  Forward  (2048→4096): Self=97%, Transfer=97% [EQUAL]")
    print(f"  Reverse  (4096→2048): Self={r_self['delta_accuracy']:.0%}, Transfer={r_transfer['delta_accuracy']:.0%}")

    results = {
        "source": source_id,
        "target": target_id,
        "alpha": alpha,
        "self": r_self,
        "cross_dim": r_transfer,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_dim_transfer_reverse.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
