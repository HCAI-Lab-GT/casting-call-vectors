#!/usr/bin/env python
"""
Transitive 3-model transfer: SmolLM3(2048d) → Qwen(3584d) → Marin(4096d).

Tests whether the 5D personality bridge COMPOSES across models.
The vectors never directly touch Marin — they go through Qwen as intermediary.

Pipeline:
1. Extract 5D coords for SmolLM3, Qwen, and Marin
2. Fit Procrustes: SmolLM3 → Qwen (independent of Marin)
3. Fit Procrustes: Qwen → Marin (independent of SmolLM3)
4. Chain: SmolLM3 5D → R1 → Qwen 5D → R2 → Marin 5D → Marin full dim
5. Steer Marin 8B with these transitively-transferred vectors

Compare:
- Self (Marin native): positive control
- Direct (SmolLM3 → Marin): single-hop
- Transitive (SmolLM3 → Qwen → Marin): two-hop, never touches Marin directly
- Random: negative control

If transitive ≈ direct ≈ self, the 5D space is CANONICALLY universal.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="transitive-transfer")

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
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d


def fit_procrustes(source_5d, target_5d):
    """Fit Procrustes rotation + scale from source 5D to target 5D."""
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
    smol_id = "HuggingFaceTB/SmolLM3-3B"           # 2048d, 36 layers
    qwen_id = "Qwen/Qwen2.5-7B-Instruct"            # 3584d, 28 layers
    marin_id = "marin-community/marin-8b-instruct"   # 4096d, 32 layers
    device = "cuda:0"
    alpha = 1.0

    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load residual vectors for all 3 models (no model loading needed)
    logger.info("Loading residual vectors for 3 models...")
    smol_residual, _ = load_residual_vectors(smol_id, riasec_dir)
    qwen_residual, _ = load_residual_vectors(qwen_id, riasec_dir)
    marin_residual, marin_mid = load_residual_vectors(marin_id, riasec_dir)

    smol_dim = smol_residual[TRAITS[0]].shape[0]
    qwen_dim = qwen_residual[TRAITS[0]].shape[0]
    marin_dim = marin_residual[TRAITS[0]].shape[0]
    logger.info("Dimensions: SmolLM3=%d, Qwen=%d, Marin=%d", smol_dim, qwen_dim, marin_dim)

    # Get 5D coordinates for each model
    smol_5d, _ = get_5d_coordinates(smol_residual)
    qwen_5d, _ = get_5d_coordinates(qwen_residual)
    marin_5d, marin_basis = get_5d_coordinates(marin_residual)

    # === DIRECT TRANSFER: SmolLM3 → Marin ===
    R_direct, s_direct = fit_procrustes(smol_5d, marin_5d)
    direct_vectors = {}
    for t in TRAITS:
        aligned = s_direct * (R_direct @ smol_5d[t])
        direct_vectors[t] = (marin_basis.T @ aligned).astype(np.float32)

    # === TRANSITIVE TRANSFER: SmolLM3 → Qwen → Marin ===
    R_sq, s_sq = fit_procrustes(smol_5d, qwen_5d)   # SmolLM3 → Qwen
    R_qm, s_qm = fit_procrustes(qwen_5d, marin_5d)  # Qwen → Marin

    transitive_vectors = {}
    for t in TRAITS:
        # Hop 1: SmolLM3 5D → Qwen 5D
        qwen_coord = s_sq * (R_sq @ smol_5d[t])
        # Hop 2: Qwen 5D → Marin 5D
        marin_coord = s_qm * (R_qm @ qwen_coord)
        # Reconstruct in Marin's full dimension
        transitive_vectors[t] = (marin_basis.T @ marin_coord).astype(np.float32)

    # === Analyze vector quality ===
    print(f"\n{'='*70}")
    print(f"TRANSITIVE 3-MODEL TRANSFER")
    print(f"SmolLM3({smol_dim}d) → Qwen({qwen_dim}d) → Marin({marin_dim}d)")
    print(f"{'='*70}")

    print(f"\n--- Cosine: transferred vs native Marin vectors ---")
    print(f"  {'Trait':>14}  {'Direct':>8}  {'Transitive':>10}")
    cos_direct_all = []
    cos_trans_all = []
    for t in TRAITS:
        cos_d = np.dot(direct_vectors[t], marin_residual[t]) / \
                (np.linalg.norm(direct_vectors[t]) * np.linalg.norm(marin_residual[t]))
        cos_t = np.dot(transitive_vectors[t], marin_residual[t]) / \
                (np.linalg.norm(transitive_vectors[t]) * np.linalg.norm(marin_residual[t]))
        cos_direct_all.append(cos_d)
        cos_trans_all.append(cos_t)
        print(f"  {t:>14}  {cos_d:>8.4f}  {cos_t:>10.4f}")
    print(f"  {'Mean':>14}  {np.mean(cos_direct_all):>8.4f}  {np.mean(cos_trans_all):>10.4f}")

    # How close are direct vs transitive vectors to each other?
    print(f"\n--- Cosine: direct vs transitive vectors ---")
    for t in TRAITS:
        cos = np.dot(direct_vectors[t], transitive_vectors[t]) / \
              (np.linalg.norm(direct_vectors[t]) * np.linalg.norm(transitive_vectors[t]))
        print(f"  {t:>14}: {cos:.4f}")

    # === Composed rotation analysis ===
    R_composed = R_qm @ R_sq  # Transitive rotation
    # How close is composed to direct?
    R_diff = R_direct - s_direct / (s_sq * s_qm) * R_composed  # Scale-adjusted
    print(f"\n--- Rotation analysis ---")
    print(f"  ||R_direct - R_composed||_F = {np.linalg.norm(R_direct - R_composed):.4f}")
    cos_R = np.trace(R_direct.T @ R_composed) / 5.0  # Normalized trace
    print(f"  Rotation similarity (normalized trace): {cos_R:.4f}")

    # === Load Marin 8B for steering evaluation ===
    logger.info("Loading target model: %s", marin_id)
    tokenizer = AutoTokenizer.from_pretrained(marin_id)
    model = AutoModelForCausalLM.from_pretrained(
        marin_id, torch_dtype=torch.float16, device_map=device)
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

    # Evaluate all conditions
    print(f"\n--- Steering evaluation (α={alpha}) ---")

    r_self = eval_discrimination(model, tokenizer, device, blocks, marin_mid,
                                marin_residual, alpha, baseline)
    print(f"  Self (Marin native):          {r_self['delta_accuracy']:.0%} ({r_self['correct']}/{r_self['total']}), "
          f"Δ={r_self['mean_delta']:+.4f}")

    r_direct = eval_discrimination(model, tokenizer, device, blocks, marin_mid,
                                   direct_vectors, alpha, baseline)
    print(f"  Direct (SmolLM3→Marin):       {r_direct['delta_accuracy']:.0%} ({r_direct['correct']}/{r_direct['total']}), "
          f"Δ={r_direct['mean_delta']:+.4f}")

    r_transitive = eval_discrimination(model, tokenizer, device, blocks, marin_mid,
                                       transitive_vectors, alpha, baseline)
    print(f"  Transitive (SmolLM3→Qwen→M):  {r_transitive['delta_accuracy']:.0%} ({r_transitive['correct']}/{r_transitive['total']}), "
          f"Δ={r_transitive['mean_delta']:+.4f}")

    # Random control
    np.random.seed(42)
    random_vecs = {}
    for t in TRAITS:
        rv = np.random.randn(marin_dim).astype(np.float32)
        rv = rv / np.linalg.norm(rv) * np.linalg.norm(marin_residual[t])
        random_vecs[t] = rv
    r_random = eval_discrimination(model, tokenizer, device, blocks, marin_mid,
                                   random_vecs, alpha, baseline)
    print(f"  Random (matched norm):        {r_random['delta_accuracy']:.0%} ({r_random['correct']}/{r_random['total']}), "
          f"Δ={r_random['mean_delta']:+.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Self (Marin native):      {r_self['delta_accuracy']:.0%}")
    print(f"  Direct (1-hop):           {r_direct['delta_accuracy']:.0%}")
    print(f"  Transitive (2-hop):       {r_transitive['delta_accuracy']:.0%}")
    print(f"  Random:                   {r_random['delta_accuracy']:.0%}")
    print(f"\n  Path: SmolLM3({smol_dim}d) → Qwen({qwen_dim}d) → Marin({marin_dim}d)")
    if r_transitive['delta_accuracy'] >= r_self['delta_accuracy'] - 0.05:
        print(f"\n  ** TRANSITIVE TRANSFER PRESERVES FULL DISCRIMINATION **")
        print(f"     The 5D personality space is canonically universal!")

    results = {
        "source": smol_id,
        "intermediary": qwen_id,
        "target": marin_id,
        "dimensions": {"source": int(smol_dim), "intermediary": int(qwen_dim), "target": int(marin_dim)},
        "alpha": alpha,
        "self": r_self,
        "direct": r_direct,
        "transitive": r_transitive,
        "random": r_random,
        "cosines_vs_native": {
            "direct": {t: float(c) for t, c in zip(TRAITS, cos_direct_all)},
            "transitive": {t: float(c) for t, c in zip(TRAITS, cos_trans_all)},
        },
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transitive_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
