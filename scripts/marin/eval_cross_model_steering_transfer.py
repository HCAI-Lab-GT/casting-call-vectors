#!/usr/bin/env python
"""
Cross-model steering transfer: Can vectors extracted from Model A steer Model B?

Tests the practical implications of cross-model geometric alignment.
Llama 1B and SmolLM3 3B share hidden dim = 2048 and have the highest
residual correlation (r=0.991). If cross-model transfer works, it means:
- Extract vectors cheaply on a small model, steer a different model
- Universal personality basis exists across architectures

We test 4 conditions:
1. Llama steered with Llama vectors (baseline)
2. Llama steered with SmolLM3 vectors (cross-model)
3. SmolLM3 steered with SmolLM3 vectors (baseline)
4. SmolLM3 steered with Llama vectors (cross-model)

And also with Procrustes-aligned vectors:
5. Llama steered with Procrustes-rotated SmolLM3 vectors
6. SmolLM3 steered with Procrustes-rotated Llama vectors
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-model-transfer")

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


def load_residual_vectors(model_id, mid_layer):
    """Load and compute residual vectors for a given model."""
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

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

    return residual


def procrustes_align(source_vecs, target_vecs):
    """
    Compute Procrustes rotation to align source vectors to target vectors.
    Returns aligned versions of source_vecs.
    """
    S = np.stack([source_vecs[t] for t in TRAITS])
    T = np.stack([target_vecs[t] for t in TRAITS])

    # Normalize to unit norm for alignment
    S_norms = np.linalg.norm(S, axis=1, keepdims=True)
    T_norms = np.linalg.norm(T, axis=1, keepdims=True)
    S_normed = S / S_norms
    T_normed = T / T_norms

    # Procrustes: find R that minimizes ||T_normed - S_normed @ R||
    M = S_normed.T @ T_normed
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Apply rotation to original (unnormalized) source vectors
    # Scale each aligned vector to match target norm
    aligned = {}
    for i, t in enumerate(TRAITS):
        rotated = S[i] @ R
        # Scale to match target magnitude
        scale = T_norms[i, 0] / S_norms[i, 0]
        aligned[t] = rotated * scale

    return aligned


def pairwise_logprob_chat(model, tokenizer, device, trait_a, trait_b):
    desc_a = TRAIT_DESCRIPTIONS[trait_a]
    desc_b = TRAIT_DESCRIPTIONS[trait_b]
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
    a_candidates = ["A", " A", "a", " a"]
    b_candidates = ["B", " B", "b", " b"]
    a_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in a_candidates if tokenizer.encode(t, add_special_tokens=False))
    b_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in b_candidates if tokenizer.encode(t, add_special_tokens=False))
    return a_lp - b_lp


def eval_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    """Evaluate delta accuracy + per-pair detail."""
    correct_delta = 0
    total = 0
    deltas = []
    pair_detail = {}

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
                    gap = pairwise_logprob_chat(model, tokenizer, device, trait_a, trait_b)
                    base_gap = baseline[f"{trait_a}-{trait_b}"]
                    if steer_trait == trait_a:
                        d = gap - base_gap
                    else:
                        d = base_gap - gap
                    correct_delta += int(d > 0)
                    total += 1
                    deltas.append(d)
                    pair_detail[f"steer_{steer_trait}_{trait_a}-{trait_b}"] = {
                        "delta": float(d),
                        "correct": d > 0,
                        "steered_gap": float(gap),
                        "baseline_gap": float(base_gap),
                    }
        finally:
            hook_handle.remove()

    return {
        "delta_accuracy": correct_delta / total if total else 0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0,
        "correct": correct_delta,
        "total": total,
        "pair_detail": pair_detail,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device_a", type=str, default="cuda:2")
    ap.add_argument("--device_b", type=str, default="cuda:3")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    model_a_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_b_id = "HuggingFaceTB/SmolLM3-3B"

    config_a = AutoConfig.from_pretrained(model_a_id)
    config_b = AutoConfig.from_pretrained(model_b_id)
    mid_a = config_a.num_hidden_layers // 2
    mid_b = config_b.num_hidden_layers // 2

    logger.info("Model A: %s (mid=%d, dim=%d)", model_a_id, mid_a, config_a.hidden_size)
    logger.info("Model B: %s (mid=%d, dim=%d)", model_b_id, mid_b, config_b.hidden_size)

    assert config_a.hidden_size == config_b.hidden_size, \
        f"Hidden dims must match: {config_a.hidden_size} vs {config_b.hidden_size}"

    # Load residual vectors for both models
    logger.info("Loading vectors...")
    vecs_a = load_residual_vectors(model_a_id, mid_a)
    vecs_b = load_residual_vectors(model_b_id, mid_b)

    # Compute Procrustes-aligned vectors
    logger.info("Computing Procrustes alignment...")
    vecs_b_aligned_to_a = procrustes_align(vecs_b, vecs_a)  # SmolLM3 vecs rotated to Llama space
    vecs_a_aligned_to_b = procrustes_align(vecs_a, vecs_b)  # Llama vecs rotated to SmolLM3 space

    # Verify alignment quality
    for t in TRAITS:
        cos_raw = np.dot(vecs_a[t], vecs_b[t]) / (np.linalg.norm(vecs_a[t]) * np.linalg.norm(vecs_b[t]))
        cos_aligned = np.dot(vecs_a[t], vecs_b_aligned_to_a[t]) / (np.linalg.norm(vecs_a[t]) * np.linalg.norm(vecs_b_aligned_to_a[t]))
        logger.info("  %s: raw cos=%.3f, aligned cos=%.3f", t, cos_raw, cos_aligned)

    # Load Model A (Llama 1B)
    logger.info("Loading Model A: %s on %s", model_a_id, args.device_a)
    tok_a = AutoTokenizer.from_pretrained(model_a_id)
    mdl_a = AutoModelForCausalLM.from_pretrained(
        model_a_id,
        torch_dtype=torch.float16,
        device_map=args.device_a,
    )
    mdl_a.eval()
    blocks_a = get_decoder_blocks(mdl_a)

    # Compute baseline for Model A
    logger.info("Computing Model A baseline...")
    baseline_a = {}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(mdl_a, tok_a, args.device_a, ta, tb)
            baseline_a[f"{ta}-{tb}"] = gap

    results = {"model_a": model_a_id, "model_b": model_b_id, "alpha": args.alpha}

    # Condition 1: Llama with own vectors
    logger.info("Condition 1: Model A with own vectors...")
    r1 = eval_discrimination(mdl_a, tok_a, args.device_a, blocks_a, mid_a, vecs_a, args.alpha, baseline_a)
    results["llama_own"] = r1
    print(f"\n  Llama + Llama vectors:     {r1['delta_accuracy']:.0%} ({r1['correct']}/{r1['total']})")

    # Condition 2: Llama with SmolLM3 vectors (raw, no alignment)
    logger.info("Condition 2: Model A with Model B vectors (raw)...")
    r2 = eval_discrimination(mdl_a, tok_a, args.device_a, blocks_a, mid_a, vecs_b, args.alpha, baseline_a)
    results["llama_smollm3_raw"] = r2
    print(f"  Llama + SmolLM3 raw:       {r2['delta_accuracy']:.0%} ({r2['correct']}/{r2['total']})")

    # Condition 3: Llama with Procrustes-aligned SmolLM3 vectors
    logger.info("Condition 3: Model A with Procrustes-aligned Model B vectors...")
    r3 = eval_discrimination(mdl_a, tok_a, args.device_a, blocks_a, mid_a, vecs_b_aligned_to_a, args.alpha, baseline_a)
    results["llama_smollm3_procrustes"] = r3
    print(f"  Llama + SmolLM3 Procrustes: {r3['delta_accuracy']:.0%} ({r3['correct']}/{r3['total']})")

    # Free Model A
    del mdl_a
    torch.cuda.empty_cache()

    # Load Model B (SmolLM3)
    logger.info("Loading Model B: %s on %s", model_b_id, args.device_b)
    tok_b = AutoTokenizer.from_pretrained(model_b_id)
    mdl_b = AutoModelForCausalLM.from_pretrained(
        model_b_id,
        torch_dtype=torch.float16,
        device_map=args.device_b,
    )
    mdl_b.eval()
    blocks_b = get_decoder_blocks(mdl_b)

    # Compute baseline for Model B
    logger.info("Computing Model B baseline...")
    baseline_b = {}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(mdl_b, tok_b, args.device_b, ta, tb)
            baseline_b[f"{ta}-{tb}"] = gap

    # Condition 4: SmolLM3 with own vectors
    logger.info("Condition 4: Model B with own vectors...")
    r4 = eval_discrimination(mdl_b, tok_b, args.device_b, blocks_b, mid_b, vecs_b, args.alpha, baseline_b)
    results["smollm3_own"] = r4
    print(f"\n  SmolLM3 + SmolLM3 vectors:  {r4['delta_accuracy']:.0%} ({r4['correct']}/{r4['total']})")

    # Condition 5: SmolLM3 with Llama vectors (raw)
    logger.info("Condition 5: Model B with Model A vectors (raw)...")
    r5 = eval_discrimination(mdl_b, tok_b, args.device_b, blocks_b, mid_b, vecs_a, args.alpha, baseline_b)
    results["smollm3_llama_raw"] = r5
    print(f"  SmolLM3 + Llama raw:        {r5['delta_accuracy']:.0%} ({r5['correct']}/{r5['total']})")

    # Condition 6: SmolLM3 with Procrustes-aligned Llama vectors
    logger.info("Condition 6: Model B with Procrustes-aligned Model A vectors...")
    r6 = eval_discrimination(mdl_b, tok_b, args.device_b, blocks_b, mid_b, vecs_a_aligned_to_b, args.alpha, baseline_b)
    results["smollm3_llama_procrustes"] = r6
    print(f"  SmolLM3 + Llama Procrustes: {r6['delta_accuracy']:.0%} ({r6['correct']}/{r6['total']})")

    # Summary
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL STEERING TRANSFER SUMMARY (alpha={args.alpha})")
    print(f"{'='*70}")
    print(f"\n  {'Condition':>35} {'Delta%':>8} {'MeanΔ':>8}")
    print(f"  {'-'*53}")
    for name, r in [
        ("Llama + Llama (self)", r1),
        ("Llama + SmolLM3 (raw)", r2),
        ("Llama + SmolLM3 (Procrustes)", r3),
        ("SmolLM3 + SmolLM3 (self)", r4),
        ("SmolLM3 + Llama (raw)", r5),
        ("SmolLM3 + Llama (Procrustes)", r6),
    ]:
        print(f"  {name:>35} {r['delta_accuracy']:>7.0%} {r['mean_delta']:>+7.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_model_steering_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
