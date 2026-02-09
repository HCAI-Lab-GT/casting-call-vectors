#!/usr/bin/env python
"""
Minimal calibration: How many trait correspondences are needed for
successful cross-dimensional transfer?

Tests Procrustes alignment with k=2,3,4,5,6 calibration traits,
evaluating discrimination on ALL 6 traits each time.
Uses cross-dim transfer (SmolLM3 2048d → Marin 8B 4096d).
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="minimal-calibration")

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


def fit_procrustes(source_5d, target_5d, cal_traits):
    """Fit Procrustes on a subset of traits."""
    S = np.stack([source_5d[t] for t in cal_traits])
    T = np.stack([target_5d[t] for t in cal_traits])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))
    return R, scale


def transfer_vectors(source_5d, target_basis, R, scale):
    """Transfer all 6 vectors using the fitted Procrustes."""
    transferred = {}
    for t in TRAITS:
        aligned = scale * (R @ source_5d[t])
        transferred[t] = (target_basis.T @ aligned).astype(np.float32)
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


def eval_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    correct = 0
    total = 0
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
        finally:
            hook_handle.remove()
    return correct / total if total else 0


def main():
    source_id = "HuggingFaceTB/SmolLM3-3B"
    target_id = "marin-community/marin-8b-instruct"
    device = "cuda:0"
    alpha = 1.0

    riasec_dir = _repo_root() / "persona_data/model_inits"

    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    target_residual, target_mid = load_residual_vectors(target_id, riasec_dir)

    source_5d, _ = get_5d_coordinates(source_residual)
    target_5d, target_basis = get_5d_coordinates(target_residual)

    # Load target model
    logger.info("Loading target model: %s", target_id)
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Baseline
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    # Self (positive control)
    self_acc = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                  target_residual, alpha, baseline)

    print(f"\n{'='*70}")
    print(f"MINIMAL CALIBRATION: How many traits needed for cross-dim transfer?")
    print(f"Source: {source_id} (2048d) → Target: {target_id} (4096d)")
    print(f"{'='*70}")
    print(f"\n  Self (Marin 8B native): {self_acc:.0%}")

    # Test k=6 (all traits) as reference
    R6, s6 = fit_procrustes(source_5d, target_5d, TRAITS)
    vecs6 = transfer_vectors(source_5d, target_basis, R6, s6)
    acc6 = eval_discrimination(model, tokenizer, device, blocks, target_mid, vecs6, alpha, baseline)
    print(f"  Cross-dim (k=6 calibration): {acc6:.0%}")

    results = {"self": float(self_acc), "k6": float(acc6), "by_k": {}}

    # Test k=2,3,4,5
    for k in [2, 3, 4, 5]:
        all_combos = list(combinations(TRAITS, k))
        accs = []

        for cal_traits in all_combos:
            R, s = fit_procrustes(source_5d, target_5d, list(cal_traits))
            vecs = transfer_vectors(source_5d, target_basis, R, s)
            acc = eval_discrimination(model, tokenizer, device, blocks, target_mid, vecs, alpha, baseline)
            accs.append(acc)

        mean_acc = np.mean(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        perfect = sum(1 for a in accs if a >= 0.97)  # Match self-steering

        results["by_k"][str(k)] = {
            "mean": float(mean_acc),
            "min": float(min_acc),
            "max": float(max_acc),
            "n_combos": len(all_combos),
            "matches_self": perfect,
        }

        print(f"\n  k={k} ({len(all_combos)} combos): mean={mean_acc:.0%}, "
              f"min={min_acc:.0%}, max={max_acc:.0%}, matches_self={perfect}/{len(all_combos)}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  {'k':>3}  {'Mean':>6}  {'Min':>6}  {'Max':>6}  {'Combos':>7}  {'Matches self':>12}")
    print(f"  {'-'*42}")
    for k in [2, 3, 4, 5]:
        d = results["by_k"][str(k)]
        print(f"  {k:>3}  {d['mean']:>5.0%}  {d['min']:>5.0%}  {d['max']:>5.0%}"
              f"  {d['n_combos']:>7}  {d['matches_self']:>5}/{d['n_combos']}")
    print(f"    6  {acc6:>5.0%}  {acc6:>5.0%}  {acc6:>5.0%}        1      1/1")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "minimal_calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
