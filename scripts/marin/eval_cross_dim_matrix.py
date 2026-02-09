#!/usr/bin/env python
"""
Cross-dimensional transfer matrix: ALL source models → Marin 8B.

Tests 5D Procrustes transfer from every available source model to Marin 8B,
creating a comprehensive transfer comparison:

Sources:
- SmolLM3-3B Instruct (2048d, 36L) — same-format, cross-dim
- SmolLM3-3B Base (2048d, 36L) — cross-format + cross-dim
- Llama-3.2-1B Instruct (2048d, 16L) — tiny model, different family
- Qwen-2.5-7B Instruct (3584d, 28L) — different arch, different dim
- Marin 32B Base (5120d, 64L) — same family but base, LARGER dim

Also tests: Transitive paths (SmolLM3→Llama→Marin, SmolLM3→Qwen→Marin).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-dim-matrix")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

SOURCES = {
    "SmolLM3-Instruct": "HuggingFaceTB/SmolLM3-3B",
    "SmolLM3-Base": "HuggingFaceTB/SmolLM3-3B-Base",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Marin-32B-Base": "marin-community/marin-32b-base",
}

TARGET_ID = "marin-community/marin-8b-instruct"


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


def transfer_vectors(source_5d, target_basis, R, scale):
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
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # === Phase 1: Load ALL vectors (no GPU needed) ===
    logger.info("Loading residual vectors for all models...")
    all_residuals = {}
    all_5d = {}

    for name, model_id in SOURCES.items():
        logger.info("  %s (%s)", name, model_id)
        residual, _ = load_residual_vectors(model_id, riasec_dir)
        all_residuals[name] = residual
        coords, _ = get_5d_coordinates(residual)
        all_5d[name] = coords

    # Target
    target_residual, target_mid = load_residual_vectors(TARGET_ID, riasec_dir)
    target_5d, target_basis = get_5d_coordinates(target_residual)
    target_dim = target_residual[TRAITS[0]].shape[0]

    # === Phase 2: Compute all transferred vector sets ===
    transferred_sets = {}

    # Direct 1-hop transfers
    for name in SOURCES:
        source_dim = all_residuals[name][TRAITS[0]].shape[0]
        R, scale = fit_procrustes(all_5d[name], target_5d)
        vecs = transfer_vectors(all_5d[name], target_basis, R, scale)
        transferred_sets[name] = vecs
        logger.info("  %s (%dd → %dd): transferred", name, source_dim, target_dim)

    # Transitive 2-hop transfers
    transitive_paths = [
        ("SmolLM3→Llama→Marin", "SmolLM3-Instruct", "Llama-1B"),
        ("SmolLM3→Qwen→Marin", "SmolLM3-Instruct", "Qwen-7B"),
        ("Llama→Qwen→Marin", "Llama-1B", "Qwen-7B"),
    ]

    for label, src_name, mid_name in transitive_paths:
        R1, s1 = fit_procrustes(all_5d[src_name], all_5d[mid_name])
        R2, s2 = fit_procrustes(all_5d[mid_name], target_5d)
        vecs = {}
        for t in TRAITS:
            mid_coord = s1 * (R1 @ all_5d[src_name][t])
            target_coord = s2 * (R2 @ mid_coord)
            vecs[t] = (target_basis.T @ target_coord).astype(np.float32)
        transferred_sets[f"transitive:{label}"] = vecs

    # === Phase 3: Cosine analysis (no GPU needed) ===
    print(f"\n{'='*70}")
    print(f"CROSS-DIMENSIONAL TRANSFER MATRIX → Marin 8B ({target_dim}d)")
    print(f"{'='*70}")

    print(f"\n--- Mean cosine vs native Marin vectors ---")
    cosine_results = {}
    for name, vecs in transferred_sets.items():
        cosines = []
        for t in TRAITS:
            cos = np.dot(vecs[t], target_residual[t]) / \
                  (np.linalg.norm(vecs[t]) * np.linalg.norm(target_residual[t]))
            cosines.append(cos)
        mean_cos = np.mean(cosines)
        cosine_results[name] = {t: float(c) for t, c in zip(TRAITS, cosines)}
        src_dim = all_residuals[name.split(":")[-1].split("→")[0].strip() if ":" in name else name][TRAITS[0]].shape[0] if ":" not in name else "multi"
        print(f"  {name:>30}: mean={mean_cos:.4f}  (min={min(cosines):.4f}, max={max(cosines):.4f})")

    # === Phase 4: Load model and evaluate ===
    logger.info("Loading target model: %s", TARGET_ID)
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, torch_dtype=torch.float16, device_map=device)
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

    # Self (positive control)
    r_self = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                target_residual, alpha, baseline)
    print(f"\n--- Steering evaluation (α={alpha}) ---")
    print(f"  {'Source':>30}  {'Dim':>6}  {'Acc':>5}  {'Correct':>8}  {'Mean Δ':>8}")
    print(f"  {'-'*65}")
    print(f"  {'Self (Marin native)':>30}  {target_dim:>5}d  {r_self['delta_accuracy']:>4.0%}  "
          f"{r_self['correct']:>3}/{r_self['total']:>3}   {r_self['mean_delta']:>+.4f}")

    results = {"target": TARGET_ID, "alpha": alpha, "self": r_self, "transfers": {}}

    # Evaluate each transfer
    for name, vecs in transferred_sets.items():
        r = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                               vecs, alpha, baseline)
        if ":" not in name:
            src_dim = all_residuals[name][TRAITS[0]].shape[0]
            print(f"  {name:>30}  {src_dim:>5}d  {r['delta_accuracy']:>4.0%}  "
                  f"{r['correct']:>3}/{r['total']:>3}   {r['mean_delta']:>+.4f}")
        else:
            print(f"  {name:>30}  {'multi':>6}  {r['delta_accuracy']:>4.0%}  "
                  f"{r['correct']:>3}/{r['total']:>3}   {r['mean_delta']:>+.4f}")
        results["transfers"][name] = r

    # Random control
    np.random.seed(42)
    random_vecs = {}
    for t in TRAITS:
        rv = np.random.randn(target_dim).astype(np.float32)
        rv = rv / np.linalg.norm(rv) * np.linalg.norm(target_residual[t])
        random_vecs[t] = rv
    r_random = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                   random_vecs, alpha, baseline)
    print(f"  {'Random':>30}  {target_dim:>5}d  {r_random['delta_accuracy']:>4.0%}  "
          f"{r_random['correct']:>3}/{r_random['total']:>3}   {r_random['mean_delta']:>+.4f}")
    results["random"] = r_random

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Transfer hierarchy")
    print(f"{'='*70}")

    all_items = [("Self (native)", r_self)]
    for name, r in results["transfers"].items():
        all_items.append((name, r))
    all_items.append(("Random", r_random))
    all_items.sort(key=lambda x: x[1]["delta_accuracy"], reverse=True)

    for name, r in all_items:
        bar = "#" * int(r["delta_accuracy"] * 30)
        print(f"  {name:>35}: {r['delta_accuracy']:>4.0%} |{bar}")

    results["cosines"] = cosine_results

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_dim_matrix.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
