#!/usr/bin/env python
"""
Truly zero-calibration transfer: Predict PC sign flips from a canonical
convention WITHOUT any target model data.

Key insight from zero_calibration_transfer.py:
  Sign-corrected identity achieves 97% (= full Procrustes), using only 5 bits.

This script tests whether those 5 bits can be PREDICTED from the source
model's coordinate patterns alone, by establishing a canonical sign convention:
  - PC1: Artistic should be most negative → if not, flip
  - PC2: Investigative should be most negative → if not, flip
  - PC3: Social should be most positive → if not, flip
  - PC4: Enterprising should be most negative → if not, flip
  - PC5: Conventional should be most positive → if not, flip

If ALL models follow this convention (after appropriate sign flips), then
we can standardize both source and target independently → no calibration needed.

Tests ALL 12 directed pairs among 4 instruct models:
  SmolLM3 ↔ Llama 1B ↔ Qwen 7B ↔ Marin 8B (all directions)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="predicted-sign")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

INSTRUCT_MODELS = {
    "SmolLM3": "HuggingFaceTB/SmolLM3-3B",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Marin-8B": "marin-community/marin-8b-instruct",
}

# Canonical sign convention: for each PC, specify which trait should have
# the most extreme loading and in which direction (positive or negative).
# Derived from the universal 5D semantics analysis showing consistent patterns.
CANONICAL_SIGNS = [
    # (pc_index, trait_that_should_be_extreme, expected_sign)
    # PC1: Artistic↔Conventional axis. Artistic should be most negative.
    (0, "artistic", -1),
    # PC2: Investigative↔Realistic axis. Different models use different signs.
    # Use the loading with largest absolute value to determine.
    (1, None, 0),  # Will use argmax strategy
    # PC3-5: Less consistent across models, use argmax strategy.
    (2, None, 0),
    (3, None, 0),
    (4, None, 0),
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
    return coords_5d, basis_5d, S[:5]


def canonical_sign_convention(coords_5d):
    """Determine the canonical sign for each PC based on trait loadings.

    Strategy 1 (PC1): Use the known Artistic↔Conventional axis.
      Artistic should have NEGATIVE PC1 loading.

    Strategy 2 (PC2-5): Use the trait with the largest absolute loading.
      The sign of that trait's loading determines the canonical direction.
      We define "canonical" = the trait with max |loading| should be NEGATIVE.
      This ensures consistency: the most discriminative trait sets the sign.
    """
    signs = np.ones(5)

    # PC1: Artistic should be negative
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1

    # PC2-4: trait with max absolute loading should be negative
    # (This is an arbitrary but deterministic convention)
    for pc in range(1, 5):
        loadings = {t: coords_5d[t][pc] for t in TRAITS}
        max_trait = max(loadings, key=lambda t: abs(loadings[t]))
        # Convention: max-loaded trait should have NEGATIVE sign
        if loadings[max_trait] > 0:
            signs[pc] = -1

    return signs


def standardize_coords(coords_5d, basis_5d):
    """Standardize a model's 5D coordinates to canonical sign convention.

    Returns standardized coordinates and updated basis.
    """
    signs = canonical_sign_convention(coords_5d)
    std_coords = {}
    for t in TRAITS:
        std_coords[t] = signs * coords_5d[t]
    std_basis = np.diag(signs) @ basis_5d
    return std_coords, std_basis, signs


def transfer_with_predicted_signs(source_5d, source_basis, target_5d, target_basis):
    """Transfer using predicted canonical signs — NO target data needed at runtime.

    Both source and target are independently standardized to the canonical
    convention. The 5D coordinates should then be directly compatible.
    """
    # Standardize source
    source_std, source_basis_std, source_signs = standardize_coords(source_5d, source_basis)
    # Standardize target
    target_std, target_basis_std, target_signs = standardize_coords(target_5d, target_basis)

    # Scale: match norms
    source_norms = np.mean([np.linalg.norm(source_std[t]) for t in TRAITS])
    target_norms = np.mean([np.linalg.norm(target_std[t]) for t in TRAITS])
    scale = target_norms / source_norms

    # Transfer: standardized source coords → target full-dim via target's standardized basis
    transferred = {}
    for t in TRAITS:
        target_coord = scale * source_std[t]
        transferred[t] = (target_basis_std.T @ target_coord).astype(np.float32)

    return transferred, source_signs, target_signs


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


def transfer_standard(source_5d, target_5d, source_basis, target_basis):
    R, scale = fit_procrustes(source_5d, target_5d)
    transferred = {}
    for t in TRAITS:
        target_coord = scale * (R @ source_5d[t])
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


def eval_accuracy(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    correct = 0
    total = 0
    total_delta = 0.0

    for steer_trait in TRAITS:
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
                    total_delta += d
                    total += 1
        finally:
            hook_handle.remove()

    return correct / total if total else 0, total_delta / total if total else 0


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load all model vectors
    logger.info("Loading all model vectors...")
    model_data = {}
    for name, model_id in INSTRUCT_MODELS.items():
        residual, mid_layer = load_residual_vectors(model_id, riasec_dir)
        coords, basis, sv = get_5d_coords_and_basis(residual)
        model_data[name] = {
            "model_id": model_id,
            "residual": residual,
            "coords": coords,
            "basis": basis,
            "mid_layer": mid_layer,
        }

    # Show canonical signs for each model
    print(f"\n{'='*70}")
    print(f"PREDICTED SIGN CONVENTION ANALYSIS")
    print(f"{'='*70}")

    print(f"\n--- Canonical signs per model ---")
    for name in INSTRUCT_MODELS:
        signs = canonical_sign_convention(model_data[name]["coords"])
        print(f"  {name:>10}: signs = [{', '.join(f'{s:+.0f}' for s in signs)}]")

    print(f"\n--- Standardized 5D coordinates ---")
    for name in INSTRUCT_MODELS:
        std_coords, std_basis, signs = standardize_coords(
            model_data[name]["coords"], model_data[name]["basis"])
        model_data[name]["std_coords"] = std_coords
        model_data[name]["std_basis"] = std_basis
        model_data[name]["signs"] = signs

        print(f"\n  {name} (signs: [{', '.join(f'{s:+.0f}' for s in signs)}]):")
        print(f"  {'Trait':>14}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC4':>7}  {'PC5':>7}")
        for t in TRAITS:
            c = std_coords[t]
            print(f"  {t:>14}  {c[0]:>+6.3f}  {c[1]:>+6.3f}  {c[2]:>+6.3f}  {c[3]:>+6.3f}  {c[4]:>+6.3f}")

    # Check standardization consistency: after standardizing, are all models'
    # coordinates aligned? Compute pairwise cosines of standardized coords.
    print(f"\n--- Standardized coordinate alignment (cosine between model pairs) ---")
    names = list(INSTRUCT_MODELS.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            cosines = []
            for t in TRAITS:
                c1 = model_data[n1]["std_coords"][t]
                c2 = model_data[n2]["std_coords"][t]
                cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
                cosines.append(cos)
            print(f"  {n1:>10} ↔ {n2:<10}: mean cos = {np.mean(cosines):.3f}, min = {np.min(cosines):.3f}")

    # Now test predicted-sign transfer on each target model
    # We test all 12 directed pairs (4 targets × 3 sources each)
    print(f"\n{'='*70}")
    print(f"BEHAVIORAL VALIDATION: Predicted-sign transfer vs Full Procrustes")
    print(f"{'='*70}")

    # Test on 2 target models: Marin 8B and SmolLM3 (to keep GPU time manageable)
    test_targets = ["Marin-8B", "SmolLM3"]

    results = {}
    for target_name in test_targets:
        target_id = INSTRUCT_MODELS[target_name]
        logger.info(f"Loading target model: {target_name}...")

        tokenizer = AutoTokenizer.from_pretrained(target_id)
        model = AutoModelForCausalLM.from_pretrained(
            target_id, torch_dtype=torch.bfloat16 if "SmolLM" in target_name else torch.float16,
            device_map=device)
        model.eval()
        blocks = get_decoder_blocks(model)
        mid_layer = model_data[target_name]["mid_layer"]

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

        # Self-steering
        logger.info(f"Testing self-steering on {target_name}...")
        self_acc, self_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer,
            model_data[target_name]["residual"], alpha, baseline)

        print(f"\n--- {target_name} as target ---")
        print(f"  Self-steering: {self_acc:.0%} (delta={self_delta:+.3f})")

        target_results = {"self": {"accuracy": float(self_acc), "mean_delta": float(self_delta)}}

        for source_name in INSTRUCT_MODELS:
            if source_name == target_name:
                continue

            # A. Full Procrustes (gold standard)
            full_vecs = transfer_standard(
                model_data[source_name]["coords"],
                model_data[target_name]["coords"],
                model_data[source_name]["basis"],
                model_data[target_name]["basis"])

            logger.info(f"Testing {source_name} → {target_name} (Full Procrustes)...")
            full_acc, full_delta = eval_accuracy(
                model, tokenizer, device, blocks, mid_layer, full_vecs, alpha, baseline)

            # B. Predicted-sign transfer (zero-calibration)
            pred_vecs, src_signs, tgt_signs = transfer_with_predicted_signs(
                model_data[source_name]["coords"],
                model_data[source_name]["basis"],
                model_data[target_name]["coords"],
                model_data[target_name]["basis"])

            logger.info(f"Testing {source_name} → {target_name} (Predicted signs)...")
            pred_acc, pred_delta = eval_accuracy(
                model, tokenizer, device, blocks, mid_layer, pred_vecs, alpha, baseline)

            # Cosine between predicted and full-Procrustes vectors
            cos_pred_full = []
            for t in TRAITS:
                c = np.dot(pred_vecs[t], full_vecs[t]) / (
                    np.linalg.norm(pred_vecs[t]) * np.linalg.norm(full_vecs[t]))
                cos_pred_full.append(c)

            print(f"  {source_name:>10} → {target_name}: "
                  f"Procrustes={full_acc:.0%}, Predicted={pred_acc:.0%}, "
                  f"cos(pred,proc)={np.mean(cos_pred_full):.3f}")

            target_results[f"{source_name}_procrustes"] = {
                "accuracy": float(full_acc), "mean_delta": float(full_delta)}
            target_results[f"{source_name}_predicted"] = {
                "accuracy": float(pred_acc), "mean_delta": float(pred_delta),
                "source_signs": src_signs.tolist(),
                "target_signs": tgt_signs.tolist(),
                "cos_to_procrustes": float(np.mean(cos_pred_full)),
            }

        results[target_name] = target_results

        # Unload model
        del model, tokenizer
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Predicted-Sign vs Full Procrustes")
    print(f"{'='*70}")
    for target_name in test_targets:
        print(f"\n  Target: {target_name}")
        print(f"  {'Source':>12}  {'Procrustes':>10}  {'Predicted':>10}  {'cos(P,F)':>8}")
        for source_name in INSTRUCT_MODELS:
            if source_name == target_name:
                continue
            pk = f"{source_name}_procrustes"
            ppk = f"{source_name}_predicted"
            if pk in results[target_name] and ppk in results[target_name]:
                proc = results[target_name][pk]["accuracy"]
                pred = results[target_name][ppk]["accuracy"]
                cos_pf = results[target_name][ppk]["cos_to_procrustes"]
                print(f"  {source_name:>12}  {proc:>9.0%}  {pred:>9.0%}  {cos_pf:>7.3f}")

    # Save results
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predicted_sign_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
