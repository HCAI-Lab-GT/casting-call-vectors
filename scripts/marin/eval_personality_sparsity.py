#!/usr/bin/env python
"""
Personality Vector Sparsity: What is the minimum information needed
to encode personality?

Previous findings show personality lives in a 5D subspace of a
4096-dimensional space. But how sparse is the representation within
that full space?

Tests:
1. Dimension thresholding: keep only top-K dimensions, measure accuracy
2. Random projection: project from 4096 to M dimensions, measure accuracy
3. Coordinate masking: which coordinates are most critical?
4. Weight sharing: how many unique values does the personality vector need?
5. Minimum bit-width: can we quantize personality vectors to low precision?
6. Structured sparsity: contiguous blocks vs scattered elements
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="sparsity")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_model_data(model_id, riasec_dir):
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
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    _, _, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def steer_and_detect(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                      steer_vec, alpha, prompt):
    """Steer with a given vector and detect personality."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured_base = {}
    captured_steer = {}

    # Baseline
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_base["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    # Steered
    hooks = []
    def cap_steer(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_steer["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_steer))

    def steer_fn(_m, _i, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out
    hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    diff = (captured_steer["act"] - captured_base["act"]).astype(np.float64)
    coords = basis_5d @ diff
    norm_5d = float(np.linalg.norm(coords))
    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0
    detected = max(sims, key=sims.get)
    return {"detected": detected, "cos": sims, "norm": norm_5d}


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]
    hidden_size = residual[TRAITS[0]].shape[0]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    alpha = 2.0
    prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY VECTOR SPARSITY ANALYSIS")
    print(f"Model: Marin 8B, hidden_size={hidden_size}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Dimension thresholding (keep top-K absolute values)
    # ================================================================
    logger.info("Part 1: Dimension thresholding...")
    print(f"\n{'='*70}")
    print("PART 1: TOP-K DIMENSION THRESHOLDING")
    print(f"{'='*70}")

    k_values = [10, 25, 50, 100, 200, 500, 1000, 2000, 4096]
    threshold_results = {}

    for k in k_values:
        correct = 0
        total = 0
        for trait in TRAITS:
            vec = residual[trait].copy()
            # Keep only top-K dimensions by absolute value
            if k < hidden_size:
                abs_vals = np.abs(vec)
                threshold = np.partition(abs_vals, -k)[-k]
                mask = abs_vals >= threshold
                vec_sparse = vec * mask
            else:
                vec_sparse = vec

            sparsity = 1.0 - (np.count_nonzero(vec_sparse) / hidden_size)
            res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, vec_sparse.astype(np.float32),
                                    alpha, prompt)
            if res["detected"] == trait:
                correct += 1
            total += 1

        accuracy = correct / total
        threshold_results[k] = {"accuracy": float(accuracy), "sparsity": float(1 - k / hidden_size)}
        print(f"  K={k:>5} ({k/hidden_size:.1%} of dims): {correct}/{total} ({accuracy:.0%})")

    results["dimension_threshold"] = threshold_results

    # Find minimum K for 100%
    min_k = None
    for k in k_values:
        if threshold_results[k]["accuracy"] == 1.0:
            min_k = k
            break
    if min_k:
        print(f"\n  Minimum K for 100%: {min_k} ({min_k/hidden_size:.1%} of dimensions)")

    # ================================================================
    # PART 2: Random projection
    # ================================================================
    logger.info("Part 2: Random projection...")
    print(f"\n{'='*70}")
    print("PART 2: RANDOM PROJECTION (Johnson-Lindenstrauss)")
    print(f"{'='*70}")

    m_values = [5, 10, 25, 50, 100, 200, 500]
    rng = np.random.RandomState(42)
    n_trials = 5  # Average over random projections

    random_proj_results = {}
    for m in m_values:
        trial_accuracies = []
        for trial in range(n_trials):
            # Random projection matrix: R of shape (m, hidden_size)
            R = rng.randn(m, hidden_size) / np.sqrt(m)
            # Pseudo-inverse to project back
            R_pinv = np.linalg.pinv(R)  # (hidden_size, m)

            correct = 0
            total = 0
            for trait in TRAITS:
                vec = residual[trait].copy()
                # Project to m dims and back
                projected = R @ vec  # (m,)
                reconstructed = R_pinv @ projected  # (hidden_size,)

                res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                        basis_5d, coords_5d, reconstructed.astype(np.float32),
                                        alpha, prompt)
                if res["detected"] == trait:
                    correct += 1
                total += 1
            trial_accuracies.append(correct / total)

        mean_acc = float(np.mean(trial_accuracies))
        std_acc = float(np.std(trial_accuracies))
        random_proj_results[m] = {"mean_accuracy": mean_acc, "std": std_acc}
        print(f"  M={m:>4} random dims: {mean_acc:.0%} ± {std_acc:.0%}")

    results["random_projection"] = random_proj_results

    # ================================================================
    # PART 3: Quantization of personality vectors
    # ================================================================
    logger.info("Part 3: Vector quantization...")
    print(f"\n{'='*70}")
    print("PART 3: PERSONALITY VECTOR QUANTIZATION")
    print(f"{'='*70}")

    bit_widths = [1, 2, 3, 4, 8, 16]
    quant_results = {}

    for bits in bit_widths:
        correct = 0
        total = 0
        for trait in TRAITS:
            vec = residual[trait].copy()
            # Quantize to N bits
            vmin, vmax = vec.min(), vec.max()
            if vmax - vmin > 0:
                n_levels = 2 ** bits
                # Uniform quantization
                quantized = np.round((vec - vmin) / (vmax - vmin) * (n_levels - 1))
                dequantized = quantized / (n_levels - 1) * (vmax - vmin) + vmin
            else:
                dequantized = vec

            res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, dequantized.astype(np.float32),
                                    alpha, prompt)
            if res["detected"] == trait:
                correct += 1
            total += 1

        accuracy = correct / total
        quant_results[bits] = {"accuracy": float(accuracy)}
        print(f"  {bits:>2}-bit quantization: {correct}/{total} ({accuracy:.0%})")

    results["quantization"] = quant_results

    # ================================================================
    # PART 4: Ternary vectors (+1, 0, -1)
    # ================================================================
    logger.info("Part 4: Ternary approximation...")
    print(f"\n{'='*70}")
    print("PART 4: TERNARY APPROXIMATION (+1, 0, -1)")
    print(f"{'='*70}")

    # Different thresholds for ternarization
    percentiles = [50, 75, 90, 95, 99]
    ternary_results = {}

    for pct in percentiles:
        correct = 0
        total = 0
        for trait in TRAITS:
            vec = residual[trait].copy()
            abs_vals = np.abs(vec)
            threshold = np.percentile(abs_vals, pct)
            # Ternary: sign(x) if |x| > threshold, else 0
            ternary = np.sign(vec) * (abs_vals > threshold).astype(float)
            # Scale to match original norm
            if np.linalg.norm(ternary) > 0:
                ternary = ternary * np.linalg.norm(vec) / np.linalg.norm(ternary)

            res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, ternary.astype(np.float32),
                                    alpha, prompt)
            if res["detected"] == trait:
                correct += 1
            total += 1

        nonzero_frac = 1 - pct / 100
        accuracy = correct / total
        ternary_results[pct] = {"accuracy": float(accuracy), "nonzero_frac": float(nonzero_frac)}
        print(f"  Percentile {pct} (keeping {nonzero_frac:.0%} dims): {correct}/{total} ({accuracy:.0%})")

    results["ternary"] = ternary_results

    # ================================================================
    # PART 5: 5D basis vectors only (ultimate sparsity test)
    # ================================================================
    logger.info("Part 5: 5D-basis-only steering...")
    print(f"\n{'='*70}")
    print("PART 5: STEERING WITH 5D BASIS RECONSTRUCTION ONLY")
    print(f"{'='*70}")

    # Instead of using the full 4096-dim residual vector, reconstruct
    # from just the 5D coordinates through the basis
    basis_5d_results = {}
    for trait in TRAITS:
        # Full residual vector
        full_vec = residual[trait]
        # 5D projection: coords_5d[trait] through basis_5d
        reconstructed = (basis_5d.T @ coords_5d[trait]).astype(np.float32)
        # What fraction of the full vector is captured?
        reconstruction_frac = np.linalg.norm(reconstructed) / np.linalg.norm(full_vec)

        # Test with reconstructed
        res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                basis_5d, coords_5d, reconstructed, alpha, prompt)

        # Test with full
        res_full = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                     basis_5d, coords_5d, full_vec.astype(np.float32),
                                     alpha, prompt)

        basis_5d_results[trait] = {
            "full_detected": res_full["detected"],
            "5d_detected": res["detected"],
            "5d_correct": res["detected"] == trait,
            "full_correct": res_full["detected"] == trait,
            "reconstruction_frac": float(reconstruction_frac),
            "5d_norm": res["norm"],
            "full_norm": res_full["norm"],
        }
        match = "✓" if res["detected"] == trait else "✗"
        print(f"  {trait:>15}: 5D={res['detected'][:6]} {match}, "
              f"recon={reconstruction_frac:.1%} of full vector")

    correct_5d = sum(1 for v in basis_5d_results.values() if v["5d_correct"])
    print(f"\n  5D-only accuracy: {correct_5d}/{len(TRAITS)}")

    results["basis_5d_only"] = basis_5d_results

    # ================================================================
    # PART 6: Per-dimension importance ranking
    # ================================================================
    logger.info("Part 6: Per-dimension importance...")
    print(f"\n{'='*70}")
    print("PART 6: PER-DIMENSION IMPORTANCE (LEAVE-ONE-OUT)")
    print(f"{'='*70}")

    # Compute importance of each dimension by measuring how much the 5D
    # projection changes when that dimension is zeroed
    dim_importance = np.zeros(hidden_size)
    for trait in TRAITS:
        vec = residual[trait]
        base_coords = basis_5d @ vec
        for d in range(hidden_size):
            perturbed = vec.copy()
            perturbed[d] = 0
            perturbed_coords = basis_5d @ perturbed
            delta = np.linalg.norm(base_coords - perturbed_coords)
            dim_importance[d] += delta

    # Normalize
    dim_importance /= len(TRAITS)

    # Gini coefficient of dimension importance
    sorted_imp = np.sort(dim_importance)
    n = len(sorted_imp)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_imp)) / (n * np.sum(sorted_imp))) - (n+1)/n

    # How many dimensions carry 90% of importance
    cum_imp = np.cumsum(sorted_imp[::-1]) / sorted_imp.sum()
    dims_for_90 = int(np.searchsorted(cum_imp, 0.9)) + 1
    dims_for_99 = int(np.searchsorted(cum_imp, 0.99)) + 1

    print(f"  Gini coefficient: {gini:.4f}")
    print(f"  Dims for 90% importance: {dims_for_90}/{hidden_size} ({dims_for_90/hidden_size:.1%})")
    print(f"  Dims for 99% importance: {dims_for_99}/{hidden_size} ({dims_for_99/hidden_size:.1%})")

    # Top 20 dimensions
    top_dims = np.argsort(dim_importance)[::-1][:20]
    print(f"\n  Top 20 most important dimensions:")
    for rank, d in enumerate(top_dims):
        print(f"    #{rank+1}: dim {d} (importance={dim_importance[d]:.6f})")

    results["dimension_importance"] = {
        "gini": float(gini),
        "dims_for_90pct": int(dims_for_90),
        "dims_for_99pct": int(dims_for_99),
        "top20_dims": [int(d) for d in top_dims],
        "top20_importance": [float(dim_importance[d]) for d in top_dims],
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"  Minimum K for 100% (top-K dims): {min_k if min_k else 'N/A'}")
    min_bits = min(b for b, v in quant_results.items() if v["accuracy"] == 1.0) if any(
        v["accuracy"] == 1.0 for v in quant_results.values()) else "N/A"
    print(f"  Minimum bits for 100% (quantization): {min_bits}")
    print(f"  5D-only steering accuracy: {correct_5d}/{len(TRAITS)}")
    print(f"  Dimension importance Gini: {gini:.4f}")
    print(f"  Dims for 90% importance: {dims_for_90} ({dims_for_90/hidden_size:.1%})")

    results["summary"] = {
        "min_k_for_100pct": int(min_k) if min_k else None,
        "min_bits_for_100pct": int(min_bits) if isinstance(min_bits, int) else None,
        "5d_only_accuracy": f"{correct_5d}/{len(TRAITS)}",
        "gini": float(gini),
        "dims_for_90pct": int(dims_for_90),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_sparsity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
