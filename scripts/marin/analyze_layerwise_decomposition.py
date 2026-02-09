#!/usr/bin/env python
"""
Layer-wise analysis of RIASEC persona vector geometry.

For each model, the safetensors files contain all_layers_response_persona_vector
(shape [num_layers+1, 1, hidden_dim]) -- the persona vector at EVERY layer.

This script analyzes how the shared/specific decomposition changes across layers,
answering: WHERE in the transformer do personality-specific representations emerge?

Key questions:
1. At which layers does the shared "agree" direction dominate?
2. Do trait-specific residuals emerge early, middle, or late?
3. Is the cross-model cosine structure consistent across layers?
4. Does the specificity index vary by layer?
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

MODELS = {
    "llama-1b": {
        "safe_id": "meta-llama__Llama-3.2-1B-Instruct",
        "size": "1B",
    },
    "marin-8b": {
        "safe_id": "marin-community__marin-8b-instruct",
        "size": "8B",
    },
    "qwen-7b": {
        "safe_id": "Qwen__Qwen2.5-7B-Instruct",
        "size": "7B",
    },
}

SAFETENSORS_DIR = Path("./persona_data/model_inits/")
OUTPUT_DIR = Path("./outputs/analysis/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_layers(model_key: str) -> dict[str, np.ndarray]:
    """Load all_layers_response_persona_vector for all 6 traits.
    Returns dict: trait -> (num_layers+1, hidden_dim)"""
    info = MODELS[model_key]
    vectors = {}
    for trait in TRAITS:
        path = SAFETENSORS_DIR / f"{trait}_persona_initialization" / f"{info['safe_id']}.safetensors"
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            return {}
        data = load_file(str(path))
        # Shape: (num_layers+1, 1, hidden_dim) -> squeeze to (num_layers+1, hidden_dim)
        v = data["all_layers_response_persona_vector"].numpy()
        if v.ndim == 3:
            v = v.squeeze(1)
        vectors[trait] = v
    return vectors


def analyze_layer(vectors_at_layer: np.ndarray):
    """Analyze 6 trait vectors at a single layer.
    vectors_at_layer: (6, hidden_dim)
    Returns dict with shared fraction, residual variance, cosine stats."""
    vecs = vectors_at_layer
    norms = np.linalg.norm(vecs, axis=1)
    mean_norm = np.mean(norms)

    if mean_norm < 1e-8:
        return {"shared_frac": 0, "mean_norm": 0, "residual_pca_var": [], "cosine_diag_mean": 0}

    # Cosine matrix
    normed = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
    cos_mat = normed @ normed.T

    # Shared direction (PC1)
    U, S, Vt = np.linalg.svd(vecs, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    projections = vecs @ shared_dir
    residuals = vecs - np.outer(projections, shared_dir)

    # Shared fraction
    total_norm_sq = np.sum(vecs ** 2)
    shared_norm_sq = np.sum(projections ** 2)
    frac_shared = shared_norm_sq / max(total_norm_sq, 1e-8)

    # Residual PCA
    res_norms = np.linalg.norm(residuals, axis=1)
    mean_res_norm = np.mean(res_norms)

    # Cross-trait cosine stats
    idx = np.triu_indices(6, k=1)
    upper_tri = cos_mat[idx]

    # Residual cosine matrix
    res_normed = residuals / np.maximum(np.linalg.norm(residuals, axis=1, keepdims=True), 1e-8)
    res_cos_mat = res_normed @ res_normed.T
    res_upper_tri = res_cos_mat[idx]

    # Holland hexagon check
    hex_order = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
    hex_idx = [TRAITS.index(t) for t in hex_order]
    adjacent, alternate, opposite = [], [], []
    for i in range(6):
        for j in range(i + 1, 6):
            dist = min(abs(i - j), 6 - abs(i - j))
            ii, jj = hex_idx[i], hex_idx[j]
            val = res_cos_mat[ii, jj]
            if dist == 1:
                adjacent.append(val)
            elif dist == 2:
                alternate.append(val)
            elif dist == 3:
                opposite.append(val)

    return {
        "shared_frac": float(frac_shared),
        "mean_norm": float(mean_norm),
        "mean_residual_norm": float(mean_res_norm),
        "cosine_mean": float(np.mean(upper_tri)),
        "cosine_std": float(np.std(upper_tri)),
        "residual_cosine_mean": float(np.mean(res_upper_tri)),
        "residual_cosine_std": float(np.std(res_upper_tri)),
        "hex_adjacent": float(np.mean(adjacent)) if adjacent else 0,
        "hex_alternate": float(np.mean(alternate)) if alternate else 0,
        "hex_opposite": float(np.mean(opposite)) if opposite else 0,
        "projections": {t: float(projections[i]) for i, t in enumerate(TRAITS)},
    }


def cross_model_correlation_at_layer(vecs_a: np.ndarray, vecs_b: np.ndarray) -> float:
    """Cosine matrix correlation between two models at a given layer."""
    def cos_mat(v):
        n = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
        return n @ n.T
    mat_a = cos_mat(vecs_a)
    mat_b = cos_mat(vecs_b)
    idx = np.triu_indices(6, k=1)
    return float(np.corrcoef(mat_a[idx], mat_b[idx])[0, 1])


def main():
    print("=" * 70)
    print("LAYER-WISE RIASEC PERSONA VECTOR ANALYSIS")
    print("=" * 70)

    # Load all-layers vectors
    all_layers = {}
    for key in MODELS:
        vecs = load_all_layers(key)
        if vecs:
            num_layers = vecs[TRAITS[0]].shape[0]
            dim = vecs[TRAITS[0]].shape[1]
            all_layers[key] = vecs
            print(f"  {key}: {num_layers} layers, dim={dim}")

    results = {}
    for key, vecs in all_layers.items():
        num_layers = vecs[TRAITS[0]].shape[0]
        print(f"\n{'=' * 50}")
        print(f"Model: {key} ({MODELS[key]['size']}) - {num_layers} layers")
        print(f"{'=' * 50}")

        layer_results = []
        for L in range(num_layers):
            # Stack 6 trait vectors at layer L
            layer_vecs = np.array([vecs[t][L] for t in TRAITS])
            res = analyze_layer(layer_vecs)
            res["layer"] = L
            layer_results.append(res)

        results[key] = layer_results

        # Print summary
        print(f"\n  {'Layer':>5} {'Norm':>8} {'SharedF':>8} {'ResNorm':>8} {'CosMean':>8} {'ResCosMn':>8} {'Adj':>7} {'Alt':>7} {'Opp':>7}")
        for r in layer_results:
            print(f"  {r['layer']:5d} {r['mean_norm']:8.3f} {r['shared_frac']:8.3f} "
                  f"{r['mean_residual_norm']:8.3f} {r['cosine_mean']:8.3f} {r['residual_cosine_mean']:8.3f} "
                  f"{r['hex_adjacent']:7.3f} {r['hex_alternate']:7.3f} {r['hex_opposite']:7.3f}")

        # Key findings
        shared_fracs = [r["shared_frac"] for r in layer_results]
        norms = [r["mean_norm"] for r in layer_results]
        peak_norm_layer = int(np.argmax(norms))
        peak_shared_layer = int(np.argmax(shared_fracs))
        min_shared_layer = int(np.argmin(shared_fracs[1:]) + 1)  # skip layer 0

        print(f"\n  Key findings:")
        print(f"    Peak norm at layer {peak_norm_layer} (norm={norms[peak_norm_layer]:.3f})")
        print(f"    Peak shared fraction at layer {peak_shared_layer} (frac={shared_fracs[peak_shared_layer]:.3f})")
        print(f"    Min shared fraction at layer {min_shared_layer} (frac={shared_fracs[min_shared_layer]:.3f})")

        # Where does hex ordering hold?
        hex_consistent = []
        for r in layer_results:
            if r["hex_adjacent"] > r["hex_alternate"] > r["hex_opposite"]:
                hex_consistent.append(r["layer"])
        print(f"    Holland hexagon consistent at layers: {hex_consistent}")

    # Cross-model layer-wise correlation
    print(f"\n{'=' * 50}")
    print("Cross-model cosine correlation at matching fractional layers")
    print(f"{'=' * 50}")
    model_keys = sorted(all_layers.keys())
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            a, b = model_keys[i], model_keys[j]
            n_a = all_layers[a][TRAITS[0]].shape[0]
            n_b = all_layers[b][TRAITS[0]].shape[0]
            # Compare at matching fractional positions (0%, 25%, 50%, 75%, 100%)
            fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
            print(f"\n  {a} vs {b}:")
            for frac in fracs:
                la = min(int(frac * (n_a - 1)), n_a - 1)
                lb = min(int(frac * (n_b - 1)), n_b - 1)
                va = np.array([all_layers[a][t][la] for t in TRAITS])
                vb = np.array([all_layers[b][t][lb] for t in TRAITS])
                r = cross_model_correlation_at_layer(va, vb)
                print(f"    frac={frac:.0%} (L{la}, L{lb}): r={r:.4f}")

    # Save
    serializable = {}
    for key, layer_results in results.items():
        serializable[key] = []
        for r in layer_results:
            sr = {k: v for k, v in r.items() if k != "projections"}
            sr["projections"] = r["projections"]
            serializable[key].append(sr)

    out_path = OUTPUT_DIR / "layerwise_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
