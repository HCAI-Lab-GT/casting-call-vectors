#!/usr/bin/env python
"""
Test whether RIASEC residual vectors form an equiangular tight frame (ETF).

For n=6 vectors in d=5 dimensions (after removing the shared direction),
the theoretical equiangular frame has all pairwise cosines = -1/(n-1) = -1/5 = -0.200.

If the 6 RIASEC residual vectors approximate an ETF, this means they are maximally
separated in the residual subspace -- they spread out as uniformly as possible.
This would be a strong structural constraint on how transformers represent personality.

We also compare against random baselines to assess statistical significance.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

MODELS = {
    "llama-1b": "meta-llama__Llama-3.2-1B-Instruct",
    "marin-8b": "marin-community__marin-8b-instruct",
    "qwen-7b": "Qwen__Qwen2.5-7B-Instruct",
}

SAFETENSORS_DIR = Path("./persona_data/model_inits/")
OUTPUT_DIR = Path("./outputs/analysis/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_vectors(safe_id: str) -> dict[str, np.ndarray]:
    vectors = {}
    for trait in TRAITS:
        path = SAFETENSORS_DIR / f"{trait}_persona_initialization" / f"{safe_id}.safetensors"
        data = load_file(str(path))
        vectors[trait] = data["response_persona_vector"].numpy().flatten()
    return vectors


def load_all_layers(safe_id: str) -> dict[str, np.ndarray]:
    vectors = {}
    for trait in TRAITS:
        path = SAFETENSORS_DIR / f"{trait}_persona_initialization" / f"{safe_id}.safetensors"
        data = load_file(str(path))
        v = data["all_layers_response_persona_vector"].numpy()
        if v.ndim == 3:
            v = v.squeeze(1)
        vectors[trait] = v
    return vectors


def decompose_and_analyze(vecs_6xD: np.ndarray):
    """Given 6 trait vectors (6, D), remove shared direction and analyze residuals."""
    # SVD to get shared direction
    U, S, Vt = np.linalg.svd(vecs_6xD, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    # Project and get residuals
    projections = vecs_6xD @ shared_dir
    residuals = vecs_6xD - np.outer(projections, shared_dir)

    # Residual pairwise cosines
    res_norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    res_normed = residuals / np.maximum(res_norms, 1e-8)
    cos_mat = res_normed @ res_normed.T

    # Extract upper triangle (15 pairs)
    idx = np.triu_indices(6, k=1)
    cosines = cos_mat[idx]

    mean_cos = np.mean(cosines)
    std_cos = np.std(cosines)
    theoretical_etf = -1.0 / 5  # = -0.200

    # Gram matrix analysis: for a perfect ETF, the Gram matrix should be
    # G = (1/(n-1)) * (n*I - J) where J is all-ones matrix
    # i.e. diagonal = 1, off-diagonal = -1/(n-1) = -0.2
    gram = cos_mat
    off_diag = cosines

    # Deviation from ETF
    etf_deviation = np.sqrt(np.mean((off_diag - theoretical_etf) ** 2))

    return {
        "mean_cosine": float(mean_cos),
        "std_cosine": float(std_cos),
        "min_cosine": float(np.min(cosines)),
        "max_cosine": float(np.max(cosines)),
        "theoretical_etf": float(theoretical_etf),
        "etf_deviation": float(etf_deviation),
        "all_cosines": cosines.tolist(),
    }


def random_baseline(dim: int, n_trials: int = 500) -> dict:
    """Generate random 6-vector sets and compute their residual cosine stats."""
    rng = np.random.default_rng(42)
    mean_cosines = []
    std_cosines = []

    for _ in range(n_trials):
        # Random 6 vectors in dim dimensions
        vecs = rng.standard_normal((6, dim))
        # Remove shared direction (PC1)
        U, S, Vt = np.linalg.svd(vecs, full_matrices=False)
        shared = Vt[0]
        proj = vecs @ shared
        residuals = vecs - np.outer(proj, shared)
        # Cosines
        norms = np.linalg.norm(residuals, axis=1, keepdims=True)
        normed = residuals / np.maximum(norms, 1e-8)
        cos_mat = normed @ normed.T
        idx = np.triu_indices(6, k=1)
        cosines = cos_mat[idx]
        mean_cosines.append(np.mean(cosines))
        std_cosines.append(np.std(cosines))

    return {
        "mean_of_mean_cosine": float(np.mean(mean_cosines)),
        "std_of_mean_cosine": float(np.std(mean_cosines)),
        "mean_of_std_cosine": float(np.mean(std_cosines)),
        "p95_mean_cosine": float(np.percentile(mean_cosines, [2.5, 97.5]).tolist()[0]),
        "theoretical_random": float(-1 / (dim - 1)),  # for dim >> 6
    }


def main():
    print("=" * 70)
    print("EQUIANGULAR TIGHT FRAME ANALYSIS OF RIASEC RESIDUALS")
    print("=" * 70)

    # Theory
    print(f"\nTheoretical ETF for n=6 vectors in d=5: cos = -1/(n-1) = -0.200")
    print(f"If residuals approximate an ETF, RIASEC traits are maximally separated.")

    # Analyze at extraction layer
    print(f"\n--- Analysis at extraction layer ---")
    results = {}
    for name, safe_id in MODELS.items():
        vecs = load_vectors(safe_id)
        mat = np.array([vecs[t] for t in TRAITS])
        res = decompose_and_analyze(mat)
        results[name] = res
        print(f"\n  {name}:")
        print(f"    Mean residual cosine: {res['mean_cosine']:.4f} (ETF = {res['theoretical_etf']:.4f})")
        print(f"    Std of cosines:       {res['std_cosine']:.4f} (ETF = 0.000)")
        print(f"    Range:                [{res['min_cosine']:.4f}, {res['max_cosine']:.4f}]")
        print(f"    RMSE from ETF:        {res['etf_deviation']:.4f}")
        print(f"    All 15 pairwise cosines: {[f'{c:.3f}' for c in res['all_cosines']]}")

    # Random baselines
    print(f"\n--- Random baseline (500 trials) ---")
    for dim in [2048, 3584, 4096]:
        bl = random_baseline(dim)
        print(f"  dim={dim}: random mean cosine = {bl['mean_of_mean_cosine']:.4f} "
              f"(std {bl['std_of_mean_cosine']:.4f}), "
              f"random std cosine = {bl['mean_of_std_cosine']:.4f}")

    # Layer-wise ETF analysis
    print(f"\n--- Layer-wise ETF deviation ---")
    for name, safe_id in MODELS.items():
        all_layers = load_all_layers(safe_id)
        num_layers = all_layers[TRAITS[0]].shape[0]
        print(f"\n  {name} ({num_layers} layers):")
        print(f"  {'Layer':>5} {'MeanCos':>8} {'StdCos':>8} {'ETF_Dev':>8} {'ETF?':>5}")

        layer_etf = []
        for L in range(num_layers):
            layer_vecs = np.array([all_layers[t][L] for t in TRAITS])
            if np.linalg.norm(layer_vecs) < 1e-6:
                continue
            res = decompose_and_analyze(layer_vecs)
            etf_like = "✓" if res["std_cosine"] < 0.05 else " "
            print(f"  {L:5d} {res['mean_cosine']:8.4f} {res['std_cosine']:8.4f} "
                  f"{res['etf_deviation']:8.4f} {etf_like:>5}")
            layer_etf.append({
                "layer": L,
                "mean_cosine": res["mean_cosine"],
                "std_cosine": res["std_cosine"],
                "etf_deviation": res["etf_deviation"],
            })

        results[f"{name}_layerwise"] = layer_etf

    # Save
    out_path = OUTPUT_DIR / "equiangular_frame_analysis.json"
    # Convert numpy arrays in results
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = v
        elif isinstance(v, list):
            serializable[k] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
