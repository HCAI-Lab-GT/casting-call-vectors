#!/usr/bin/env python
"""
Analyze what the shared direction represents across models.

The shared direction (PC1) captures ~60-72% of variance. Understanding what it
encodes is crucial for interpretation. Key questions:
1. Is it really "agreeableness/compliance"?
2. Are projections onto shared direction consistent across models?
3. Does projection magnitude predict steering strength?
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from scipy.stats import pearsonr, spearmanr

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "Llama 1B": "meta-llama/Llama-3.2-1B-Instruct",
    "SmolLM3 3B": "HuggingFaceTB/SmolLM3-3B",
    "Qwen 7B": "Qwen/Qwen2.5-7B-Instruct",
    "Marin 8B": "marin-community/marin-8b-instruct",
    "Marin 32B": "marin-community/marin-32b-base",
}


def load_vectors(model_id):
    safe = model_id.replace("/", "__")
    vecs = {}
    for trait in TRAITS:
        path = ROOT / f"persona_data/model_inits/{trait}_persona_initialization/{safe}.safetensors"
        data = load_file(str(path))
        v = data["all_layers_response_persona_vector"].numpy()
        if v.ndim == 3:
            v = v[:, 0, :]
        num_layers = v.shape[0] - 1
        mid = num_layers // 2
        vecs[trait] = v[mid + 1]
    return vecs


def analyze_shared_direction(vecs):
    V = np.stack([vecs[t] for t in TRAITS])
    norms = np.linalg.norm(V, axis=1)
    V_normed = V / norms[:, None]
    _, s, Vt = np.linalg.svd(V_normed, full_matrices=False)

    shared = Vt[0]
    projections = {t: float(np.dot(V_normed[i], shared)) for i, t in enumerate(TRAITS)}
    variance_explained = s[0]**2 / np.sum(s**2)

    return {
        "shared_direction": shared,
        "projections": projections,
        "variance_explained": float(variance_explained),
        "singular_values": s.tolist(),
        "norms": {t: float(norms[i]) for i, t in enumerate(TRAITS)},
    }


def main():
    all_data = {}
    for name, model_id in MODELS.items():
        vecs = load_vectors(model_id)
        all_data[name] = analyze_shared_direction(vecs)

    print("=" * 70)
    print("SHARED DIRECTION ANALYSIS ACROSS 5 MODELS")
    print("=" * 70)

    # 1. Shared direction projections
    print(f"\n--- Projection onto Shared Direction (PC1) ---")
    print(f"  {'Model':>12}", end="")
    for t in TRAITS:
        print(f"  {t[:5]:>7}", end="")
    print(f"  {'Var%':>6}")
    print(f"  {'-'*72}")

    proj_matrix = {}
    for name in MODELS:
        d = all_data[name]
        print(f"  {name:>12}", end="")
        for t in TRAITS:
            print(f"  {d['projections'][t]:>+6.3f}", end="")
        print(f"  {d['variance_explained']:>5.1%}")
        proj_matrix[name] = [d['projections'][t] for t in TRAITS]

    # 2. Cross-model consistency of projections
    print(f"\n--- Cross-model Projection Consistency ---")
    names = list(MODELS.keys())
    print(f"  {'':>12}", end="")
    for n in names:
        print(f"{n:>12}", end="")
    print()
    for n1 in names:
        print(f"  {n1:>12}", end="")
        for n2 in names:
            if n1 == n2:
                print(f"{'1.000':>12}", end="")
            else:
                r, _ = pearsonr(proj_matrix[n1], proj_matrix[n2])
                print(f"{r:>12.3f}", end="")
        print()

    # 3. Projection ordering (consistent across models?)
    print(f"\n--- Projection Ranking (lowest to highest) ---")
    for name in MODELS:
        ranked = sorted(all_data[name]["projections"].items(), key=lambda x: x[1])
        rank_str = " < ".join(f"{t[:3]}({v:+.3f})" for t, v in ranked)
        print(f"  {name:>12}: {rank_str}")

    # 4. Is projection rank perfectly consistent?
    rankings = {}
    for name in MODELS:
        ranked = sorted(all_data[name]["projections"].items(), key=lambda x: x[1])
        rankings[name] = [t for t, _ in ranked]

    # Check if all rankings are identical
    base_ranking = rankings[names[0]]
    all_same = all(rankings[n] == base_ranking for n in names[1:])
    print(f"\n  All rankings identical: {all_same}")

    if not all_same:
        # Show which traits change position
        for n in names[1:]:
            diffs = [(i, base_ranking[i], rankings[n][i])
                     for i in range(6) if base_ranking[i] != rankings[n][i]]
            if diffs:
                print(f"  {names[0]:>12} vs {n}: differs at positions {[d[0] for d in diffs]}")

    # 5. Norm analysis
    print(f"\n--- Vector Norms (unnormalized) ---")
    print(f"  {'Model':>12}", end="")
    for t in TRAITS:
        print(f"  {t[:5]:>7}", end="")
    print()
    for name in MODELS:
        d = all_data[name]
        print(f"  {name:>12}", end="")
        for t in TRAITS:
            print(f"  {d['norms'][t]:>7.2f}", end="")
        print()

    # 6. Shared direction cosine across models
    # In same-dim models (Llama 1B and SmolLM3 3B both have dim=2048), can we compare shared directions?
    print(f"\n--- Shared Direction Alignment (same-dim models only) ---")
    same_dim_pairs = [
        ("Llama 1B", "SmolLM3 3B"),  # both 2048
    ]
    for n1, n2 in same_dim_pairs:
        sd1 = all_data[n1]["shared_direction"]
        sd2 = all_data[n2]["shared_direction"]
        cos = np.dot(sd1, sd2) / (np.linalg.norm(sd1) * np.linalg.norm(sd2))
        print(f"  {n1} ↔ {n2}: cos = {cos:.4f}")

    # 7. Singular value spectra comparison
    print(f"\n--- Singular Value Spectra (% of total) ---")
    print(f"  {'Model':>12}", end="")
    for i in range(6):
        print(f"  {'s'+str(i+1):>6}", end="")
    print()
    for name in MODELS:
        s = np.array(all_data[name]["singular_values"])
        pcts = s**2 / np.sum(s**2) * 100
        print(f"  {name:>12}", end="")
        for p in pcts:
            print(f"  {p:>5.1f}%", end="")
        print()

    # Summary
    print(f"\n{'='*70}")
    print("KEY OBSERVATIONS")
    print(f"{'='*70}")
    print("""
  1. PROJECTION ORDERING: Is it consistent across ALL 5 models?
  2. SHARED FRACTION: Base model (32B) has LOWER shared fraction (0.588)
     suggesting pretraining encodes weaker shared "agree" bias
  3. NORMS: Do norm patterns predict anything about steering effectiveness?
  4. SINGULAR SPECTRA: Are residual dimensions approximately equal?
""")


if __name__ == "__main__":
    main()
