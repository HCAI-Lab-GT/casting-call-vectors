#!/usr/bin/env python
"""
Exhaustive permutation + sign search for zero-calibration transfer.

PCA can produce principal components in arbitrary order AND with arbitrary signs.
Between two models, the PCs might not just be sign-flipped but also reordered.

This script exhaustively searches ALL 5! × 2^5 = 3840 possible mappings
(permutations of 5 PCs × sign flips) to find which mapping best aligns
source and target 5D coordinates WITHOUT using any behavioral data.

Scoring criterion: mean cosine similarity between transformed source coords
and target coords (purely geometric, no model inference needed).

Key questions:
1. Does the identity permutation (what we currently use) produce the best alignment?
2. Or is there a better permutation that PCA arbitrarily missed?
3. How much room for improvement exists beyond sign correction alone?
"""

import json
from itertools import permutations, product
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from transformers import AutoConfig

from pvx import setup_logging

logger = setup_logging(name="perm-sign-search")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

INSTRUCT_MODELS = {
    "SmolLM3": "HuggingFaceTB/SmolLM3-3B",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Marin-8B": "marin-community/marin-8b-instruct",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    return residual


def get_5d_coords(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, S[:5]


def score_alignment(source_coords, target_coords, perm, signs):
    """Score a permutation+sign mapping by mean cosine similarity.

    Applies: target_estimated[pc] = signs[pc] * source[perm[pc]]
    Then computes mean cosine between estimated and actual target coords.
    """
    # Build transformation matrix
    transformed = {}
    for t in TRAITS:
        s = source_coords[t]
        # Apply permutation and sign flip
        new_coord = np.array([signs[i] * s[perm[i]] for i in range(5)])
        transformed[t] = new_coord

    # Scale to match target norms
    s_norms = np.mean([np.linalg.norm(transformed[t]) for t in TRAITS])
    t_norms = np.mean([np.linalg.norm(target_coords[t]) for t in TRAITS])
    scale = t_norms / s_norms if s_norms > 0 else 1.0

    # Cosine similarity
    cosines = []
    for t in TRAITS:
        a = scale * transformed[t]
        b = target_coords[t]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        cosines.append(cos)

    return np.mean(cosines)


def procrustes_score(source_coords, target_coords):
    """Score using full Procrustes alignment."""
    S = np.stack([source_coords[t] for t in TRAITS])
    T = np.stack([target_coords[t] for t in TRAITS])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))

    cosines = []
    for t in TRAITS:
        a = scale * (R @ source_coords[t])
        b = target_coords[t]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        cosines.append(cos)
    return np.mean(cosines)


def main():
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load all model coordinates
    logger.info("Loading all model vectors...")
    model_coords = {}
    for name, model_id in INSTRUCT_MODELS.items():
        residual = load_residual_vectors(model_id, riasec_dir)
        coords, sv = get_5d_coords(residual)
        model_coords[name] = coords

    print(f"\n{'='*70}")
    print(f"EXHAUSTIVE PERMUTATION + SIGN SEARCH (5!×2^5 = 3840 combos)")
    print(f"{'='*70}")

    # Generate all permutations and sign combinations
    all_perms = list(permutations(range(5)))  # 120
    all_signs = list(product([-1, 1], repeat=5))  # 32

    names = list(INSTRUCT_MODELS.keys())
    results = {}

    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            src, tgt = names[i], names[j]
            pair_key = f"{src}→{tgt}"

            # Exhaustive search
            best_score = -2
            best_perm = None
            best_signs = None
            identity_score = None

            for perm in all_perms:
                for signs in all_signs:
                    score = score_alignment(model_coords[src], model_coords[tgt], perm, signs)
                    if score > best_score:
                        best_score = score
                        best_perm = perm
                        best_signs = signs
                    # Track identity permutation
                    if perm == (0, 1, 2, 3, 4):
                        if identity_score is None or score > identity_score:
                            identity_score = score

            # Compare to Procrustes
            proc_score = procrustes_score(model_coords[src], model_coords[tgt])

            # Best sign-only (identity permutation, best signs)
            best_sign_only_score = -2
            best_sign_only = None
            for signs in all_signs:
                score = score_alignment(model_coords[src], model_coords[tgt], (0,1,2,3,4), signs)
                if score > best_sign_only_score:
                    best_sign_only_score = score
                    best_sign_only = signs

            is_identity = best_perm == (0, 1, 2, 3, 4)

            results[pair_key] = {
                "best_perm": list(best_perm),
                "best_signs": list(best_signs),
                "best_score": float(best_score),
                "identity_perm_best_score": float(best_sign_only_score),
                "procrustes_score": float(proc_score),
                "is_identity_best": is_identity,
                "gain_over_identity": float(best_score - best_sign_only_score),
                "gap_to_procrustes": float(proc_score - best_score),
            }

            perm_str = str(best_perm)
            sign_str = "[" + ",".join(f"{s:+d}" for s in best_signs) + "]"
            print(f"\n  {pair_key}:")
            print(f"    Best perm+sign: {perm_str} {sign_str} → cos={best_score:.4f}")
            print(f"    Identity+sign:  (0,1,2,3,4) [{','.join(f'{s:+d}' for s in best_sign_only)}] → cos={best_sign_only_score:.4f}")
            print(f"    Procrustes:     → cos={proc_score:.4f}")
            print(f"    Identity is best? {'YES' if is_identity else 'NO'}")
            print(f"    Gain over identity: {best_score - best_sign_only_score:+.4f}")
            print(f"    Gap to Procrustes: {proc_score - best_score:+.4f}")

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    identity_count = sum(1 for r in results.values() if r["is_identity_best"])
    total = len(results)
    print(f"\n  Identity permutation is optimal: {identity_count}/{total} pairs ({identity_count/total:.0%})")

    gains = [r["gain_over_identity"] for r in results.values()]
    gaps = [r["gap_to_procrustes"] for r in results.values()]
    print(f"\n  Gain from permutation search over sign-only:")
    print(f"    Mean: {np.mean(gains):+.4f}")
    print(f"    Max:  {np.max(gains):+.4f}")

    print(f"\n  Gap between best perm+sign and Procrustes:")
    print(f"    Mean: {np.mean(gaps):+.4f}")
    print(f"    Max:  {np.max(gaps):+.4f}")

    # Check: are there consistent permutations across pairs?
    print(f"\n--- Best permutations (across all pairs) ---")
    perm_counts = {}
    for r in results.values():
        p = tuple(r["best_perm"])
        perm_counts[p] = perm_counts.get(p, 0) + 1
    for perm, count in sorted(perm_counts.items(), key=lambda x: -x[1]):
        print(f"  {perm}: {count} pairs")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "permutation_sign_search.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
