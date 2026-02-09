#!/usr/bin/env python
"""
Analyze whether the 5D principal components have consistent semantic
meaning across different models.

Questions:
1. Do the same PCs encode the same trait contrasts in every model?
2. After Procrustes alignment, are the coordinate patterns identical?
3. Is the RIASEC hexagonal arrangement reflected in the 5D geometry?
4. What is the geometry of the simplex? (angles, distances, regularity)

This is pure analysis — no model loading needed.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from transformers import AutoConfig

from pvx import setup_logging

logger = setup_logging(name="5d-semantics")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Holland hexagonal order (clockwise): R, I, A, S, E, C
HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
# Adjacent pairs (distance=1 on hexagon)
ADJACENT = [("realistic", "investigative"), ("investigative", "artistic"),
            ("artistic", "social"), ("social", "enterprising"),
            ("enterprising", "conventional"), ("conventional", "realistic")]
# Opposite pairs (distance=3)
OPPOSITES = [("realistic", "social"), ("investigative", "enterprising"),
             ("artistic", "conventional")]

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
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual


def get_5d_with_svd(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, S[:5]


def procrustes_align(source_coords, target_coords):
    S = np.stack([source_coords[t] for t in TRAITS])
    T = np.stack([target_coords[t] for t in TRAITS])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))
    aligned = {t: scale * (R @ source_coords[t]) for t in TRAITS}
    return aligned


def main():
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load all models
    logger.info("Loading residual vectors...")
    model_data = {}
    for name, model_id in INSTRUCT_MODELS.items():
        residual = load_residual_vectors(model_id, riasec_dir)
        coords, svs = get_5d_with_svd(residual)
        dim = residual[TRAITS[0]].shape[0]
        model_data[name] = {"coords": coords, "svs": svs, "dim": dim}

    # === 1. Raw 5D coordinates per model ===
    print(f"\n{'='*70}")
    print(f"5D SEMANTIC ANALYSIS ACROSS 4 INSTRUCT MODELS")
    print(f"{'='*70}")

    for name in INSTRUCT_MODELS:
        print(f"\n--- {name} ({model_data[name]['dim']}d) ---")
        print(f"  {'Trait':>14}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC4':>7}  {'PC5':>7}")
        for t in TRAITS:
            c = model_data[name]["coords"][t]
            print(f"  {t:>14}  {c[0]:>+6.3f}  {c[1]:>+6.3f}  {c[2]:>+6.3f}  {c[3]:>+6.3f}  {c[4]:>+6.3f}")

    # === 2. Procrustes-aligned coordinates (align all to Marin-8B) ===
    ref = "Marin-8B"
    print(f"\n{'='*70}")
    print(f"PROCRUSTES-ALIGNED COORDINATES (reference: {ref})")
    print(f"{'='*70}")

    aligned_all = {ref: model_data[ref]["coords"]}
    for name in INSTRUCT_MODELS:
        if name == ref:
            continue
        aligned = procrustes_align(model_data[name]["coords"], model_data[ref]["coords"])
        aligned_all[name] = aligned

    # Show aligned coordinates
    for name in INSTRUCT_MODELS:
        print(f"\n--- {name} (aligned to {ref}) ---")
        print(f"  {'Trait':>14}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC4':>7}  {'PC5':>7}")
        for t in TRAITS:
            c = aligned_all[name][t]
            print(f"  {t:>14}  {c[0]:>+6.3f}  {c[1]:>+6.3f}  {c[2]:>+6.3f}  {c[3]:>+6.3f}  {c[4]:>+6.3f}")

    # === 3. Per-trait coordinate consistency after alignment ===
    print(f"\n--- Coordinate consistency (std across models for each trait/PC) ---")
    print(f"  {'Trait':>14}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC4':>7}  {'PC5':>7}")
    models_list = list(INSTRUCT_MODELS.keys())
    for t in TRAITS:
        stds = []
        for pc in range(5):
            vals = [aligned_all[name][t][pc] for name in models_list]
            stds.append(np.std(vals))
        print(f"  {t:>14}  {stds[0]:>6.3f}  {stds[1]:>6.3f}  {stds[2]:>6.3f}  {stds[3]:>6.3f}  {stds[4]:>6.3f}")

    # === 4. Pairwise cosines in 5D (aligned) ===
    print(f"\n--- Pairwise cosines in aligned 5D space (averaged across models) ---")
    avg_cosines = {}
    for i, t1 in enumerate(TRAITS):
        for j, t2 in enumerate(TRAITS):
            if i >= j:
                continue
            cosines = []
            for name in models_list:
                c1 = aligned_all[name][t1]
                c2 = aligned_all[name][t2]
                cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
                cosines.append(cos)
            avg_cosines[f"{t1}-{t2}"] = np.mean(cosines)

    # Display as matrix
    print(f"\n  {'':>14}", end="")
    for t in TRAITS:
        print(f"  {t[:4]:>5}", end="")
    print()
    for i, t1 in enumerate(TRAITS):
        print(f"  {t1:>14}", end="")
        for j, t2 in enumerate(TRAITS):
            if i == j:
                print(f"  {'1.00':>5}", end="")
            elif i < j:
                print(f"  {avg_cosines[f'{t1}-{t2}']:>+4.2f}", end="")
            else:
                print(f"  {avg_cosines[f'{t2}-{t1}']:>+4.2f}", end="")
        print()

    # === 5. Holland hexagonal structure test ===
    print(f"\n--- Holland hexagonal structure ---")

    adj_cosines = [avg_cosines[f"{min(a,b)}-{max(a,b)}"] if f"{min(a,b)}-{max(a,b)}" in avg_cosines
                   else avg_cosines[f"{a}-{b}"] for a, b in ADJACENT
                   if f"{a}-{b}" in avg_cosines or f"{b}-{a}" in avg_cosines]

    # Recalculate properly
    adj_cos = []
    for a, b in ADJACENT:
        key = f"{a}-{b}" if f"{a}-{b}" in avg_cosines else f"{b}-{a}"
        if key in avg_cosines:
            adj_cos.append(avg_cosines[key])

    opp_cos = []
    for a, b in OPPOSITES:
        key = f"{a}-{b}" if f"{a}-{b}" in avg_cosines else f"{b}-{a}"
        if key in avg_cosines:
            opp_cos.append(avg_cosines[key])

    # All other pairs are "alternate" (distance=2)
    all_pairs = set(avg_cosines.keys())
    adj_keys = set()
    for a, b in ADJACENT:
        adj_keys.add(f"{a}-{b}" if f"{a}-{b}" in avg_cosines else f"{b}-{a}")
    opp_keys = set()
    for a, b in OPPOSITES:
        opp_keys.add(f"{a}-{b}" if f"{a}-{b}" in avg_cosines else f"{b}-{a}")
    alt_keys = all_pairs - adj_keys - opp_keys
    alt_cos = [avg_cosines[k] for k in alt_keys]

    print(f"  Adjacent (d=1): mean={np.mean(adj_cos):.3f} ({len(adj_cos)} pairs)")
    for k in sorted(adj_keys):
        print(f"    {k}: {avg_cosines[k]:+.3f}")
    print(f"  Alternate (d=2): mean={np.mean(alt_cos):.3f} ({len(alt_cos)} pairs)")
    for k in sorted(alt_keys):
        print(f"    {k}: {avg_cosines[k]:+.3f}")
    print(f"  Opposite (d=3): mean={np.mean(opp_cos):.3f} ({len(opp_cos)} pairs)")
    for k in sorted(opp_keys):
        print(f"    {k}: {avg_cosines[k]:+.3f}")

    # Holland prediction: adjacent > alternate > opposite (in cosine)
    holland_consistent = np.mean(adj_cos) > np.mean(alt_cos) > np.mean(opp_cos)
    print(f"\n  Holland order (adj > alt > opp): {'CONFIRMED' if holland_consistent else 'VIOLATED'}")
    print(f"  {np.mean(adj_cos):.3f} > {np.mean(alt_cos):.3f} > {np.mean(opp_cos):.3f}")

    # === 6. 2D projection (first 2 PCs) for hexagonal visualization ===
    print(f"\n--- 2D projection (PC1 vs PC2) of aligned coordinates ---")
    print(f"  {'Model':>10}  ", end="")
    for t in HOLLAND_ORDER:
        print(f"  ({t[:3]}_1, {t[:3]}_2)", end="")
    print()
    for name in models_list:
        print(f"  {name:>10}  ", end="")
        for t in HOLLAND_ORDER:
            c = aligned_all[name][t]
            print(f"  ({c[0]:>+5.1f}, {c[1]:>+5.1f})", end="")
        print()

    # Check: do traits arranged in Holland order form a hexagonal pattern in 2D?
    # Compute angular position of each trait in PC1-PC2 plane
    print(f"\n--- Angular position in PC1-PC2 plane (Holland order) ---")
    for name in models_list:
        print(f"  {name}:")
        angles = []
        for t in HOLLAND_ORDER:
            c = aligned_all[name][t]
            angle = np.degrees(np.arctan2(c[1], c[0]))
            angles.append(angle)
            print(f"    {t:>14}: {angle:>+7.1f}°")

        # Check if angles increase monotonically (modulo 360)
        # For a hexagon, they should be ~60° apart
        diffs = []
        for k in range(len(angles)-1):
            diff = (angles[k+1] - angles[k]) % 360
            if diff > 180:
                diff -= 360
            diffs.append(diff)
        print(f"    Angular gaps: {', '.join(f'{d:+.0f}°' for d in diffs)}")

    # Save results
    results = {
        "aligned_coordinates": {},
        "avg_cosines": avg_cosines,
        "holland_structure": {
            "adjacent_mean": float(np.mean(adj_cos)),
            "alternate_mean": float(np.mean(alt_cos)),
            "opposite_mean": float(np.mean(opp_cos)),
            "holland_consistent": bool(holland_consistent),
        },
    }
    for name in models_list:
        results["aligned_coordinates"][name] = {
            t: [float(x) for x in aligned_all[name][t]] for t in TRAITS
        }
    results["avg_cosines"] = {k: float(v) for k, v in avg_cosines.items()}

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "5d_semantics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
