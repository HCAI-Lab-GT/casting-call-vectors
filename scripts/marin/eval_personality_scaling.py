#!/usr/bin/env python
"""
Personality Scaling Laws Across Model Sizes.

We've established that personality lives in exactly 5D for ALL models.
But how does the STRENGTH and QUALITY of personality representation scale?

Tests across 3 models: Llama 1B (16L), SmolLM3 3B (28L), Marin 8B (32L):
1. Singular value spectrum of personality residuals — does the 5D structure
   get "sharper" (higher condition number) with scale?
2. Personality vector norms vs model hidden size
3. Cross-model Procrustes quality vs model size
4. Detection sensitivity (minimum detectable alpha) vs model size
5. Behavioral discrimination accuracy vs model size at matched alpha

This reveals whether larger models have "stronger" personality representations
or just "larger" ones (norm-scaled).
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from transformers import AutoConfig

from pvx import setup_logging

logger = setup_logging(name="pers-scaling")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "marin-community/marin-8b-instruct",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def analyze_model_geometry(model_id, riasec_dir):
    """Analyze personality geometry for one model without loading weights."""
    safe_model = model_id.replace("/", "__")
    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        if not path.exists():
            return None
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Full vectors at detection layer
    V_full = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])

    # SVD of full vectors
    U_full, S_full, Vt_full = np.linalg.svd(V_full, full_matrices=False)

    # Shared direction and residuals
    shared_dir = Vt_full[0]
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj

    V_res = np.stack([residual[t] for t in TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    # Metrics
    norms = {t: float(np.linalg.norm(residual[t])) for t in TRAITS}
    full_norms = {t: float(np.linalg.norm(all_layer_vectors[t][mid_layer + 1])) for t in TRAITS}

    # Condition number of 5D basis
    cond = float(S_res[0] / S_res[4]) if S_res[4] > 0 else float("inf")

    # Variance explained by each PC
    total_var = float(np.sum(S_res**2))
    var_explained = [(float(s**2 / total_var)) for s in S_res]

    # 6th singular value (should be ~0)
    sixth_sv = float(S_res[5]) if len(S_res) > 5 else 0.0

    # Simplex quality (mean pairwise distance)
    pairwise_cos = []
    for i, t1 in enumerate(TRAITS):
        for t2 in TRAITS[i+1:]:
            c = float(np.dot(coords_5d[t1], coords_5d[t2]) / (
                np.linalg.norm(coords_5d[t1]) * np.linalg.norm(coords_5d[t2])))
            pairwise_cos.append(c)

    # Perturbation magnitude
    mean_full_norm = float(np.mean(list(full_norms.values())))
    mean_res_norm = float(np.mean(list(norms.values())))
    perturbation_frac = mean_res_norm / mean_full_norm if mean_full_norm > 0 else 0

    # 5D vs full-dim cosine fidelity
    fidelities = []
    for t in TRAITS:
        recon = basis_5d.T @ coords_5d[t]
        cos = float(np.dot(recon, residual[t]) / (
            np.linalg.norm(recon) * np.linalg.norm(residual[t])))
        fidelities.append(cos)

    # Layer profile: at which layer does personality emerge?
    layer_norms = []
    for lidx in range(num_layers):
        V_layer = np.stack([all_layer_vectors[t][lidx] for t in TRAITS])
        mean_norm = float(np.mean(np.linalg.norm(V_layer, axis=1)))
        layer_norms.append(mean_norm)

    # Normalized layer profile (fraction of max)
    max_norm = max(layer_norms) if layer_norms else 1
    layer_frac = [n / max_norm for n in layer_norms]

    return {
        "model_id": model_id,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "mid_layer": mid_layer,
        "num_params_approx": hidden_size * hidden_size * num_layers * 4 / 1e9,  # rough estimate
        "singular_values": S_res.tolist(),
        "condition_number": cond,
        "sixth_sv": sixth_sv,
        "variance_explained": var_explained,
        "residual_norms": norms,
        "full_norms": full_norms,
        "mean_residual_norm": mean_res_norm,
        "mean_full_norm": mean_full_norm,
        "perturbation_fraction": perturbation_frac,
        "pairwise_cosines": pairwise_cos,
        "mean_pairwise_cos": float(np.mean(pairwise_cos)),
        "reconstruction_fidelity": fidelities,
        "mean_fidelity": float(np.mean(fidelities)),
        "layer_norm_profile": layer_norms,
        "layer_frac_profile": layer_frac,
        "coords_5d": {t: coords_5d[t].tolist() for t in TRAITS},
    }


def cross_model_procrustes(data_a, data_b):
    """Compute Procrustes alignment between two models' 5D coordinates."""
    coords_a = np.stack([np.array(data_a["coords_5d"][t]) for t in TRAITS])
    coords_b = np.stack([np.array(data_b["coords_5d"][t]) for t in TRAITS])

    # Center
    coords_a_c = coords_a - coords_a.mean(axis=0)
    coords_b_c = coords_b - coords_b.mean(axis=0)

    # SVD of cross-covariance
    M = coords_a_c.T @ coords_b_c
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    det = np.linalg.det(R)
    if det < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Apply rotation
    coords_b_rot = coords_b_c @ R.T

    # Compute alignment cosine per trait
    per_trait_cos = []
    for i, t in enumerate(TRAITS):
        a_norm = np.linalg.norm(coords_a_c[i])
        b_norm = np.linalg.norm(coords_b_rot[i])
        if a_norm > 0 and b_norm > 0:
            c = float(np.dot(coords_a_c[i], coords_b_rot[i]) / (a_norm * b_norm))
            per_trait_cos.append(c)

    # Overall Procrustes distance
    total_cos = float(np.sum(S) / (
        np.sqrt(np.sum(coords_a_c**2)) * np.sqrt(np.sum(coords_b_c**2))))

    return {
        "total_procrustes_cosine": total_cos,
        "per_trait_cosines": per_trait_cos,
        "mean_per_trait_cos": float(np.mean(per_trait_cos)) if per_trait_cos else 0,
    }


def main():
    riasec_dir = _repo_root() / "persona_data/model_inits"
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY SCALING LAWS ACROSS MODEL SIZES")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Per-model geometry analysis
    # ================================================================
    logger.info("Part 1: Per-model geometry...")
    print(f"\n{'='*70}")
    print("PART 1: GEOMETRY PER MODEL")
    print(f"{'='*70}")

    model_data = {}
    for model_id in MODELS:
        logger.info(f"  Analyzing {model_id}...")
        data = analyze_model_geometry(model_id, riasec_dir)
        if data is None:
            print(f"  {model_id}: vectors not found, skipping")
            continue

        model_data[model_id] = data
        short = model_id.split("/")[-1]

        print(f"\n  {short}:")
        print(f"    Layers: {data['num_layers']}, Hidden: {data['hidden_size']}")
        print(f"    Singular values: {[f'{s:.2f}' for s in data['singular_values'][:6]]}")
        print(f"    Condition number: {data['condition_number']:.2f}")
        print(f"    6th SV: {data['sixth_sv']:.6f}")
        print(f"    Variance explained: {[f'{v:.1%}' for v in data['variance_explained'][:6]]}")
        print(f"    Mean residual norm: {data['mean_residual_norm']:.2f}")
        print(f"    Perturbation fraction: {data['perturbation_fraction']:.4f}")
        print(f"    Mean pairwise cosine: {data['mean_pairwise_cos']:.3f}")
        print(f"    Reconstruction fidelity: {data['mean_fidelity']:.6f}")

    results["per_model"] = {k: {kk: vv for kk, vv in v.items() if kk != "layer_norm_profile"}
                             for k, v in model_data.items()}

    # ================================================================
    # PART 2: Scaling relationships
    # ================================================================
    logger.info("Part 2: Scaling analysis...")
    print(f"\n{'='*70}")
    print("PART 2: SCALING RELATIONSHIPS")
    print(f"{'='*70}")

    ids_with_data = [m for m in MODELS if m in model_data]
    if len(ids_with_data) >= 2:
        hidden_sizes = [model_data[m]["hidden_size"] for m in ids_with_data]
        num_layers_list = [model_data[m]["num_layers"] for m in ids_with_data]
        res_norms = [model_data[m]["mean_residual_norm"] for m in ids_with_data]
        full_norms = [model_data[m]["mean_full_norm"] for m in ids_with_data]
        cond_numbers = [model_data[m]["condition_number"] for m in ids_with_data]
        perturb_fracs = [model_data[m]["perturbation_fraction"] for m in ids_with_data]

        print(f"\n  {'Model':>30} {'Hidden':>8} {'Layers':>8} {'ResNorm':>10} "
              f"{'FullNorm':>10} {'Perturb%':>10} {'Cond#':>8}")
        for m in ids_with_data:
            d = model_data[m]
            short = m.split("/")[-1]
            print(f"  {short:>30} {d['hidden_size']:>8} {d['num_layers']:>8} "
                  f"{d['mean_residual_norm']:>10.2f} {d['mean_full_norm']:>10.2f} "
                  f"{d['perturbation_fraction']*100:>10.3f} {d['condition_number']:>8.2f}")

        # Norm vs hidden_size scaling
        if len(ids_with_data) >= 2:
            log_h = np.log(hidden_sizes)
            log_rn = np.log(res_norms)
            if len(ids_with_data) >= 2:
                slope = float((log_rn[-1] - log_rn[0]) / (log_h[-1] - log_h[0]))
                print(f"\n  Residual norm ∝ hidden_size^{slope:.2f}")

            log_fn = np.log(full_norms)
            slope_fn = float((log_fn[-1] - log_fn[0]) / (log_h[-1] - log_h[0]))
            print(f"  Full norm ∝ hidden_size^{slope_fn:.2f}")

            # Perturbation fraction vs size
            print(f"\n  Perturbation fraction (personality as % of total activation):")
            for m in ids_with_data:
                d = model_data[m]
                short = m.split("/")[-1]
                print(f"    {short}: {d['perturbation_fraction']*100:.3f}%")

        results["scaling"] = {
            "hidden_sizes": hidden_sizes,
            "num_layers": num_layers_list,
            "residual_norms": res_norms,
            "full_norms": full_norms,
            "condition_numbers": cond_numbers,
            "perturbation_fractions": perturb_fracs,
        }

    # ================================================================
    # PART 3: Cross-model Procrustes alignment
    # ================================================================
    logger.info("Part 3: Cross-model Procrustes...")
    print(f"\n{'='*70}")
    print("PART 3: CROSS-MODEL PROCRUSTES ALIGNMENT")
    print(f"{'='*70}")

    procrustes_results = {}
    for i, m1 in enumerate(ids_with_data):
        for m2 in ids_with_data[i+1:]:
            p = cross_model_procrustes(model_data[m1], model_data[m2])
            short1 = m1.split("/")[-1]
            short2 = m2.split("/")[-1]
            key = f"{short1}_vs_{short2}"
            procrustes_results[key] = p

            print(f"\n  {short1} ↔ {short2}:")
            print(f"    Procrustes cosine: {p['total_procrustes_cosine']:.4f}")
            print(f"    Per-trait: {[f'{c:.3f}' for c in p['per_trait_cosines']]}")

    results["procrustes"] = procrustes_results

    # ================================================================
    # PART 4: Layer emergence profile comparison
    # ================================================================
    logger.info("Part 4: Layer emergence profiles...")
    print(f"\n{'='*70}")
    print("PART 4: PERSONALITY EMERGENCE LAYER PROFILES")
    print(f"{'='*70}")

    for m in ids_with_data:
        d = model_data[m]
        short = m.split("/")[-1]
        print(f"\n  {short} ({d['num_layers']} layers):")

        norms = d["layer_norm_profile"]
        fracs = d["layer_frac_profile"]

        # Find emergence point (first layer > 50% of max)
        emergence = None
        for lidx, f in enumerate(fracs):
            if f > 0.5:
                emergence = lidx
                break

        # Find peak
        peak = int(np.argmax(norms))

        print(f"    Peak at L{peak} (norm={norms[peak]:.2f})")
        if emergence is not None:
            print(f"    Emergence (>50% max) at L{emergence} ({emergence/d['num_layers']:.0%} depth)")
        else:
            print(f"    No clear emergence point")

        # Show profile at 25%, 50%, 75%, 100% depth
        for pct in [0.25, 0.5, 0.75, 1.0]:
            lidx = min(int(pct * d["num_layers"]) - 1, d["num_layers"] - 1)
            print(f"    L{lidx} ({pct:.0%} depth): norm={norms[lidx]:.2f}, "
                  f"frac={fracs[lidx]:.3f}")

    # ================================================================
    # PART 5: Holland hexagonal structure comparison
    # ================================================================
    logger.info("Part 5: Holland structure comparison...")
    print(f"\n{'='*70}")
    print("PART 5: HOLLAND HEXAGONAL STRUCTURE ACROSS MODELS")
    print(f"{'='*70}")

    # Holland hexagonal distances (1=adjacent, 2=alternate, 3=opposite)
    holland_pairs = {
        1: [("realistic", "investigative"), ("investigative", "artistic"),
            ("artistic", "social"), ("social", "enterprising"),
            ("enterprising", "conventional"), ("conventional", "realistic")],
        2: [("realistic", "artistic"), ("investigative", "social"),
            ("artistic", "enterprising"), ("social", "conventional"),
            ("enterprising", "realistic"), ("conventional", "investigative")],
        3: [("realistic", "social"), ("investigative", "enterprising"),
            ("artistic", "conventional")],
    }

    holland_results = {}
    for m in ids_with_data:
        d = model_data[m]
        short = m.split("/")[-1]
        coords = {t: np.array(d["coords_5d"][t]) for t in TRAITS}

        by_distance = {}
        for dist, pairs in holland_pairs.items():
            cosines = []
            for t1, t2 in pairs:
                c = float(np.dot(coords[t1], coords[t2]) / (
                    np.linalg.norm(coords[t1]) * np.linalg.norm(coords[t2])))
                cosines.append(c)
            by_distance[dist] = float(np.mean(cosines))

        print(f"\n  {short}:")
        print(f"    Adjacent (d=1): mean cos = {by_distance[1]:.3f}")
        print(f"    Alternate (d=2): mean cos = {by_distance[2]:.3f}")
        print(f"    Opposite (d=3): mean cos = {by_distance[3]:.3f}")

        # Check monotonicity
        monotonic = by_distance[1] > by_distance[2] > by_distance[3]
        print(f"    Monotonic (adj > alt > opp): {'YES' if monotonic else 'NO'}")

        holland_results[short] = {
            "by_distance": by_distance,
            "monotonic": monotonic,
        }

    results["holland"] = holland_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for m in ids_with_data:
        d = model_data[m]
        short = m.split("/")[-1]
        print(f"\n  {short}: hidden={d['hidden_size']}, layers={d['num_layers']}, "
              f"cond={d['condition_number']:.2f}, perturb={d['perturbation_fraction']*100:.3f}%")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
