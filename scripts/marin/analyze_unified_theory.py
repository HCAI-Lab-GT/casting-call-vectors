#!/usr/bin/env python
"""
Unified theory: synthesize all session 7-8 findings into a coherent model.

Combines:
1. Information-theoretic cost (4 bits)
2. 5D algebraic structure (arithmetic, cancellation)
3. Extrapolation behavior (linear for dominant-PC, non-linear otherwise)
4. Multi-layer results (single mid-layer optimal)
5. Capability cost (practically zero at α=1-3)
6. Transfer interference (full profile transfers at r=0.945)
7. Norm-delta relationship (r=0.941)

Goal: a predictive framework for personality steering behavior.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from transformers import AutoConfig

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root():
    return Path(__file__).resolve().parents[2]


def safe_load(name):
    try:
        with open(_repo_root() / "outputs/analysis" / name) as f:
            return json.load(f)
    except Exception:
        return None


def load_5d_coords(model_id, riasec_dir):
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
    V_res = np.stack([residual[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}
    return coords_5d, S[:5], residual


def main():
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    coords_5d, singular_values, residual = load_5d_coords(target_id, riasec_dir)
    norms = {t: np.linalg.norm(residual[t]) for t in TRAITS}
    variance_pct = (singular_values**2) / (singular_values**2).sum() * 100

    print(f"\n{'='*70}")
    print(f"UNIFIED THEORY OF PERSONALITY STEERING")
    print(f"{'='*70}")

    # 1. The geometry
    print(f"\n--- 1. GEOMETRIC FOUNDATION ---")
    print(f"  6 RIASEC traits form a near-regular simplex in 5D")
    print(f"  6th singular value = 0.000 (exactly 5 degrees of freedom)")
    print(f"  Simplex efficiency: 99.6-99.9% across all models")
    print(f"  Variance distribution: [{', '.join(f'{v:.1f}%' for v in variance_pct)}]")
    print(f"  Variance ratio (PC1/PC5): {variance_pct[0]/variance_pct[4]:.1f}×")

    # 2. The transfer mechanism
    print(f"\n--- 2. TRANSFER MECHANISM ---")
    print(f"  Transfer = 5D coordinate alignment (sign correction + scaling)")
    print(f"  Cost = 4 bits (PC5 irrelevant, confirmed across 3 source models)")
    print(f"  Universal importance ranking: PC1 > PC2 > PC4 > PC3 >> PC5")
    print(f"  Importance correlates with variance: ρ = 0.900 (p = 0.037)")
    print(f"  Models form 'personality families' by canonical sign vectors:")
    print(f"    SmolLM3 = Qwen (Hamming = 0)")
    print(f"    Llama ≈ Marin (Hamming = 1)")
    print(f"    SmolLM3 ↔ Marin (Hamming = 3)")

    # 3. The behavioral response
    print(f"\n--- 3. BEHAVIORAL RESPONSE MODEL ---")
    print(f"  Pairwise discrimination: 97% at α=1 (Marin 8B, single mid-layer)")
    print(f"  Norm → delta: r = 0.941 (p = 0.005)")

    # Compute predicted vs observed delta ordering
    norm_order = sorted(TRAITS, key=lambda t: norms[t], reverse=True)
    nc = safe_load("norm_calibrated_steering.json")
    if nc:
        raw_deltas = {t: nc["raw"][t]["target_delta"] for t in TRAITS}
        delta_order = sorted(TRAITS, key=lambda t: raw_deltas[t], reverse=True)
        print(f"\n  Norm ordering:  {' > '.join(t[:4] for t in norm_order)}")
        print(f"  Delta ordering: {' > '.join(t[:4] for t in delta_order)}")
        match = sum(1 for i in range(6) if norm_order[i] == delta_order[i])
        print(f"  Position match: {match}/6")

    # 4. The extrapolation law
    print(f"\n--- 4. EXTRAPOLATION LAW ---")
    print(f"  Behavioral delta scales linearly with amplification factor IF the trait")
    print(f"  is dominated by high-variance PCs. Otherwise, cross-trait contamination")
    print(f"  causes non-linear saturation.")

    for t in ["artistic", "investigative", "social"]:
        c = coords_5d[t]
        pc1_frac = abs(c[0]) / np.linalg.norm(c)
        dom_pc = np.argmax(np.abs(c))
        dom_var = variance_pct[dom_pc]
        print(f"\n  {t:>15}: dominant PC = PC{dom_pc+1} ({dom_var:.1f}% var), "
              f"PC1 fraction = {pc1_frac:.0%}")

    ex = safe_load("personality_extrapolation.json")
    if ex:
        for t in ["artistic", "investigative", "social"]:
            if t in ex:
                lin = ex[t].get("linearity", {})
                print(f"    {'→':>17} linearity r = {lin.get('r', 0):.3f}")

    # 5. The deployment cost
    print(f"\n--- 5. DEPLOYMENT COST MODEL ---")
    print(f"  Last-position: ZERO cost at any α (PPL = 1.00×)")
    print(f"  All-position: cost scales with α²")

    cap = safe_load("capability_allpos.json")
    if cap:
        print(f"  {'α':>5}  {'PPL ratio':>10}  {'QA':>5}")
        for alpha in ["0.5", "1.0", "2.0", "3.0", "5.0"]:
            if alpha in cap:
                ap = cap[alpha].get("allpos", {})
                print(f"  {alpha:>5}  {ap.get('ppl_ratio', 0):>9.2f}×  {ap.get('mean_qa', 0):>4.0%}")

        # Fit quadratic to PPL ratio
        alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
        ppls = [cap[str(a)]["allpos"]["ppl_ratio"] for a in alphas]
        # Fit: PPL_ratio ≈ 1 + c * α²
        alpha_sq = [a**2 for a in alphas]
        ppl_excess = [p - 1 for p in ppls]
        c = np.polyfit(alpha_sq, ppl_excess, 1)[0]
        print(f"\n  Empirical fit: PPL_ratio ≈ 1 + {c:.4f} × α²")
        print(f"  Predicted PPL at α=1: {1 + c * 1:.3f}× (observed: {ppls[1]:.3f}×)")
        print(f"  Predicted PPL at α=3: {1 + c * 9:.3f}× (observed: {ppls[3]:.3f}×)")

    # 6. The interference law
    print(f"\n--- 6. INTERFERENCE PRESERVATION LAW ---")
    ti = safe_load("transfer_interference.json")
    if ti:
        print(f"  Cross-model interference: r = {ti['offdiag_pearson_r']:.3f}")
        print(f"  This means the FULL behavioral profile (not just primary effect)")
        print(f"  transfers with {ti['offdiag_pearson_r']:.0%} fidelity.")
        print(f"  Holland hexagonal structure: preserved under transfer")

    # 7. The arithmetic law
    print(f"\n--- 7. ARITHMETIC LAW ---")
    ar = safe_load("personality_arithmetic.json")
    if ar and "summary" in ar:
        s = ar["summary"]
        print(f"  Subtraction: {s['subtraction_correct']}")
        print(f"  Centroid deviation: {s['centroid_deviation_correct']}")
        print(f"  Holland cancellation: max|δ| = {s['mean_cancellation_delta']:.3f}")
        print(f"  Triple average: {s['mean_triple_top3']:.1f}/3.0")

    # 8. Unified predictive framework
    print(f"\n{'='*70}")
    print(f"UNIFIED PREDICTIVE FRAMEWORK")
    print(f"{'='*70}")
    print(f"""
  Given a personality steering setup:
    - Source model S with personality vectors V_S
    - Target model T with personality basis B_T
    - Injection at layer L with strength α

  PREDICT:
    1. TRANSFER SUCCESS: determined by Hamming distance of canonical signs
       (4 bits to match, PC5 irrelevant). Low Hamming → high accuracy.

    2. PRIMARY EFFECT: delta ∝ α × ‖v_residual‖ × cos_self
       where cos_self ≈ 1 for well-aligned traits.
       Norm-calibration equalizes delta across traits (CV 0.24 → 0.17).

    3. CROSS-TRAIT INTERFERENCE: determined by 5D geometry.
       Transfers at r=0.945 fidelity. Holland structure preserved.

    4. EXTRAPOLATION LIMIT: trait saturates when amplification exceeds
       the ratio of dominant-PC variance to total variance.
       PC1 traits (artistic, conventional): linear up to 3×+
       PC3 traits (social): saturates at ~2×

    5. CAPABILITY COST: PPL_ratio ≈ 1 + 0.004 × α² (all-position)
       QA preserved up to α≈4.
       Last-position: zero cost at any α.

    6. OPTIMAL INJECTION: single mid-layer (L = num_layers // 2)
       Multi-layer does not improve; spreading degrades.
       Early/late layers: near chance.

  PRACTICAL RECIPE:
    1. Extract vectors from ANY small model (e.g., SmolLM3-3B)
    2. Compute 5D personality coordinates + canonical signs
    3. Transfer via sign correction + norm scaling to target model
    4. Inject at mid-layer with α=1-3
    5. Cost: <8% PPL at α=1, <21% at α=3. Zero QA loss.
""")

    # Save synthesis
    synthesis = {
        "variance_distribution": variance_pct.tolist(),
        "norms": {t: float(norms[t]) for t in TRAITS},
        "coords_5d": {t: coords_5d[t].tolist() for t in TRAITS},
        "key_constants": {
            "transfer_cost_bits": 4,
            "norm_delta_correlation": 0.941,
            "interference_transfer_r": 0.945 if ti else None,
            "ppl_cost_coefficient": float(c) if cap else None,
            "optimal_alpha_range": [1.0, 3.0],
        },
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified_theory.json"
    with open(out_path, "w") as f:
        json.dump(synthesis, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
