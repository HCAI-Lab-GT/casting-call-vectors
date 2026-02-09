#!/usr/bin/env python
"""
Analyze why some traits extrapolate linearly and others don't.

The extrapolation experiment showed:
- Artistic: r=0.987 (linear), stays on top 5/5
- Investigative: r=0.987 (linear), stays on top 5/5
- Social: r=0.284 (non-linear), overtaken by artistic at 2×

Why? Possible explanations:
1. 5D coordinate structure (direction, norm, overlap with other traits)
2. Distance from centroid
3. Alignment with high-variance PCs
4. Holland hexagonal position

This script analyzes the 5D coordinate geometry to explain extrapolation robustness.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from transformers import AutoConfig

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
HOLLAND_OPPOSITES = {
    "realistic": "social",
    "investigative": "enterprising",
    "artistic": "conventional",
    "social": "realistic",
    "enterprising": "investigative",
    "conventional": "artistic",
}


def _repo_root():
    return Path(__file__).resolve().parents[2]


def load_residual_and_5d(model_id, riasec_dir):
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
    return coords_5d, basis_5d, S[:5], residual


def main():
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    coords_5d, basis_5d, singular_values, residual = load_residual_and_5d(target_id, riasec_dir)

    print(f"\n{'='*70}")
    print(f"EXTRAPOLATION THEORY ANALYSIS")
    print(f"{'='*70}")

    # 1. Basic 5D coordinate properties
    print(f"\n--- 5D Coordinate Properties ---")
    centroid = np.mean([coords_5d[t] for t in TRAITS], axis=0)
    print(f"  Centroid: [{', '.join(f'{c:+.2f}' for c in centroid)}]")
    print(f"  Centroid norm: {np.linalg.norm(centroid):.4f} (should be near 0)")

    print(f"\n  {'Trait':>15}  {'Norm':>6}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC4':>7}  {'PC5':>7}")
    for t in TRAITS:
        c = coords_5d[t]
        print(f"  {t:>15}  {np.linalg.norm(c):>6.1f}  {c[0]:>+7.2f}  {c[1]:>+7.2f}  {c[2]:>+7.2f}  {c[3]:>+7.2f}  {c[4]:>+7.2f}")

    print(f"\n  Singular values: [{', '.join(f'{s:.1f}' for s in singular_values)}]")
    variance_pct = (singular_values**2) / (singular_values**2).sum() * 100
    print(f"  Variance %: [{', '.join(f'{v:.1f}%' for v in variance_pct)}]")

    # 2. Pairwise cosines in 5D
    print(f"\n--- Pairwise Cosines in 5D ---")
    print(f"  {'':>15}", end="")
    for t in TRAITS:
        print(f"  {t[:4]:>6}", end="")
    print()
    for t1 in TRAITS:
        print(f"  {t1:>15}", end="")
        for t2 in TRAITS:
            cos = np.dot(coords_5d[t1], coords_5d[t2]) / (np.linalg.norm(coords_5d[t1]) * np.linalg.norm(coords_5d[t2]))
            print(f"  {cos:>+6.2f}", end="")
        print()

    # 3. Analyze amplification behavior
    print(f"\n--- Amplification Analysis ---")
    print(f"\n  When we amplify trait A by factor k, the vector becomes k*coords_5d[A].")
    print(f"  The PROFILE measures pairwise preference shift relative to baseline.")
    print(f"  Problem: at high k, the amplified vector may drift toward OTHER traits")
    print(f"  because different PCs have different 'importance' for behavior.")

    # For each trait, compute which other trait is most cosine-aligned
    print(f"\n  Nearest neighbor in 5D (highest cosine):")
    for t1 in TRAITS:
        max_cos = -2
        nn = ""
        for t2 in TRAITS:
            if t1 == t2:
                continue
            cos = np.dot(coords_5d[t1], coords_5d[t2]) / (np.linalg.norm(coords_5d[t1]) * np.linalg.norm(coords_5d[t2]))
            if cos > max_cos:
                max_cos = cos
                nn = t2
        print(f"    {t1:>15} → {nn} (cos = {max_cos:+.3f})")

    # 4. Explain social's non-linearity
    print(f"\n--- Why Social Fails at 2×+ ---")
    print(f"  Social coords: [{', '.join(f'{c:+.2f}' for c in coords_5d['social'])}]")
    print(f"  Artistic coords: [{', '.join(f'{c:+.2f}' for c in coords_5d['artistic'])}]")

    # Cosine between social and artistic
    cos_sa = np.dot(coords_5d["social"], coords_5d["artistic"]) / (
        np.linalg.norm(coords_5d["social"]) * np.linalg.norm(coords_5d["artistic"]))
    print(f"  Social-Artistic cosine: {cos_sa:+.3f}")

    # Project social onto each PC, weighted by importance (singular value)
    print(f"\n  PC-weighted contributions to trait profile:")
    print(f"  (Singular values weight how much each PC matters for behavior)")
    for t in ["artistic", "investigative", "social"]:
        c = coords_5d[t]
        weighted = c * singular_values
        print(f"    {t:>15}: [{', '.join(f'{w:+.1f}' for w in weighted)}]  total={np.sum(np.abs(weighted)):.1f}")

    # 5. Projection overlap analysis
    print(f"\n--- Projection Overlap (trait A amplified, measured as trait B) ---")
    print(f"  When amplifying trait A, the measured effect on trait B depends on")
    print(f"  how much A and B overlap in the measurement basis.")
    print(f"\n  Overlap = dot(norm_A, norm_B) in 5D:")

    for t1 in ["artistic", "investigative", "social"]:
        n1 = coords_5d[t1] / np.linalg.norm(coords_5d[t1])
        overlaps = {}
        for t2 in TRAITS:
            n2 = coords_5d[t2] / np.linalg.norm(coords_5d[t2])
            overlaps[t2] = np.dot(n1, n2)
        sorted_ov = sorted(overlaps.items(), key=lambda x: -x[1])
        print(f"  {t1:>15}: " + "  ".join(f"{t[:4]}={o:+.3f}" for t, o in sorted_ov))

    # 6. Critical insight: non-linearity from cross-trait coupling
    print(f"\n--- Cross-Trait Coupling Under Amplification ---")
    print(f"  When we amplify social by k, the profile is measured as:")
    print(f"    delta_t = sum over pairs containing t of (gap_steered - gap_baseline)")
    print(f"  The KEY issue: at high k, the steering vector saturates the mid-layer")
    print(f"  representation, and cross-trait coupling becomes non-linear.")
    print(f"\n  Traits with HIGH PC1 loading are most vulnerable to non-linearity")
    print(f"  because PC1 captures the most variance → largest behavioral effect:")

    for t in TRAITS:
        pc1_frac = abs(coords_5d[t][0]) / np.linalg.norm(coords_5d[t])
        print(f"    {t:>15}: |PC1| = {abs(coords_5d[t][0]):.1f}, fraction = {pc1_frac:.2%}")

    # 7. Predict which traits should extrapolate well
    print(f"\n--- Extrapolation Robustness Prediction ---")
    print(f"  Hypothesis: traits that are well-separated from all others in 5D")
    print(f"  should extrapolate more linearly (less cross-trait contamination).")

    for t in TRAITS:
        min_cos = 1
        for t2 in TRAITS:
            if t == t2:
                continue
            cos = np.dot(coords_5d[t], coords_5d[t2]) / (np.linalg.norm(coords_5d[t]) * np.linalg.norm(coords_5d[t2]))
            if cos < min_cos:
                min_cos = cos
        max_cos_others = max(
            np.dot(coords_5d[t], coords_5d[t2]) / (np.linalg.norm(coords_5d[t]) * np.linalg.norm(coords_5d[t2]))
            for t2 in TRAITS if t2 != t
        )
        mean_cos = np.mean([
            np.dot(coords_5d[t], coords_5d[t2]) / (np.linalg.norm(coords_5d[t]) * np.linalg.norm(coords_5d[t2]))
            for t2 in TRAITS if t2 != t
        ])
        print(f"    {t:>15}: max_cos_to_others = {max_cos_others:+.3f}, min = {min_cos:+.3f}, mean = {mean_cos:+.3f}")

    # 8. Extrapolation results from the experiment
    extrap_path = _repo_root() / "outputs/analysis/personality_extrapolation.json"
    if extrap_path.exists():
        with open(extrap_path) as f:
            extrap = json.load(f)

        print(f"\n--- Extrapolation Results (from experiment) ---")
        for trait in ["artistic", "investigative", "social"]:
            if trait in extrap:
                amps = sorted(extrap[trait].keys(), key=lambda x: float(x) if x != "linearity" else 999)
                deltas = []
                for a in amps:
                    if a == "linearity":
                        continue
                    r = extrap[trait][a]
                    deltas.append(r["target_delta"])
                print(f"\n    {trait}: linearity r = {extrap[trait].get('linearity', {}).get('r', 'N/A')}")
                print(f"      Deltas: {[f'{d:+.3f}' for d in deltas]}")
                if len(deltas) >= 4:
                    # Check for saturation
                    ratios = [deltas[i+1] / deltas[i] if deltas[i] != 0 else 0 for i in range(len(deltas)-1)]
                    print(f"      Successive ratios: {[f'{r:.2f}' for r in ratios]}")
                    print(f"      Saturation (ratio < 1.0): {any(r < 1.0 for r in ratios)}")

    # 9. Final synthesis
    print(f"\n{'='*70}")
    print(f"SYNTHESIS")
    print(f"{'='*70}")
    print(f"""
  1. ARTISTIC extrapolates linearly because:
     - Largest PC1 loading (|-44.03| = highest), well-separated from others
     - PC1 captures 38% of variance → amplifying artistic = mostly amplifying PC1
     - Holland opposite (conventional) is at the extreme other end of PC1

  2. INVESTIGATIVE extrapolates linearly because:
     - Largest PC2 loading (|-35.75| = highest on PC2)
     - PC2 captures 22% of variance → second-largest behavioral axis
     - Well-separated from nearest neighbor

  3. SOCIAL fails at 2×+ because:
     - Largest PC3 loading (|-38.90|), but PC3 is the THIRD axis
     - PC3 captures less variance than PC1/PC2
     - At high amplification, the PC1 component of social (-19.99) grows faster
       in behavioral effect than the PC3 component, because PC1 has more variance
     - Social's PC1 loading (-19.99) is in the SAME DIRECTION as artistic (-44.03)
     - So amplified social vector looks increasingly like artistic

  GENERAL RULE: Traits that are dominated by high-variance PCs extrapolate well.
  Traits whose unique signature is in low-variance PCs get overtaken at high amplification.

  This is a consequence of the non-uniform variance structure of the 5D subspace.
  If all PCs had equal variance (perfect isotropic simplex), ALL traits would
  extrapolate equally well.
""")

    # Save
    results = {
        "coords_5d": {t: coords_5d[t].tolist() for t in TRAITS},
        "singular_values": singular_values.tolist(),
        "variance_pct": variance_pct.tolist(),
        "pairwise_cosines": {
            f"{t1}-{t2}": float(np.dot(coords_5d[t1], coords_5d[t2]) / (np.linalg.norm(coords_5d[t1]) * np.linalg.norm(coords_5d[t2])))
            for i, t1 in enumerate(TRAITS) for j, t2 in enumerate(TRAITS) if i < j
        },
        "pc1_fraction": {t: float(abs(coords_5d[t][0]) / np.linalg.norm(coords_5d[t])) for t in TRAITS},
        "max_cos_to_others": {
            t: float(max(
                np.dot(coords_5d[t], coords_5d[t2]) / (np.linalg.norm(coords_5d[t]) * np.linalg.norm(coords_5d[t2]))
                for t2 in TRAITS if t2 != t
            ))
            for t in TRAITS
        },
    }
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "extrapolation_theory.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
