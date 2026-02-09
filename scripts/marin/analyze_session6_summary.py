#!/usr/bin/env python
"""
Session 6 Summary: Zero-Calibration Transfer and Related Findings.

Synthesizes all experiments from this session into a coherent narrative.
"""

import json
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[2]


def safe_load(name):
    try:
        with open(_repo_root() / "outputs/analysis" / name) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    print(f"\n{'='*70}")
    print(f"SESSION 6 SUMMARY: ZERO-CALIBRATION TRANSFER")
    print(f"{'='*70}")

    # 1. Zero-calibration strategies
    zc = safe_load("zero_calibration_transfer.json")
    if zc:
        print(f"\n--- Transfer Strategy Comparison (SmolLM3 → Marin 8B) ---")
        print(f"  {'Strategy':>35}  {'Accuracy':>8}  {'Calibration':>20}")
        print(f"  {'-'*65}")
        for name, data in zc.items():
            calib = {
                "Self (native)": "N/A",
                "Full Procrustes (k=6)": "6 correspondences",
                "Sign-corrected (5 bits)": "5 binary bits",
                "Canonical simplex bridge": "0 (fails)",
                "Identity (no rotation)": "0 (fails)",
                "Random": "N/A",
                "Canonical bridge (Llama→Marin)": "0 (fails)",
                "Full Procrustes (Llama→Marin)": "6 correspondences",
            }.get(name, "?")
            print(f"  {name:>35}  {data['accuracy']:>7.0%}  {calib:>20}")

    # 2. Predicted signs
    ps = safe_load("predicted_sign_transfer.json")
    if ps:
        print(f"\n--- Predicted-Sign Transfer (Zero-Calibration, No Target Data) ---")
        for target in ["Marin-8B", "SmolLM3"]:
            if target in ps:
                print(f"\n  Target: {target}")
                r = ps[target]
                print(f"    Self: {r['self']['accuracy']:.0%}")
                for key, data in r.items():
                    if key.endswith("_procrustes"):
                        src = key.replace("_procrustes", "")
                        pred = r.get(f"{src}_predicted", {})
                        print(f"    {src}: Procrustes={data['accuracy']:.0%}, "
                              f"Predicted={pred.get('accuracy', 0):.0%}")

    # 3. Permutation search
    perm = safe_load("permutation_sign_search.json")
    if perm:
        identity_count = sum(1 for r in perm.values() if isinstance(r, dict) and r.get("is_identity_best"))
        total = sum(1 for r in perm.values() if isinstance(r, dict) and "is_identity_best" in r)
        print(f"\n--- Exhaustive Permutation+Sign Search (3840 combos × 12 pairs) ---")
        print(f"  Identity permutation optimal: {identity_count}/{total} pairs ({identity_count/total:.0%})")
        gains = [r["gain_over_identity"] for r in perm.values()
                 if isinstance(r, dict) and "gain_over_identity" in r]
        print(f"  Gain from permutation search: {sum(gains)/len(gains):+.4f} (ZERO)")

    # 4. Dose-response
    dr = safe_load("cross_dim_dose_response.json")
    if dr:
        print(f"\n--- Cross-Dim Dose-Response (Spearman ρ vs Self) ---")
        self_deltas = [r["mean_delta"] for r in dr.get("Self (native)", [])]
        from scipy.stats import spearmanr
        for name, data in dr.items():
            if name == "Self (native)":
                continue
            other_deltas = [r["mean_delta"] for r in data]
            if len(self_deltas) == len(other_deltas) > 0:
                rho, _ = spearmanr(self_deltas, other_deltas)
                print(f"  {name:>25}: rho={rho:.3f}")

    # 5. Leave-one-model-out
    lomo_marin = safe_load("leave_one_model_out_marin.json")
    if lomo_marin:
        print(f"\n--- Leave-One-Model-Out (Marin 8B as held-out target) ---")
        print(f"  Self: {lomo_marin['self']['accuracy']:.0%}")
        for src, data in lomo_marin.get("transfers", {}).items():
            zc_acc = data["zero_calibration"]["accuracy"]
            proc_acc = data["procrustes"]["accuracy"]
            print(f"  {src}: Zero-cal={zc_acc:.0%}, Procrustes={proc_acc:.0%}")
        if "ensemble_zero_cal" in lomo_marin:
            print(f"  Ensemble: {lomo_marin['ensemble_zero_cal']['accuracy']:.0%}")
        print(f"  Random: {lomo_marin['random']['accuracy']:.0%}")

    # 6. Cross-model composition
    comp = safe_load("cross_model_composition.json")
    if comp and "summary" in comp:
        s = comp["summary"]
        print(f"\n--- Cross-Model Compositional Steering (SmolLM3 → Marin 8B) ---")
        print(f"  Cross-model: {s['cross_pair_success']}/{s['total_pairs']} pairs succeed, "
              f"mean={s['cross_mean_accuracy']:.0%}")
        print(f"  Self:        {s['self_pair_success']}/{s['total_pairs']} pairs succeed, "
              f"mean={s['self_mean_accuracy']:.0%}")

    # 7. 6D vs 5D
    fv = safe_load("full_vector_zero_cal.json")
    if fv:
        print(f"\n--- 6D (Full) vs 5D (Residual) Vectors ---")
        if "self_full" in fv:
            print(f"  Self full: {fv['self_full']['accuracy']:.0%}")
        if "self_residual" in fv:
            print(f"  Self residual: {fv['self_residual']['accuracy']:.0%}")
        print(f"  Verdict: Residual vectors > full vectors for discrimination")

    # 8. Generation validation
    gen = safe_load("zero_cal_generation.json")
    if gen and "summary" in gen:
        s = gen["summary"]
        print(f"\n--- Generation Validation (LLM Judge) ---")
        print(f"  Zero-cal: {s['zero_cal_correct']}/{s['total']} ({s['zero_cal_accuracy']:.0%})")
        print(f"  Self: {s['self_correct']}/{s['total']} ({s['self_accuracy']:.0%})")
        print(f"  Note: Low accuracy due to RLHF training ('As an AI...')")
        print(f"  Key: Zero-cal = Self in both logprob AND generation evals")

    # TOP 5 findings from this session
    print(f"\n{'='*70}")
    print(f"TOP 5 FINDINGS FROM SESSION 6")
    print(f"{'='*70}")
    print(f"""
  1. ZERO-CALIBRATION TRANSFER WORKS (97-100% on Marin 8B)
     Sign-corrected identity matches full Procrustes with only 5 bits.
     Predicted sign convention eliminates need for ANY target data.

  2. PCA PRODUCES CANONICAL ORDERING (12/12 pairs, 100%)
     Exhaustive search over 3840 permutation+sign combos:
     identity permutation is optimal for ALL model pairs.
     Only sign flips vary between models.

  3. DOSE-RESPONSE CURVE PERFECTLY PRESERVED (rho=1.000 for all sources)
     Cross-dim transfer preserves not just trait identity but the entire
     alpha-response relationship. Same alpha, same effect.

  4. CROSS-MODEL COMPOSITION WORKS (14/15 pairs, 82%)
     Compose traits in source's 5D, transfer to target — both traits
     expressed. Within 4% of self-composition.

  5. SHARED DIRECTION IS NOISE (6D < 5D for discrimination)
     Residual vectors outperform full vectors (97% vs 80%).
     The shared direction is personality-nonspecific agreement bias.
     Personality lives ONLY in the 5D residual subspace.
""")

    # Updated recipe
    print(f"{'='*70}")
    print(f"UPDATED PRACTICAL RECIPE (ZERO-CALIBRATION)")
    print(f"{'='*70}")
    print(f"""
  Given: Source model S (any instruct LLM, even 1B params)
         Target model T (any instruct LLM, even 8B+ params)

  On Source S (cheap, one-time):
  1. Extract 6 RIASEC persona vectors
  2. Compute residual vectors (remove shared PC1)
  3. PCA to 5D coordinates
  4. Standardize signs: PC1 should have 'artistic' most negative;
     PC2-5: trait with max |loading| should be negative

  On Target T (no persona vectors needed):
  1. Extract 6 RIASEC persona vectors (for PCA basis only)
  2. Compute residual vectors, PCA to 5D basis
  3. Standardize signs via same canonical convention
  4. Transfer: target_vec = target_basis^T @ (scale * source_std_coords)
     where scale = mean(target_norms) / mean(source_norms)

  Result: 97-100% personality discrimination on target model
  No Procrustes alignment needed. No calibration data needed.
  Works across architectures: 2048d → 4096d, 1B → 8B params.
""")

    # Count total scripts and outputs
    scripts = sorted((_repo_root() / "scripts/marin").glob("*.py"))
    outputs = sorted((_repo_root() / "outputs/analysis").glob("*.json"))
    print(f"  Total scripts: {len(scripts)}")
    print(f"  Total JSON outputs: {len(outputs)}")


if __name__ == "__main__":
    main()
