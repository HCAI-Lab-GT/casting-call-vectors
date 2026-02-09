#!/usr/bin/env python
"""
Session 8 Summary: Personality Algebra, Extrapolation, and Deployment Readiness.

Synthesizes all experiments from this session.
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
    print(f"SESSION 8 SUMMARY: PERSONALITY ALGEBRA & DEPLOYMENT READINESS")
    print(f"{'='*70}")

    # 1. Personality Extrapolation
    ex = safe_load("personality_extrapolation.json")
    if ex:
        print(f"\n--- Personality Extrapolation (Marin 8B, 0.5×-3×) ---")
        for trait in ["artistic", "investigative", "social"]:
            if trait in ex:
                lin = ex[trait].get("linearity", {})
                amps = sorted([k for k in ex[trait] if k != "linearity"], key=float)
                correct = sum(1 for a in amps if ex[trait][a].get("is_target_top", False))
                print(f"  {trait:>15}: r={lin.get('r', 0):.3f}, correct top={correct}/{len(amps)}")

    # 2. Vector Arithmetic
    ar = safe_load("personality_arithmetic.json")
    if ar and "summary" in ar:
        s = ar["summary"]
        print(f"\n--- Vector Arithmetic (5D) ---")
        print(f"  Subtraction (A-B): {s['subtraction_correct']} correct direction")
        print(f"  Centroid deviation: {s['centroid_deviation_correct']} correct top")
        print(f"  Holland cancellation: mean max|δ| = {s['mean_cancellation_delta']:.3f}")
        print(f"  Triple average: {s['mean_triple_top3']:.1f}/3.0 targets in top 3")
        print(f"  Double subtraction: {s['double_subtraction_correct']} all correct")

    # 3. Multi-Layer Steering
    ml = safe_load("multi_layer_steering.json")
    if ml:
        print(f"\n--- Multi-Layer Steering (Marin 8B) ---")
        for name in ["single_mid", "triple_tight", "triple_spread", "distributed",
                      "early_only", "late_only", "additive_triple"]:
            if name in ml:
                r = ml[name]
                print(f"  {r['description']:>45}: {r['accuracy']:.0%}, Δ={r['mean_delta']:+.3f}")

    # 4. Capability Preservation
    cap = safe_load("capability_allpos.json")
    if cap:
        print(f"\n--- Capability Preservation (All-Position Steering) ---")
        print(f"  Baseline PPL: {cap['baseline']['mean_ppl']:.2f}")
        for alpha in ["0.5", "1.0", "2.0", "3.0", "5.0"]:
            if alpha in cap:
                ap = cap[alpha].get("allpos", {})
                lp = cap[alpha].get("lastpos", {})
                print(f"  α={alpha}: lastpos PPL={lp.get('ppl_ratio', 0):.2f}×, "
                      f"allpos PPL={ap.get('ppl_ratio', 0):.2f}×, "
                      f"allpos QA={ap.get('mean_qa', 0):.0%}")

    # 5. Transfer Interference
    ti = safe_load("transfer_interference.json")
    if ti:
        print(f"\n--- Transfer Interference (SmolLM3→Marin) ---")
        print(f"  Full matrix: r = {ti['full_pearson_r']:.3f}")
        print(f"  Off-diagonal: r = {ti['offdiag_pearson_r']:.3f}")
        print(f"  Spearman ρ = {ti['offdiag_spearman_rho']:.3f}")

    # 6. Isotropic Extrapolation
    iso = safe_load("isotropic_extrapolation.json")
    if iso:
        print(f"\n--- Isotropic Extrapolation (Whitening Test) ---")
        for trait in ["artistic", "social"]:
            if trait in iso:
                orig_lin = iso[trait].get("original_linearity", {})
                iso_lin = iso[trait].get("isotropic_linearity", {})
                print(f"  {trait}: original r={orig_lin.get('r', 0):.3f}, "
                      f"isotropic r={iso_lin.get('r', 0):.3f}")

    # 7. Norm Calibration
    nc = safe_load("norm_calibrated_steering.json")
    if nc and "analysis" in nc:
        a = nc["analysis"]
        print(f"\n--- Norm-Calibrated Steering ---")
        print(f"  Raw delta std: {a['raw_delta_std']:.3f}")
        print(f"  Calibrated delta std: {a['calibrated_delta_std']:.3f}")
        print(f"  Norm→Delta: r={a['norm_delta_pearson']['r']:.3f} (p={a['norm_delta_pearson']['p']:.3f})")

    # TOP 5
    print(f"\n{'='*70}")
    print(f"TOP 5 FINDINGS FROM SESSION 8")
    print(f"{'='*70}")
    print(f"""
  1. PERSONALITY STEERING IS PRACTICALLY FREE
     All-position, α=1: 2% PPL increase, 100% QA. α=3: 21% PPL, 100% QA.
     Only α=5 degrades capability (92% PPL, 88% QA).
     Last-position steering: literally zero cost at any alpha.

  2. FULL BEHAVIORAL PROFILE TRANSFERS (r=0.945)
     Not just the diagonal: the ENTIRE 6×6 interference matrix transfers
     from SmolLM3→Marin with off-diagonal r=0.945. All 6 traits preserve
     their top and bottom interference targets. Holland hex structure identical.

  3. VECTOR ARITHMETIC WORKS IN 5D
     Subtraction: 5/5 correct. Centroid deviation: 6/6 correct top trait.
     Holland opposite cancellation: artistic+conventional = near-zero (0.068).
     Triple averages: 2.7/3.0 targets in top 3.
     The 5D personality subspace has genuine algebraic structure.

  4. EXTRAPOLATION IS LINEAR FOR DOMINANT-PC TRAITS
     Artistic (r=0.987) and investigative (r=0.987) scale linearly up to 3×.
     Social fails at 2× (r=0.284) because PC1 effect dominates PC3 identity.
     Isotropic whitening doesn't fix it — the non-linearity is MODEL-intrinsic.
     General rule: extrapolation robustness = variance fraction on high-gain PCs.

  5. SINGLE MID-LAYER IS OPTIMAL
     Multi-layer injection cannot improve on single L16: 97% for both.
     Spreading to wider ranges degrades (93%). Additive multi-layer (5× alpha)
     only reaches 90%. Early (L8) = chance, Late (L24) = 60%.
""")

    # Count artifacts
    scripts = sorted((_repo_root() / "scripts/marin").glob("*.py"))
    outputs = sorted((_repo_root() / "outputs/analysis").glob("*.json"))
    print(f"  Total scripts: {len(scripts)}")
    print(f"  Total JSON outputs: {len(outputs)}")


if __name__ == "__main__":
    main()
