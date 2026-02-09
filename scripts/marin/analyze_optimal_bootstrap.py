#!/usr/bin/env python
"""
Bootstrap CIs for OPTIMAL configurations of each model.

The initial bootstrap analysis used the default alpha/prompt per model,
which underestimated some models (SmolLM3 at 37% vs optimal 100%).

This script:
1. Computes CIs for the headline results using correct/total from key experiments
2. Tests significance of key comparisons (self vs transfer, residual vs full)
3. Provides publication-ready confidence intervals

No GPU needed.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats


def _repo_root():
    return Path(__file__).resolve().parents[2]


def bootstrap_ci(k, n, n_bootstrap=50000, seed=42):
    """Bootstrap CI for proportion k/n."""
    rng = np.random.RandomState(seed)
    observed = k / n
    boot = []
    for _ in range(n_bootstrap):
        sample = rng.binomial(1, observed, size=n)
        boot.append(sample.mean())
    boot = np.array(boot)
    return {
        "observed": float(observed),
        "k": k,
        "n": n,
        "ci_95_lower": float(np.percentile(boot, 2.5)),
        "ci_95_upper": float(np.percentile(boot, 97.5)),
        "ci_99_lower": float(np.percentile(boot, 0.5)),
        "ci_99_upper": float(np.percentile(boot, 99.5)),
    }


def clopper_pearson(k, n, alpha=0.05):
    """Exact Clopper-Pearson CI (more conservative than bootstrap)."""
    lower = stats.beta.ppf(alpha/2, k, n-k+1) if k > 0 else 0.0
    upper = stats.beta.ppf(1-alpha/2, k+1, n-k) if k < n else 1.0
    return float(lower), float(upper)


def binomial_p(k, n, p0=0.5):
    """One-sided binomial test: is k/n > p0?"""
    return float(stats.binomtest(k, n, p0, alternative="greater").pvalue)


def compare_two_proportions(k1, n1, k2, n2):
    """Two-proportion z-test."""
    p1, p2 = k1/n1, k2/n2
    p_pool = (k1+k2) / (n1+n2)
    se = np.sqrt(p_pool * (1-p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return {"z": 0, "p_value": 1.0, "diff": float(p1-p2)}
    z = (p1 - p2) / se
    p = 2 * stats.norm.sf(abs(z))
    return {"z": float(z), "p_value": float(p), "diff": float(p1-p2)}


def main():
    print(f"\n{'='*70}")
    print(f"PUBLICATION-READY CONFIDENCE INTERVALS")
    print(f"{'='*70}")

    results = {}

    # Key results from across all sessions (correct/total from actual experiments)
    headline_results = {
        # Self-steering (residual, optimal alpha)
        "Marin-8B self (residual, α=1)": (29, 30),
        "Marin-8B self (residual, α=2)": (29, 30),
        "SmolLM3 self (completion, α=1)": (30, 30),
        "Llama-1B self (direct)": (19, 30),
        "Qwen-7B self (residual)": (25, 30),

        # Cross-dim transfer → Marin 8B
        "SmolLM3→Marin (Procrustes)": (29, 30),
        "Llama→Marin (Procrustes)": (30, 30),
        "Qwen→Marin (Procrustes)": (29, 30),
        "Ensemble→Marin": (30, 30),

        # Zero-calibration transfer → Marin 8B
        "SmolLM3→Marin (zero-cal)": (29, 30),
        "Llama→Marin (zero-cal)": (30, 30),
        "Qwen→Marin (zero-cal)": (30, 30),
        "Ensemble→Marin (zero-cal)": (30, 30),

        # Special conditions
        "Transitive SmolLM3→Llama→Marin": (30, 30),
        "SmolLM3→Marin (sign-corrected, 5 bits)": (29, 30),
        "SmolLM3→Marin (identity, 0 bits)": (11, 30),
        "SmolLM3→Marin (4 bits, greedy)": (29, 30),
        "Random vectors (Marin 8B)": (13, 30),

        # Compositional
        "Dual-trait composition (SmolLM3)": (115, 120),
        "Composition scaling k=1": (30, 30),
        "Composition scaling k=3": (169, 180),
        "Composition scaling k=5": (30, 30),

        # Negative steering
        "Negative α=-1 (SmolLM3)": (30, 30),
        "Negative α=-1 (Marin 8B)": (30, 30),

        # Generalization
        "Novel activity descriptions (SmolLM3, α=1)": (30, 30),
        "Novel value descriptions (SmolLM3, α=1)": (30, 30),

        # Generation validation
        "Zero-cal generation (LLM judge)": (4, 18),
        "Self generation (LLM judge)": (4, 18),

        # Cross-model composition
        "Cross-model composition (SmolLM3→Marin)": (14, 15),
    }

    print(f"\n  {'Result':>45}  {'k/n':>7}  {'Acc':>5}  {'95% CI (CP)':>18}  {'p(>50%)':>10}")
    print(f"  {'-'*95}")

    for name, (k, n) in headline_results.items():
        acc = k / n
        cp_lo, cp_hi = clopper_pearson(k, n)
        p = binomial_p(k, n)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        print(f"  {name:>45}  {k:>3}/{n:<3}  {acc:>4.0%}  [{cp_lo:.0%}, {cp_hi:.0%}]  {p:>9.2e} {sig}")

        results[name] = {
            "k": k, "n": n, "accuracy": float(acc),
            "ci_95_clopper_pearson": [float(cp_lo), float(cp_hi)],
            "p_value_vs_50pct": float(p),
        }

    # Key pairwise comparisons
    print(f"\n{'='*70}")
    print(f"KEY COMPARISONS (Two-proportion z-test)")
    print(f"{'='*70}")

    comparisons = [
        ("Self vs Zero-cal (SmolLM3→Marin)", 29, 30, 29, 30),
        ("Self vs Procrustes (SmolLM3→Marin)", 29, 30, 29, 30),
        ("Procrustes vs Zero-cal (SmolLM3→Marin)", 29, 30, 29, 30),
        ("Self vs Llama transfer (Marin)", 29, 30, 30, 30),
        ("Residual vs Full (Marin, α=1)", 29, 30, 24, 30),
        ("5 bits vs 4 bits transfer", 29, 30, 29, 30),
        ("5 bits vs identity (0 bits)", 29, 30, 11, 30),
        ("Zero-cal vs Random", 29, 30, 13, 30),
        ("Self-steering vs Random", 29, 30, 13, 30),
        ("Dual composition vs chance", 115, 120, 60, 120),
    ]

    comp_results = {}
    print(f"\n  {'Comparison':>45}  {'k1/n1':>7}  {'k2/n2':>7}  {'Diff':>6}  {'p':>10}  {'Sig':>4}")
    for name, k1, n1, k2, n2 in comparisons:
        r = compare_two_proportions(k1, n1, k2, n2)
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "n.s."
        print(f"  {name:>45}  {k1:>3}/{n1:<3}  {k2:>3}/{n2:<3}  {r['diff']:>+5.0%}  {r['p_value']:>9.3f}  {sig}")
        comp_results[name] = r

    results["comparisons"] = comp_results

    # Effect sizes (Cohen's h for proportions)
    print(f"\n{'='*70}")
    print(f"EFFECT SIZES (Cohen's h)")
    print(f"{'='*70}")

    def cohens_h(p1, p2):
        return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

    effect_sizes = [
        ("Residual vs Full (Marin α=1)", 29/30, 24/30),
        ("Self vs Random (Marin)", 29/30, 13/30),
        ("Zero-cal vs Random", 29/30, 13/30),
        ("5 bits vs 0 bits", 29/30, 11/30),
        ("SmolLM3 (completion) vs Llama", 30/30, 19/30),
        ("Composition vs chance (50%)", 115/120, 0.5),
    ]

    for name, p1, p2 in effect_sizes:
        h = cohens_h(p1, p2)
        size = "large" if abs(h) > 0.8 else "medium" if abs(h) > 0.5 else "small"
        print(f"  {name:>40}: h = {h:+.3f} ({size})")

    # Summary table for paper
    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE FOR PAPER")
    print(f"{'='*70}")

    paper_rows = [
        ("Self (Marin 8B)", 29, 30),
        ("Cross-dim (SmolLM3→Marin)", 29, 30),
        ("Cross-dim (Llama→Marin)", 30, 30),
        ("Cross-dim (Qwen→Marin)", 29, 30),
        ("Zero-cal (SmolLM3→Marin)", 29, 30),
        ("Zero-cal (Llama→Marin)", 30, 30),
        ("Zero-cal ensemble", 30, 30),
        ("Transitive (SmolLM3→Llama→Marin)", 30, 30),
        ("Random baseline", 13, 30),
    ]

    print(f"\n  {'Condition':>40}  {'Accuracy':>8}  {'95% CI':>18}  {'p < 0.001':>10}")
    for name, k, n in paper_rows:
        acc = k/n
        lo, hi = clopper_pearson(k, n)
        p = binomial_p(k, n)
        print(f"  {name:>40}  {acc:>7.0%}  [{lo:>5.0%}, {hi:>5.0%}]  {'Yes' if p < 0.001 else 'No':>10}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_path = out_dir / "optimal_bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
