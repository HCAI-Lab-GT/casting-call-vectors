#!/usr/bin/env python
"""
Compare compositional steering results across models.

Analyzes:
1. Single-trait accuracy (does residual_X boost trait X most?)
2. Composition accuracy (does A+B boost both A and B?)
3. Linearity test: is gap(A+B) ≈ gap(A) + gap(B) - gap(baseline)?
4. Subtraction effectiveness: does A-B boost A and suppress B?
5. Cross-model comparison of all metrics
"""

import json
from pathlib import Path
import numpy as np
from itertools import combinations

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
SHORT = {"artistic": "art", "conventional": "con", "enterprising": "ent",
         "investigative": "inv", "realistic": "rea", "social": "soc"}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def rank_of(gaps, trait):
    """Return 1-indexed rank of trait in gaps dict (1 = highest)."""
    sorted_traits = sorted(gaps.keys(), key=lambda t: gaps[t], reverse=True)
    return sorted_traits.index(trait) + 1


def analyze_model(data):
    model_id = data["model_id"]
    conds = data["conditions"]
    baseline = conds["baseline"]

    results = {"model_id": model_id}

    # 1. Single-trait accuracy
    print(f"\n{'='*70}")
    print(f"Model: {model_id}")
    print(f"{'='*70}")

    print(f"\n--- Single Residual Trait Steering ---")
    print(f"{'Trait':>15} {'Top Trait':>12} {'Rank':>5} {'Gap':>8} {'Baseline':>8} {'Lift':>8}")

    single_ranks = []
    single_lifts = []
    for trait in TRAITS:
        cond = f"residual_{trait}"
        if cond not in conds:
            continue
        gaps = conds[cond]
        top_trait = max(gaps, key=lambda t: gaps[t])
        r = rank_of(gaps, trait)
        lift = gaps[trait] - baseline[trait]
        single_ranks.append(r)
        single_lifts.append(lift)
        marker = "✓" if r == 1 else ""
        print(f"{trait:>15} {top_trait:>12} {r:>5} {gaps[trait]:>8.2f} {baseline[trait]:>8.2f} {lift:>+8.2f}  {marker}")

    results["single_top1"] = sum(1 for r in single_ranks if r == 1)
    results["single_top3"] = sum(1 for r in single_ranks if r <= 3)
    results["single_mean_rank"] = np.mean(single_ranks)
    results["single_mean_lift"] = np.mean(single_lifts)

    print(f"\n  Top-1 accuracy: {results['single_top1']}/6")
    print(f"  Top-3 accuracy: {results['single_top3']}/6")
    print(f"  Mean rank: {results['single_mean_rank']:.2f}")
    print(f"  Mean lift over baseline: {results['single_mean_lift']:+.2f}")

    # 2. Composition accuracy and linearity
    print(f"\n--- Additive Compositions ---")
    print(f"{'Pair':>25} {'RankA':>6} {'RankB':>6} {'Both≤3':>7} {'Predicted':>10} {'Actual':>10} {'Linearity':>10}")

    comp_both_top3 = 0
    comp_total = 0
    linearity_errors = []

    for t1, t2 in combinations(TRAITS, 2):
        cond = f"residual_{t1}+{t2}"
        if cond not in conds:
            continue
        comp_total += 1
        gaps = conds[cond]
        r1 = rank_of(gaps, t1)
        r2 = rank_of(gaps, t2)
        both = "✓" if r1 <= 3 and r2 <= 3 else ""
        if r1 <= 3 and r2 <= 3:
            comp_both_top3 += 1

        # Linearity test: gap(A+B, trait) ≈ gap(A, trait) + gap(B, trait) - gap(baseline, trait)
        # Test for the two target traits
        for target in [t1, t2]:
            predicted = conds[f"residual_{t1}"][target] + conds[f"residual_{t2}"][target] - baseline[target]
            actual = gaps[target]
            linearity_errors.append(actual - predicted)

        pred_sum = (conds[f"residual_{t1}"][t1] + conds[f"residual_{t2}"][t1] - baseline[t1],
                    conds[f"residual_{t1}"][t2] + conds[f"residual_{t2}"][t2] - baseline[t2])
        actual_vals = (gaps[t1], gaps[t2])

        print(f"  {t1[:3]}+{t2[:3]:>3}: "
              f"  r({SHORT[t1]})={r1}  r({SHORT[t2]})={r2}  {both:>4}  "
              f"pred=({pred_sum[0]:+.2f},{pred_sum[1]:+.2f})  "
              f"act=({actual_vals[0]:+.2f},{actual_vals[1]:+.2f})  "
              f"err=({actual_vals[0]-pred_sum[0]:+.2f},{actual_vals[1]-pred_sum[1]:+.2f})")

    results["comp_both_top3"] = comp_both_top3
    results["comp_total"] = comp_total
    results["linearity_mean_error"] = np.mean(linearity_errors)
    results["linearity_std_error"] = np.std(linearity_errors)
    results["linearity_rmse"] = np.sqrt(np.mean(np.array(linearity_errors)**2))

    print(f"\n  Both in top-3: {comp_both_top3}/{comp_total}")
    print(f"  Linearity error: mean={results['linearity_mean_error']:+.3f}, "
          f"std={results['linearity_std_error']:.3f}, RMSE={results['linearity_rmse']:.3f}")

    # 3. Full linearity analysis (all 6 traits for all compositions)
    all_predicted = []
    all_actual = []
    for t1, t2 in combinations(TRAITS, 2):
        cond = f"residual_{t1}+{t2}"
        if cond not in conds:
            continue
        for target in TRAITS:
            predicted = conds[f"residual_{t1}"][target] + conds[f"residual_{t2}"][target] - baseline[target]
            actual = conds[cond][target]
            all_predicted.append(predicted)
            all_actual.append(actual)

    all_predicted = np.array(all_predicted)
    all_actual = np.array(all_actual)
    corr = np.corrcoef(all_predicted, all_actual)[0, 1]
    full_rmse = np.sqrt(np.mean((all_actual - all_predicted)**2))

    print(f"\n  Full linearity (all 6 traits x 15 pairs = {len(all_predicted)} points):")
    print(f"    Correlation: r = {corr:.4f}")
    print(f"    RMSE: {full_rmse:.3f}")
    print(f"    Mean absolute error: {np.mean(np.abs(all_actual - all_predicted)):.3f}")

    results["full_linearity_r"] = corr
    results["full_linearity_rmse"] = full_rmse

    # 4. Subtraction analysis
    print(f"\n--- Subtractive Compositions ---")
    sub_pairs = [
        ("artistic", "conventional"),
        ("conventional", "artistic"),
        ("investigative", "social"),
        ("social", "investigative"),
        ("realistic", "enterprising"),
        ("enterprising", "realistic"),
    ]

    sub_correct_boost = 0
    sub_correct_suppress = 0
    sub_total = 0

    for t_boost, t_suppress in sub_pairs:
        cond = f"residual_{t_boost}-{t_suppress}"
        if cond not in conds:
            continue
        sub_total += 1
        gaps = conds[cond]

        # Check if boosted trait increased relative to baseline
        boost_lift = gaps[t_boost] - baseline[t_boost]
        suppress_lift = gaps[t_suppress] - baseline[t_suppress]
        r_boost = rank_of(gaps, t_boost)
        r_suppress = rank_of(gaps, t_suppress)

        boost_ok = boost_lift > 0
        suppress_ok = suppress_lift < 0 or r_suppress >= 4  # Suppressed = below median or decreased

        if boost_ok:
            sub_correct_boost += 1
        if r_suppress > r_boost:
            sub_correct_suppress += 1

        print(f"  {t_boost[:3]}-{t_suppress[:3]}: "
              f"boost({SHORT[t_boost]})={boost_lift:+.2f} rank={r_boost}, "
              f"suppress({SHORT[t_suppress]})={suppress_lift:+.2f} rank={r_suppress}  "
              f"{'✓ rank' if r_suppress > r_boost else '✗ rank'}")

    if sub_total > 0:
        results["sub_correct_rank_ordering"] = sub_correct_suppress
        results["sub_total"] = sub_total
        print(f"\n  Correct rank ordering (boosted > suppressed): {sub_correct_suppress}/{sub_total}")

    return results


def main():
    files = sorted(Path("outputs/compositional").glob("*_compositional_steering.json"))

    if not files:
        print("No compositional steering files found!")
        return

    all_results = []
    for f in files:
        data = load_results(f)
        results = analyze_model(data)
        all_results.append(results)

    # Cross-model comparison
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")

        print(f"\n{'Metric':>35}", end="")
        for r in all_results:
            model_short = r["model_id"].split("/")[-1][:20]
            print(f"{model_short:>22}", end="")
        print()
        print("-" * (35 + 22 * len(all_results)))

        metrics = [
            ("Single top-1 accuracy", "single_top1", "/6"),
            ("Single top-3 accuracy", "single_top3", "/6"),
            ("Single mean rank", "single_mean_rank", ""),
            ("Single mean lift", "single_mean_lift", ""),
            ("Comp both-in-top3", "comp_both_top3", "/15"),
            ("Linearity r", "full_linearity_r", ""),
            ("Linearity RMSE", "full_linearity_rmse", ""),
            ("Subtraction rank ordering", "sub_correct_rank_ordering", "/6"),
        ]

        for label, key, suffix in metrics:
            print(f"{label:>35}", end="")
            for r in all_results:
                val = r.get(key, "N/A")
                if isinstance(val, float):
                    print(f"{val:>18.3f}{suffix:>4}", end="")
                else:
                    print(f"{val!s:>18}{suffix:>4}", end="")
            print()

    # Save results
    out_path = Path("outputs/analysis/compositional_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
