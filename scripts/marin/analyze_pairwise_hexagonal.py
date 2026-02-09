#!/usr/bin/env python
"""
Test whether pairwise discrimination accuracy correlates with Holland hexagonal distance.

Holland's hexagon: R-I-A-S-E-C (going around clockwise).
Adjacent pairs should be harder to discriminate (more similar traits).
Opposite pairs should be easier (maximally different traits).
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
ROOT = Path(__file__).resolve().parents[2]

# Holland hexagonal ordering: R, I, A, S, E, C
HEXAGON = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
HEX_POS = {t: i for i, t in enumerate(HEXAGON)}


def hex_distance(t1, t2):
    """Distance on hexagon (1=adjacent, 2=alternate, 3=opposite)."""
    d = abs(HEX_POS[t1] - HEX_POS[t2])
    return min(d, 6 - d)


def load_results(model_safe):
    path = ROOT / "outputs" / "analysis" / f"pairwise_discrimination_{model_safe}.json"
    with open(path) as f:
        return json.load(f)


def analyze_hexagonal_connection():
    models = {
        "Llama 1B": "meta-llama__Llama-3.2-1B-Instruct",
        "Marin 8B": "marin-community__marin-8b-instruct",
        "Qwen 7B": "Qwen__Qwen2.5-7B-Instruct",
    }

    print("=" * 70)
    print("PAIRWISE DISCRIMINATION vs HOLLAND HEXAGONAL DISTANCE")
    print("=" * 70)

    # Enumerate all 15 pairs with their hex distances
    pairs = []
    for i, t1 in enumerate(TRAITS):
        for j, t2 in enumerate(TRAITS):
            if i >= j:
                continue
            pairs.append((t1, t2, hex_distance(t1, t2)))

    print(f"\nAll 15 pairs with hex distance:")
    for t1, t2, d in sorted(pairs, key=lambda x: x[2]):
        print(f"  {t1:>15}-{t2:<15} dist={d}")

    # For each model: compute delta accuracy per pair and correlate with hex distance
    print(f"\n{'='*70}")
    print("Per-pair delta accuracy by hex distance")
    print(f"{'='*70}")

    all_pair_deltas = {d: [] for d in [1, 2, 3]}
    per_model_results = {}

    for name, safe in models.items():
        data = load_results(safe)
        baseline = data["baseline"]

        print(f"\n--- {name} (residual vectors) ---")

        pair_data = []
        for t1, t2, hd in pairs:
            pair_key = f"{t1}-{t2}"
            # For BOTH steering directions
            deltas = []
            for steer_trait in [t1, t2]:
                gap = data["residual"][steer_trait][pair_key]
                base_gap = baseline[pair_key]
                if steer_trait == t1:
                    d = gap - base_gap
                else:
                    d = base_gap - gap
                deltas.append(d)

            # Both steer directions positive = both correct
            both_correct = all(d > 0 for d in deltas)
            any_correct = any(d > 0 for d in deltas)
            mean_delta = np.mean(deltas)

            pair_data.append({
                "pair": pair_key,
                "hex_dist": hd,
                "both_correct": both_correct,
                "any_correct": any_correct,
                "mean_delta": mean_delta,
            })
            all_pair_deltas[hd].extend(deltas)

        # Group by hex distance
        for d in [1, 2, 3]:
            group = [p for p in pair_data if p["hex_dist"] == d]
            both = sum(p["both_correct"] for p in group)
            any_c = sum(p["any_correct"] for p in group)
            n = len(group)
            mean_d = np.mean([p["mean_delta"] for p in group])
            label = {1: "Adjacent", 2: "Alternate", 3: "Opposite"}[d]
            print(f"  {label} (d={d}): both={both}/{n}, any={any_c}/{n}, meanΔ={mean_d:+.3f}")

        # Correlation
        hex_dists = [p["hex_dist"] for p in pair_data]
        mean_deltas = [p["mean_delta"] for p in pair_data]
        rho, p_val = spearmanr(hex_dists, mean_deltas)
        per_model_results[name] = (rho, p_val)
        print(f"  Spearman(hex_dist, meanΔ): ρ={rho:.3f}, p={p_val:.3f}")

    # Cross-model summary
    print(f"\n{'='*70}")
    print("SUMMARY: Mean delta accuracy by hex distance (ALL models pooled)")
    print(f"{'='*70}")

    for d in [1, 2, 3]:
        deltas = all_pair_deltas[d]
        correct = sum(1 for x in deltas if x > 0)
        total = len(deltas)
        label = {1: "Adjacent", 2: "Alternate", 3: "Opposite"}[d]
        print(f"  {label:>10} (d={d}): {correct}/{total} ({correct/total:.0%}), meanΔ={np.mean(deltas):+.3f}")

    # Also check: which specific pairs are universally hard/easy?
    print(f"\n--- Pair-level consistency ---")
    pair_consistency = {}
    for t1, t2, hd in pairs:
        pair_key = f"{t1}-{t2}"
        correct = 0
        total = 0
        for name, safe in models.items():
            data = load_results(safe)
            baseline = data["baseline"]
            for steer_trait in [t1, t2]:
                gap = data["residual"][steer_trait][pair_key]
                base_gap = baseline[pair_key]
                if steer_trait == t1:
                    d = gap - base_gap
                else:
                    d = base_gap - gap
                correct += int(d > 0)
                total += 1
        pair_consistency[pair_key] = (correct / total, hd)

    for pair, (acc, hd) in sorted(pair_consistency.items(), key=lambda x: x[1][0]):
        label = {1: "adj", 2: "alt", 3: "opp"}[hd]
        bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
        print(f"  {pair:>28} (d={hd}, {label}): {acc:.0%} {bar}")

    print(f"\n--- Prediction ---")
    print(f"  If Holland hexagon predicts discrimination difficulty:")
    print(f"  Adjacent (d=1) should be HARDEST (most similar traits)")
    print(f"  Opposite (d=3) should be EASIEST (most different traits)")

    accs = {}
    for d in [1, 2, 3]:
        deltas = all_pair_deltas[d]
        accs[d] = sum(1 for x in deltas if x > 0) / len(deltas)

    if accs[3] > accs[2] > accs[1]:
        print(f"  Result: CONFIRMED - opposite ({accs[3]:.0%}) > alternate ({accs[2]:.0%}) > adjacent ({accs[1]:.0%})")
    elif accs[3] > accs[1]:
        print(f"  Result: PARTIALLY CONFIRMED - opposite ({accs[3]:.0%}) > adjacent ({accs[1]:.0%}), but alternate ({accs[2]:.0%}) not monotonic")
    else:
        print(f"  Result: NOT CONFIRMED - adjacent ({accs[1]:.0%}), alternate ({accs[2]:.0%}), opposite ({accs[3]:.0%})")


if __name__ == "__main__":
    analyze_hexagonal_connection()
