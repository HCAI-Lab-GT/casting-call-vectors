#!/usr/bin/env python
"""
Analyze pairwise discrimination results across models.

Key insight: raw win/loss doesn't account for baseline model preferences.
The proper metric is the DELTA from baseline: does steering with trait A
move the A-vs-B gap in A's favor?
"""

import json
from pathlib import Path

import numpy as np

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
ROOT = Path(__file__).resolve().parents[2]


def load_results(model_safe):
    path = ROOT / "outputs" / "analysis" / f"pairwise_discrimination_{model_safe}.json"
    with open(path) as f:
        return json.load(f)


def analyze_model(name, model_safe):
    data = load_results(model_safe)
    baseline = data["baseline"]

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    for vec_type in ["full", "residual"]:
        steered = data[vec_type]
        print(f"\n--- {vec_type.upper()} vectors ---")

        # Build delta matrix: for each steered trait, how much did steering
        # shift preferences toward the steered trait vs baseline?
        correct_raw = 0
        correct_delta = 0
        total = 0
        delta_magnitudes = []

        per_trait_raw = {t: {"correct": 0, "total": 0} for t in TRAITS}
        per_trait_delta = {t: {"correct": 0, "total": 0, "deltas": []} for t in TRAITS}

        for steer_trait in TRAITS:
            pairs = steered[steer_trait]
            for pair_key, gap in pairs.items():
                t1, t2 = pair_key.split("-")
                if steer_trait not in (t1, t2):
                    continue

                base_gap = baseline[pair_key]

                # Convention: gap > 0 means prefers t1
                # Delta: positive means steering moved toward steered trait
                if steer_trait == t1:
                    delta = gap - base_gap   # Want gap to increase
                    raw_correct = gap > 0
                else:
                    delta = base_gap - gap   # Want gap to decrease (become more negative)
                    raw_correct = gap < 0

                delta_correct = delta > 0

                per_trait_raw[steer_trait]["correct"] += int(raw_correct)
                per_trait_raw[steer_trait]["total"] += 1
                per_trait_delta[steer_trait]["correct"] += int(delta_correct)
                per_trait_delta[steer_trait]["total"] += 1
                per_trait_delta[steer_trait]["deltas"].append(delta)

                correct_raw += int(raw_correct)
                correct_delta += int(delta_correct)
                total += 1
                delta_magnitudes.append(delta)

        print(f"\n  {'Trait':>15} {'Raw':>6} {'Delta':>6} {'MeanΔ':>8} {'MinΔ':>8} {'MaxΔ':>8}")
        print(f"  {'-'*55}")
        for t in TRAITS:
            r = per_trait_raw[t]
            d = per_trait_delta[t]
            deltas = d["deltas"]
            mean_d = np.mean(deltas)
            min_d = np.min(deltas)
            max_d = np.max(deltas)
            print(f"  {t:>15} {r['correct']}/{r['total']}   {d['correct']}/{d['total']}   {mean_d:+7.3f}  {min_d:+7.3f}  {max_d:+7.3f}")

        print(f"\n  Overall Raw:   {correct_raw}/{total} ({correct_raw/total*100:.0f}%)")
        print(f"  Overall Delta: {correct_delta}/{total} ({correct_delta/total*100:.0f}%)")
        print(f"  Mean delta:    {np.mean(delta_magnitudes):+.3f}")
        print(f"  Median delta:  {np.median(delta_magnitudes):+.3f}")

        return_data = {
            "raw": correct_raw / total,
            "delta": correct_delta / total,
            "mean_delta": float(np.mean(delta_magnitudes)),
        }

    return return_data


def cross_model_summary():
    """Aggregate comparison across models."""
    models = {
        "Llama 1B":  "meta-llama__Llama-3.2-1B-Instruct",
        "Marin 8B":  "marin-community__marin-8b-instruct",
        "Qwen 7B":   "Qwen__Qwen2.5-7B-Instruct",
    }

    print(f"\n\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Model':>12} {'Vec Type':>10} {'Raw':>8} {'Delta':>8} {'MeanΔ':>8}")
    print(f"  {'-'*50}")

    for name, safe in models.items():
        data = load_results(safe)
        baseline = data["baseline"]

        for vec_type in ["full", "residual"]:
            steered = data[vec_type]
            correct_raw = 0
            correct_delta = 0
            total = 0
            deltas = []

            for steer_trait in TRAITS:
                pairs = steered[steer_trait]
                for pair_key, gap in pairs.items():
                    t1, t2 = pair_key.split("-")
                    if steer_trait not in (t1, t2):
                        continue
                    base_gap = baseline[pair_key]
                    if steer_trait == t1:
                        delta = gap - base_gap
                        raw_correct = gap > 0
                    else:
                        delta = base_gap - gap
                        raw_correct = gap < 0
                    correct_raw += int(raw_correct)
                    correct_delta += int(delta > 0)
                    total += 1
                    deltas.append(delta)

            print(f"  {name:>12} {vec_type:>10}  {correct_raw}/{total}  {correct_delta}/{total}  {np.mean(deltas):+7.3f}")

    # Per-trait consistency across models
    print(f"\n\nPer-trait delta accuracy (residual vectors):")
    print(f"  {'Trait':>15}", end="")
    for name in models:
        print(f"  {name:>10}", end="")
    print(f"  {'Mean':>8}")
    print(f"  {'-'*55}")

    for trait in TRAITS:
        print(f"  {trait:>15}", end="")
        scores = []
        for name, safe in models.items():
            data = load_results(safe)
            baseline = data["baseline"]
            steered = data["residual"][trait]
            correct = 0
            total = 0
            for pair_key, gap in steered.items():
                t1, t2 = pair_key.split("-")
                if trait not in (t1, t2):
                    continue
                base_gap = baseline[pair_key]
                if trait == t1:
                    delta = gap - base_gap
                else:
                    delta = base_gap - gap
                correct += int(delta > 0)
                total += 1
            score = correct / total
            scores.append(score)
            print(f"  {correct}/{total} ({score:.0%})", end="")
        print(f"  {np.mean(scores):.0%}")

    # Interesting: which specific pairs are hardest?
    print(f"\n\nHardest pairs (lowest cross-model delta accuracy, residual):")
    pair_scores = {}
    for i, t1 in enumerate(TRAITS):
        for j, t2 in enumerate(TRAITS):
            if i >= j:
                continue
            pair_key = f"{t1}-{t2}"
            correct = 0
            total = 0
            for steer_trait in [t1, t2]:
                for name, safe in models.items():
                    data = load_results(safe)
                    baseline = data["baseline"]
                    gap = data["residual"][steer_trait][pair_key]
                    base_gap = baseline[pair_key]
                    if steer_trait == t1:
                        delta = gap - base_gap
                    else:
                        delta = base_gap - gap
                    correct += int(delta > 0)
                    total += 1
            pair_scores[pair_key] = correct / total

    for pair, score in sorted(pair_scores.items(), key=lambda x: x[1]):
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  {pair:>28}: {score:.0%} {bar}")


def main():
    models = {
        "Llama 1B":  "meta-llama__Llama-3.2-1B-Instruct",
        "Marin 8B":  "marin-community__marin-8b-instruct",
        "Qwen 7B":   "Qwen__Qwen2.5-7B-Instruct",
    }

    for name, safe in models.items():
        analyze_model(name, safe)

    cross_model_summary()


if __name__ == "__main__":
    main()
