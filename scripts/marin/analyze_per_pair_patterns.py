#!/usr/bin/env python
"""
Analyze per-pair and per-trait patterns in pairwise discrimination
across all available models and prompt formats.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Holland hexagonal distances
HEX_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
HEX_DISTANCES = {}
for i, t1 in enumerate(HEX_ORDER):
    for j, t2 in enumerate(HEX_ORDER):
        if t1 == t2:
            continue
        d = min(abs(i - j), 6 - abs(i - j))
        key = tuple(sorted([t1, t2]))
        HEX_DISTANCES[key] = d


def load_cross_model_detail():
    """Load per-pair detail from cross-model transfer experiment."""
    path = _repo_root() / "outputs" / "analysis" / "cross_model_steering_transfer.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data


def analyze_per_trait(pair_detail):
    """Analyze per-trait accuracy from pair detail."""
    trait_correct = defaultdict(int)
    trait_total = defaultdict(int)

    for key, info in pair_detail.items():
        # key format: steer_TRAIT_TRAITA-TRAITB
        parts = key.split("_")
        steer_trait = parts[1]
        trait_correct[steer_trait] += int(info["correct"])
        trait_total[steer_trait] += 1

    result = {}
    for t in TRAITS:
        if t in trait_total:
            result[t] = {
                "accuracy": trait_correct[t] / trait_total[t],
                "correct": trait_correct[t],
                "total": trait_total[t],
            }
    return result


def analyze_per_pair(pair_detail):
    """Analyze per-pair accuracy (across all steering conditions involving that pair)."""
    pair_correct = defaultdict(int)
    pair_total = defaultdict(int)
    pair_deltas = defaultdict(list)

    for key, info in pair_detail.items():
        parts = key.split("_")
        pair = parts[2]  # e.g., artistic-conventional
        pair_correct[pair] += int(info["correct"])
        pair_total[pair] += 1
        pair_deltas[pair].append(info["delta"])

    result = {}
    for pair in sorted(pair_correct.keys()):
        result[pair] = {
            "accuracy": pair_correct[pair] / pair_total[pair],
            "correct": pair_correct[pair],
            "total": pair_total[pair],
            "mean_delta": float(np.mean(pair_deltas[pair])),
        }
    return result


def main():
    data = load_cross_model_detail()
    if not data:
        print("No cross-model transfer data found")
        return

    print("="*70)
    print("PER-TRAIT AND PER-PAIR ANALYSIS")
    print("="*70)

    conditions = [
        ("llama_own", "Llama + Llama"),
        ("llama_smollm3_procrustes", "Llama + SmolLM3(P)"),
        ("smollm3_own", "SmolLM3 + SmolLM3"),
        ("smollm3_llama_procrustes", "SmolLM3 + Llama(P)"),
    ]

    # Per-trait analysis
    print(f"\n{'='*70}")
    print("PER-TRAIT ACCURACY (each trait steered against its 5 partners)")
    print(f"{'='*70}")

    trait_data = {}
    for cond_key, cond_name in conditions:
        if cond_key in data and "pair_detail" in data[cond_key]:
            trait_data[cond_name] = analyze_per_trait(data[cond_key]["pair_detail"])

    print(f"\n  {'Trait':>15}", end="")
    for name in trait_data:
        print(f"  {name:>20}", end="")
    print(f"  {'Mean':>8}")
    print(f"  {'-'*15}", end="")
    for _ in trait_data:
        print(f"  {'-'*20}", end="")
    print(f"  {'-'*8}")

    trait_means = defaultdict(list)
    for t in TRAITS:
        print(f"  {t:>15}", end="")
        for name, td in trait_data.items():
            if t in td:
                acc = td[t]["accuracy"]
                print(f"  {acc:>19.0%}", end="")
                trait_means[t].append(acc)
            else:
                print(f"  {'N/A':>19}", end="")
        mean_acc = np.mean(trait_means[t]) if trait_means[t] else 0
        print(f"  {mean_acc:>7.0%}")

    # Per-pair analysis
    print(f"\n{'='*70}")
    print("PER-PAIR ACCURACY (each pair across both steering directions)")
    print(f"{'='*70}")

    pair_data = {}
    for cond_key, cond_name in conditions:
        if cond_key in data and "pair_detail" in data[cond_key]:
            pair_data[cond_name] = analyze_per_pair(data[cond_key]["pair_detail"])

    all_pairs = sorted(set().union(*(pd.keys() for pd in pair_data.values())))

    print(f"\n  {'Pair':>30} {'HexDist':>8}", end="")
    for name in pair_data:
        print(f"  {name:>16}", end="")
    print(f"  {'Mean':>8}")
    print(f"  {'-'*30} {'-'*8}", end="")
    for _ in pair_data:
        print(f"  {'-'*16}", end="")
    print(f"  {'-'*8}")

    pair_means = {}
    for pair in all_pairs:
        t1, t2 = pair.split("-")
        hex_d = HEX_DISTANCES.get(tuple(sorted([t1, t2])), "?")
        print(f"  {pair:>30} {hex_d:>8}", end="")
        accs = []
        for name, pd in pair_data.items():
            if pair in pd:
                acc = pd[pair]["accuracy"]
                print(f"  {acc:>15.0%}", end="")
                accs.append(acc)
            else:
                print(f"  {'N/A':>15}", end="")
        mean_acc = np.mean(accs) if accs else 0
        pair_means[pair] = mean_acc
        print(f"  {mean_acc:>7.0%}")

    # Summary by hex distance
    print(f"\n{'='*70}")
    print("ACCURACY BY HOLLAND HEXAGONAL DISTANCE")
    print(f"{'='*70}")

    by_distance = defaultdict(list)
    for pair, mean_acc in pair_means.items():
        t1, t2 = pair.split("-")
        d = HEX_DISTANCES.get(tuple(sorted([t1, t2])), 0)
        by_distance[d].append(mean_acc)

    for d in sorted(by_distance):
        labels = {1: "Adjacent", 2: "Alternate", 3: "Opposite"}
        mean = np.mean(by_distance[d])
        n = len(by_distance[d])
        print(f"  Distance {d} ({labels.get(d, '?'):>9}): {mean:.0%} mean (n={n})")

    # Hardest and easiest pairs
    sorted_pairs = sorted(pair_means.items(), key=lambda x: x[1])
    print(f"\n  Hardest pairs:")
    for pair, acc in sorted_pairs[:5]:
        t1, t2 = pair.split("-")
        d = HEX_DISTANCES.get(tuple(sorted([t1, t2])), "?")
        print(f"    {pair:>30} (d={d}): {acc:.0%}")

    print(f"\n  Easiest pairs:")
    for pair, acc in sorted_pairs[-5:]:
        t1, t2 = pair.split("-")
        d = HEX_DISTANCES.get(tuple(sorted([t1, t2])), "?")
        print(f"    {pair:>30} (d={d}): {acc:.0%}")

    # Per-trait summary
    print(f"\n{'='*70}")
    print("PER-TRAIT SUMMARY (mean across conditions)")
    print(f"{'='*70}")
    sorted_traits = sorted(trait_means.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for t, accs in sorted_traits:
        print(f"  {t:>15}: {np.mean(accs):.0%}")

    # Save comprehensive results
    output = {
        "per_trait": {name: {t: td[t] for t in TRAITS if t in td} for name, td in trait_data.items()},
        "per_pair": {name: pd for name, pd in pair_data.items()},
        "hex_distance_means": {str(d): float(np.mean(accs)) for d, accs in by_distance.items()},
        "pair_ranking": {pair: float(acc) for pair, acc in sorted_pairs},
    }

    out_path = _repo_root() / "outputs" / "analysis" / "per_pair_patterns.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
