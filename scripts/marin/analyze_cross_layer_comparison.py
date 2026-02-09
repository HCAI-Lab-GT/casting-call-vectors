#!/usr/bin/env python
"""
Synthesize cross-layer steering results across models.

Key questions:
1. Is the optimal injection layer above the middle for all models?
2. Do different traits prefer different injection layers?
3. How does transfer robustness vary with model scale?
"""

import json
from pathlib import Path
import numpy as np

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def analyze_model(data):
    model_id = data["model_id"]
    num_layers = data["num_layers"]
    mid_layer = data["mid_layer"]
    test_layers = data["test_layers"]
    baseline = data["conditions"]["baseline"]

    print(f"\n{'='*70}")
    print(f"Model: {model_id} ({num_layers} layers, mid={mid_layer})")
    print(f"{'='*70}")

    # Find which traits are present
    traits_present = []
    for t in TRAITS:
        if f"{t}_matched_L{mid_layer}" in data["conditions"]:
            traits_present.append(t)

    print(f"\n--- Optimal Injection Layer Per Trait ---")
    print(f"{'Trait':>15} {'Best Matched':>14} {'Best Transfer':>14} {'Mid Gap':>10} {'Improv%':>10} {'Opt Frac':>10}")

    for trait in traits_present:
        best_m_gap = -999
        best_m_layer = -1
        best_t_gap = -999
        best_t_layer = -1
        mid_gap = 0

        for L in test_layers:
            m_key = f"{trait}_matched_L{L}"
            t_key = f"{trait}_transfer_L{L}"

            m_val = data["conditions"].get(m_key, {}).get(trait, -999)
            t_val = data["conditions"].get(t_key, {}).get(trait, -999)

            if m_val > best_m_gap:
                best_m_gap = m_val
                best_m_layer = L
            if t_val > best_t_gap:
                best_t_gap = t_val
                best_t_layer = L
            if L == mid_layer:
                mid_gap = m_val

        improvement = (best_m_gap - mid_gap) / max(abs(mid_gap), 0.01) * 100
        opt_frac = best_m_layer / num_layers

        print(f"{trait:>15} L{best_m_layer} ({best_m_gap:.2f}) L{best_t_layer} ({best_t_gap:.2f}) "
              f"{mid_gap:>10.2f} {improvement:>+9.0f}% {opt_frac:>10.1%}")

    # Transfer robustness: how stable is the mid-layer vector across injection points?
    print(f"\n--- Transfer Robustness (mid-layer vector across injection points) ---")
    for trait in traits_present:
        gaps = []
        for L in test_layers:
            t_key = f"{trait}_transfer_L{L}"
            val = data["conditions"].get(t_key, {}).get(trait, None)
            if val is not None:
                gaps.append(val)
        gaps = np.array(gaps)
        print(f"  {trait:>15}: mean={gaps.mean():.2f}, std={gaps.std():.2f}, "
              f"min={gaps.min():.2f}, max={gaps.max():.2f}, "
              f"CV={gaps.std()/max(abs(gaps.mean()),0.01):.2f}")

    # Layer-specificity: how much does matched outperform transfer?
    print(f"\n--- Layer Specificity (matched - transfer) by depth ---")
    depths = []
    deltas_by_depth = {}

    for trait in traits_present:
        for L in test_layers:
            m_key = f"{trait}_matched_L{L}"
            t_key = f"{trait}_transfer_L{L}"
            m_val = data["conditions"].get(m_key, {}).get(trait, None)
            t_val = data["conditions"].get(t_key, {}).get(trait, None)
            if m_val is not None and t_val is not None:
                frac = L / num_layers
                if frac not in deltas_by_depth:
                    deltas_by_depth[frac] = []
                deltas_by_depth[frac].append(m_val - t_val)

    print(f"{'Depth':>8} {'Mean Δ':>8} {'Interpretation'}")
    for frac in sorted(deltas_by_depth.keys()):
        d = np.mean(deltas_by_depth[frac])
        interp = "local>transfer" if d > 0.5 else ("transfer>local" if d < -0.5 else "≈ equivalent")
        marker = "  ← mid" if abs(frac - 0.5) < 0.05 else ""
        print(f"{frac:>7.0%}{marker:>7} {d:>+8.2f}   {interp}")


def main():
    results_dir = Path("outputs/analysis")
    files = sorted(results_dir.glob("cross_layer_steering_*.json"))

    if not files:
        print("No cross-layer steering files found!")
        return

    all_data = []
    for f in files:
        data = load_results(f)
        all_data.append(data)
        analyze_model(data)

    # Cross-model comparison
    if len(all_data) >= 2:
        print(f"\n{'='*70}")
        print("CROSS-MODEL SYNTHESIS")
        print(f"{'='*70}")

        print(f"\nKey finding: Optimal injection layer fraction by model:")
        for data in all_data:
            model = data["model_id"].split("/")[-1]
            num_layers = data["num_layers"]
            mid = data["mid_layer"]

            opt_layers = []
            for t in TRAITS:
                best_gap = -999
                best_L = -1
                for L in data["test_layers"]:
                    m_key = f"{t}_matched_L{L}"
                    val = data["conditions"].get(m_key, {}).get(t, -999)
                    if val > best_gap:
                        best_gap = val
                        best_L = L
                if best_L >= 0:
                    opt_layers.append(best_L / num_layers)

            if opt_layers:
                print(f"  {model:>25}: mean optimal = {np.mean(opt_layers):.1%} "
                      f"(mid = {mid/num_layers:.0%}), range = {min(opt_layers):.0%}-{max(opt_layers):.0%}")

        # Common traits comparison
        common_traits = set(TRAITS)
        for data in all_data:
            model_traits = set()
            for t in TRAITS:
                if f"{t}_matched_L{data['mid_layer']}" in data["conditions"]:
                    model_traits.add(t)
            common_traits &= model_traits

        if common_traits and len(common_traits) > 0:
            print(f"\n  Transfer robustness comparison (common trait: {list(common_traits)[0]}):")
            trait = list(common_traits)[0]
            for data in all_data:
                model = data["model_id"].split("/")[-1]
                gaps = []
                for L in data["test_layers"]:
                    t_key = f"{trait}_transfer_L{L}"
                    val = data["conditions"].get(t_key, {}).get(trait, None)
                    if val is not None:
                        gaps.append(val)
                gaps = np.array(gaps)
                frac_positive = np.mean(gaps > 0) * 100
                print(f"    {model:>25}: mean={gaps.mean():.2f}, std={gaps.std():.2f}, "
                      f"{frac_positive:.0f}% positive")


if __name__ == "__main__":
    main()
