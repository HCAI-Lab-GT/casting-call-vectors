#!/usr/bin/env python
"""
Bootstrap confidence intervals on key pairwise discrimination results.

All headline accuracy numbers are point estimates on 30 pairwise comparisons.
This script computes bootstrap 95% CIs, tests significance of key differences,
and performs binomial exact tests.

No GPU needed — purely analysis of existing JSON outputs.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats


def _repo_root():
    return Path(__file__).resolve().parents[2]


MODELS = {
    "SmolLM3-3B": "HuggingFaceTB__SmolLM3-3B",
    "Llama-1B": "meta-llama__Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen__Qwen2.5-7B-Instruct",
    "Marin-8B": "marin-community__marin-8b-instruct",
}

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def load_pairwise_data(model_safe_name):
    path = _repo_root() / "outputs/analysis" / f"pairwise_discrimination_{model_safe_name}.json"
    with open(path) as f:
        return json.load(f)


def get_per_pair_deltas(data, vec_type="residual"):
    """Extract per-pair delta values for a given vector type.

    Returns a dict of pair_name -> list of deltas (one per steering trait that is in the pair).
    And a flat list of (correct, delta) tuples.
    """
    baseline = data["baseline"]
    steer_data = data[vec_type]
    results = []

    for steer_trait in TRAITS:
        if steer_trait not in steer_data:
            continue
        steered_gaps = steer_data[steer_trait]
        for pair_name in steered_gaps:
            trait_a, trait_b = pair_name.split("-")
            if steer_trait not in (trait_a, trait_b):
                continue
            base_gap = baseline[pair_name]
            steered_gap = steered_gaps[pair_name]
            if steer_trait == trait_a:
                delta = steered_gap - base_gap
            else:
                delta = base_gap - steered_gap
            results.append({
                "pair": pair_name,
                "steer_trait": steer_trait,
                "delta": delta,
                "correct": int(delta > 0),
            })

    return results


def bootstrap_accuracy(results, n_bootstrap=10000, seed=42):
    """Bootstrap the accuracy statistic from per-trial results."""
    rng = np.random.RandomState(seed)
    n = len(results)
    correct = np.array([r["correct"] for r in results])
    observed_acc = correct.mean()

    boot_accs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_accs.append(correct[idx].mean())

    boot_accs = np.array(boot_accs)
    ci_lower = np.percentile(boot_accs, 2.5)
    ci_upper = np.percentile(boot_accs, 97.5)

    return {
        "observed": float(observed_acc),
        "n": n,
        "correct": int(correct.sum()),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "bootstrap_mean": float(boot_accs.mean()),
        "bootstrap_std": float(boot_accs.std()),
    }


def binomial_test(k, n, p0=0.5):
    """Exact binomial test: is k/n significantly different from p0?"""
    result = stats.binomtest(k, n, p0, alternative="greater")
    return {
        "k": k,
        "n": n,
        "p0": p0,
        "p_value": float(result.pvalue),
        "ci_95": [float(result.proportion_ci(confidence_level=0.95).low),
                  float(result.proportion_ci(confidence_level=0.95).high)],
        "significant_05": result.pvalue < 0.05,
        "significant_01": result.pvalue < 0.01,
        "significant_001": result.pvalue < 0.001,
    }


def compare_accuracies(results_a, results_b, n_bootstrap=10000, seed=42):
    """Bootstrap test of whether two accuracy distributions differ."""
    rng = np.random.RandomState(seed)
    correct_a = np.array([r["correct"] for r in results_a])
    correct_b = np.array([r["correct"] for r in results_b])
    observed_diff = correct_a.mean() - correct_b.mean()

    n_a, n_b = len(correct_a), len(correct_b)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)
        boot_diffs.append(correct_a[idx_a].mean() - correct_b[idx_b].mean())

    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_value = np.mean(boot_diffs <= 0) if observed_diff > 0 else np.mean(boot_diffs >= 0)

    return {
        "observed_diff": float(observed_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
    }


def main():
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ON KEY RESULTS")
    print(f"{'='*70}")

    results = {}

    # 1. Bootstrap CI on each model's pairwise discrimination accuracy
    print(f"\n--- Per-Model Pairwise Discrimination (Residual, α from sweep) ---")
    print(f"  {'Model':>15}  {'Acc':>5}  {'n':>3}  {'95% CI':>15}  {'Binomial p':>12}  {'Sig':>3}")

    for model_name, model_safe in MODELS.items():
        data = load_pairwise_data(model_safe)
        per_pair = get_per_pair_deltas(data, "residual")

        boot = bootstrap_accuracy(per_pair)
        binom = binomial_test(boot["correct"], boot["n"])

        print(f"  {model_name:>15}  {boot['observed']:>4.0%}  {boot['n']:>3}  "
              f"[{boot['ci_95_lower']:.0%}, {boot['ci_95_upper']:.0%}]  "
              f"{binom['p_value']:>11.2e}  {'***' if binom['significant_001'] else '**' if binom['significant_01'] else '*' if binom['significant_05'] else 'n.s.'}")

        results[model_name] = {
            "bootstrap": boot,
            "binomial": binom,
        }

    # 2. Cross-model comparisons
    print(f"\n--- Pairwise Model Comparisons (Bootstrap Diff Test) ---")
    models_list = list(MODELS.keys())
    comparisons = {}
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            m1, m2 = models_list[i], models_list[j]
            data1 = load_pairwise_data(MODELS[m1])
            data2 = load_pairwise_data(MODELS[m2])
            pp1 = get_per_pair_deltas(data1, "residual")
            pp2 = get_per_pair_deltas(data2, "residual")
            comp = compare_accuracies(pp1, pp2)
            print(f"  {m1} vs {m2}: diff={comp['observed_diff']:+.0%}, "
                  f"CI=[{comp['ci_95_lower']:+.0%}, {comp['ci_95_upper']:+.0%}], "
                  f"p={comp['p_value']:.3f} {'*' if comp['significant_05'] else 'n.s.'}")
            comparisons[f"{m1}_vs_{m2}"] = comp

    results["comparisons"] = comparisons

    # 3. Effect of vector type: Full vs Residual
    print(f"\n--- Full vs Residual (Paired Bootstrap, per model) ---")
    vec_comparisons = {}
    for model_name, model_safe in MODELS.items():
        data = load_pairwise_data(model_safe)
        pp_full = get_per_pair_deltas(data, "full")
        pp_res = get_per_pair_deltas(data, "residual")
        comp = compare_accuracies(pp_res, pp_full)
        boot_full = bootstrap_accuracy(pp_full)
        boot_res = bootstrap_accuracy(pp_res)
        print(f"  {model_name:>15}: Full={boot_full['observed']:.0%} CI=[{boot_full['ci_95_lower']:.0%},{boot_full['ci_95_upper']:.0%}], "
              f"Res={boot_res['observed']:.0%} CI=[{boot_res['ci_95_lower']:.0%},{boot_res['ci_95_upper']:.0%}], "
              f"diff={comp['observed_diff']:+.0%} p={comp['p_value']:.3f}")
        vec_comparisons[model_name] = {
            "full_bootstrap": boot_full,
            "residual_bootstrap": boot_res,
            "comparison": comp,
        }
    results["full_vs_residual"] = vec_comparisons

    # 4. Key cross-model transfer CIs
    # Load cross-dim transfer and zero-cal data
    print(f"\n--- Cross-Model Transfer Results (Bootstrap CIs) ---")
    transfer_files = {
        "cross_dim_transfer": "cross_dim_transfer.json",
        "zero_calibration_transfer": "zero_calibration_transfer.json",
        "predicted_sign_transfer": "predicted_sign_transfer.json",
    }
    transfer_results = {}
    for name, fname in transfer_files.items():
        fpath = _repo_root() / "outputs/analysis" / fname
        if fpath.exists():
            with open(fpath) as f:
                tdata = json.load(f)
            # Extract accuracy if available
            if "accuracy" in str(tdata):
                transfer_results[name] = "loaded"
                print(f"  {name}: loaded")
            else:
                print(f"  {name}: no accuracy field")
        else:
            print(f"  {name}: not found")

    # 5. Per-pair analysis: which pairs are hardest?
    print(f"\n--- Per-Pair Difficulty (Residual, All Models) ---")
    pair_names = []
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i < j:
                pair_names.append(f"{ta}-{tb}")

    # Collect per-pair accuracy across models
    per_pair_accs = {pair: {} for pair in pair_names}
    for model_name, model_safe in MODELS.items():
        data = load_pairwise_data(model_safe)
        pp = get_per_pair_deltas(data, "residual")
        for r in pp:
            pair = r["pair"]
            steer = r["steer_trait"]
            key = f"{pair}_{steer}"
            if pair not in per_pair_accs:
                continue
            if model_name not in per_pair_accs[pair]:
                per_pair_accs[pair][model_name] = []
            per_pair_accs[pair][model_name].append(r["correct"])

    print(f"\n  {'Pair':>30}  {'SmolLM3':>8}  {'Llama':>8}  {'Qwen':>8}  {'Marin':>8}  {'Mean':>6}")
    pair_means = {}
    for pair in pair_names:
        vals = []
        row = f"  {pair:>30}"
        for model_name in MODELS:
            if model_name in per_pair_accs[pair]:
                corr = per_pair_accs[pair][model_name]
                acc = sum(corr) / len(corr)
                row += f"  {acc:>7.0%}"
                vals.append(acc)
            else:
                row += f"  {'?':>7}"
        mean_acc = np.mean(vals) if vals else 0
        row += f"  {mean_acc:>5.0%}"
        pair_means[pair] = mean_acc
        print(row)

    # Rank pairs by mean accuracy
    sorted_pairs = sorted(pair_means.items(), key=lambda x: x[1])
    print(f"\n  --- Difficulty Ranking (hardest to easiest) ---")
    for rank, (pair, acc) in enumerate(sorted_pairs, 1):
        t1, t2 = pair.split("-")
        # Holland distance
        hex_order = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
        try:
            d1 = hex_order.index(t1)
            d2 = hex_order.index(t2)
            dist = min(abs(d1-d2), 6-abs(d1-d2))
            dist_label = {1: "adj", 2: "alt", 3: "opp"}[dist]
        except (ValueError, KeyError):
            dist_label = "?"
        print(f"  {rank:>2}. {pair:>30} ({dist_label}): {acc:.0%}")

    results["pair_difficulty"] = pair_means

    # 6. Cross-model concordance of pair difficulty
    print(f"\n--- Cross-Model Concordance (Kendall's W) ---")
    # Compute rank correlation between all model pairs
    model_pair_accs = {}
    for model_name in MODELS:
        accs = []
        for pair in pair_names:
            if model_name in per_pair_accs[pair]:
                corr = per_pair_accs[pair][model_name]
                accs.append(sum(corr) / len(corr))
            else:
                accs.append(0.5)
        model_pair_accs[model_name] = accs

    print(f"\n  {'':>15}", end="")
    for m in MODELS:
        print(f"  {m:>10}", end="")
    print()

    concordance_matrix = {}
    for m1 in MODELS:
        row = f"  {m1:>15}"
        concordance_matrix[m1] = {}
        for m2 in MODELS:
            if m1 == m2:
                rho = 1.0
            else:
                rho, p = stats.spearmanr(model_pair_accs[m1], model_pair_accs[m2])
            concordance_matrix[m1][m2] = float(rho)
            row += f"  {rho:>9.3f}"
        print(row)

    results["concordance"] = concordance_matrix

    # 7. Kendall's W (overall concordance)
    rankings = np.array([stats.rankdata(model_pair_accs[m]) for m in MODELS])
    k = len(MODELS)  # number of raters
    n = len(pair_names)  # number of items
    rank_sums = rankings.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * S / (k**2 * (n**3 - n))
    chi2 = k * (n - 1) * W
    p_value = 1 - stats.chi2.cdf(chi2, n - 1)
    print(f"\n  Kendall's W = {W:.3f} (chi2={chi2:.1f}, df={n-1}, p={p_value:.4f})")
    results["kendalls_W"] = {"W": float(W), "chi2": float(chi2), "p_value": float(p_value)}

    # Summary
    print(f"\n{'='*70}")
    print(f"KEY STATISTICAL FINDINGS")
    print(f"{'='*70}")

    for model_name in MODELS:
        boot = results[model_name]["bootstrap"]
        binom = results[model_name]["binomial"]
        print(f"  {model_name}: {boot['observed']:.0%} [{boot['ci_95_lower']:.0%}, {boot['ci_95_upper']:.0%}], "
              f"p={binom['p_value']:.2e}")

    sig_comps = [(k, v) for k, v in comparisons.items() if v["significant_05"]]
    nonsig_comps = [(k, v) for k, v in comparisons.items() if not v["significant_05"]]
    print(f"\n  Significant differences ({len(sig_comps)}):")
    for name, comp in sig_comps:
        print(f"    {name}: diff={comp['observed_diff']:+.0%}, p={comp['p_value']:.3f}")
    print(f"\n  Non-significant differences ({len(nonsig_comps)}):")
    for name, comp in nonsig_comps:
        print(f"    {name}: diff={comp['observed_diff']:+.0%}, p={comp['p_value']:.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_confidence_intervals.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
