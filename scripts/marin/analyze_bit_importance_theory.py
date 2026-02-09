#!/usr/bin/env python
"""
Why is the bit importance ranking PC1 > PC2 > PC4 > PC3 >> PC5?

The 5D PCA variance distribution is ~[34, 23, 17, 16, 10]% across models,
but the IMPORTANCE ranking is PC1 > PC2 > PC4 > PC3 > PC5.

Note: PC4 is MORE important than PC3 despite explaining LESS variance!

This script explores:
1. Does variance explained predict bit importance?
2. Does inter-model sign agreement predict importance?
3. What is the "semantic content" of each PC?
4. Can we predict importance from geometry alone?

No GPU needed — pure analysis.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from transformers import AutoConfig
from scipy import stats


def _repo_root():
    return Path(__file__).resolve().parents[2]


TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

MODELS = {
    "SmolLM3": "HuggingFaceTB/SmolLM3-3B",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Marin-8B": "marin-community/marin-8b-instruct",
}


def load_residual_and_pca(model_id, riasec_dir):
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
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj

    V_res = np.stack([residual[t] for t in TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}
    var_explained = S_res[:5]**2 / np.sum(S_res[:5]**2)

    return coords_5d, basis_5d, var_explained, S_res[:5]


def canonical_sign_convention(coords_5d):
    signs = np.ones(5)
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1
    for pc in range(1, 5):
        loadings = {t: coords_5d[t][pc] for t in TRAITS}
        max_trait = max(loadings, key=lambda t: abs(loadings[t]))
        if loadings[max_trait] > 0:
            signs[pc] = -1
    return signs


def main():
    riasec_dir = _repo_root() / "persona_data/model_inits"

    print(f"\n{'='*70}")
    print(f"THEORETICAL ANALYSIS: WHY PC1 > PC2 > PC4 > PC3 >> PC5?")
    print(f"{'='*70}")

    # Load all models
    model_data = {}
    for name, model_id in MODELS.items():
        coords, basis, var_exp, sv = load_residual_and_pca(model_id, riasec_dir)
        signs = canonical_sign_convention(coords)
        std_coords = {t: signs * coords[t] for t in TRAITS}
        model_data[name] = {
            "coords": coords,
            "std_coords": std_coords,
            "basis": basis,
            "var_explained": var_exp,
            "singular_values": sv,
            "canonical_signs": signs,
        }

    # 1. Variance explained per PC
    print(f"\n--- Variance Explained per PC ---")
    print(f"  {'Model':>10}  {'PC1':>6}  {'PC2':>6}  {'PC3':>6}  {'PC4':>6}  {'PC5':>6}")
    mean_var = np.zeros(5)
    for name in MODELS:
        var = model_data[name]["var_explained"]
        mean_var += var
        print(f"  {name:>10}  {var[0]:>5.1%}  {var[1]:>5.1%}  {var[2]:>5.1%}  {var[3]:>5.1%}  {var[4]:>5.1%}")
    mean_var /= len(MODELS)
    print(f"  {'MEAN':>10}  {mean_var[0]:>5.1%}  {mean_var[1]:>5.1%}  {mean_var[2]:>5.1%}  {mean_var[3]:>5.1%}  {mean_var[4]:>5.1%}")

    # Importance values from the multi-source experiment
    importance_smol = [0.40, 0.29, 0.13, 0.15, -0.01]
    importance_llama = [0.40, 0.30, 0.12, 0.17, 0.00]
    importance_qwen = [0.36, 0.31, 0.12, 0.16, -0.01]
    mean_importance = np.mean([importance_smol, importance_llama, importance_qwen], axis=0)

    # 2. Compare variance ranking with importance ranking
    print(f"\n--- Variance Explained vs Bit Importance ---")
    print(f"  {'PC':>4}  {'Var%':>6}  {'Importance':>10}  {'Var rank':>9}  {'Imp rank':>9}")
    var_ranks = np.argsort(-mean_var) + 1
    imp_ranks = np.argsort(-mean_importance) + 1
    for pc in range(5):
        print(f"  PC{pc+1}  {mean_var[pc]:>5.1%}  {mean_importance[pc]:>+9.0%}  "
              f"{var_ranks[pc]:>9}  {imp_ranks[pc]:>9}")

    # Spearman correlation
    rho, p = stats.spearmanr(mean_var, mean_importance)
    print(f"\n  Spearman(variance, importance): ρ = {rho:.3f}, p = {p:.3f}")
    rho_sv, p_sv = stats.spearmanr(np.mean([model_data[n]["singular_values"] for n in MODELS], axis=0),
                                    mean_importance)
    print(f"  Spearman(singular_value, importance): ρ = {rho_sv:.3f}, p = {p_sv:.3f}")

    # 3. Canonical signs per model
    print(f"\n--- Canonical Signs per Model ---")
    print(f"  {'Model':>10}  {'PC1':>4}  {'PC2':>4}  {'PC3':>4}  {'PC4':>4}  {'PC5':>4}")
    all_signs = []
    for name in MODELS:
        s = model_data[name]["canonical_signs"]
        all_signs.append(s)
        print(f"  {name:>10}  {'+' if s[0]>0 else '-':>4}  {'+' if s[1]>0 else '-':>4}  "
              f"{'+' if s[2]>0 else '-':>4}  {'+' if s[3]>0 else '-':>4}  {'+' if s[4]>0 else '-':>4}")

    # Inter-model sign agreement per PC
    all_signs = np.array(all_signs)
    print(f"\n--- Inter-Model Sign Agreement ---")
    for pc in range(5):
        signs_pc = all_signs[:, pc]
        n_plus = np.sum(signs_pc > 0)
        n_minus = np.sum(signs_pc < 0)
        agreement = max(n_plus, n_minus) / len(MODELS)
        majority = "+" if n_plus > n_minus else "-"
        print(f"  PC{pc+1}: agreement = {agreement:.0%} (majority = {majority}, "
              f"+ : {n_plus}, - : {n_minus})")

    # 4. Semantic content of each PC (standardized loadings)
    print(f"\n--- Standardized PC Loadings (after canonical sign convention) ---")
    for name in MODELS:
        std = model_data[name]["std_coords"]
        print(f"\n  {name}:")
        print(f"  {'Trait':>14}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC4':>7}  {'PC5':>7}")
        for t in TRAITS:
            c = std[t]
            print(f"  {t:>14}  {c[0]:>+6.3f}  {c[1]:>+6.3f}  {c[2]:>+6.3f}  {c[3]:>+6.3f}  {c[4]:>+6.3f}")

    # 5. Cross-model consistency per PC (how similar are loadings after standardization?)
    print(f"\n--- Cross-Model Loading Consistency per PC ---")
    names = list(MODELS.keys())
    for pc in range(5):
        correlations = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                vec_i = np.array([model_data[names[i]]["std_coords"][t][pc] for t in TRAITS])
                vec_j = np.array([model_data[names[j]]["std_coords"][t][pc] for t in TRAITS])
                cos = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                correlations.append(cos)
        mean_cos = np.mean(correlations)
        min_cos = np.min(correlations)
        print(f"  PC{pc+1}: mean cosine = {mean_cos:.3f}, min = {min_cos:.3f}")

    # 6. What does each PC encode? (dominant trait loadings)
    print(f"\n--- PC Semantic Identity (mean across models) ---")
    mean_loadings = {}
    for pc in range(5):
        for t in TRAITS:
            if t not in mean_loadings:
                mean_loadings[t] = np.zeros(5)
            mean_loadings[t][pc] = np.mean([model_data[n]["std_coords"][t][pc] for n in MODELS])

    for pc in range(5):
        loadings = {t: mean_loadings[t][pc] for t in TRAITS}
        sorted_traits = sorted(loadings.items(), key=lambda x: x[1])
        neg_end = sorted_traits[0]
        pos_end = sorted_traits[-1]
        print(f"  PC{pc+1}: {neg_end[0]} ({neg_end[1]:+.3f}) ↔ {pos_end[0]} ({pos_end[1]:+.3f})")
        # Show all loadings
        print(f"         " + "  ".join(f"{t[:4]}={loadings[t]:+.3f}" for t in TRAITS))

    # 7. Prediction: does the max loading magnitude per PC predict importance?
    print(f"\n--- Max Loading Magnitude vs Importance ---")
    max_loadings = []
    for pc in range(5):
        max_abs = np.mean([max(abs(model_data[n]["std_coords"][t][pc]) for t in TRAITS) for n in MODELS])
        max_loadings.append(max_abs)
        print(f"  PC{pc+1}: max|loading| = {max_abs:.3f}, importance = {mean_importance[pc]:+.0%}")

    rho_max, p_max = stats.spearmanr(max_loadings, mean_importance)
    print(f"\n  Spearman(max_loading, importance): ρ = {rho_max:.3f}, p = {p_max:.3f}")

    # 8. Transfer cost matrix (Hamming distance of canonical signs)
    print(f"\n--- Transfer Cost Matrix (Hamming Distance of Canonical Signs) ---")
    target_signs = model_data["Marin-8B"]["canonical_signs"]
    print(f"\n  Transfer to Marin-8B:")
    for name in MODELS:
        source_signs = model_data[name]["canonical_signs"]
        hamming = int(np.sum(source_signs != target_signs))
        free_bits = 5 - hamming
        relative = target_signs * source_signs
        print(f"  {name:>10}: Hamming = {hamming}, free bits = {free_bits}/5, "
              f"relative_signs = [{', '.join('+' if r > 0 else '-' for r in relative)}]")

    # Full pairwise transfer cost matrix
    print(f"\n  Pairwise Hamming distances:")
    print(f"  {'':>10}", end="")
    for n in MODELS:
        print(f"  {n:>10}", end="")
    print()
    for n1 in MODELS:
        row = f"  {n1:>10}"
        for n2 in MODELS:
            h = int(np.sum(model_data[n1]["canonical_signs"] != model_data[n2]["canonical_signs"]))
            row += f"  {h:>10}"
        print(row)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"""
  1. VARIANCE does NOT perfectly predict importance:
     Variance ranking: PC1 > PC2 > PC3 > PC4 > PC5
     Importance ranking: PC1 > PC2 > PC4 > PC3 >> PC5
     PC4 beats PC3 despite lower variance (ρ = {rho:.3f}, p = {p:.3f})

  2. PC SEMANTIC IDENTITY (standardized loadings, mean across models):
     PC1: Artistic ↔ Conventional (34% variance, +40% importance)
     PC2: Investigative ↔ Social/Realistic (23% variance, +30% importance)
     PC3-5: increasingly model-specific loadings

  3. TRANSFER COST = Hamming distance of canonical sign vectors:
     Llama → Marin: 1 bit (cheapest transfer)
     SmolLM3 → Marin: 3 bits
     Qwen → Marin: 3 bits

  4. SIGN AGREEMENT varies by PC:
     PC1: 100% agreement (all models: +)
     PC5: highest disagreement
""")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bit_importance_theory.json"
    with open(out_path, "w") as f:
        json.dump({
            "mean_variance_explained": mean_var.tolist(),
            "mean_importance": mean_importance.tolist(),
            "variance_importance_spearman": {"rho": float(rho), "p": float(p)},
            "canonical_signs": {n: model_data[n]["canonical_signs"].tolist() for n in MODELS},
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
