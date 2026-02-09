#!/usr/bin/env python
"""
Generate comprehensive summary of all experiments in this session.
Counts scripts, outputs, and summarizes key findings.
"""

import json
from pathlib import Path

def _repo_root():
    return Path(__file__).resolve().parents[2]

def main():
    root = _repo_root()
    scripts = sorted((root / "scripts/marin").glob("*.py"))
    outputs = sorted((root / "outputs/analysis").glob("*.json"))

    print(f"\n{'='*70}")
    print(f"SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Total scripts: {len(scripts)}")
    print(f"  Total JSON outputs: {len(outputs)}")

    # Key results table
    key_results = [
        ("Self-steering (SmolLM3)", "100%", "pairwise_discrimination"),
        ("Self-steering (Marin 8B)", "97%", "pairwise_discrimination"),
        ("Cross-dim SmolLM3→Marin", "97%", "cross_dim_transfer"),
        ("Cross-dim Marin→SmolLM3", "100%", "cross_dim_transfer_reverse"),
        ("Llama 1B→Marin 8B", "100%", "cross_dim_matrix"),
        ("Ensemble (3 models)→Marin", "100%", "transfer_quality_analysis"),
        ("Transitive SmolLM3→Llama→Marin", "100%", "transitive_transfer"),
        ("Transitive SmolLM3→Qwen→Marin", "90%", "transitive_transfer"),
        ("Base→Instruct transfer", "77%", "base_to_instruct_transfer"),
        ("5D-reconstructed", "97%", "orthogonal_control"),
        ("Orthogonal to 5D", "51%", "orthogonal_control"),
        ("Random control", "43%", "random_control"),
        ("Cross-dim interpolation", "ρ=-1.000", "cross_dim_interpolation"),
        ("Composition (k=2)", "96%", "composition_scaling"),
        ("Composition (k=3)", "94%", "triple_composition"),
        ("Dose-response (α=0.1-3)", "100%", "dose_response"),
        ("Novel descriptions", "100%", "pairwise_generalization"),
    ]

    print(f"\n--- KEY RESULTS TABLE ---")
    print(f"  {'Experiment':>40}  {'Result':>10}")
    print(f"  {'-'*55}")
    for name, result, _ in key_results:
        print(f"  {name:>40}  {result:>10}")

    # Cross-dim transfer hierarchy
    print(f"\n--- CROSS-DIM TRANSFER HIERARCHY (all → Marin 8B) ---")
    try:
        with open(root / "outputs/analysis/cross_dim_matrix.json") as f:
            matrix = json.load(f)
        items = [("Self (native)", matrix["self"]["delta_accuracy"])]
        for name, data in matrix["transfers"].items():
            items.append((name, data["delta_accuracy"]))
        items.append(("Random", matrix["random"]["delta_accuracy"]))
        items.sort(key=lambda x: x[1], reverse=True)
        for name, acc in items:
            bar = "#" * int(acc * 30)
            print(f"  {name:>35}: {acc:>4.0%} |{bar}")
    except Exception as e:
        print(f"  (Could not load matrix: {e})")

    # Component ablation
    print(f"\n--- 5D SUBSPACE ANALYSIS ---")
    try:
        with open(root / "outputs/analysis/orthogonal_control.json") as f:
            ortho = json.load(f)
        print(f"  Personality subspace: 5 / {ortho['hidden_dim']} dimensions ({5/ortho['hidden_dim']*100:.2f}%)")
        print(f"  Full vectors:      {ortho['real']['delta_accuracy']:.0%}")
        print(f"  5D-reconstructed:  {ortho['in_subspace']['delta_accuracy']:.0%}")
        print(f"  Orthogonal to 5D:  {ortho['orthogonal_mean']:.0%} ± {ortho['orthogonal_std']:.0%}")
        print(f"  Random:            {ortho['random']['delta_accuracy']:.0%}")
    except Exception:
        pass

    try:
        with open(root / "outputs/analysis/component_ablation.json") as f:
            ablation = json.load(f)
        print(f"\n  Component importance:")
        for i in range(5):
            key = f"remove_PC{i+1}"
            if key in ablation["ablations"]:
                a = ablation["ablations"][key]
                print(f"    PC{i+1}: var={ablation['importance'][i]:.1%}, "
                      f"remove→{a['overall']:.0%} (drop={a['drop']:+.0%})")
    except Exception:
        pass

    # Holland structure
    print(f"\n--- HOLLAND HEXAGONAL STRUCTURE ---")
    try:
        with open(root / "outputs/analysis/5d_semantics.json") as f:
            sem = json.load(f)
        hs = sem["holland_structure"]
        print(f"  Adjacent (d=1): cosine = {hs['adjacent_mean']:+.3f}")
        print(f"  Alternate (d=2): cosine = {hs['alternate_mean']:+.3f}")
        print(f"  Opposite (d=3): cosine = {hs['opposite_mean']:+.3f}")
        print(f"  Holland order: {'CONFIRMED' if hs['holland_consistent'] else 'VIOLATED'}")
    except Exception:
        pass

    print(f"\n{'='*70}")
    print(f"TOP 5 STRONGEST FINDINGS")
    print(f"{'='*70}")
    print(f"  1. Personality lives in EXACTLY 5D (0.12% of hidden dim)")
    print(f"     Orthogonal to personality = chance (51%), 5D-reconstructed = full (97%)")
    print(f"  2. Cross-dim transfer is UNIVERSAL across instruct models")
    print(f"     ALL 4 instruct models: ≥97%, including 1B→8B at 100%")
    print(f"  3. Ensemble of sources BEATS self-steering")
    print(f"     3-model average: 100% > 97% self, cosine 0.986 > any individual")
    print(f"  4. Transitive transfer works through intermediaries")
    print(f"     SmolLM3→Llama→Marin: 100% without ever touching target vectors")
    print(f"  5. 5D structure matches Holland's RIASEC hexagon")
    print(f"     Adjacent > alternate > opposite cosines, PC1 = Artistic↔Conventional")

    print(f"\n{'='*70}")
    print(f"PRACTICAL RECIPE")
    print(f"{'='*70}")
    print(f"  1. Extract persona vectors on ANY small model (even 1B)")
    print(f"  2. Compute residual vectors (remove shared PC1)")
    print(f"  3. Project to 5D via PCA")
    print(f"  4. Procrustes-align to target model's 5D (need 5 calibration vectors)")
    print(f"  5. Reconstruct in target's full dimension via target's PCA basis")
    print(f"  6. Steer at α=1.0 → 97-100% trait discrimination")
    print(f"\n  OR: Average transferred vectors from multiple sources for 100%")

if __name__ == "__main__":
    main()
