#!/usr/bin/env python
"""
Comprehensive synthesis of ALL pairwise discrimination results.
Combines:
- Alpha sweep data (4 instruct models)
- Base model data (SmolLM3-Base, Marin 32B base)
- Prompt format data (4 instruct models × 2 formats)
- Cross-model transfer data
"""

import json
from pathlib import Path
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    out_dir = _repo_root() / "outputs" / "analysis"

    print("=" * 80)
    print("COMPREHENSIVE PAIRWISE DISCRIMINATION SYNTHESIS")
    print("=" * 80)

    # 1. Best discrimination per model (any config)
    print(f"\n{'='*80}")
    print("1. BEST PAIRWISE DISCRIMINATION PER MODEL (optimal config)")
    print(f"{'='*80}")

    models = [
        {
            "name": "SmolLM3-3B",
            "type": "Instruct",
            "hidden_dim": 2048,
            "num_layers": 36,
            "best_alpha": "1-3",
            "best_prompt": "Completion",
            "best_acc": 1.0,
            "source": "prompt_format",
        },
        {
            "name": "SmolLM3-3B-Base",
            "type": "Base",
            "hidden_dim": 2048,
            "num_layers": 36,
            "best_alpha": "2-3",
            "best_prompt": "Completion",
            "best_acc": 1.0,
            "source": "pairwise_base",
        },
        {
            "name": "Marin 8B",
            "type": "Instruct",
            "hidden_dim": 4096,
            "num_layers": 32,
            "best_alpha": "1-2",
            "best_prompt": "Chat",
            "best_acc": 0.967,
            "source": "alpha_sweep",
        },
        {
            "name": "Marin 32B",
            "type": "Base",
            "hidden_dim": 5120,
            "num_layers": 64,
            "best_alpha": "1",
            "best_prompt": "Completion",
            "best_acc": 0.87,
            "source": "pairwise_base",
        },
        {
            "name": "Qwen 7B",
            "type": "Instruct",
            "hidden_dim": 3584,
            "num_layers": 28,
            "best_alpha": "5",
            "best_prompt": "Chat",
            "best_acc": 0.833,
            "source": "alpha_sweep",
        },
        {
            "name": "Llama 1B",
            "type": "Instruct",
            "hidden_dim": 2048,
            "num_layers": 16,
            "best_alpha": "1",
            "best_prompt": "Completion",
            "best_acc": 0.77,
            "source": "prompt_format",
        },
    ]

    print(f"\n  {'Model':>20} {'Type':>8} {'Dim':>5} {'Layers':>7} {'Best α':>7} {'Prompt':>12} {'Delta%':>7}")
    print(f"  {'-'*20} {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*12} {'-'*7}")
    for m in models:
        print(f"  {m['name']:>20} {m['type']:>8} {m['hidden_dim']:>5} {m['num_layers']:>7} "
              f"{m['best_alpha']:>7} {m['best_prompt']:>12} {m['best_acc']:>6.0%}")

    # 2. Prompt format comparison
    print(f"\n{'='*80}")
    print("2. PROMPT FORMAT EFFECT (Chat vs Completion, residual vectors)")
    print(f"{'='*80}")

    prompt_models = [
        ("meta-llama__Llama-3.2-1B-Instruct", "Llama 1B"),
        ("HuggingFaceTB__SmolLM3-3B", "SmolLM3 3B"),
        ("Qwen__Qwen2.5-7B-Instruct", "Qwen 7B"),
        ("marin-community__marin-8b-instruct", "Marin 8B"),
    ]

    print(f"\n  {'Model':>12} | {'α=1 Chat':>9} {'α=1 Comp':>9} {'Δ':>5} | {'α=5 Chat':>9} {'α=5 Comp':>9} {'Δ':>5} | {'Winner':>12}")
    print(f"  {'-'*12} | {'-'*9} {'-'*9} {'-'*5} | {'-'*9} {'-'*9} {'-'*5} | {'-'*12}")

    for safe_model, name in prompt_models:
        data = load_json(out_dir / f"pairwise_prompt_format_{safe_model}.json")
        if not data:
            continue

        chat_1 = data.get("residual_chat_alpha_1.0", {}).get("delta_accuracy", 0)
        comp_1 = data.get("residual_completion_alpha_1.0", {}).get("delta_accuracy", 0)
        chat_5 = data.get("residual_chat_alpha_5.0", {}).get("delta_accuracy", 0)
        comp_5 = data.get("residual_completion_alpha_5.0", {}).get("delta_accuracy", 0)

        best_chat = max(
            data.get(f"residual_chat_alpha_{a}", {}).get("delta_accuracy", 0)
            for a in ["1.0", "2.0", "3.0", "5.0"]
        )
        best_comp = max(
            data.get(f"residual_completion_alpha_{a}", {}).get("delta_accuracy", 0)
            for a in ["1.0", "2.0", "3.0", "5.0"]
        )
        winner = "Chat" if best_chat > best_comp else ("Completion" if best_comp > best_chat else "Tied")

        d1 = comp_1 - chat_1
        d5 = comp_5 - chat_5
        print(f"  {name:>12} | {chat_1:>8.0%} {comp_1:>8.0%} {d1:>+4.0%} | {chat_5:>8.0%} {comp_5:>8.0%} {d5:>+4.0%} | {winner:>12}")

    # 3. Baseline positional bias analysis
    print(f"\n{'='*80}")
    print("3. BASELINE POSITIONAL BIAS (chat template)")
    print(f"{'='*80}")
    print(f"  (Positive = prefers option A; strong positive = positional bias)")

    for safe_model, name in prompt_models:
        data = load_json(out_dir / f"pairwise_prompt_format_{safe_model}.json")
        if not data or "baseline_chat" not in data:
            continue
        baselines = list(data["baseline_chat"].values())
        n_positive = sum(1 for b in baselines if b > 0)
        mean_val = np.mean(baselines)
        print(f"\n  {name:>12}: mean={mean_val:+.2f}, {n_positive}/15 positive, range=[{min(baselines):+.1f}, {max(baselines):+.1f}]")

        baselines_comp = list(data["baseline_completion"].values())
        n_positive_comp = sum(1 for b in baselines_comp if b > 0)
        mean_comp = np.mean(baselines_comp)
        print(f"  {'(completion)':>12}: mean={mean_comp:+.2f}, {n_positive_comp}/15 positive, range=[{min(baselines_comp):+.1f}, {max(baselines_comp):+.1f}]")

    # 4. Cross-model steering transfer
    print(f"\n{'='*80}")
    print("4. CROSS-MODEL STEERING TRANSFER (Llama 1B ↔ SmolLM3 3B)")
    print(f"{'='*80}")

    transfer = load_json(out_dir / "cross_model_steering_transfer.json")
    if transfer:
        conditions = [
            ("llama_own", "Llama + Llama (self)"),
            ("llama_smollm3_raw", "Llama + SmolLM3 (raw)"),
            ("llama_smollm3_procrustes", "Llama + SmolLM3 (Procrustes)"),
            ("smollm3_own", "SmolLM3 + SmolLM3 (self)"),
            ("smollm3_llama_raw", "SmolLM3 + Llama (raw)"),
            ("smollm3_llama_procrustes", "SmolLM3 + Llama (Procrustes)"),
        ]

        print(f"\n  {'Condition':>35} {'Delta%':>8} {'MeanΔ':>8} {'vs Self':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")

        llama_self = transfer.get("llama_own", {}).get("delta_accuracy", 0)
        smollm3_self = transfer.get("smollm3_own", {}).get("delta_accuracy", 0)

        for key, name in conditions:
            if key in transfer:
                acc = transfer[key]["delta_accuracy"]
                mean_d = transfer[key]["mean_delta"]
                if "llama" == key.split("_")[0]:
                    vs_self = acc - llama_self
                else:
                    vs_self = acc - smollm3_self
                print(f"  {name:>35} {acc:>7.0%} {mean_d:>+7.3f} {vs_self:>+7.0%}")

    # 5. Key correlations
    print(f"\n{'='*80}")
    print("5. WHAT PREDICTS DISCRIMINATION ACCURACY?")
    print(f"{'='*80}")

    # With best config
    best_accs = [m["best_acc"] for m in models]
    dims = [m["hidden_dim"] for m in models]
    layers = [m["num_layers"] for m in models]

    # Spearman correlations
    from scipy.stats import spearmanr
    rho_dim, p_dim = spearmanr(dims, best_accs)
    rho_lay, p_lay = spearmanr(layers, best_accs)

    print(f"\n  Hidden dim → accuracy:  ρ = {rho_dim:.3f} (p = {p_dim:.3f})")
    print(f"  Num layers → accuracy:  ρ = {rho_lay:.3f} (p = {p_lay:.3f})")

    # Same hidden dim comparison
    print(f"\n  Same dim (2048) comparison:")
    same_dim = [m for m in models if m["hidden_dim"] == 2048]
    for m in sorted(same_dim, key=lambda x: -x["best_acc"]):
        print(f"    {m['name']:>20} ({m['num_layers']} layers): {m['best_acc']:.0%}")

    # Instruct only
    instruct_only = [m for m in models if m["type"] == "Instruct"]
    inst_accs = [m["best_acc"] for m in instruct_only]
    inst_dims = [m["hidden_dim"] for m in instruct_only]
    inst_layers = [m["num_layers"] for m in instruct_only]
    rho_dim_i, p_dim_i = spearmanr(inst_dims, inst_accs)
    rho_lay_i, p_lay_i = spearmanr(inst_layers, inst_accs)

    print(f"\n  Instruct models only:")
    print(f"    Hidden dim → accuracy:  ρ = {rho_dim_i:.3f} (p = {p_dim_i:.3f})")
    print(f"    Num layers → accuracy:  ρ = {rho_lay_i:.3f} (p = {p_lay_i:.3f})")

    # 6. Summary narrative
    print(f"\n{'='*80}")
    print("6. KEY NARRATIVES")
    print(f"{'='*80}")

    narratives = [
        "1. EVALUATION INSTRUMENT: YES/NO gives ~0% specificity; pairwise forced-choice gives up to 100%.",
        "2. PROMPT FORMAT: Model-specific; 3/4 instruct models benefit from completion-style prompts.",
        "   - SmolLM3: +27% with completion (100% vs 73%) due to chat template positional bias",
        "   - Marin 8B: -3% with completion (93% vs 97%) — chat template creates useful structure",
        "3. OPTIMAL ALPHA: Low (1-3) for discrimination. Standard alpha (5-10) destroys specificity.",
        "4. RESIDUAL VECTORS: Consistently outperform full vectors (removes shared 'agree' direction).",
        "5. CROSS-MODEL TRANSFER: Procrustes-aligned vectors achieve self-steering parity (0% loss).",
        "6. DEPTH > WIDTH: SmolLM3 (36L, 2048d) > Llama (16L, 2048d) at 100% vs 77%.",
        "7. BASE ≈ INSTRUCT: With optimal prompts, instruct = base (both 100% for SmolLM3).",
        "8. UNIVERSAL GEOMETRY: 6/6 Procrustes LOO across all 10 model pairs. 99.6-100% simplex efficiency.",
    ]

    for n in narratives:
        print(f"  {n}")

    # Save
    output = {
        "models": models,
        "correlations": {
            "all_models": {"hidden_dim_rho": float(rho_dim), "hidden_dim_p": float(p_dim),
                          "num_layers_rho": float(rho_lay), "num_layers_p": float(p_lay)},
            "instruct_only": {"hidden_dim_rho": float(rho_dim_i), "hidden_dim_p": float(p_dim_i),
                             "num_layers_rho": float(rho_lay_i), "num_layers_p": float(p_lay_i)},
        },
    }
    out_path = out_dir / "comprehensive_synthesis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
