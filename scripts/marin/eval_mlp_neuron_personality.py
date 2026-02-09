#!/usr/bin/env python
"""
MLP Neuron Personality Analysis.

We know personality is 93-98% MLP per layer. This experiment goes deeper:
which specific MLP neurons carry personality signal?

1. Capture MLP intermediate activations (gate/up projections) for steered vs baseline
2. Identify neurons with largest personality-correlated activation changes
3. Compute Gini coefficient of personality signal across neurons
4. Test if a small subset of "personality neurons" is sufficient
5. Compare personality neurons across traits — are they shared or distinct?

Marin 8B uses a SwiGLU MLP: gate_proj (4096→14336), up_proj (4096→14336),
down_proj (14336→4096). The intermediate dim is 14336.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="mlp-neurons")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_model_data(model_id, riasec_dir):
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
    _, _, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
        "intermediate_size": getattr(config, "intermediate_size", 14336),
    }


def capture_mlp_internals(model, tokenizer, device, blocks, mid_layer,
                           detect_prompt, steer_vec=None, alpha=0.0,
                           target_layers=None):
    """
    Capture MLP intermediate activations at specified layers.

    For SwiGLU: output = down_proj(act_fn(gate_proj(x)) * up_proj(x))
    We capture:
    - gate_proj output (pre-activation)
    - The activated intermediate (gate * up, post SwiGLU)
    - down_proj output (final MLP output)
    """
    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    if target_layers is None:
        target_layers = [mid_layer, mid_layer + 1]

    mlp_outputs = {}  # layer -> MLP output [hidden_size]
    gate_outputs = {}  # layer -> gate_proj output [intermediate_size]
    up_outputs = {}  # layer -> up_proj output [intermediate_size]
    block_outputs = {}  # layer -> block output [hidden_size]

    hooks = []

    for lidx in target_layers:
        block = blocks[lidx]
        mlp = block.mlp

        def make_mlp_hook(layer_idx):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                mlp_outputs[layer_idx] = hs[0, -1, :].detach().cpu().float().numpy().copy()
                return out
            return hook_fn

        def make_gate_hook(layer_idx):
            def hook_fn(_module, _inp, out):
                gate_outputs[layer_idx] = out[0, -1, :].detach().cpu().float().numpy().copy()
                return out
            return hook_fn

        def make_up_hook(layer_idx):
            def hook_fn(_module, _inp, out):
                up_outputs[layer_idx] = out[0, -1, :].detach().cpu().float().numpy().copy()
                return out
            return hook_fn

        def make_block_hook(layer_idx):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                block_outputs[layer_idx] = hs[0, -1, :].detach().cpu().float().numpy().copy()
                return out
            return hook_fn

        hooks.append(mlp.register_forward_hook(make_mlp_hook(lidx)))
        hooks.append(block.register_forward_hook(make_block_hook(lidx)))

        # Hook gate and up projections
        if hasattr(mlp, "gate_proj"):
            hooks.append(mlp.gate_proj.register_forward_hook(make_gate_hook(lidx)))
        if hasattr(mlp, "up_proj"):
            hooks.append(mlp.up_proj.register_forward_hook(make_up_hook(lidx)))

    # Steering hook
    if steer_vec is not None and alpha != 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return {
        "mlp": mlp_outputs,
        "gate": gate_outputs,
        "up": up_outputs,
        "block": block_outputs,
    }


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]
    intermediate_size = model_data["intermediate_size"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompt = "Tell me about yourself."
    # Focus on layers around injection (mid=16)
    target_layers = [mid_layer - 1, mid_layer, mid_layer + 1, mid_layer + 2]
    alpha = 2.0
    results = {}

    print(f"\n{'='*70}")
    print("MLP NEURON PERSONALITY ANALYSIS")
    print(f"Model: Marin 8B, intermediate_size={intermediate_size}")
    print(f"Target layers: {target_layers}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Baseline vs steered MLP internals
    # ================================================================
    logger.info("Part 1: Capturing baseline and steered MLP activations...")
    print(f"\n{'='*70}")
    print("PART 1: MLP ACTIVATION DIFFERENCES")
    print(f"{'='*70}")

    baseline = capture_mlp_internals(
        model, tokenizer, device, blocks, mid_layer,
        detect_prompt, target_layers=target_layers)

    trait_neuron_diffs = {}

    for test_trait in ["artistic", "investigative", "social"]:
        vec = residual[test_trait].astype(np.float32)

        steered = capture_mlp_internals(
            model, tokenizer, device, blocks, mid_layer,
            detect_prompt, steer_vec=vec, alpha=alpha,
            target_layers=target_layers)

        print(f"\n  {test_trait} (α={alpha}):")

        for lidx in target_layers:
            if lidx not in baseline["gate"] or lidx not in steered["gate"]:
                continue

            gate_diff = steered["gate"][lidx] - baseline["gate"][lidx]
            up_diff = steered["up"][lidx] - baseline["up"][lidx]
            mlp_diff = steered["mlp"][lidx] - baseline["mlp"][lidx]

            # Which neurons changed most?
            gate_abs = np.abs(gate_diff)
            up_abs = np.abs(up_diff)

            # Top neurons by activation change
            gate_top_idx = np.argsort(gate_abs)[::-1][:20]
            up_top_idx = np.argsort(up_abs)[::-1][:20]

            # Concentration metrics
            gate_total = np.sum(gate_abs)
            gate_sorted = np.sort(gate_abs)[::-1]
            gate_cumsum = np.cumsum(gate_sorted) / gate_total if gate_total > 0 else np.zeros(len(gate_sorted))

            # How many neurons needed for 50%, 90%, 99% of the signal?
            n_50 = int(np.searchsorted(gate_cumsum, 0.5)) + 1
            n_90 = int(np.searchsorted(gate_cumsum, 0.9)) + 1
            n_99 = int(np.searchsorted(gate_cumsum, 0.99)) + 1

            # Gini coefficient
            n = len(gate_abs)
            sorted_vals = np.sort(gate_abs)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_vals)) / (n * np.sum(sorted_vals))) - (n+1)/n if np.sum(sorted_vals) > 0 else 0

            print(f"\n    L{lidx}:")
            print(f"      Gate diff: mean={np.mean(gate_abs):.4f}, max={np.max(gate_abs):.4f}")
            print(f"      Neurons for 50%: {n_50}/{intermediate_size} ({n_50/intermediate_size:.1%})")
            print(f"      Neurons for 90%: {n_90}/{intermediate_size} ({n_90/intermediate_size:.1%})")
            print(f"      Neurons for 99%: {n_99}/{intermediate_size} ({n_99/intermediate_size:.1%})")
            print(f"      Gini coefficient: {gini:.3f}")
            print(f"      Top 5 gate neurons: {gate_top_idx[:5].tolist()} "
                  f"(diffs: {gate_abs[gate_top_idx[:5]].tolist()})")

            if test_trait not in trait_neuron_diffs:
                trait_neuron_diffs[test_trait] = {}
            trait_neuron_diffs[test_trait][lidx] = {
                "gate_diff": gate_diff,
                "up_diff": up_diff,
                "gate_gini": float(gini),
                "n_50": int(n_50),
                "n_90": int(n_90),
                "n_99": int(n_99),
                "top20_gate": gate_top_idx[:20].tolist(),
            }

        results[f"{test_trait}_concentration"] = {
            lidx: {
                "gate_gini": trait_neuron_diffs[test_trait][lidx]["gate_gini"],
                "n_50": trait_neuron_diffs[test_trait][lidx]["n_50"],
                "n_90": trait_neuron_diffs[test_trait][lidx]["n_90"],
                "n_99": trait_neuron_diffs[test_trait][lidx]["n_99"],
                "top20_gate": trait_neuron_diffs[test_trait][lidx]["top20_gate"],
            }
            for lidx in trait_neuron_diffs[test_trait]
        }

    # ================================================================
    # PART 2: Cross-trait neuron overlap
    # ================================================================
    logger.info("Part 2: Cross-trait neuron overlap...")
    print(f"\n{'='*70}")
    print("PART 2: CROSS-TRAIT NEURON OVERLAP")
    print(f"{'='*70}")

    for lidx in target_layers:
        traits_with_data = [t for t in trait_neuron_diffs if lidx in trait_neuron_diffs[t]]
        if len(traits_with_data) < 2:
            continue

        print(f"\n  L{lidx}:")

        # Top-100 neuron overlap
        top_k = 100
        top_sets = {}
        for t in traits_with_data:
            gate_abs = np.abs(trait_neuron_diffs[t][lidx]["gate_diff"])
            top_idx = set(np.argsort(gate_abs)[::-1][:top_k].tolist())
            top_sets[t] = top_idx

        # Pairwise Jaccard similarity
        for i, t1 in enumerate(traits_with_data):
            for t2 in traits_with_data[i+1:]:
                intersection = len(top_sets[t1] & top_sets[t2])
                union = len(top_sets[t1] | top_sets[t2])
                jaccard = intersection / union if union > 0 else 0
                print(f"    {t1} vs {t2}: top-{top_k} overlap = {intersection}/{top_k} "
                      f"(Jaccard={jaccard:.3f})")

        # Correlation of activation diffs
        for i, t1 in enumerate(traits_with_data):
            for t2 in traits_with_data[i+1:]:
                diff1 = trait_neuron_diffs[t1][lidx]["gate_diff"]
                diff2 = trait_neuron_diffs[t2][lidx]["gate_diff"]
                r = float(np.corrcoef(diff1, diff2)[0, 1])
                print(f"    {t1} vs {t2}: activation diff correlation r={r:.3f}")

    # ================================================================
    # PART 3: Down-projection 5D analysis
    # ================================================================
    logger.info("Part 3: Down-projection analysis...")
    print(f"\n{'='*70}")
    print("PART 3: WHICH DOWN_PROJ COLUMNS PROJECT INTO 5D?")
    print(f"{'='*70}")

    # For each neuron, compute how much of its down_proj column
    # lies in the 5D personality subspace
    neuron_5d_importance = {}

    for lidx in target_layers:
        block = blocks[lidx]
        down_proj_w = block.mlp.down_proj.weight.detach().cpu().float().numpy()
        # down_proj_w: [hidden_size, intermediate_size]

        importance = np.zeros(intermediate_size)
        for n in range(intermediate_size):
            col = down_proj_w[:, n]  # [hidden_size]
            proj_5d = basis_5d @ col  # [5]
            importance[n] = np.sum(proj_5d**2) / np.sum(col**2) if np.sum(col**2) > 0 else 0

        neuron_5d_importance[lidx] = importance

        # Statistics
        sorted_imp = np.sort(importance)[::-1]
        mean_imp = float(np.mean(importance))
        max_imp = float(np.max(importance))
        top_idx = np.argsort(importance)[::-1][:10]

        print(f"\n  L{lidx}:")
        print(f"    Mean neuron→5D fraction: {mean_imp:.6f}")
        print(f"    Max neuron→5D fraction: {max_imp:.6f} ({max_imp/mean_imp:.1f}× mean)")
        print(f"    Top 10 personality neurons: {top_idx.tolist()}")
        print(f"    Top 10 5D fractions: {importance[top_idx].tolist()}")

        # Gini of down_proj personality importance
        n = len(importance)
        sv = np.sort(importance)
        gini = (2 * np.sum((np.arange(1, n+1) * sv)) / (n * np.sum(sv))) - (n+1)/n if np.sum(sv) > 0 else 0
        print(f"    Gini coefficient: {gini:.3f}")

        results[f"L{lidx}_down_proj_5d"] = {
            "mean": float(mean_imp),
            "max": float(max_imp),
            "max_over_mean": float(max_imp / mean_imp) if mean_imp > 0 else 0,
            "gini": float(gini),
            "top10_neurons": top_idx.tolist(),
            "top10_fractions": importance[top_idx].tolist(),
        }

    # ================================================================
    # PART 4: Cross-layer neuron consistency
    # ================================================================
    logger.info("Part 4: Cross-layer consistency...")
    print(f"\n{'='*70}")
    print("PART 4: SAME NEURONS ACROSS LAYERS?")
    print(f"{'='*70}")

    layers_with_imp = [l for l in target_layers if l in neuron_5d_importance]
    for i, l1 in enumerate(layers_with_imp):
        for l2 in layers_with_imp[i+1:]:
            imp1 = neuron_5d_importance[l1]
            imp2 = neuron_5d_importance[l2]
            r = float(np.corrcoef(imp1, imp2)[0, 1])

            top100_1 = set(np.argsort(imp1)[::-1][:100].tolist())
            top100_2 = set(np.argsort(imp2)[::-1][:100].tolist())
            overlap = len(top100_1 & top100_2)

            print(f"  L{l1} vs L{l2}: importance corr r={r:.3f}, "
                  f"top-100 overlap={overlap}/100")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for test_trait in ["artistic", "investigative", "social"]:
        if test_trait in trait_neuron_diffs:
            for lidx in target_layers:
                if lidx in trait_neuron_diffs[test_trait]:
                    d = trait_neuron_diffs[test_trait][lidx]
                    print(f"  {test_trait} L{lidx}: "
                          f"Gini={d['gate_gini']:.3f}, "
                          f"50%={d['n_50']}, "
                          f"90%={d['n_90']}, "
                          f"99%={d['n_99']}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mlp_neuron_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
