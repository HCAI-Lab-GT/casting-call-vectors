#!/usr/bin/env python
"""
Neuron Ablation: Amplification vs Suppression.

Key finding: ablating neurons that project INTO 5D AMPLIFIES personality.
This suggests those neurons DAMPEN the personality signal.

New tests:
1. Ablate ANTI-personality neurons (those that project AWAY from 5D)
   → Does this suppress personality?
2. Ablate RANDOM neurons as control
   → Establish that the effect is specific to personality-projecting neurons
3. Layer sweep: test ablation at L15-L20 (not just L17)
4. Trait-specific neuron ablation: ablate only artistic-projecting neurons
   → Does this selectively ablate artistic while preserving social?
5. Full-rank ablation: instead of zeroing neurons, project the MLP output
   away from the 5D subspace at L17
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="neuron-amp")

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
    }


def get_neuron_5d_importance(blocks, basis_5d, layer_idx, intermediate_size):
    """Get per-neuron 5D importance via down_proj."""
    block = blocks[layer_idx]
    down_proj_w = block.mlp.down_proj.weight.detach().cpu().float().numpy()
    importance = np.zeros(intermediate_size)
    for n in range(intermediate_size):
        col = down_proj_w[:, n]
        proj_5d = basis_5d @ col
        total = np.sum(col**2)
        importance[n] = np.sum(proj_5d**2) / total if total > 0 else 0
    return importance


def capture_5d_signal(model, tokenizer, device, blocks, mid_layer, basis_5d,
                       coords_5d, detect_prompt, steer_vec=None, alpha=0.0,
                       ablate_neurons=None, ablate_layer=None,
                       project_away_5d=False, project_layer=None):
    """Capture 5D personality signal with optional modifications."""
    capture_layer = mid_layer + 1

    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}
    hooks = []

    # Capture
    def cap_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[capture_layer].register_forward_hook(cap_hook))

    # Steering
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

    # Neuron ablation
    if ablate_neurons is not None and ablate_layer is not None:
        block = blocks[ablate_layer]
        neuron_idx = torch.tensor(ablate_neurons, dtype=torch.long, device=device)
        def ablate_fn(_module, _inp, out):
            out[:, -1, neuron_idx] = 0
            return out
        if hasattr(block.mlp, "gate_proj"):
            hooks.append(block.mlp.gate_proj.register_forward_hook(ablate_fn))

    # Project MLP output away from 5D
    if project_away_5d and project_layer is not None:
        basis_tensor = torch.tensor(basis_5d, dtype=model.dtype).to(device)
        block = blocks[project_layer]
        def proj_hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            mlp_out = hs[:, -1, :]  # [batch, hidden]
            # Project out 5D component
            coords = mlp_out @ basis_tensor.T  # [batch, 5]
            recon = coords @ basis_tensor  # [batch, hidden]
            hs[:, -1, :] = mlp_out - recon
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs
        hooks.append(block.mlp.register_forward_hook(proj_hook))

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured.get("act")


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

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    config = AutoConfig.from_pretrained(model_id)
    intermediate_size = getattr(config, "intermediate_size", 14336)
    detect_prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("NEURON ABLATION: AMPLIFICATION vs SUPPRESSION")
    print(f"Model: Marin 8B, {intermediate_size} neurons/layer")
    print(f"{'='*70}")

    # Get baseline
    baseline_act = capture_5d_signal(
        model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
        detect_prompt)

    def analyze(act, target_trait):
        diff = (act - baseline_act).astype(np.float64)
        coords = basis_5d @ diff
        norm_5d = float(np.linalg.norm(coords))
        sims = {}
        for t in TRAITS:
            if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                sims[t] = float(np.dot(coords, coords_5d[t]) / (
                    norm_5d * np.linalg.norm(coords_5d[t])))
            else:
                sims[t] = 0
        best = max(sims, key=sims.get)
        return {"norm": norm_5d, "detected": best, "cos": sims.get(target_trait, 0)}

    # ================================================================
    # PART 1: Pro-5D vs Anti-5D vs Random ablation
    # ================================================================
    detect_layer = mid_layer + 1
    importance = get_neuron_5d_importance(blocks, basis_5d, detect_layer, intermediate_size)
    sorted_idx = np.argsort(importance)[::-1]  # highest first
    anti_sorted = np.argsort(importance)  # lowest first

    logger.info("Part 1: Pro vs Anti vs Random ablation...")
    print(f"\n{'='*70}")
    print("PART 1: PRO-5D vs ANTI-5D vs RANDOM NEURON ABLATION")
    print(f"{'='*70}")

    n_ablate = 1000
    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0

    # No ablation control
    control_act = capture_5d_signal(
        model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
        detect_prompt, steer_vec=vec, alpha=alpha)
    control_res = analyze(control_act, test_trait)

    # Pro-5D ablation (top neurons by 5D importance)
    pro_neurons = sorted_idx[:n_ablate].tolist()
    pro_act = capture_5d_signal(
        model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
        detect_prompt, steer_vec=vec, alpha=alpha,
        ablate_neurons=pro_neurons, ablate_layer=detect_layer)
    pro_res = analyze(pro_act, test_trait)

    # Anti-5D ablation (bottom neurons by 5D importance)
    anti_neurons = anti_sorted[:n_ablate].tolist()
    anti_act = capture_5d_signal(
        model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
        detect_prompt, steer_vec=vec, alpha=alpha,
        ablate_neurons=anti_neurons, ablate_layer=detect_layer)
    anti_res = analyze(anti_act, test_trait)

    # Random ablation (3 seeds)
    random_results = []
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        rand_neurons = rng.choice(intermediate_size, n_ablate, replace=False).tolist()
        rand_act = capture_5d_signal(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            detect_prompt, steer_vec=vec, alpha=alpha,
            ablate_neurons=rand_neurons, ablate_layer=detect_layer)
        random_results.append(analyze(rand_act, test_trait))

    rand_mean = np.mean([r["norm"] for r in random_results])

    print(f"\n  {test_trait} α={alpha}, ablating {n_ablate} neurons at L{detect_layer}:")
    print(f"    No ablation:  norm={control_res['norm']:.1f}, cos={control_res['cos']:.3f}")
    print(f"    Pro-5D:       norm={pro_res['norm']:.1f}, cos={pro_res['cos']:.3f} "
          f"(change: {(pro_res['norm']/control_res['norm']-1)*100:+.1f}%)")
    print(f"    Anti-5D:      norm={anti_res['norm']:.1f}, cos={anti_res['cos']:.3f} "
          f"(change: {(anti_res['norm']/control_res['norm']-1)*100:+.1f}%)")
    print(f"    Random (mean): norm={rand_mean:.1f} "
          f"(change: {(rand_mean/control_res['norm']-1)*100:+.1f}%)")

    results["pro_vs_anti"] = {
        "n_ablate": n_ablate,
        "control": control_res,
        "pro_5d": pro_res,
        "anti_5d": anti_res,
        "random_mean_norm": float(rand_mean),
    }

    # ================================================================
    # PART 2: Layer sweep for ablation effect
    # ================================================================
    logger.info("Part 2: Layer sweep...")
    print(f"\n{'='*70}")
    print("PART 2: ABLATION AT DIFFERENT LAYERS")
    print(f"{'='*70}")

    layer_sweep_results = {}
    for test_layer in range(mid_layer - 1, min(mid_layer + 5, len(blocks))):
        layer_imp = get_neuron_5d_importance(blocks, basis_5d, test_layer, intermediate_size)
        layer_sorted = np.argsort(layer_imp)[::-1]
        layer_neurons = layer_sorted[:n_ablate].tolist()

        act = capture_5d_signal(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            detect_prompt, steer_vec=vec, alpha=alpha,
            ablate_neurons=layer_neurons, ablate_layer=test_layer)
        res = analyze(act, test_trait)

        change_pct = (res["norm"] / control_res["norm"] - 1) * 100
        print(f"  L{test_layer}: norm={res['norm']:.1f}, cos={res['cos']:.3f} "
              f"(change: {change_pct:+.1f}%)")
        layer_sweep_results[test_layer] = {**res, "change_pct": float(change_pct)}

    results["layer_sweep"] = {str(k): v for k, v in layer_sweep_results.items()}

    # ================================================================
    # PART 3: Direct MLP output projection (remove 5D component)
    # ================================================================
    logger.info("Part 3: MLP output 5D projection...")
    print(f"\n{'='*70}")
    print("PART 3: PROJECT MLP OUTPUT AWAY FROM 5D")
    print(f"{'='*70}")

    for proj_layer in [mid_layer, mid_layer + 1, mid_layer + 2]:
        act = capture_5d_signal(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
            detect_prompt, steer_vec=vec, alpha=alpha,
            project_away_5d=True, project_layer=proj_layer)
        res = analyze(act, test_trait)

        change_pct = (res["norm"] / control_res["norm"] - 1) * 100
        print(f"  Project away 5D at L{proj_layer}: norm={res['norm']:.1f}, "
              f"cos={res['cos']:.3f} (change: {change_pct:+.1f}%)")

    # ================================================================
    # PART 4: Trait-selective ablation
    # ================================================================
    logger.info("Part 4: Trait-selective ablation...")
    print(f"\n{'='*70}")
    print("PART 4: TRAIT-SELECTIVE ABLATION")
    print(f"{'='*70}")

    # Get trait-specific neuron importance
    down_proj_w = blocks[detect_layer].mlp.down_proj.weight.detach().cpu().float().numpy()

    for steer_trait in ["artistic", "social"]:
        trait_dir = coords_5d[steer_trait]
        trait_dir_norm = trait_dir / np.linalg.norm(trait_dir)

        # Neuron importance for this specific trait direction
        trait_importance = np.zeros(intermediate_size)
        for n in range(intermediate_size):
            col = down_proj_w[:, n]
            proj_5d = basis_5d @ col
            trait_proj = np.dot(trait_dir_norm, proj_5d)
            trait_importance[n] = trait_proj**2

        trait_sorted = np.argsort(trait_importance)[::-1]
        trait_neurons = trait_sorted[:n_ablate].tolist()

        print(f"\n  Ablating top-{n_ablate} {steer_trait}-specific neurons:")

        for test_steer in ["artistic", "social"]:
            test_vec = residual[test_steer].astype(np.float32)
            act = capture_5d_signal(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                detect_prompt, steer_vec=test_vec, alpha=alpha,
                ablate_neurons=trait_neurons, ablate_layer=detect_layer)

            # Get reference
            ref_act = capture_5d_signal(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                detect_prompt, steer_vec=test_vec, alpha=alpha)
            ref_res = analyze(ref_act, test_steer)
            ablated_res = analyze(act, test_steer)

            change_pct = (ablated_res["norm"] / ref_res["norm"] - 1) * 100
            print(f"    Steer {test_steer}: "
                  f"norm {ref_res['norm']:.1f}→{ablated_res['norm']:.1f} ({change_pct:+.1f}%), "
                  f"cos {ref_res['cos']:.3f}→{ablated_res['cos']:.3f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Control (no ablation): norm={control_res['norm']:.1f}")
    print(f"  Pro-5D ablation: norm={pro_res['norm']:.1f} ({(pro_res['norm']/control_res['norm']-1)*100:+.1f}%)")
    print(f"  Anti-5D ablation: norm={anti_res['norm']:.1f} ({(anti_res['norm']/control_res['norm']-1)*100:+.1f}%)")
    print(f"  Random ablation: norm={rand_mean:.1f} ({(rand_mean/control_res['norm']-1)*100:+.1f}%)")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "neuron_amplifier.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
