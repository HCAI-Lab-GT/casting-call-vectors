#!/usr/bin/env python
"""
Weight-Space Personality Injection.

Instead of runtime hooks, modify model weights directly:
1. Add a rank-1 update to a specific weight matrix (e.g., down_proj at L16)
   that permanently encodes personality
2. Test if the model produces personality-consistent outputs WITHOUT hooks
3. Compare weight-space injection vs activation-space steering
4. Test if weight modification is reversible (subtract to remove)
5. Compose multiple trait modifications in weight space

This tests the fundamental question: is personality a linear perturbation
that can be absorbed into the model's weights?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="weight-pers")

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


def capture_5d_from_activation(model, tokenizer, device, blocks, mid_layer, basis_5d,
                                 coords_5d, prompt, steer_vec=None, alpha=0.0):
    """Get 5D signal via activation capture (baseline method)."""
    capture_layer = mid_layer + 1
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}
    hooks = []

    def cap_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[capture_layer].register_forward_hook(cap_hook))

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

    return captured.get("act")


def generate_text(model, tokenizer, device, prompt, max_tokens=50):
    """Generate text from the model."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def analyze_5d(act, baseline_act, basis_5d, coords_5d, target_trait):
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
    detected = max(sims, key=sims.get)
    return {"norm": norm_5d, "detected": detected, "correct": detected == target_trait,
            "cos_target": sims[target_trait]}


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

    detect_prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("WEIGHT-SPACE PERSONALITY INJECTION")
    print(f"Model: Marin 8B, injection layer L{mid_layer}")
    print(f"{'='*70}")

    # Get baseline
    baseline_act = capture_5d_from_activation(
        model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d, detect_prompt)

    # ================================================================
    # PART 1: Bias injection — add personality vector directly to layer bias
    # ================================================================
    logger.info("Part 1: Bias injection...")
    print(f"\n{'='*70}")
    print("PART 1: BIAS INJECTION (add vector to layer output bias)")
    print(f"{'='*70}")

    # Strategy: modify the self_attn output projection bias or add a bias
    # to the block's output. Simplest: modify the MLP's down_proj bias.
    # Most LLMs don't have bias, so we'll add one temporarily.

    inject_layer = mid_layer
    block = blocks[inject_layer]
    mlp = block.mlp

    bias_results = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)

        for alpha in [1.0, 2.0]:
            # Save original bias state
            had_bias = mlp.down_proj.bias is not None
            if had_bias:
                orig_bias = mlp.down_proj.bias.data.clone()

            # Add personality as bias
            bias_val = alpha * torch.tensor(vec, dtype=model.dtype).to(device)
            if had_bias:
                mlp.down_proj.bias.data += bias_val
            else:
                mlp.down_proj.bias = torch.nn.Parameter(bias_val)

            # Capture signal
            act = capture_5d_from_activation(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d, detect_prompt)
            res = analyze_5d(act, baseline_act, basis_5d, coords_5d, trait)

            # Also capture activation-based reference
            # Restore first
            if had_bias:
                mlp.down_proj.bias.data = orig_bias
            else:
                mlp.down_proj.bias = None

            ref_act = capture_5d_from_activation(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                detect_prompt, steer_vec=vec, alpha=alpha)
            ref_res = analyze_5d(ref_act, baseline_act, basis_5d, coords_5d, trait)

            print(f"  {trait} α={alpha}: "
                  f"weight-space: detected={res['detected']}, cos={res['cos_target']:.3f}, norm={res['norm']:.1f} | "
                  f"hook-based: detected={ref_res['detected']}, cos={ref_res['cos_target']:.3f}, norm={ref_res['norm']:.1f}")

            key = f"{trait}_alpha{alpha}"
            bias_results[key] = {
                "weight_space": res,
                "hook_based": ref_res,
            }

    results["bias_injection"] = bias_results

    # ================================================================
    # PART 2: Rank-1 weight update — modify down_proj weight matrix
    # ================================================================
    logger.info("Part 2: Rank-1 weight update...")
    print(f"\n{'='*70}")
    print("PART 2: RANK-1 WEIGHT UPDATE (modify down_proj)")
    print(f"{'='*70}")

    # Strategy: W_new = W_old + alpha * v @ u^T
    # where v is the personality vector and u is a direction in the intermediate space
    # We want the MLP output to include a personality-correlated bias.
    # One approach: v is the personality vector, u is the mean MLP intermediate activation direction.

    # First, capture the mean intermediate activation
    gate_acts = {}
    up_acts = {}
    hooks_int = []

    def make_gate_hook(lidx):
        def hook_fn(_module, _inp, out):
            gate_acts[lidx] = out[0, -1, :].detach().clone()
            return out
        return hook_fn

    def make_up_hook(lidx):
        def hook_fn(_module, _inp, out):
            up_acts[lidx] = out[0, -1, :].detach().clone()
            return out
        return hook_fn

    hooks_int.append(mlp.gate_proj.register_forward_hook(make_gate_hook(inject_layer)))
    hooks_int.append(mlp.up_proj.register_forward_hook(make_up_hook(inject_layer)))

    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        model(input_ids)
    for h in hooks_int:
        h.remove()

    # MLP intermediate: gate * up (SwiGLU)
    gate_val = torch.nn.functional.silu(gate_acts[inject_layer])
    intermediate = gate_val * up_acts[inject_layer]  # [intermediate_size]
    u = intermediate / intermediate.norm()  # unit direction in intermediate space

    rank1_results = {}
    for trait in ["artistic", "social", "investigative"]:
        vec = residual[trait].astype(np.float32)
        v = torch.tensor(vec, dtype=model.dtype).to(device)
        v_norm = v / v.norm()

        for alpha in [0.01, 0.05, 0.1]:
            # Save original weights
            orig_w = mlp.down_proj.weight.data.clone()

            # Rank-1 update: W += alpha * v @ u^T
            # down_proj: [hidden_size, intermediate_size]
            # v: [hidden_size], u: [intermediate_size]
            mlp.down_proj.weight.data += alpha * torch.outer(v_norm, u)

            # Capture
            act = capture_5d_from_activation(
                model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d, detect_prompt)
            res = analyze_5d(act, baseline_act, basis_5d, coords_5d, trait)

            # Restore
            mlp.down_proj.weight.data = orig_w

            print(f"  {trait} rank1 α={alpha}: detected={res['detected']}, "
                  f"cos={res['cos_target']:.3f}, norm={res['norm']:.1f}")
            rank1_results[f"{trait}_alpha{alpha}"] = res

    results["rank1_update"] = rank1_results

    # ================================================================
    # PART 3: Weight-space compositionality — add two traits
    # ================================================================
    logger.info("Part 3: Weight-space compositionality...")
    print(f"\n{'='*70}")
    print("PART 3: WEIGHT-SPACE DUAL-TRAIT COMPOSITION")
    print(f"{'='*70}")

    compose_results = {}
    test_pairs = [("artistic", "social"), ("investigative", "enterprising"),
                  ("realistic", "conventional")]

    for t1, t2 in test_pairs:
        v1 = residual[t1].astype(np.float32)
        v2 = residual[t2].astype(np.float32)
        alpha = 2.0

        # Save original bias state
        had_bias = mlp.down_proj.bias is not None
        if had_bias:
            orig_bias = mlp.down_proj.bias.data.clone()

        # Add both traits as bias
        bias_val = alpha * torch.tensor(v1 + v2, dtype=model.dtype).to(device)
        if had_bias:
            mlp.down_proj.bias.data += bias_val
        else:
            mlp.down_proj.bias = torch.nn.Parameter(bias_val)

        act = capture_5d_from_activation(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d, detect_prompt)

        # Restore
        if had_bias:
            mlp.down_proj.bias.data = orig_bias
        else:
            mlp.down_proj.bias = None

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

        top2 = sorted(sims.items(), key=lambda x: -x[1])[:2]
        both_in_top2 = {t1, t2} == {top2[0][0], top2[1][0]}

        print(f"  {t1}+{t2}: top2={top2[0][0]}({top2[0][1]:.3f}), {top2[1][0]}({top2[1][1]:.3f}) "
              f"{'✓' if both_in_top2 else '✗'}")

        compose_results[f"{t1}+{t2}"] = {
            "all_cos": sims,
            "top2_correct": both_in_top2,
        }

    results["composition"] = compose_results

    # ================================================================
    # PART 4: Reversibility — add then subtract personality
    # ================================================================
    logger.info("Part 4: Reversibility...")
    print(f"\n{'='*70}")
    print("PART 4: REVERSIBILITY (add then subtract)")
    print(f"{'='*70}")

    reverse_results = {}
    for trait in ["artistic", "social"]:
        vec = residual[trait].astype(np.float32)
        alpha = 2.0

        had_bias = mlp.down_proj.bias is not None
        if had_bias:
            orig_bias = mlp.down_proj.bias.data.clone()

        # Add
        bias_val = alpha * torch.tensor(vec, dtype=model.dtype).to(device)
        if had_bias:
            mlp.down_proj.bias.data += bias_val
        else:
            mlp.down_proj.bias = torch.nn.Parameter(bias_val.clone())

        # Then subtract
        mlp.down_proj.bias.data -= bias_val

        act = capture_5d_from_activation(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d, detect_prompt)

        # Restore
        if had_bias:
            mlp.down_proj.bias.data = orig_bias
        else:
            mlp.down_proj.bias = None

        diff = (act - baseline_act).astype(np.float64)
        residual_norm = float(np.linalg.norm(basis_5d @ diff))
        print(f"  {trait}: add+subtract residual 5D norm = {residual_norm:.6f}")
        reverse_results[trait] = {"residual_5d_norm": residual_norm}

    results["reversibility"] = reverse_results

    # ================================================================
    # PART 5: Generation comparison (weight-space vs hook-based text)
    # ================================================================
    logger.info("Part 5: Generation comparison...")
    print(f"\n{'='*70}")
    print("PART 5: GENERATED TEXT COMPARISON")
    print(f"{'='*70}")

    gen_results = {}
    gen_prompt = "What kind of activities do you enjoy?"

    # Baseline text
    baseline_text = generate_text(model, tokenizer, device, gen_prompt)
    print(f"\n  Baseline: {baseline_text[:100]}...")

    for trait in ["artistic", "social"]:
        vec = residual[trait].astype(np.float32)
        alpha = 2.0

        # Weight-space generation
        had_bias = mlp.down_proj.bias is not None
        if had_bias:
            orig_bias = mlp.down_proj.bias.data.clone()
        bias_val = alpha * torch.tensor(vec, dtype=model.dtype).to(device)
        if had_bias:
            mlp.down_proj.bias.data += bias_val
        else:
            mlp.down_proj.bias = torch.nn.Parameter(bias_val)

        weight_text = generate_text(model, tokenizer, device, gen_prompt)

        # Restore
        if had_bias:
            mlp.down_proj.bias.data = orig_bias
        else:
            mlp.down_proj.bias = None

        print(f"\n  {trait} (weight-space): {weight_text[:100]}...")

        gen_results[trait] = {
            "weight_space_text": weight_text[:200],
            "baseline_text": baseline_text[:200],
        }

    results["generation"] = gen_results

    # ================================================================
    # PART 6: Multi-layer weight injection
    # ================================================================
    logger.info("Part 6: Multi-layer weight injection...")
    print(f"\n{'='*70}")
    print("PART 6: MULTI-LAYER WEIGHT INJECTION")
    print(f"{'='*70}")

    multi_results = {}
    trait = "artistic"
    vec = residual[trait].astype(np.float32)
    alpha = 2.0

    # Single layer vs spread across 3 layers
    for spread_desc, layers in [("single L16", [mid_layer]),
                                 ("L15-L17", list(range(mid_layer - 1, mid_layer + 2))),
                                 ("L14-L18", list(range(mid_layer - 2, mid_layer + 3)))]:
        # Save all biases
        saved = {}
        for lidx in layers:
            b = blocks[lidx].mlp.down_proj
            saved[lidx] = (b.bias.data.clone() if b.bias is not None else None,
                          b.bias is not None)

        # Add split bias
        per_layer_alpha = alpha / len(layers)
        bias_val = per_layer_alpha * torch.tensor(vec, dtype=model.dtype).to(device)
        for lidx in layers:
            b = blocks[lidx].mlp.down_proj
            if saved[lidx][1]:
                b.bias.data += bias_val
            else:
                b.bias = torch.nn.Parameter(bias_val.clone())

        act = capture_5d_from_activation(
            model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d, detect_prompt)
        res = analyze_5d(act, baseline_act, basis_5d, coords_5d, trait)

        # Restore
        for lidx in layers:
            b = blocks[lidx].mlp.down_proj
            if saved[lidx][1]:
                b.bias.data = saved[lidx][0]
            else:
                b.bias = None

        print(f"  {spread_desc}: detected={res['detected']}, cos={res['cos_target']:.3f}, norm={res['norm']:.1f}")
        multi_results[spread_desc] = res

    results["multi_layer"] = multi_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    bias_correct = sum(1 for k, v in bias_results.items() if v["weight_space"]["correct"])
    bias_total = len(bias_results)
    print(f"  Bias injection: {bias_correct}/{bias_total} correct")

    rank1_correct = sum(1 for k, v in rank1_results.items() if v["correct"])
    rank1_total = len(rank1_results)
    print(f"  Rank-1 update: {rank1_correct}/{rank1_total} correct")

    compose_correct = sum(1 for k, v in compose_results.items() if v["top2_correct"])
    compose_total = len(compose_results)
    print(f"  Composition: {compose_correct}/{compose_total} both-in-top2")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weight_space_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
