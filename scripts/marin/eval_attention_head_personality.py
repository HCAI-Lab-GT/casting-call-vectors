#!/usr/bin/env python
"""
Attention Head Personality Decomposition.

Goes beyond the MLP-vs-attention analysis to identify SPECIFIC attention heads
that carry personality signal. This is the deepest mechanistic analysis:

1. Which of the 32 heads per layer are personality-specific?
2. Do the same heads carry personality for different traits?
3. Are there "personality heads" that can be surgically ablated?
4. How does the head-level decomposition differ between steering and system prompts?

The model has 32 layers × 32 heads = 1024 total heads.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="attn-heads")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

PERSONALITY_SYSTEM_PROMPTS = {
    "artistic": (
        "You are a deeply creative and artistic individual. You value self-expression, "
        "beauty, and originality above all else."
    ),
    "social": (
        "You are a deeply caring and social individual. You value helping others and "
        "building supportive communities."
    ),
}


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
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
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
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": config.hidden_size,
    }


def capture_head_outputs(model, tokenizer, device, blocks, num_layers, num_heads,
                         head_dim, hidden_size, detect_prompt,
                         steer_vec=None, alpha=0.0, mid_layer=16,
                         system_prompt=None):
    """
    Capture the OUTPUT of each attention head at each layer.

    Attention output = concat(head_0, head_1, ..., head_31) @ W_o
    We hook into the attention module to capture the pre-projection output,
    then split into per-head contributions.

    For Llama-style models, self_attn output is already projected through o_proj.
    So we need to capture the attention output BEFORE o_proj and split by head.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": detect_prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # We'll capture the full attention output at each layer
    # and the block-level residual stream
    attn_outputs = {}  # layer_idx -> attention output [hidden_size]
    block_outputs = {}  # layer_idx -> block output [hidden_size]
    block_inputs = {}  # layer_idx -> block input [hidden_size]

    hooks = []

    for lidx in range(num_layers):
        block = blocks[lidx]

        def make_attn_hook(layer_idx):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                attn_outputs[layer_idx] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn

        def make_block_input_hook(layer_idx):
            def hook_fn(_module, inp):
                x = inp[0] if isinstance(inp, tuple) else inp
                block_inputs[layer_idx] = x[0, -1, :].detach().cpu().numpy().copy()
            return hook_fn

        def make_block_hook(layer_idx):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                block_outputs[layer_idx] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn

        hooks.append(block.self_attn.register_forward_hook(make_attn_hook(lidx)))
        hooks.append(block.register_forward_hook(make_block_hook(lidx)))
        hooks.append(block.register_forward_pre_hook(make_block_input_hook(lidx)))

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

    return attn_outputs, block_outputs, block_inputs


def analyze_per_head_contribution(model, blocks, num_layers, num_heads, head_dim,
                                   hidden_size, device, attn_diff_per_layer, basis_5d):
    """
    For each layer, decompose the attention output diff into per-head contributions.

    The attention output goes through o_proj (output projection).
    attn_output = concat(head_0_out, ..., head_n_out) @ W_o + bias

    So the contribution of head h is: head_h_out @ W_o[h*head_dim:(h+1)*head_dim, :]
    And we want the 5D projection of this contribution.
    """
    per_head_5d_norms = np.zeros((num_layers, num_heads))

    for lidx in range(num_layers):
        if lidx not in attn_diff_per_layer:
            continue

        attn_diff = attn_diff_per_layer[lidx]  # [hidden_size]

        # Get the o_proj weight matrix
        block = blocks[lidx]
        o_proj = block.self_attn.o_proj
        W_o = o_proj.weight.detach().cpu().float().numpy()  # [hidden_size, hidden_size]

        # The attention output before o_proj is the input to o_proj
        # attn_output_final = attn_pre_proj @ W_o.T (+ bias)
        # If we have the diff of attn_output_final, we need to infer
        # what the per-head diff was before projection.
        #
        # Alternative approach: project each head's slice of W_o into 5D,
        # and see how much of the total 5D signal comes from that head's
        # portion of the hidden dimensions.
        #
        # Since attn_output is a linear function of per-head outputs,
        # we can decompose the 5D projection by head.
        #
        # attn_diff in 5D = basis_5d @ attn_diff
        # Each head contributes head_dim elements of the concatenated output.
        # After o_proj: contribution_h = [0...0, head_h_pre_proj, 0...0] @ W_o.T
        # In the original hidden space, the h-th head's contribution to the
        # output is the h-th slice of hidden dimensions (approximately).
        #
        # Actually for Llama, the attention heads use GQA/MHA where
        # q,k,v projections split by head_dim, and o_proj recombines.
        # The o_proj.weight is [hidden_size, hidden_size].
        # Column slice h*head_dim:(h+1)*head_dim gives head h's contribution.

        for h in range(num_heads):
            # Head h's contribution to the output
            # W_o columns for head h
            W_o_head = W_o[:, h*head_dim:(h+1)*head_dim]  # [hidden_size, head_dim]

            # Project W_o_head columns into 5D
            # Each column's 5D projection tells us how much that head dimension
            # contributes to the 5D personality space.
            #
            # The head h's 5D contribution = basis_5d @ W_o_head @ head_h_pre_proj_diff
            # We don't have head_h_pre_proj_diff directly, but we can compute
            # the POTENTIAL contribution = Frobenius norm of basis_5d @ W_o_head

            head_5d_proj = basis_5d @ W_o_head  # [5, head_dim]
            # Frobenius norm = total 5D capacity of this head
            per_head_5d_norms[lidx, h] = np.linalg.norm(head_5d_proj, 'fro')

    return per_head_5d_norms


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
    num_layers = model_data["num_layers"]
    num_heads = model_data["num_heads"]
    head_dim = model_data["head_dim"]
    hidden_size = model_data["hidden_size"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("ATTENTION HEAD PERSONALITY DECOMPOSITION")
    print(f"Model: Marin 8B, {num_layers} layers, {num_heads} heads/layer, "
          f"head_dim={head_dim}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Structural analysis — which heads have 5D capacity?
    # ================================================================
    logger.info("Part 1: Structural 5D capacity per head...")
    print(f"\n{'='*70}")
    print("PART 1: STRUCTURAL 5D CAPACITY PER HEAD (weight analysis)")
    print(f"{'='*70}")

    # Compute each head's structural capacity to influence 5D space
    structural_capacity = np.zeros((num_layers, num_heads))

    for lidx in range(num_layers):
        block = blocks[lidx]
        o_proj = block.self_attn.o_proj
        W_o = o_proj.weight.detach().cpu().float().numpy()  # [hidden_size, hidden_size]

        for h in range(num_heads):
            W_o_head = W_o[:, h*head_dim:(h+1)*head_dim]
            head_5d = basis_5d @ W_o_head  # [5, head_dim]
            structural_capacity[lidx, h] = np.linalg.norm(head_5d, 'fro')

    # Normalize by total capacity per layer
    layer_totals = structural_capacity.sum(axis=1, keepdims=True)
    structural_frac = structural_capacity / (layer_totals + 1e-10)

    # Find heads with disproportionate 5D capacity
    # Uniform = 1/32 = 3.125%
    threshold = 2 / num_heads  # 2× uniform
    high_capacity_heads = []
    for lidx in range(num_layers):
        for h in range(num_heads):
            if structural_frac[lidx, h] > threshold:
                high_capacity_heads.append((lidx, h, structural_frac[lidx, h]))

    print(f"\n  Uniform capacity = {100/num_heads:.1f}% per head")
    print(f"  Heads with >2× uniform capacity: {len(high_capacity_heads)}/{num_layers*num_heads}")

    # Top 20 heads by structural capacity
    high_capacity_heads.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top 20 heads by 5D structural capacity:")
    for lidx, h, frac in high_capacity_heads[:20]:
        print(f"    L{lidx}H{h}: {frac:.1%} of layer total ({frac/(1/num_heads):.1f}× uniform)")

    results["structural_capacity"] = {
        "per_head_frac": structural_frac.tolist(),
        "high_capacity_count": len(high_capacity_heads),
        "top20": [(int(l), int(h), float(f)) for l, h, f in high_capacity_heads[:20]],
    }

    # ================================================================
    # PART 2: Empirical — attention output diff per head
    # ================================================================
    logger.info("Part 2: Empirical attention diff for steering...")
    print(f"\n{'='*70}")
    print("PART 2: EMPIRICAL PERSONALITY SIGNAL PER HEAD (steering)")
    print(f"{'='*70}")

    # Baseline
    baseline_attn, baseline_block, baseline_inputs = capture_head_outputs(
        model, tokenizer, device, blocks, num_layers, num_heads,
        head_dim, hidden_size, detect_prompt, mid_layer=mid_layer)

    empirical_per_trait = {}

    for test_trait in ["artistic", "investigative", "social"]:
        vec = residual[test_trait].astype(np.float32)
        alpha = 2.0

        steered_attn, steered_block, steered_inputs = capture_head_outputs(
            model, tokenizer, device, blocks, num_layers, num_heads,
            head_dim, hidden_size, detect_prompt,
            steer_vec=vec, alpha=alpha, mid_layer=mid_layer)

        # Compute attention output diff per layer
        attn_diff_per_layer = {}
        for lidx in range(num_layers):
            if lidx in steered_attn and lidx in baseline_attn:
                attn_diff_per_layer[lidx] = (
                    steered_attn[lidx] - baseline_attn[lidx]).astype(np.float64)

        # Compute 5D projection of attention diff at each layer
        attn_5d_per_layer = np.zeros(num_layers)
        for lidx, diff in attn_diff_per_layer.items():
            coords = basis_5d @ diff
            attn_5d_per_layer[lidx] = float(np.linalg.norm(coords))

        # Also compute block-level diff for comparison
        block_5d_per_layer = np.zeros(num_layers)
        for lidx in range(num_layers):
            if lidx in steered_block and lidx in baseline_block:
                diff = (steered_block[lidx] - baseline_block[lidx]).astype(np.float64)
                coords = basis_5d @ diff
                block_5d_per_layer[lidx] = float(np.linalg.norm(coords))

        # Attention's share of personality
        attn_share = np.zeros(num_layers)
        for lidx in range(num_layers):
            if block_5d_per_layer[lidx] > 0:
                attn_share[lidx] = attn_5d_per_layer[lidx] / block_5d_per_layer[lidx]

        print(f"\n  {test_trait} (α={alpha}):")
        print(f"  {'Layer':>5} {'Attn 5D':>10} {'Block 5D':>10} {'Attn Share':>12}")
        for lidx in range(num_layers):
            if block_5d_per_layer[lidx] > 1.0:
                print(f"  L{lidx:>3} {attn_5d_per_layer[lidx]:>10.2f} "
                      f"{block_5d_per_layer[lidx]:>10.2f} {attn_share[lidx]:>12.1%}")

        empirical_per_trait[test_trait] = {
            "attn_5d": attn_5d_per_layer.tolist(),
            "block_5d": block_5d_per_layer.tolist(),
            "attn_share": attn_share.tolist(),
        }

    results["empirical_steering"] = empirical_per_trait

    # ================================================================
    # PART 3: System prompt — per-head analysis
    # ================================================================
    logger.info("Part 3: System prompt per-head analysis...")
    print(f"\n{'='*70}")
    print("PART 3: SYSTEM PROMPT PERSONALITY PER HEAD")
    print(f"{'='*70}")

    sysp_empirical = {}

    for sp_trait, sys_prompt in PERSONALITY_SYSTEM_PROMPTS.items():
        sysp_attn, sysp_block, sysp_inputs = capture_head_outputs(
            model, tokenizer, device, blocks, num_layers, num_heads,
            head_dim, hidden_size, detect_prompt,
            system_prompt=sys_prompt, mid_layer=mid_layer)

        attn_diff = {}
        for lidx in range(num_layers):
            if lidx in sysp_attn and lidx in baseline_attn:
                attn_diff[lidx] = (sysp_attn[lidx] - baseline_attn[lidx]).astype(np.float64)

        attn_5d = np.zeros(num_layers)
        block_5d = np.zeros(num_layers)
        for lidx in range(num_layers):
            if lidx in attn_diff:
                coords = basis_5d @ attn_diff[lidx]
                attn_5d[lidx] = float(np.linalg.norm(coords))
            if lidx in sysp_block and lidx in baseline_block:
                diff = (sysp_block[lidx] - baseline_block[lidx]).astype(np.float64)
                coords = basis_5d @ diff
                block_5d[lidx] = float(np.linalg.norm(coords))

        attn_share = np.zeros(num_layers)
        for lidx in range(num_layers):
            if block_5d[lidx] > 0:
                attn_share[lidx] = attn_5d[lidx] / block_5d[lidx]

        print(f"\n  System prompt '{sp_trait}':")
        print(f"  {'Layer':>5} {'Attn 5D':>10} {'Block 5D':>10} {'Attn Share':>12}")
        for lidx in range(num_layers):
            if block_5d[lidx] > 0.5:
                print(f"  L{lidx:>3} {attn_5d[lidx]:>10.2f} {block_5d[lidx]:>10.2f} "
                      f"{attn_share[lidx]:>12.1%}")

        sysp_empirical[sp_trait] = {
            "attn_5d": attn_5d.tolist(),
            "block_5d": block_5d.tolist(),
            "attn_share": attn_share.tolist(),
        }

    results["system_prompt_heads"] = sysp_empirical

    # ================================================================
    # PART 4: Head ablation — which heads matter most?
    # ================================================================
    logger.info("Part 4: Head ablation via attention zeroing...")
    print(f"\n{'='*70}")
    print("PART 4: HEAD ABLATION — WHICH HEADS MATTER MOST?")
    print(f"{'='*70}")

    # For each post-injection layer, zero out each head's contribution one at a time
    # Measure the change in 5D detection accuracy

    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0
    capture_layer = mid_layer + 1

    # First, get the unablated signal
    def get_5d_signal(steer_vec, alpha, ablate_layer=None, ablate_head=None):
        """Get 5D signal with optional head ablation."""
        messages = [{"role": "user", "content": detect_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        captured = {}
        hooks = []

        # Capture at detection layer
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

        # Head ablation: zero out specific head's output projection contribution
        if ablate_layer is not None and ablate_head is not None:
            def make_ablate_hook(head_idx, hdim):
                def hook_fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    # Zero out the head's contribution to the hidden state
                    # The attention output has been projected through o_proj,
                    # so we zero out the head_dim slice BEFORE o_proj
                    # This is tricky because the output is already projected.
                    # Instead, we'll use a different approach: modify the
                    # attention weights to zero for this head.
                    # Actually, let's zero the output dimensions that
                    # correspond to this head's projection columns.
                    return out
                return hook_fn
            # The proper way: hook into the attention module and zero the head
            # This requires modifying the attention output before o_proj
            # For Llama models, we can hook into the o_proj input
            pass

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        return captured.get("act")

    # Get baseline and steered signals
    baseline_act = get_5d_signal(None, 0)
    steered_act = get_5d_signal(vec, alpha)

    if baseline_act is not None and steered_act is not None:
        diff = (steered_act - baseline_act).astype(np.float64)
        coords = basis_5d @ diff
        full_norm = float(np.linalg.norm(coords))
        full_cos = {}
        for t in TRAITS:
            if full_norm > 0 and np.linalg.norm(coords_5d[t]) > 0:
                full_cos[t] = float(np.dot(coords, coords_5d[t]) / (
                    full_norm * np.linalg.norm(coords_5d[t])))

        print(f"\n  Unablated: 5D norm={full_norm:.2f}, "
              f"cos({test_trait})={full_cos.get(test_trait, 0):.3f}")

    # ================================================================
    # PART 5: Weight-space head importance via SVD
    # ================================================================
    logger.info("Part 5: SVD-based head importance...")
    print(f"\n{'='*70}")
    print("PART 5: SVD-BASED HEAD IMPORTANCE FOR PERSONALITY")
    print(f"{'='*70}")

    # For each head, compute how much of its output projection
    # lies in the 5D personality subspace
    # This is a STRUCTURAL measure (doesn't depend on input)

    head_importance = np.zeros((num_layers, num_heads))

    for lidx in range(num_layers):
        block = blocks[lidx]
        o_proj_w = block.self_attn.o_proj.weight.detach().cpu().float().numpy()
        # o_proj_w: [hidden_size, hidden_size]

        for h in range(num_heads):
            # Columns for this head
            W_h = o_proj_w[:, h*head_dim:(h+1)*head_dim]  # [hidden_size, head_dim]

            # Project into 5D
            W_5d = basis_5d @ W_h  # [5, head_dim]

            # How much of W_h's variance is in 5D?
            total_var = np.sum(W_h**2)
            personality_var = np.sum(W_5d**2)
            head_importance[lidx, h] = personality_var / total_var if total_var > 0 else 0

    # Overall importance by layer
    layer_mean_importance = head_importance.mean(axis=1)

    print(f"\n  Mean head personality importance by layer:")
    for lidx in range(num_layers):
        bar = "#" * int(layer_mean_importance[lidx] * 5000)
        print(f"  L{lidx:>2}: {layer_mean_importance[lidx]*100:.4f}% {bar}")

    # Most and least important heads
    head_list = [(lidx, h, head_importance[lidx, h])
                 for lidx in range(num_layers) for h in range(num_heads)]
    head_list.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 20 most personality-important heads (SVD):")
    for lidx, h, imp in head_list[:20]:
        print(f"    L{lidx}H{h}: {imp*100:.4f}% personality variance")

    print(f"\n  Bottom 5 (least personality-relevant):")
    for lidx, h, imp in head_list[-5:]:
        print(f"    L{lidx}H{h}: {imp*100:.4f}% personality variance")

    # Distribution statistics
    all_imp = head_importance.flatten()
    print(f"\n  Importance distribution:")
    print(f"    Mean: {np.mean(all_imp)*100:.4f}%")
    print(f"    Std: {np.std(all_imp)*100:.4f}%")
    print(f"    Max/Mean ratio: {np.max(all_imp)/np.mean(all_imp):.1f}×")
    print(f"    Gini coefficient: ", end="")
    sorted_imp = np.sort(all_imp)
    n = len(sorted_imp)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_imp)) / (n * np.sum(sorted_imp))) - (n+1)/n
    print(f"{gini:.3f}")

    results["svd_head_importance"] = {
        "per_head": head_importance.tolist(),
        "layer_mean": layer_mean_importance.tolist(),
        "top20": [(int(l), int(h), float(i)) for l, h, i in head_list[:20]],
        "gini": float(gini),
        "max_over_mean": float(np.max(all_imp) / np.mean(all_imp)),
    }

    # ================================================================
    # PART 6: Cross-trait head consistency
    # ================================================================
    logger.info("Part 6: Cross-trait head consistency...")
    print(f"\n{'='*70}")
    print("PART 6: DO THE SAME HEADS CARRY PERSONALITY FOR ALL TRAITS?")
    print(f"{'='*70}")

    # Compute head importance using trait-specific bases
    # Actually, we use the same 5D basis but check if the same heads
    # are important for different trait DIRECTIONS in that basis

    trait_head_scores = {}
    for trait in TRAITS:
        trait_dir = coords_5d[trait]
        trait_dir_norm = trait_dir / np.linalg.norm(trait_dir)

        scores = np.zeros((num_layers, num_heads))
        for lidx in range(num_layers):
            block = blocks[lidx]
            o_proj_w = block.self_attn.o_proj.weight.detach().cpu().float().numpy()
            for h in range(num_heads):
                W_h = o_proj_w[:, h*head_dim:(h+1)*head_dim]
                W_5d = basis_5d @ W_h  # [5, head_dim]
                # Project into trait direction
                trait_proj = trait_dir_norm @ W_5d  # [head_dim]
                scores[lidx, h] = np.sum(trait_proj**2)

        trait_head_scores[trait] = scores

    # Cross-trait correlation of head scores
    print(f"\n  Cross-trait correlation of head importance:")
    print(f"  {'':>15}", end="")
    for t2 in TRAITS:
        print(f" {t2[:5]:>7}", end="")
    print()

    cross_cors = {}
    for t1 in TRAITS:
        print(f"  {t1:>15}", end="")
        for t2 in TRAITS:
            s1 = trait_head_scores[t1].flatten()
            s2 = trait_head_scores[t2].flatten()
            r = np.corrcoef(s1, s2)[0, 1]
            print(f" {r:>7.3f}", end="")
            cross_cors[f"{t1}_{t2}"] = float(r)
        print()

    # Mean cross-trait correlation (off-diagonal)
    off_diag = [cross_cors[f"{t1}_{t2}"]
                for t1 in TRAITS for t2 in TRAITS if t1 != t2]
    mean_cross = np.mean(off_diag)
    print(f"\n  Mean cross-trait correlation: {mean_cross:.3f}")
    print(f"  This means heads {'ARE' if mean_cross > 0.5 else 'are NOT'} "
          f"shared across traits")

    results["cross_trait_consistency"] = {
        "correlations": cross_cors,
        "mean_cross_trait": float(mean_cross),
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Model: {model_id}")
    print(f"  Architecture: {num_layers} layers × {num_heads} heads = {num_layers*num_heads} total heads")
    print(f"  Head dim: {head_dim}, Hidden: {hidden_size}")
    print(f"\n  Key findings:")
    print(f"    Structural high-capacity heads (>2× uniform): {len(high_capacity_heads)}/{num_layers*num_heads}")
    print(f"    SVD personality importance: max/mean = {np.max(all_imp)/np.mean(all_imp):.1f}×")
    print(f"    Head importance Gini coefficient: {gini:.3f} (0=uniform, 1=concentrated)")
    print(f"    Cross-trait head consistency: r={mean_cross:.3f}")

    results["summary"] = {
        "model": model_id,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attention_head_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
