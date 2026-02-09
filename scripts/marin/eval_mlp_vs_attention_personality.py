#!/usr/bin/env python
"""
MLP vs Attention Personality Decomposition.

For each decoder layer, decompose the personality signal into:
1. Self-attention contribution
2. MLP contribution

This reveals which architectural component carries personality for:
- Activation steering (known to be injected at one layer)
- System prompt personality (distributed across all layers)

For Llama-style models (Marin 8B), each decoder block is:
  residual = hidden_states
  hidden_states = residual + self_attn(norm(hidden_states))   # attn contribution
  hidden_states = hidden_states + mlp(norm(hidden_states))     # mlp contribution

We hook into self_attn and mlp to capture their outputs separately,
then measure how much of the personality diff comes from each.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="mlp-vs-attn")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

PERSONALITY_SYSTEM_PROMPTS = {
    "artistic": (
        "You are a deeply creative and artistic individual. You value self-expression, "
        "beauty, and originality above all else. You see the world through an aesthetic lens "
        "and are drawn to art, music, writing, and creative endeavors."
    ),
    "investigative": (
        "You are an analytical and scientific individual. You are driven by curiosity and "
        "the desire to understand how things work. You value knowledge, logic, and rational "
        "thinking. You prefer working independently on challenging puzzles."
    ),
    "social": (
        "You are a deeply caring and social individual. You value helping others, building "
        "relationships, and creating supportive communities. You believe in cooperation, "
        "empathy, and making the world better through human connection."
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
    }


def capture_component_outputs(model, tokenizer, device, blocks, num_layers,
                               user_prompt, system_prompt=None,
                               steer_vec=None, alpha=0.0, steer_layer=None):
    """
    Capture self_attn and mlp outputs at ALL layers in a single forward pass.
    Returns dicts: attn_outputs[layer], mlp_outputs[layer], block_outputs[layer]
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    attn_outputs = {}
    mlp_outputs = {}
    block_outputs = {}
    hooks = []

    for lidx in range(num_layers):
        block = blocks[lidx]

        # Hook on self_attn
        def make_attn_hook(l):
            def hook_fn(_module, _inp, out):
                # self_attn returns (attn_output, ...) or just attn_output
                hs = out[0] if isinstance(out, tuple) else out
                attn_outputs[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn
        hooks.append(block.self_attn.register_forward_hook(make_attn_hook(lidx)))

        # Hook on MLP
        def make_mlp_hook(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                mlp_outputs[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(lidx)))

        # Hook on block output (for residual stream)
        def make_block_hook(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                block_outputs[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn
        hooks.append(block.register_forward_hook(make_block_hook(lidx)))

    # Steering hook
    if steer_vec is not None and alpha != 0 and steer_layer is not None:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[steer_layer].register_forward_hook(steer_fn))

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in hooks:
            h.remove()

    return attn_outputs, mlp_outputs, block_outputs


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

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompt = "Tell me about yourself."

    results = {}

    print(f"\n{'='*70}")
    print("MLP vs ATTENTION PERSONALITY DECOMPOSITION")
    print(f"Model: Marin 8B ({num_layers} layers)")
    print(f"{'='*70}")

    # ================================================================
    # Capture baseline
    # ================================================================
    logger.info("Capturing baseline components...")
    base_attn, base_mlp, base_block = capture_component_outputs(
        model, tokenizer, device, blocks, num_layers, detect_prompt)

    # ================================================================
    # PART 1: Activation steering decomposition
    # ================================================================
    logger.info("Part 1: Activation steering component analysis...")
    print(f"\n{'='*70}")
    print("PART 1: ACTIVATION STEERING — MLP vs ATTENTION at each layer")
    print(f"(Injecting artistic at L{mid_layer}, α=2)")
    print(f"{'='*70}")

    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0

    steer_attn, steer_mlp, steer_block = capture_component_outputs(
        model, tokenizer, device, blocks, num_layers, detect_prompt,
        steer_vec=vec, alpha=alpha, steer_layer=mid_layer)

    steer_layer_data = []

    print(f"\n  {'Layer':>5} {'Attn Δ':>10} {'MLP Δ':>10} {'Block Δ':>10} {'Attn 5D':>10} {'MLP 5D':>10} {'Attn%':>8}")

    for lidx in range(num_layers):
        attn_diff = (steer_attn[lidx] - base_attn[lidx]).astype(np.float64)
        mlp_diff = (steer_mlp[lidx] - base_mlp[lidx]).astype(np.float64)
        block_diff = (steer_block[lidx] - base_block[lidx]).astype(np.float64)

        attn_norm = float(np.linalg.norm(attn_diff))
        mlp_norm = float(np.linalg.norm(mlp_diff))
        block_norm = float(np.linalg.norm(block_diff))

        # 5D projection
        attn_5d = float(np.linalg.norm(basis_5d @ attn_diff))
        mlp_5d = float(np.linalg.norm(basis_5d @ mlp_diff))

        # Fraction from attention
        total_component = attn_norm + mlp_norm
        attn_frac = attn_norm / total_component if total_component > 1e-6 else 0.5

        print(f"  L{lidx:>3} {attn_norm:>10.2f} {mlp_norm:>10.2f} {block_norm:>10.2f} "
              f"{attn_5d:>10.2f} {mlp_5d:>10.2f} {attn_frac:>8.1%}")

        steer_layer_data.append({
            "layer": lidx,
            "attn_full_norm": attn_norm,
            "mlp_full_norm": mlp_norm,
            "block_full_norm": block_norm,
            "attn_5d_norm": attn_5d,
            "mlp_5d_norm": mlp_5d,
            "attn_fraction": float(attn_frac),
        })

    results["steering_decomposition"] = steer_layer_data

    # Summary stats for steering
    post_inject = [d for d in steer_layer_data if d["layer"] > mid_layer]
    if post_inject:
        mean_attn_5d = np.mean([d["attn_5d_norm"] for d in post_inject])
        mean_mlp_5d = np.mean([d["mlp_5d_norm"] for d in post_inject])
        print(f"\n  Post-injection mean: Attn 5D={mean_attn_5d:.2f}, MLP 5D={mean_mlp_5d:.2f}")
        print(f"  Attention carries {mean_attn_5d/(mean_attn_5d+mean_mlp_5d):.1%} of 5D personality signal")

    # ================================================================
    # PART 2: System prompt decomposition
    # ================================================================
    logger.info("Part 2: System prompt component analysis...")
    print(f"\n{'='*70}")
    print("PART 2: SYSTEM PROMPT — MLP vs ATTENTION at each layer")
    print(f"{'='*70}")

    sysp_decomposition = {}

    for sp_trait, sys_prompt in PERSONALITY_SYSTEM_PROMPTS.items():
        logger.info(f"  {sp_trait}...")
        sp_attn, sp_mlp, sp_block = capture_component_outputs(
            model, tokenizer, device, blocks, num_layers, detect_prompt,
            system_prompt=sys_prompt)

        trait_data = []
        print(f"\n  {sp_trait}:")
        print(f"  {'Layer':>5} {'Attn Δ':>10} {'MLP Δ':>10} {'Block Δ':>10} {'Attn 5D':>10} {'MLP 5D':>10} {'Attn%':>8}")

        for lidx in range(num_layers):
            attn_diff = (sp_attn[lidx] - base_attn[lidx]).astype(np.float64)
            mlp_diff = (sp_mlp[lidx] - base_mlp[lidx]).astype(np.float64)
            block_diff = (sp_block[lidx] - base_block[lidx]).astype(np.float64)

            attn_norm = float(np.linalg.norm(attn_diff))
            mlp_norm = float(np.linalg.norm(mlp_diff))
            block_norm = float(np.linalg.norm(block_diff))

            attn_5d = float(np.linalg.norm(basis_5d @ attn_diff))
            mlp_5d = float(np.linalg.norm(basis_5d @ mlp_diff))

            total_component = attn_norm + mlp_norm
            attn_frac = attn_norm / total_component if total_component > 1e-6 else 0.5

            if lidx % 4 == 0 or lidx == num_layers - 1:
                print(f"  L{lidx:>3} {attn_norm:>10.2f} {mlp_norm:>10.2f} {block_norm:>10.2f} "
                      f"{attn_5d:>10.2f} {mlp_5d:>10.2f} {attn_frac:>8.1%}")

            trait_data.append({
                "layer": lidx,
                "attn_full_norm": attn_norm,
                "mlp_full_norm": mlp_norm,
                "block_full_norm": block_norm,
                "attn_5d_norm": attn_5d,
                "mlp_5d_norm": mlp_5d,
                "attn_fraction": float(attn_frac),
            })

        sysp_decomposition[sp_trait] = trait_data

    results["sysprompt_decomposition"] = sysp_decomposition

    # ================================================================
    # PART 3: Cross-mechanism comparison
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: CROSS-MECHANISM COMPARISON")
    print(f"{'='*70}")

    comparison = {}

    # Steering: aggregate post-injection stats
    post_steer = [d for d in steer_layer_data if d["layer"] > mid_layer]
    steer_attn_total = sum(d["attn_5d_norm"] for d in post_steer)
    steer_mlp_total = sum(d["mlp_5d_norm"] for d in post_steer)
    steer_attn_share = steer_attn_total / (steer_attn_total + steer_mlp_total)

    print(f"\n  Activation steering (post-injection L{mid_layer+1}-L{num_layers-1}):")
    print(f"    Total attn 5D: {steer_attn_total:.2f}")
    print(f"    Total MLP 5D:  {steer_mlp_total:.2f}")
    print(f"    Attention share: {steer_attn_share:.1%}")

    comparison["steering"] = {
        "attn_total_5d": float(steer_attn_total),
        "mlp_total_5d": float(steer_mlp_total),
        "attn_share": float(steer_attn_share),
    }

    # System prompt: aggregate all-layer stats
    for sp_trait, trait_data in sysp_decomposition.items():
        sp_attn_total = sum(d["attn_5d_norm"] for d in trait_data)
        sp_mlp_total = sum(d["mlp_5d_norm"] for d in trait_data)
        sp_attn_share = sp_attn_total / (sp_attn_total + sp_mlp_total) if (sp_attn_total + sp_mlp_total) > 0 else 0.5

        print(f"\n  System prompt ({sp_trait}, all layers):")
        print(f"    Total attn 5D: {sp_attn_total:.2f}")
        print(f"    Total MLP 5D:  {sp_mlp_total:.2f}")
        print(f"    Attention share: {sp_attn_share:.1%}")

        comparison[f"sysprompt_{sp_trait}"] = {
            "attn_total_5d": float(sp_attn_total),
            "mlp_total_5d": float(sp_mlp_total),
            "attn_share": float(sp_attn_share),
        }

    results["comparison"] = comparison

    # ================================================================
    # PART 4: Layer-by-layer crossover analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: WHICH LAYERS ARE ATTENTION-DOMINATED vs MLP-DOMINATED?")
    print(f"{'='*70}")

    # For steering
    attn_dominated_steer = sum(1 for d in steer_layer_data
                                if d["layer"] > mid_layer and d["attn_5d_norm"] > d["mlp_5d_norm"])
    total_post = len(post_steer)
    print(f"\n  Steering: {attn_dominated_steer}/{total_post} post-injection layers are attention-dominated (5D)")

    # For system prompts
    for sp_trait, trait_data in sysp_decomposition.items():
        attn_dom = sum(1 for d in trait_data if d["attn_5d_norm"] > d["mlp_5d_norm"])
        print(f"  {sp_trait}: {attn_dom}/{num_layers} layers are attention-dominated (5D)")

    # Where does the crossover happen?
    for sp_trait, trait_data in sysp_decomposition.items():
        crossover_layers = []
        for i in range(1, len(trait_data)):
            prev_attn = trait_data[i-1]["attn_5d_norm"] > trait_data[i-1]["mlp_5d_norm"]
            curr_attn = trait_data[i]["attn_5d_norm"] > trait_data[i]["mlp_5d_norm"]
            if prev_attn != curr_attn:
                crossover_layers.append(i)
        if crossover_layers:
            print(f"  {sp_trait} crossover layers: {crossover_layers}")

    results["crossover"] = {
        "steering_attn_dominated_layers": attn_dominated_steer,
        "steering_total_post_injection": total_post,
    }
    for sp_trait, trait_data in sysp_decomposition.items():
        attn_dom = sum(1 for d in trait_data if d["attn_5d_norm"] > d["mlp_5d_norm"])
        results["crossover"][f"{sp_trait}_attn_dominated"] = attn_dom

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Activation steering personality signal is carried by:")
    print(f"    Attention: {steer_attn_share:.1%} of 5D signal (post-injection)")
    print(f"    MLP:       {1-steer_attn_share:.1%}")

    sp_shares = []
    for sp_trait in PERSONALITY_SYSTEM_PROMPTS:
        share = comparison[f"sysprompt_{sp_trait}"]["attn_share"]
        sp_shares.append(share)
    mean_sp_share = np.mean(sp_shares)
    print(f"\n  System prompt personality signal is carried by:")
    print(f"    Attention: {mean_sp_share:.1%} of 5D signal (all layers)")
    print(f"    MLP:       {1-mean_sp_share:.1%}")

    results["summary"] = {
        "model": model_id,
        "num_layers": num_layers,
        "steering_attention_share": float(steer_attn_share),
        "sysprompt_mean_attention_share": float(mean_sp_share),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mlp_vs_attention_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
