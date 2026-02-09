#!/usr/bin/env python
"""
Gradient-Based Personality Attribution: Which model parameters contribute
most to personality encoding?

Instead of testing the model black-box (hooking activations), this analyzes
the GRADIENT of the personality signal with respect to model parameters.

Tests:
1. Parameter gradient magnitude: which weight matrices have the largest
   gradients when optimizing for personality?
2. Layer-wise gradient distribution: confirms L16-L17 localization
3. MLP vs Attention gradient: which contributes more?
4. Per-trait gradient specificity: do different traits use different parameters?
5. Gradient overlap: how much do trait gradients overlap?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="grad-attr")

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

    logger.info("Loading Marin 8B (with gradients)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    # Keep in eval mode but enable gradients for analysis
    model.eval()
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad_(True)
    blocks = get_decoder_blocks(model)
    num_layers = len(blocks)
    detect_layer = mid_layer + 1

    prompt = "Tell me about yourself."
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    results = {}
    basis_5d_tensor = torch.tensor(basis_5d, dtype=torch.float16).to(device)

    print(f"\n{'='*70}")
    print("GRADIENT-BASED PERSONALITY ATTRIBUTION")
    print(f"Model: Marin 8B, {num_layers} layers")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Compute gradient of 5D projection w.r.t. parameters
    # ================================================================
    logger.info("Part 1: Parameter gradients for personality...")
    print(f"\n{'='*70}")
    print("PART 1: PARAMETER GRADIENT MAGNITUDES")
    print(f"{'='*70}")

    # Strategy: for each trait, compute the 5D norm of the activation
    # at the detect layer, then backprop to get parameter gradients.
    # The gradient magnitude tells us which parameters most affect
    # the personality signal.

    trait_param_grads = {}

    for trait in ["artistic", "social", "conventional"]:
        logger.info(f"  Computing gradients for {trait}...")
        vec = residual[trait].astype(np.float32)
        delta = 2.0 * torch.tensor(vec, dtype=model.dtype).to(device)

        # Need to enable gradients for this computation
        model.zero_grad()

        # Forward with steering hook that allows gradients
        captured = {}

        def cap_fn(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :]  # Keep as tensor for gradient
            return out

        hook_cap = blocks[detect_layer].register_forward_hook(cap_fn)

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] = hs[:, -1, :] + delta
                return (hs,) + out[1:]
            out[:, -1, :] = out[:, -1, :] + delta
            return out

        hook_steer = blocks[mid_layer].register_forward_hook(steer_fn)

        # Forward pass (with grad)
        outputs = model(input_ids)

        hook_cap.remove()
        hook_steer.remove()

        if "act" not in captured:
            print(f"  Failed to capture activation for {trait}")
            continue

        # Project onto 5D and compute norm
        act = captured["act"]
        coords_5d_val = basis_5d_tensor @ act  # [5]
        personality_norm = torch.norm(coords_5d_val)

        # Backward
        personality_norm.backward()

        # Collect per-parameter gradient norms
        param_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                param_grads[name] = grad_norm

        trait_param_grads[trait] = param_grads
        model.zero_grad()

        # Summarize by layer and component
        layer_mlp_grad = np.zeros(num_layers)
        layer_attn_grad = np.zeros(num_layers)
        layer_total_grad = np.zeros(num_layers)

        for name, gnorm in param_grads.items():
            for lidx in range(num_layers):
                if f"layers.{lidx}." in name:
                    if "mlp" in name:
                        layer_mlp_grad[lidx] += gnorm
                    elif "self_attn" in name:
                        layer_attn_grad[lidx] += gnorm
                    layer_total_grad[lidx] += gnorm

        print(f"\n  {trait}:")
        print(f"  {'Layer':>5} {'MLP Grad':>10} {'Attn Grad':>10} {'Total':>10} {'MLP%':>6}")
        for lidx in range(num_layers):
            if layer_total_grad[lidx] > 0.001:
                mlp_frac = layer_mlp_grad[lidx] / layer_total_grad[lidx] if layer_total_grad[lidx] > 0 else 0
                print(f"  L{lidx:>3} {layer_mlp_grad[lidx]:>10.4f} "
                      f"{layer_attn_grad[lidx]:>10.4f} {layer_total_grad[lidx]:>10.4f} "
                      f"{mlp_frac:>6.0%}")

        # Store layer-level summary
        results[f"layer_gradients_{trait}"] = {
            "mlp": layer_mlp_grad.tolist(),
            "attn": layer_attn_grad.tolist(),
            "total": layer_total_grad.tolist(),
        }

    # ================================================================
    # PART 2: Top parameters across traits
    # ================================================================
    logger.info("Part 2: Top parameters...")
    print(f"\n{'='*70}")
    print("PART 2: TOP PARAMETERS BY GRADIENT MAGNITUDE")
    print(f"{'='*70}")

    # Average across traits
    all_params = set()
    for pg in trait_param_grads.values():
        all_params.update(pg.keys())

    avg_grads = {}
    for name in all_params:
        vals = [trait_param_grads[t].get(name, 0) for t in trait_param_grads]
        avg_grads[name] = np.mean(vals)

    sorted_params = sorted(avg_grads.items(), key=lambda x: -x[1])
    print(f"\n  Top 30 parameters by average gradient norm:")
    top_params = []
    for name, gnorm in sorted_params[:30]:
        print(f"    {gnorm:.6f}  {name}")
        top_params.append({"name": name, "grad_norm": float(gnorm)})

    results["top_parameters"] = top_params

    # ================================================================
    # PART 3: Cross-trait gradient similarity
    # ================================================================
    logger.info("Part 3: Cross-trait gradient similarity...")
    print(f"\n{'='*70}")
    print("PART 3: CROSS-TRAIT GRADIENT SIMILARITY")
    print(f"{'='*70}")

    # Build gradient vectors per trait (only for shared parameters)
    shared_params = sorted(all_params)
    trait_grad_vecs = {}
    for trait, pg in trait_param_grads.items():
        vec = np.array([pg.get(name, 0) for name in shared_params])
        trait_grad_vecs[trait] = vec

    # Pairwise cosine similarity
    tested_traits = list(trait_param_grads.keys())
    print(f"\n  Gradient cosine similarity:")
    print(f"  {'':>15}", end="")
    for t2 in tested_traits:
        print(f" {t2[:7]:>8}", end="")
    print()

    grad_cos = {}
    for t1 in tested_traits:
        print(f"  {t1:>15}", end="")
        for t2 in tested_traits:
            v1 = trait_grad_vecs[t1]
            v2 = trait_grad_vecs[t2]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            grad_cos[f"{t1}_{t2}"] = cos
            print(f" {cos:>8.3f}", end="")
        print()

    results["gradient_cosine"] = grad_cos

    # ================================================================
    # PART 4: Component-level breakdown
    # ================================================================
    logger.info("Part 4: Component breakdown...")
    print(f"\n{'='*70}")
    print("PART 4: COMPONENT-LEVEL GRADIENT BREAKDOWN")
    print(f"{'='*70}")

    components = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "input_layernorm", "post_attention_layernorm"]
    comp_results = {}

    for comp in components:
        total_grad = 0
        count = 0
        for name, gnorm in avg_grads.items():
            if comp in name:
                total_grad += gnorm
                count += 1
        comp_results[comp] = {"total_grad": float(total_grad), "count": count}

    # Sort by total gradient
    sorted_comps = sorted(comp_results.items(), key=lambda x: -x[1]["total_grad"])
    print(f"\n  Component gradient ranking:")
    for comp, data in sorted_comps:
        print(f"    {comp:>25}: total_grad={data['total_grad']:.4f} ({data['count']} params)")

    results["component_breakdown"] = comp_results

    # ================================================================
    # PART 5: Layer concentration (Gini coefficient)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: GRADIENT CONCENTRATION ANALYSIS")
    print(f"{'='*70}")

    for trait in trait_param_grads:
        total_grads = results[f"layer_gradients_{trait}"]["total"]
        total_arr = np.array(total_grads)
        if total_arr.sum() > 0:
            # Gini of layer-level gradients
            sorted_g = np.sort(total_arr)
            n = len(sorted_g)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_g)) / (n * np.sum(sorted_g))) - (n+1)/n

            # Peak layer
            peak_layer = int(np.argmax(total_arr))
            peak_frac = total_arr[peak_layer] / total_arr.sum()

            # Layers with >5% of total
            significant = sum(1 for g in total_arr if g > 0.05 * total_arr.sum())

            print(f"  {trait:>15}: Gini={gini:.3f}, peak=L{peak_layer} ({peak_frac:.0%} of total), "
                  f"significant layers={significant}/{num_layers}")

            results[f"gradient_concentration_{trait}"] = {
                "gini": float(gini),
                "peak_layer": peak_layer,
                "peak_fraction": float(peak_frac),
                "significant_layers": significant,
            }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    off_diag_cos = [v for k, v in grad_cos.items() if k.split("_")[0] != k.split("_")[1]]
    mean_cross_cos = np.mean(off_diag_cos) if off_diag_cos else 0
    print(f"  Cross-trait gradient cosine: {mean_cross_cos:.3f}")
    print(f"  Top component: {sorted_comps[0][0]} (grad={sorted_comps[0][1]['total_grad']:.4f})")

    results["summary"] = {
        "mean_cross_trait_gradient_cos": float(mean_cross_cos),
        "top_component": sorted_comps[0][0],
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_gradient_attribution.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
