#!/usr/bin/env python
"""
Multi-Layer Generation Dynamics: Personality Signal Across All Layers During Generation.

Tracks the 5D personality signal at EVERY layer (L0-L31) during autoregressive
generation. This answers critical mechanistic questions:

1. Does the personality "wavefront" propagate from injection layer (L16) upward
   during generation, or does it appear everywhere simultaneously?
2. At which layers does the signal grow vs decay during generation?
3. Is there a "personality echo" — does the signal appear at layers BELOW
   injection during later generation steps?
4. How does the layer profile evolve from token 1 to token 80?

This provides the most complete picture of personality dynamics in the model.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="multilayer-gen")

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


def generate_with_multilayer_tracking(model, tokenizer, device, blocks, mid_layer,
                                       num_layers, user_prompt,
                                       steer_vec=None, alpha=0.0,
                                       max_tokens=60):
    """
    Generate text while capturing activations at ALL layers for each token.
    Returns per-step, per-layer activations.
    """
    messages = [{"role": "user", "content": user_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Storage: list of dicts, one per forward pass
    # Each dict maps layer_idx -> activation vector
    all_step_acts = []

    hooks = []

    # Register capture hooks on ALL layers
    for lidx in range(num_layers):
        def make_hook(layer_idx):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                act = hs[0, -1, :].detach().cpu().numpy().copy()
                # Store in the current step's dict
                if all_step_acts:
                    all_step_acts[-1][layer_idx] = act
                return out
            return hook_fn
        hooks.append(blocks[lidx].register_forward_hook(make_hook(lidx)))

    # Steering hook at mid_layer
    steer_handle = None
    if steer_vec is not None and alpha != 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        steer_handle = blocks[mid_layer].register_forward_hook(steer_fn)

    # Use manual generation to guarantee hooks fire
    generated_ids = []
    past_kv = None
    current_ids = input_ids

    try:
        with torch.no_grad():
            for step in range(max_tokens):
                # Create new step entry BEFORE forward pass
                all_step_acts.append({})

                if past_kv is not None:
                    outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                else:
                    outputs = model(current_ids, use_cache=True)

                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_id = logits.argmax(-1)
                generated_ids.append(next_id.item())
                current_ids = next_id.unsqueeze(0)

                # Check for EOS
                if next_id.item() == tokenizer.eos_token_id:
                    break
    finally:
        for h in hooks:
            h.remove()
        if steer_handle:
            steer_handle.remove()

    # The first step captures prefill activations (context + first generated token)
    # Steps 1+ capture per-generated-token activations
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    gen_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

    return {
        "text": gen_text,
        "tokens": gen_tokens,
        "per_step_per_layer_acts": all_step_acts,
        "prompt_len": input_ids.shape[1],
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
    num_layers = model_data["num_layers"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    gen_prompt = "Tell me about your interests and what you enjoy doing."
    results = {}

    print(f"\n{'='*70}")
    print("MULTI-LAYER GENERATION DYNAMICS")
    print(f"Model: Marin 8B, {num_layers} layers, mid={mid_layer}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Baseline generation — all layers
    # ================================================================
    logger.info("Part 1: Baseline generation (all layers)...")
    baseline = generate_with_multilayer_tracking(
        model, tokenizer, device, blocks, mid_layer,
        num_layers, gen_prompt, max_tokens=60)

    print(f"\n  Baseline: {baseline['text'][:150]}...")
    print(f"  Generated {len(baseline['tokens'])} tokens")

    baseline_acts = baseline["per_step_per_layer_acts"]

    # ================================================================
    # PART 2: Steered generation — track all layers
    # ================================================================
    logger.info("Part 2: Steered generation (artistic, α=2.0)...")
    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0

    steered = generate_with_multilayer_tracking(
        model, tokenizer, device, blocks, mid_layer,
        num_layers, gen_prompt, steer_vec=vec, alpha=alpha, max_tokens=60)

    print(f"\n  Steered ({test_trait} α={alpha}): {steered['text'][:150]}...")

    steered_acts = steered["per_step_per_layer_acts"]

    # Analyze: for each generation step, compute 5D signal at each layer
    num_steps = min(len(baseline_acts), len(steered_acts))

    print(f"\n{'='*70}")
    print("PART 2: LAYER-BY-STEP PERSONALITY HEATMAP")
    print(f"{'='*70}")

    # Build heatmap: [step, layer] -> 5d_norm
    heatmap_norm = np.zeros((num_steps, num_layers))
    heatmap_cos = np.zeros((num_steps, num_layers))
    heatmap_detected = [[None]*num_layers for _ in range(num_steps)]

    for step in range(num_steps):
        for lidx in range(num_layers):
            if lidx in steered_acts[step] and lidx in baseline_acts[step]:
                diff = (steered_acts[step][lidx] - baseline_acts[step][lidx]).astype(np.float64)
                coords = basis_5d @ diff
                norm_5d = float(np.linalg.norm(coords))
                heatmap_norm[step, lidx] = norm_5d

                sims = {}
                for t in TRAITS:
                    if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                        sims[t] = float(np.dot(coords, coords_5d[t]) / (
                            norm_5d * np.linalg.norm(coords_5d[t])))
                    else:
                        sims[t] = 0
                best = max(sims, key=sims.get)
                heatmap_detected[step][lidx] = best
                heatmap_cos[step, lidx] = sims.get(test_trait, 0)

    # Print compact heatmap of 5D norms
    # Show every 4th layer, every 5th step
    layers_to_show = list(range(0, num_layers, 4)) + [num_layers-1]
    layers_to_show = sorted(set(layers_to_show))

    print(f"\n  5D Norm Heatmap (rows=steps, cols=layers):")
    header = f"  {'Step':>5} " + " ".join(f"L{l:>2}" for l in layers_to_show)
    print(header)

    for step in range(num_steps):
        if step < 10 or step % 5 == 0 or step == num_steps - 1:
            row = f"  {step:>5} "
            for lidx in layers_to_show:
                val = heatmap_norm[step, lidx]
                row += f" {val:>4.0f}" if val > 0.5 else "    ."
            print(row)

    # Print cosine heatmap
    print(f"\n  Target Cosine Heatmap (rows=steps, cols=layers):")
    header = f"  {'Step':>5} " + " ".join(f"L{l:>2}" for l in layers_to_show)
    print(header)

    for step in range(num_steps):
        if step < 10 or step % 5 == 0 or step == num_steps - 1:
            row = f"  {step:>5} "
            for lidx in layers_to_show:
                val = heatmap_cos[step, lidx]
                if heatmap_norm[step, lidx] > 1.0:
                    row += f" {val:>4.2f}"[0:5]
                else:
                    row += "    ."
            print(row)

    # ================================================================
    # PART 3: Layer profile evolution
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: LAYER PROFILE EVOLUTION")
    print(f"{'='*70}")

    # Compare early vs late steps
    early_steps = list(range(min(5, num_steps)))
    late_steps = list(range(max(0, num_steps-5), num_steps))

    early_profile = np.mean(heatmap_norm[early_steps, :], axis=0)
    late_profile = np.mean(heatmap_norm[late_steps, :], axis=0)

    print(f"\n  Layer profile (mean 5D norm):")
    print(f"  {'Layer':>5} {'Early(0-4)':>12} {'Late(last5)':>12} {'Growth':>10}")
    for lidx in range(num_layers):
        growth = (late_profile[lidx] / early_profile[lidx] - 1) if early_profile[lidx] > 0.1 else float('nan')
        print(f"  L{lidx:>3} {early_profile[lidx]:>12.2f} {late_profile[lidx]:>12.2f} "
              f"{growth:>+10.1%}" if not np.isnan(growth) else
              f"  L{lidx:>3} {early_profile[lidx]:>12.2f} {late_profile[lidx]:>12.2f} {'n/a':>10}")

    # Find which layers show personality signal
    mean_norm_per_layer = np.mean(heatmap_norm, axis=0)
    signal_layers = [l for l in range(num_layers) if mean_norm_per_layer[l] > 5.0]
    onset_layer = signal_layers[0] if signal_layers else -1

    print(f"\n  Signal onset layer: L{onset_layer}")
    print(f"  Signal layers (norm > 5): {signal_layers}")
    print(f"  Peak signal layer: L{np.argmax(mean_norm_per_layer)} "
          f"(mean norm={np.max(mean_norm_per_layer):.1f})")

    # ================================================================
    # PART 4: Below-injection echo
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: BELOW-INJECTION PERSONALITY ECHO")
    print(f"{'='*70}")

    # Do layers BELOW mid_layer develop personality signal during generation?
    below_injection_norms = heatmap_norm[:, :mid_layer]
    above_injection_norms = heatmap_norm[:, mid_layer+1:]

    print(f"\n  Mean 5D norm below injection (L0-L{mid_layer-1}):")
    for step in range(num_steps):
        if step < 10 or step % 5 == 0 or step == num_steps - 1:
            below_mean = np.mean(below_injection_norms[step, :])
            above_mean = np.mean(above_injection_norms[step, :])
            print(f"    Step {step:>3}: below={below_mean:.2f}, above={above_mean:.2f}, "
                  f"ratio={below_mean/above_mean:.3f}" if above_mean > 0 else
                  f"    Step {step:>3}: below={below_mean:.2f}, above={above_mean:.2f}")

    # Is there a growing echo?
    if num_steps > 5:
        early_below = np.mean(below_injection_norms[:5, :])
        late_below = np.mean(below_injection_norms[-5:, :])
        early_above = np.mean(above_injection_norms[:5, :])
        late_above = np.mean(above_injection_norms[-5:, :])

        print(f"\n  Echo analysis:")
        print(f"    Early below-injection mean: {early_below:.2f}")
        print(f"    Late below-injection mean: {late_below:.2f}")
        print(f"    Growth: {(late_below/early_below - 1):+.1%}" if early_below > 0.1 else "    Growth: n/a")
        print(f"    Early above-injection mean: {early_above:.2f}")
        print(f"    Late above-injection mean: {late_above:.2f}")
        print(f"    Growth: {(late_above/early_above - 1):+.1%}" if early_above > 0.1 else "    Growth: n/a")

    # ================================================================
    # PART 5: Detection accuracy heatmap
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: DETECTION ACCURACY PER LAYER PER STEP")
    print(f"{'='*70}")

    # Compute detection accuracy per layer across all steps
    layer_accuracy = {}
    for lidx in range(num_layers):
        correct = sum(1 for step in range(num_steps)
                     if heatmap_detected[step][lidx] == test_trait)
        layer_accuracy[lidx] = correct / num_steps if num_steps > 0 else 0

    print(f"\n  Detection accuracy ({test_trait}) per layer:")
    for lidx in range(num_layers):
        bar = "#" * int(layer_accuracy[lidx] * 40)
        print(f"  L{lidx:>2}: {layer_accuracy[lidx]:>5.1%} {bar}")

    # ================================================================
    # PART 6: Second trait for comparison
    # ================================================================
    logger.info("Part 6: Social trait comparison...")
    test_trait2 = "social"
    vec2 = residual[test_trait2].astype(np.float32)

    steered2 = generate_with_multilayer_tracking(
        model, tokenizer, device, blocks, mid_layer,
        num_layers, gen_prompt, steer_vec=vec2, alpha=alpha, max_tokens=60)

    steered2_acts = steered2["per_step_per_layer_acts"]
    num_steps2 = min(len(baseline_acts), len(steered2_acts))

    heatmap_norm2 = np.zeros((num_steps2, num_layers))
    heatmap_cos2 = np.zeros((num_steps2, num_layers))

    for step in range(num_steps2):
        for lidx in range(num_layers):
            if lidx in steered2_acts[step] and lidx in baseline_acts[step]:
                diff = (steered2_acts[step][lidx] - baseline_acts[step][lidx]).astype(np.float64)
                coords = basis_5d @ diff
                norm_5d = float(np.linalg.norm(coords))
                heatmap_norm2[step, lidx] = norm_5d

                sims = {}
                for t in TRAITS:
                    if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                        sims[t] = float(np.dot(coords, coords_5d[t]) / (
                            norm_5d * np.linalg.norm(coords_5d[t])))
                    else:
                        sims[t] = 0
                heatmap_cos2[step, lidx] = sims.get(test_trait2, 0)

    print(f"\n{'='*70}")
    print(f"PART 6: SOCIAL COMPARISON")
    print(f"{'='*70}")
    print(f"  Social text: {steered2['text'][:150]}...")

    mean_norm2 = np.mean(heatmap_norm2, axis=0)
    peak2 = np.argmax(mean_norm2)
    print(f"  Peak layer: L{peak2} (mean norm={mean_norm2[peak2]:.1f})")

    # Compare artistic vs social layer profiles
    print(f"\n  Layer profile comparison:")
    print(f"  {'Layer':>5} {'Artistic':>10} {'Social':>10}")
    for lidx in range(num_layers):
        print(f"  L{lidx:>3} {mean_norm_per_layer[lidx]:>10.2f} {mean_norm2[lidx]:>10.2f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Key metrics
    print(f"\n  Artistic (α={alpha}):")
    print(f"    Signal onset: L{onset_layer}")
    print(f"    Peak layer: L{np.argmax(mean_norm_per_layer)} (norm={np.max(mean_norm_per_layer):.1f})")
    print(f"    Signal growth (early→late): "
          f"above-injection {(late_above/early_above - 1):+.1%}" if early_above > 0.1 else "    n/a")
    print(f"    Below-injection echo: "
          f"{late_below:.2f} (was {early_below:.2f})" if early_below > 0.01 else "    No echo")

    # Layers with 100% detection
    perfect_layers = [l for l, acc in layer_accuracy.items() if acc >= 0.99]
    print(f"    100% detection layers: {perfect_layers}")

    # Save results
    results = {
        "heatmap_norm_artistic": heatmap_norm.tolist(),
        "heatmap_cos_artistic": heatmap_cos.tolist(),
        "heatmap_norm_social": heatmap_norm2.tolist(),
        "heatmap_cos_social": heatmap_cos2.tolist(),
        "mean_norm_per_layer_artistic": mean_norm_per_layer.tolist(),
        "mean_norm_per_layer_social": mean_norm2.tolist(),
        "layer_accuracy_artistic": layer_accuracy,
        "signal_onset_layer": onset_layer,
        "signal_layers": signal_layers,
        "peak_layer": int(np.argmax(mean_norm_per_layer)),
        "num_steps": num_steps,
        "artistic_text": steered["text"],
        "social_text": steered2["text"],
        "baseline_text": baseline["text"],
        "summary": {
            "model": model_id,
            "num_layers": num_layers,
            "mid_layer": mid_layer,
            "alpha": alpha,
        },
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multilayer_generation_dynamics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
