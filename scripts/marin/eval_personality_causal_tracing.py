#!/usr/bin/env python
"""
Personality Causal Tracing.

Which layers are NECESSARY for personality expression?
Uses activation patching: replace one layer's output with baseline (clean)
activations and measure the effect on personality detection and behavior.

If patching layer L causes personality to drop, that layer is CRITICAL.
If patching has no effect, personality has already bypassed that layer.

Tests:
1. Layer patching for activation steering: which layers must the steered
   signal pass through to produce behavioral effects?
2. Layer patching for system prompts: which layers process the system prompt
   personality?
3. Restore-one patching: in a fully-patched model, which single layer
   restores the most personality?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="causal-trace")

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
        "num_layers": config.num_hidden_layers,
    }


def measure_profile_with_patches(model, tokenizer, device, blocks, mid_layer,
                                  baseline_logprobs, baseline_acts_per_layer,
                                  steer_vec=None, alpha=0.0, system_prompt=None,
                                  patch_layers=None):
    """
    Measure behavioral profile with optional steering and layer patching.

    patch_layers: set of layer indices where we REPLACE the output
    with the cached baseline activation (removing personality at that layer).
    """
    hooks = []

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

    # Patching hooks: replace output at specific layers with baseline activations
    if patch_layers:
        for lidx in patch_layers:
            baseline_act = baseline_acts_per_layer[lidx]
            base_tensor = torch.tensor(baseline_act, dtype=model.dtype).to(device)
            def make_patch(bt):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[0, -1, :] = bt
                        return (hs,) + out[1:]
                    out[0, -1, :] = bt
                    return out
                return hook_fn
            hooks.append(blocks[lidx].register_forward_hook(make_patch(base_tensor)))

    try:
        logprobs = {}
        for i, ta in enumerate(TRAITS):
            for j, tb in enumerate(TRAITS):
                if i >= j:
                    continue
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content":
                    f"Which describes you better? Answer with just A or B.\n"
                    f"A) I am {TRAIT_DESCRIPTIONS[ta]}\n"
                    f"B) I am {TRAIT_DESCRIPTIONS[tb]}"})
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(formatted, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    out = model(input_ids=input_ids)
                lp = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
                a_id = tokenizer.encode("A", add_special_tokens=False)[0]
                b_id = tokenizer.encode("B", add_special_tokens=False)[0]
                logprobs[f"{ta}-{tb}"] = lp[a_id].item() - lp[b_id].item()
    finally:
        for h in hooks:
            h.remove()

    deltas = {t: 0.0 for t in TRAITS}
    counts = {t: 0 for t in TRAITS}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            shift = logprobs[f"{ta}-{tb}"] - baseline_logprobs[f"{ta}-{tb}"]
            deltas[ta] += shift; counts[ta] += 1
            deltas[tb] -= shift; counts[tb] += 1
    for t in TRAITS:
        if counts[t] > 0:
            deltas[t] /= counts[t]
    return deltas


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

    # Compute behavioral baseline
    logger.info("Computing behavioral baseline...")
    baseline = {}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [{"role": "user", "content":
                f"Which describes you better? Answer with just A or B.\n"
                f"A) I am {TRAIT_DESCRIPTIONS[ta]}\nB) I am {TRAIT_DESCRIPTIONS[tb]}"}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids)
            lp = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
            a_id = tokenizer.encode("A", add_special_tokens=False)[0]
            b_id = tokenizer.encode("B", add_special_tokens=False)[0]
            baseline[f"{ta}-{tb}"] = lp[a_id].item() - lp[b_id].item()

    # Cache baseline activations at ALL layers for patching
    logger.info("Caching baseline activations at all layers...")
    # We need per-prompt baseline activations for the forced-choice prompts
    # For simplicity, cache baselines for a detection prompt
    detect_prompt = "Tell me about yourself."
    baseline_acts = {}
    hooks = []
    for lidx in range(num_layers):
        def make_cap(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                baseline_acts[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn
        hooks.append(blocks[lidx].register_forward_hook(make_cap(lidx)))

    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    with torch.no_grad():
        model(input_ids=enc["input_ids"].to(device))
    for h in hooks:
        h.remove()

    # Also cache per-pair baseline activations (needed for patching during eval)
    logger.info("Caching per-pair baseline activations...")
    pair_baseline_acts = {}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            pair_key = f"{ta}-{tb}"
            messages = [{"role": "user", "content":
                f"Which describes you better? Answer with just A or B.\n"
                f"A) I am {TRAIT_DESCRIPTIONS[ta]}\nB) I am {TRAIT_DESCRIPTIONS[tb]}"}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")

            pair_acts = {}
            hooks = []
            for lidx in range(num_layers):
                def make_cap(l):
                    def hook_fn(_module, _inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        pair_acts[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                        return out
                    return hook_fn
                hooks.append(blocks[lidx].register_forward_hook(make_cap(lidx)))

            with torch.no_grad():
                model(input_ids=enc["input_ids"].to(device))
            for h in hooks:
                h.remove()
            pair_baseline_acts[pair_key] = pair_acts

    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY CAUSAL TRACING")
    print(f"Model: Marin 8B ({num_layers} layers)")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Activation steering — patch one layer at a time
    # ================================================================
    logger.info("Part 1: Steering causal tracing (patch one layer)...")
    print(f"\n{'='*70}")
    print("PART 1: ACTIVATION STEERING — PATCH ONE LAYER")
    print(f"(artistic α=2 at L{mid_layer}, patch one layer to baseline)")
    print(f"{'='*70}")

    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0

    # First, measure unpatched steered profile
    unpatched = measure_profile_with_patches(
        model, tokenizer, device, blocks, mid_layer, baseline,
        baseline_acts, steer_vec=vec, alpha=alpha)
    unpatched_mag = float(np.sqrt(sum(v**2 for v in unpatched.values())))
    unpatched_delta = unpatched[test_trait]

    steer_causal = []
    print(f"\n  Unpatched: magnitude={unpatched_mag:.3f}, {test_trait}_delta={unpatched_delta:+.3f}")
    print(f"\n  {'Layer':>5} {'Magnitude':>10} {'Δ Retained':>12} {'Effect':>10}")

    for patch_layer in range(num_layers):
        patched = measure_profile_with_patches(
            model, tokenizer, device, blocks, mid_layer, baseline,
            baseline_acts, steer_vec=vec, alpha=alpha,
            patch_layers={patch_layer})
        pat_mag = float(np.sqrt(sum(v**2 for v in patched.values())))
        retention = pat_mag / unpatched_mag if unpatched_mag > 0.01 else 0
        effect = 1.0 - retention  # How much patching this layer removes

        print(f"  L{patch_layer:>3} {pat_mag:>10.3f} {retention:>12.1%} {effect:>10.1%}")

        steer_causal.append({
            "layer": patch_layer,
            "patched_magnitude": pat_mag,
            "retention": float(retention),
            "causal_effect": float(effect),
        })

    results["steering_causal"] = steer_causal

    # Find critical layers
    critical_steer = [d for d in steer_causal if d["causal_effect"] > 0.1]
    print(f"\n  Critical layers (>10% effect): {[d['layer'] for d in critical_steer]}")

    # ================================================================
    # PART 2: System prompt — patch one layer at a time
    # ================================================================
    logger.info("Part 2: System prompt causal tracing...")
    print(f"\n{'='*70}")
    print("PART 2: SYSTEM PROMPT — PATCH ONE LAYER")
    print(f"{'='*70}")

    sysp_causal = {}

    for sp_trait in ["artistic", "social"]:
        sys_prompt = PERSONALITY_SYSTEM_PROMPTS[sp_trait]
        logger.info(f"  {sp_trait}...")

        unpatched = measure_profile_with_patches(
            model, tokenizer, device, blocks, mid_layer, baseline,
            baseline_acts, system_prompt=sys_prompt)
        unpatched_mag = float(np.sqrt(sum(v**2 for v in unpatched.values())))

        trait_data = []
        print(f"\n  {sp_trait} (unpatched: {unpatched_mag:.3f}):")
        print(f"  {'Layer':>5} {'Magnitude':>10} {'Retained':>10} {'Effect':>10}")

        for patch_layer in range(num_layers):
            patched = measure_profile_with_patches(
                model, tokenizer, device, blocks, mid_layer, baseline,
                baseline_acts, system_prompt=sys_prompt,
                patch_layers={patch_layer})
            pat_mag = float(np.sqrt(sum(v**2 for v in patched.values())))
            retention = pat_mag / unpatched_mag if unpatched_mag > 0.01 else 0
            effect = 1.0 - retention

            print(f"  L{patch_layer:>3} {pat_mag:>10.3f} {retention:>10.1%} {effect:>10.1%}")

            trait_data.append({
                "layer": patch_layer,
                "patched_magnitude": pat_mag,
                "retention": float(retention),
                "causal_effect": float(effect),
            })

        sysp_causal[sp_trait] = trait_data
        critical = [d for d in trait_data if d["causal_effect"] > 0.1]
        print(f"\n  Critical layers (>10% effect): {[d['layer'] for d in critical]}")

    results["sysprompt_causal"] = sysp_causal

    # ================================================================
    # PART 3: Window patching (patch contiguous blocks)
    # ================================================================
    logger.info("Part 3: Window patching...")
    print(f"\n{'='*70}")
    print("PART 3: WINDOW PATCHING (patch 4-layer contiguous blocks)")
    print(f"{'='*70}")

    window_size = 4
    window_results = {}

    for condition_name, condition_kwargs in [
        ("steering", {"steer_vec": vec, "alpha": alpha}),
        ("sysprompt_artistic", {"system_prompt": PERSONALITY_SYSTEM_PROMPTS["artistic"]}),
    ]:
        unpatched = measure_profile_with_patches(
            model, tokenizer, device, blocks, mid_layer, baseline,
            baseline_acts, **condition_kwargs)
        unpatched_mag = float(np.sqrt(sum(v**2 for v in unpatched.values())))

        window_data = []
        print(f"\n  {condition_name} (unpatched: {unpatched_mag:.3f}):")
        print(f"  {'Window':>12} {'Magnitude':>10} {'Retained':>10} {'Effect':>10}")

        for start in range(0, num_layers - window_size + 1, window_size):
            patch_set = set(range(start, min(start + window_size, num_layers)))
            patched = measure_profile_with_patches(
                model, tokenizer, device, blocks, mid_layer, baseline,
                baseline_acts, patch_layers=patch_set, **condition_kwargs)
            pat_mag = float(np.sqrt(sum(v**2 for v in patched.values())))
            retention = pat_mag / unpatched_mag if unpatched_mag > 0.01 else 0
            effect = 1.0 - retention

            label = f"L{start}-L{min(start+window_size-1, num_layers-1)}"
            print(f"  {label:>12} {pat_mag:>10.3f} {retention:>10.1%} {effect:>10.1%}")

            window_data.append({
                "start": start,
                "end": min(start + window_size - 1, num_layers - 1),
                "patched_magnitude": pat_mag,
                "retention": float(retention),
                "causal_effect": float(effect),
            })

        window_results[condition_name] = window_data

    results["window_patching"] = window_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Steering: identify the single most critical layer
    steer_sorted = sorted(steer_causal, key=lambda x: -x["causal_effect"])
    print(f"\n  Steering most critical layer: L{steer_sorted[0]['layer']} "
          f"({steer_sorted[0]['causal_effect']:.1%} effect)")
    top5_steer = [(d["layer"], d["causal_effect"]) for d in steer_sorted[:5]]
    print(f"  Steering top-5 layers: {top5_steer}")

    for sp_trait, data in sysp_causal.items():
        sp_sorted = sorted(data, key=lambda x: -x["causal_effect"])
        print(f"\n  {sp_trait} most critical layer: L{sp_sorted[0]['layer']} "
              f"({sp_sorted[0]['causal_effect']:.1%} effect)")
        top5_sp = [(d["layer"], d["causal_effect"]) for d in sp_sorted[:5]]
        print(f"  {sp_trait} top-5 layers: {top5_sp}")

    results["summary"] = {
        "model": model_id,
        "num_layers": num_layers,
        "steering_most_critical": steer_sorted[0]["layer"],
        "steering_critical_effect": steer_sorted[0]["causal_effect"],
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_causal_tracing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
