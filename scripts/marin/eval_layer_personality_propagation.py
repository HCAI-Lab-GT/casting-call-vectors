#!/usr/bin/env python
"""
Layer-by-Layer Personality Signal Propagation.

Maps where personality "lives" across model layers for BOTH mechanisms:
1. Activation steering: known to be 0 at L0-L16, 100% at L17-L31
2. System prompt personality: where does it appear and concentrate?

This reveals:
- Does system prompt personality emerge gradually or suddenly?
- Which layers carry the most personality information?
- Is there a "personality-free" zone in either mechanism?
- Do the two mechanisms overlap in any layers?

Also tests RIASEC 5D capture ratio at each layer — does the 5D basis
work equally well at all layers, or is personality geometry layer-dependent?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="layer-prop")

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
        "beauty, and originality above all else."
    ),
    "investigative": (
        "You are an analytical and scientific individual. You are driven by curiosity "
        "and the desire to understand how things work."
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
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Build per-layer 5D bases
    per_layer_bases = {}
    per_layer_coords = {}
    for layer_idx in range(num_layers):
        V = np.stack([all_layer_vectors[t][layer_idx] for t in TRAITS])
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
        shared_dir = Vt[0]
        shared_dir = shared_dir / np.linalg.norm(shared_dir)

        residual = {}
        for t in TRAITS:
            vec = all_layer_vectors[t][layer_idx]
            proj = np.dot(vec, shared_dir) * shared_dir
            residual[t] = vec - proj

        V_res = np.stack([residual[t] for t in TRAITS])
        U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
        per_layer_bases[layer_idx] = Vt_res[:5]
        per_layer_coords[layer_idx] = {t: Vt_res[:5] @ residual[t] for t in TRAITS}

    # Also build the canonical (mid-layer) basis
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
    canonical_basis = Vt_res[:5]
    canonical_coords = {t: canonical_basis @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "per_layer_bases": per_layer_bases,
        "per_layer_coords": per_layer_coords,
        "canonical_basis": canonical_basis,
        "canonical_coords": canonical_coords,
        "mid_layer": mid_layer,
        "num_layers": num_layers,
        "all_layer_vectors": all_layer_vectors,
    }


def capture_all_layers(model, tokenizer, device, blocks, num_layers,
                        user_prompt, system_prompt=None,
                        steer_vec=None, alpha=0.0, steer_layer=None):
    """Capture activations at ALL layers in a single forward pass."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}
    hooks = []

    for lidx in range(num_layers):
        def make_cap(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn
        hooks.append(blocks[lidx].register_forward_hook(make_cap(lidx)))

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

    return captured


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    mid_layer = model_data["mid_layer"]
    num_layers = model_data["num_layers"]
    residual = model_data["residual"]
    canonical_basis = model_data["canonical_basis"]
    canonical_coords = model_data["canonical_coords"]
    per_layer_bases = model_data["per_layer_bases"]
    per_layer_coords = model_data["per_layer_coords"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompt = "Tell me about yourself."

    # Capture baselines at all layers
    logger.info("Capturing baseline at all layers...")
    baseline_all = capture_all_layers(
        model, tokenizer, device, blocks, num_layers, detect_prompt)

    results = {}

    print(f"\n{'='*70}")
    print(f"LAYER-BY-LAYER PERSONALITY PROPAGATION")
    print(f"Model: Marin 8B ({num_layers} layers)")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Activation steering signal at each layer
    # ================================================================
    logger.info("Part 1: Activation steering layer sweep...")
    print(f"\n{'='*70}")
    print("PART 1: ACTIVATION STEERING — PERSONALITY SIGNAL AT EACH LAYER")
    print(f"(Injecting at L{mid_layer}, measuring at each layer)")
    print(f"{'='*70}")

    test_trait = "artistic"
    vec = residual[test_trait].astype(np.float32)
    alpha = 2.0

    steered_all = capture_all_layers(
        model, tokenizer, device, blocks, num_layers, detect_prompt,
        steer_vec=vec, alpha=alpha, steer_layer=mid_layer)

    steer_layer_data = []

    print(f"\n  {'Layer':>5} {'5D Norm':>10} {'Full Norm':>10} {'Capture':>10} {'Correct':>8}")

    for lidx in range(num_layers):
        diff = (steered_all[lidx] - baseline_all[lidx]).astype(np.float64)
        full_norm = float(np.linalg.norm(diff))

        # Project onto canonical basis
        detected_coords = canonical_basis @ diff
        detected_norm = float(np.linalg.norm(detected_coords))
        capture = detected_norm / full_norm if full_norm > 1e-6 else 0

        # Identify trait
        sims = {}
        for t in TRAITS:
            if detected_norm > 0 and np.linalg.norm(canonical_coords[t]) > 0:
                sims[t] = float(np.dot(detected_coords, canonical_coords[t]) / (
                    detected_norm * np.linalg.norm(canonical_coords[t])))
            else:
                sims[t] = 0
        best = max(sims, key=sims.get)
        correct = best == test_trait

        print(f"  L{lidx:>3} {detected_norm:>10.2f} {full_norm:>10.2f} {capture:>10.3f} "
              f"{'OK' if correct else 'MISS':>8}")

        steer_layer_data.append({
            "layer": lidx,
            "5d_norm": detected_norm,
            "full_norm": full_norm,
            "capture_ratio": capture,
            "correct": bool(correct),
            "detected_trait": best,
            "cosine": float(sims.get(test_trait, 0)),
        })

    results["steering_propagation"] = steer_layer_data

    # ================================================================
    # PART 2: System prompt personality at each layer
    # ================================================================
    logger.info("Part 2: System prompt layer sweep...")
    print(f"\n{'='*70}")
    print("PART 2: SYSTEM PROMPT — PERSONALITY SIGNAL AT EACH LAYER")
    print(f"{'='*70}")

    sysprompt_layer_data = {}

    for sp_trait, sys_prompt in PERSONALITY_SYSTEM_PROMPTS.items():
        logger.info(f"  {sp_trait}...")
        sysp_all = capture_all_layers(
            model, tokenizer, device, blocks, num_layers, detect_prompt,
            system_prompt=sys_prompt)

        trait_data = []
        print(f"\n  {sp_trait}:")
        print(f"  {'Layer':>5} {'5D Norm':>10} {'Full Norm':>10} {'Capture':>10} {'Detected':>15}")

        for lidx in range(num_layers):
            diff = (sysp_all[lidx] - baseline_all[lidx]).astype(np.float64)
            full_norm = float(np.linalg.norm(diff))

            detected_coords = canonical_basis @ diff
            detected_norm = float(np.linalg.norm(detected_coords))
            capture = detected_norm / full_norm if full_norm > 1e-6 else 0

            sims = {}
            for t in TRAITS:
                if detected_norm > 0 and np.linalg.norm(canonical_coords[t]) > 0:
                    sims[t] = float(np.dot(detected_coords, canonical_coords[t]) / (
                        detected_norm * np.linalg.norm(canonical_coords[t])))
                else:
                    sims[t] = 0
            best = max(sims, key=sims.get)

            # Only print every 4th layer for brevity
            if lidx % 4 == 0 or lidx == num_layers - 1:
                print(f"  L{lidx:>3} {detected_norm:>10.2f} {full_norm:>10.2f} {capture:>10.3f} {best:>15}")

            trait_data.append({
                "layer": lidx,
                "5d_norm": detected_norm,
                "full_norm": full_norm,
                "capture_ratio": capture,
                "detected_trait": best,
                "cosine": float(sims.get(sp_trait, 0)),
            })

        sysprompt_layer_data[sp_trait] = trait_data

    results["sysprompt_propagation"] = sysprompt_layer_data

    # ================================================================
    # PART 3: Comparison — peak layers and distributions
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: MECHANISM COMPARISON")
    print(f"{'='*70}")

    # Steering: find onset and peak
    steer_norms = [d["5d_norm"] for d in steer_layer_data]
    steer_peak_layer = int(np.argmax(steer_norms))
    steer_onset = next(i for i, d in enumerate(steer_layer_data) if d["5d_norm"] > 1.0)

    # System prompt: find peak for each trait
    for sp_trait, data in sysprompt_layer_data.items():
        sp_norms = [d["5d_norm"] for d in data]
        sp_full_norms = [d["full_norm"] for d in data]
        sp_peak = int(np.argmax(sp_norms))
        sp_full_peak = int(np.argmax(sp_full_norms))
        sp_captures = [d["capture_ratio"] for d in data]
        sp_peak_capture = int(np.argmax(sp_captures))

        print(f"\n  {sp_trait}:")
        print(f"    Peak 5D signal:     L{sp_peak} (norm={sp_norms[sp_peak]:.2f})")
        print(f"    Peak full signal:   L{sp_full_peak} (norm={sp_full_norms[sp_full_peak]:.2f})")
        print(f"    Peak capture ratio: L{sp_peak_capture} ({sp_captures[sp_peak_capture]:.3f})")
        print(f"    Mean capture:       {np.mean(sp_captures):.3f}")

    print(f"\n  Steering:")
    print(f"    Onset:              L{steer_onset}")
    print(f"    Peak:               L{steer_peak_layer} (norm={steer_norms[steer_peak_layer]:.2f})")
    print(f"    Mean capture (post-injection): {np.mean([d['capture_ratio'] for d in steer_layer_data[mid_layer+1:]]):.3f}")

    results["comparison"] = {
        "steering_onset": steer_onset,
        "steering_peak": steer_peak_layer,
        "steering_peak_norm": steer_norms[steer_peak_layer],
    }

    for sp_trait, data in sysprompt_layer_data.items():
        sp_norms = [d["5d_norm"] for d in data]
        sp_captures = [d["capture_ratio"] for d in data]
        results["comparison"][f"{sp_trait}_peak"] = int(np.argmax(sp_norms))
        results["comparison"][f"{sp_trait}_mean_capture"] = float(np.mean(sp_captures))

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Activation steering: binary onset at L{steer_onset} (one after injection at L{mid_layer})")
    for sp_trait in PERSONALITY_SYSTEM_PROMPTS:
        data = sysprompt_layer_data[sp_trait]
        norms = [d["5d_norm"] for d in data]
        print(f"  System prompt ({sp_trait}): peak at L{int(np.argmax(norms))}, "
              f"mean capture {np.mean([d['capture_ratio'] for d in data]):.3f}")

    results["summary"] = {
        "model": model_id,
        "num_layers": num_layers,
        "injection_layer": mid_layer,
        "steering_onset": steer_onset,
        "steering_peak": steer_peak_layer,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_personality_propagation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
