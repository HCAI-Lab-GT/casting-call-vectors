#!/usr/bin/env python
"""
Causal Patching: Test causal necessity of personality at each layer.

For each layer, replace the hidden state from trait-A steering with
trait-B steering (or baseline). If the detected trait changes, that
layer is causally necessary for personality.

Tests:
1. Layer-by-layer patching: patch each layer independently
2. Cumulative patching: patch L0→Lk (first k layers) with baseline
3. Reverse cumulative: patch Lk→L31 (last layers) with baseline
4. Cross-trait substitution: patch specific layers from artistic→social
5. Residual stream patching: patch only the residual stream contribution
   (not MLP/attention output) at each layer
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="causal-patch")

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

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    num_layers = len(blocks)

    detect_prompt = "Tell me about yourself."
    alpha = 2.0
    detect_layer = mid_layer + 1
    results = {}

    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    print(f"\n{'='*70}")
    print("CAUSAL PATCHING: PERSONALITY AT EACH LAYER")
    print(f"Model: Marin 8B, {num_layers} layers")
    print(f"{'='*70}")

    # ================================================================
    # First, collect all hidden states for baseline and each trait
    # ================================================================
    logger.info("Collecting hidden states for all conditions...")

    def collect_hidden_states(steer_vec=None, steer_alpha=0.0):
        """Run model and collect hidden state at last position after each layer."""
        states = {}
        hooks = []

        for lidx in range(num_layers):
            def make_hook(l):
                def hook_fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    states[l] = hs[0, -1, :].detach().clone()
                    return out
                return hook_fn
            hooks.append(blocks[lidx].register_forward_hook(make_hook(lidx)))

        if steer_vec is not None and steer_alpha != 0:
            delta = steer_alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
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

        return states

    baseline_states = collect_hidden_states()
    trait_states = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        trait_states[trait] = collect_hidden_states(steer_vec=vec, steer_alpha=alpha)

    def analyze(act_np, target_trait):
        baseline_act_np = baseline_states[detect_layer].cpu().numpy().astype(np.float64)
        diff = (act_np - baseline_act_np)
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
        return {"detected": detected, "correct": detected == target_trait,
                "cos_target": sims[target_trait], "norm": norm_5d}

    # ================================================================
    # PART 1: Layer-by-layer patching (replace one layer with baseline)
    # ================================================================
    logger.info("Part 1: Single-layer patching...")
    print(f"\n{'='*70}")
    print("PART 1: SINGLE-LAYER PATCHING (replace with baseline)")
    print(f"{'='*70}")

    single_patch_results = {}
    for trait in ["artistic", "social"]:
        vec = residual[trait].astype(np.float32)
        delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)

        # Reference: full steering, no patching
        ref_act = trait_states[trait][detect_layer].cpu().numpy().astype(np.float64)
        ref_res = analyze(ref_act, trait)

        trait_results = {}
        print(f"\n  {trait} (ref norm={ref_res['norm']:.1f}):")

        for patch_layer in range(num_layers):
            # Patch: replace hidden state at patch_layer with baseline
            patch_target = baseline_states[patch_layer].clone()

            hooks = []
            captured = {}

            # Steer at mid_layer
            def make_steer():
                def steer_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += delta
                        return (hs,) + out[1:]
                    out[:, -1, :] += delta
                    return out
                return steer_fn
            hooks.append(blocks[mid_layer].register_forward_hook(make_steer()))

            # Patch at patch_layer
            def make_patch(target):
                def patch_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[0, -1, :] = target
                        return (hs,) + out[1:]
                    out[0, -1, :] = target
                    return out
                return patch_fn
            hooks.append(blocks[patch_layer].register_forward_hook(make_patch(patch_target)))

            # Capture at detect_layer
            def cap_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

            try:
                with torch.no_grad():
                    model(input_ids)
            finally:
                for h in hooks:
                    h.remove()

            res = analyze(captured["act"], trait)
            norm_change = (res["norm"] / ref_res["norm"] - 1) * 100 if ref_res["norm"] > 0 else 0

            if patch_layer in [0, mid_layer - 1, mid_layer, mid_layer + 1, mid_layer + 2, num_layers - 1] or not res["correct"]:
                marker = " ← injection" if patch_layer == mid_layer else ""
                print(f"    Patch L{patch_layer}: detected={res['detected']}, "
                      f"norm={res['norm']:.1f} ({norm_change:+.1f}%){marker}")

            trait_results[patch_layer] = {**res, "norm_change_pct": float(norm_change)}

        single_patch_results[trait] = {str(k): v for k, v in trait_results.items()}

    results["single_layer_patch"] = single_patch_results

    # ================================================================
    # PART 2: Cumulative patching (replace L0..Lk with baseline)
    # ================================================================
    logger.info("Part 2: Cumulative patching...")
    print(f"\n{'='*70}")
    print("PART 2: CUMULATIVE PATCHING (replace L0..Lk with baseline)")
    print(f"{'='*70}")

    cumulative_results = {}
    trait = "artistic"
    vec = residual[trait].astype(np.float32)
    delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
    ref_act = trait_states[trait][detect_layer].cpu().numpy().astype(np.float64)
    ref_res = analyze(ref_act, trait)

    for k in [0, 4, 8, 12, mid_layer - 1, mid_layer, mid_layer + 1, mid_layer + 2, 24, 28, num_layers - 1]:
        if k >= num_layers:
            continue
        hooks = []
        captured = {}

        # Steer
        hooks.append(blocks[mid_layer].register_forward_hook(make_steer()))

        # Patch L0..Lk with baseline
        for patch_l in range(k + 1):
            hooks.append(blocks[patch_l].register_forward_hook(
                make_patch(baseline_states[patch_l].clone())))

        # Capture
        hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        res = analyze(captured["act"], trait)
        norm_change = (res["norm"] / ref_res["norm"] - 1) * 100 if ref_res["norm"] > 0 else 0
        marker = " ← injection" if k == mid_layer else ""
        print(f"  Patch L0..L{k}: detected={res['detected']}, "
              f"norm={res['norm']:.1f} ({norm_change:+.1f}%){marker}")
        cumulative_results[k] = {**res, "norm_change_pct": float(norm_change)}

    results["cumulative_patch"] = {str(k): v for k, v in cumulative_results.items()}

    # ================================================================
    # PART 3: Reverse cumulative (replace Lk..L31 with baseline)
    # ================================================================
    logger.info("Part 3: Reverse cumulative patching...")
    print(f"\n{'='*70}")
    print("PART 3: REVERSE CUMULATIVE (replace Lk..L31 with baseline)")
    print(f"{'='*70}")

    reverse_results = {}
    for k in [num_layers - 1, 28, 24, mid_layer + 2, mid_layer + 1, mid_layer, mid_layer - 1, 12, 8, 4, 0]:
        if k >= num_layers or k < 0:
            continue
        hooks = []
        captured = {}

        hooks.append(blocks[mid_layer].register_forward_hook(make_steer()))

        # Patch Lk..L31 with baseline
        for patch_l in range(k, num_layers):
            hooks.append(blocks[patch_l].register_forward_hook(
                make_patch(baseline_states[patch_l].clone())))

        hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        res = analyze(captured["act"], trait)
        norm_change = (res["norm"] / ref_res["norm"] - 1) * 100 if ref_res["norm"] > 0 else 0
        marker = " ← injection" if k == mid_layer else ""
        print(f"  Patch L{k}..L{num_layers-1}: detected={res['detected']}, "
              f"norm={res['norm']:.1f} ({norm_change:+.1f}%){marker}")
        reverse_results[k] = {**res, "norm_change_pct": float(norm_change)}

    results["reverse_cumulative"] = {str(k): v for k, v in reverse_results.items()}

    # ================================================================
    # PART 4: Cross-trait substitution (patch from trait A to trait B)
    # ================================================================
    logger.info("Part 4: Cross-trait substitution...")
    print(f"\n{'='*70}")
    print("PART 4: CROSS-TRAIT SUBSTITUTION")
    print(f"{'='*70}")

    cross_results = {}
    for source_trait, target_trait in [("artistic", "social"), ("social", "artistic"),
                                        ("artistic", "conventional")]:
        source_vec = residual[source_trait].astype(np.float32)
        source_delta = alpha * torch.tensor(source_vec, dtype=model.dtype).unsqueeze(0).to(device)

        target_vec = residual[target_trait].astype(np.float32)
        target_delta = alpha * torch.tensor(target_vec, dtype=model.dtype).unsqueeze(0).to(device)

        print(f"\n  Steer {source_trait}, patch with {target_trait} hidden states:")

        pair_results = {}
        for patch_layer in [mid_layer, mid_layer + 1, mid_layer + 2]:
            target_state = trait_states[target_trait][patch_layer].clone()

            hooks = []
            captured = {}

            # Steer with source
            def make_src_steer(d):
                def fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += d
                        return (hs,) + out[1:]
                    out[:, -1, :] += d
                    return out
                return fn
            hooks.append(blocks[mid_layer].register_forward_hook(make_src_steer(source_delta)))

            # Patch with target's hidden state
            def make_tgt_patch(tgt):
                def fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[0, -1, :] = tgt
                        return (hs,) + out[1:]
                    out[0, -1, :] = tgt
                    return out
                return fn
            hooks.append(blocks[patch_layer].register_forward_hook(make_tgt_patch(target_state)))

            def cap_cross(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            hooks.append(blocks[detect_layer].register_forward_hook(cap_cross))

            try:
                with torch.no_grad():
                    model(input_ids)
            finally:
                for h in hooks:
                    h.remove()

            # Check: does it look like source or target?
            act = captured["act"]
            baseline_np = baseline_states[detect_layer].cpu().numpy().astype(np.float64)
            diff = (act - baseline_np)
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

            print(f"    Patch L{patch_layer}: detected={detected}, "
                  f"cos({source_trait})={sims[source_trait]:.3f}, "
                  f"cos({target_trait})={sims[target_trait]:.3f}")

            pair_results[patch_layer] = {
                "detected": detected,
                "cos_source": sims[source_trait],
                "cos_target": sims[target_trait],
                "norm": norm_5d,
            }

        cross_results[f"{source_trait}→{target_trait}"] = {str(k): v for k, v in pair_results.items()}

    results["cross_trait_patch"] = cross_results

    # ================================================================
    # PART 5: 5D-only patching (remove only the 5D component at each layer)
    # ================================================================
    logger.info("Part 5: 5D-only patching...")
    print(f"\n{'='*70}")
    print("PART 5: REMOVE ONLY 5D COMPONENT AT EACH LAYER")
    print(f"{'='*70}")

    basis_tensor = torch.tensor(basis_5d, dtype=model.dtype).to(device)
    fived_patch_results = {}
    trait = "artistic"
    vec = residual[trait].astype(np.float32)
    delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)

    for patch_layer in range(mid_layer, min(mid_layer + 6, num_layers)):
        # Get the 5D difference between steered and baseline at this layer
        steered_state = trait_states[trait][patch_layer]
        base_state = baseline_states[patch_layer]
        diff_state = steered_state - base_state

        # Project diff onto 5D
        coords_diff = diff_state @ basis_tensor.T  # [5]
        proj_5d = coords_diff @ basis_tensor  # [hidden]

        # Create patched state: steered - 5D component = keep only non-5D part of steering
        patched = steered_state - proj_5d

        hooks = []
        captured = {}

        hooks.append(blocks[mid_layer].register_forward_hook(make_steer()))
        hooks.append(blocks[patch_layer].register_forward_hook(make_patch(patched)))

        def cap_5d(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cap_5d))

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        res = analyze(captured["act"], trait)
        marker = " ← injection" if patch_layer == mid_layer else ""
        print(f"  Remove 5D at L{patch_layer}: detected={res['detected']}, "
              f"cos={res['cos_target']:.3f}, norm={res['norm']:.1f}{marker}")
        fived_patch_results[patch_layer] = res

    results["fived_patch"] = {str(k): v for k, v in fived_patch_results.items()}

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for trait in ["artistic", "social"]:
        if trait in single_patch_results:
            critical_layers = [l for l, r in single_patch_results[trait].items()
                              if not r["correct"]]
            print(f"  {trait}: critical layers (detection fails when patched) = {critical_layers}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "causal_patching.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
