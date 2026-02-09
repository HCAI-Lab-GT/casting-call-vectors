#!/usr/bin/env python
"""
Cross-DIMENSIONAL steering transfer via 5D coordinate bridge.

The existing cross_model_steering_transfer.py tests models with MATCHING
hidden dimensions (SmolLM3 2048d ↔ Llama 1B 2048d) via Procrustes rotation.

This script tests the harder case: DIFFERENT hidden dimensions.
SmolLM3 (2048d) ↔ Marin 8B (4096d) — vectors can't be directly applied.

The 5D coordinate bridge:
1. Extract SmolLM3's 5D personality coordinates
2. Map those coordinates into Marin 8B's activation space using Marin's 5D basis
3. The resulting 4096d vector encodes SmolLM3's personality knowledge in Marin's space
4. Use it to steer Marin and measure if correct personality emerges

This is the ultimate test of geometric universality:
- Different model families (SmolLM3 vs Llama-based Marin)
- Different hidden dimensions (2048 vs 4096)
- Different number of layers (36 vs 32)
- Connected ONLY through abstract 5D personality coordinates
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="crossdim-transfer")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_all_data(model_id, riasec_dir):
    safe_model = model_id.replace("/", "__")
    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    hidden_dim = config.hidden_size
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

    return residual, coords_5d, basis_5d, S_res, mid_layer, hidden_dim


def canonical_sign_convention(coords_5d):
    signs = np.ones(5)
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1
    for pc in range(1, 5):
        loadings = {t: coords_5d[t][pc] for t in TRAITS}
        max_trait = max(loadings, key=lambda t: abs(loadings[t]))
        if loadings[max_trait] > 0:
            signs[pc] = -1
    return signs


def compute_baseline(model, tokenizer, device):
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [
                {"role": "user", "content": f"Which describes you better? Answer with just A or B.\n"
                                             f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                                             f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            a_ids = tokenizer.encode("A", add_special_tokens=False)
            b_ids = tokenizer.encode("B", add_special_tokens=False)
            baseline[f"{trait_a}-{trait_b}"] = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()
    return baseline


def measure_profile(model, tokenizer, device, blocks, mid_layer,
                    vec, alpha, baseline):
    hook_handle = None
    if vec is not None and alpha > 0:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))

    try:
        trait_logprobs = {}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                messages = [
                    {"role": "user", "content": f"Which describes you better? Answer with just A or B.\n"
                                                 f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                                                 f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(formatted, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                a_ids = tokenizer.encode("A", add_special_tokens=False)
                b_ids = tokenizer.encode("B", add_special_tokens=False)
                gap = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()
                trait_logprobs[f"{trait_a}-{trait_b}"] = gap
    finally:
        if hook_handle:
            hook_handle.remove()

    trait_deltas = {t: 0.0 for t in TRAITS}
    trait_counts = {t: 0 for t in TRAITS}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            key = f"{trait_a}-{trait_b}"
            shift = trait_logprobs[key] - baseline[key]
            trait_deltas[trait_a] += shift
            trait_counts[trait_a] += 1
            trait_deltas[trait_b] -= shift
            trait_counts[trait_b] += 1
    for t in TRAITS:
        if trait_counts[t] > 0:
            trait_deltas[t] /= trait_counts[t]
    return trait_deltas


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"

    target_id = "marin-community/marin-8b-instruct"  # 4096d
    source_id = "HuggingFaceTB/SmolLM3-3B"            # 2048d

    # Load vector data for both models
    logger.info("Loading vector data...")
    target_res, target_coords, target_basis, target_S, target_mid, target_dim = \
        load_all_data(target_id, riasec_dir)
    source_res, source_coords, source_basis, source_S, source_mid, source_dim = \
        load_all_data(source_id, riasec_dir)

    # Sign correction
    target_signs = canonical_sign_convention(target_coords)
    source_signs = canonical_sign_convention(source_coords)
    correct_signs = target_signs * source_signs

    print(f"\n{'='*70}")
    print(f"CROSS-DIMENSIONAL STEERING TRANSFER VIA 5D BRIDGE")
    print(f"Source: SmolLM3-3B ({source_dim}d, {source_mid*2} layers)")
    print(f"Target: Marin 8B ({target_dim}d, {target_mid*2} layers)")
    print(f"Bridge: 5D personality coordinate space")
    print(f"{'='*70}")

    # Show coordinate alignment
    print(f"\n--- 5D Coordinate Comparison ---")
    for t in TRAITS:
        src_c = correct_signs * source_coords[t]
        tgt_c = target_coords[t]
        cos = np.dot(src_c, tgt_c) / (np.linalg.norm(src_c) * np.linalg.norm(tgt_c))
        print(f"  {t:>15}: src={src_c}, tgt={tgt_c}, cos={cos:.3f}")

    # Construct transferred vectors using 3 methods
    print(f"\n--- Transfer Methods ---")

    # Method 1: Direct coordinate mapping (use source coords, reconstruct in target space)
    transferred_direct = {}
    for t in TRAITS:
        src_coords_corrected = correct_signs * source_coords[t]
        # Scale to match target coordinate magnitudes
        src_norm = np.linalg.norm(src_coords_corrected)
        tgt_norm = np.linalg.norm(target_coords[t])
        scale = tgt_norm / src_norm if src_norm > 1e-10 else 1.0
        scaled_coords = scale * src_coords_corrected
        transferred_direct[t] = (target_basis.T @ scaled_coords).astype(np.float32)

    # Method 2: Per-PC scaled (match each PC's contribution individually)
    transferred_perpc = {}
    for t in TRAITS:
        src_coords_corrected = correct_signs * source_coords[t]
        scaled_coords = np.zeros(5)
        for pc in range(5):
            # Scale each PC individually based on singular value ratios
            src_scale = source_S[pc] if pc < len(source_S) else 1.0
            tgt_scale = target_S[pc] if pc < len(target_S) else 1.0
            pc_scale = tgt_scale / src_scale if abs(src_scale) > 1e-10 else 1.0
            scaled_coords[pc] = pc_scale * src_coords_corrected[pc]
        transferred_perpc[t] = (target_basis.T @ scaled_coords).astype(np.float32)

    # Method 3: Naive mean-centered (just use target coords as-is — this is the "oracle" ceiling)
    transferred_oracle = {}
    for t in TRAITS:
        transferred_oracle[t] = (target_basis.T @ target_coords[t]).astype(np.float32)

    # Print norms
    for method_name, tvecs in [("Direct", transferred_direct),
                                 ("Per-PC", transferred_perpc),
                                 ("Oracle", transferred_oracle)]:
        norms = [np.linalg.norm(tvecs[t]) for t in TRAITS]
        native_norms = [np.linalg.norm(target_res[t]) for t in TRAITS]
        norm_ratio = np.mean(norms) / np.mean(native_norms)
        print(f"  {method_name:>10}: mean norm={np.mean(norms):.3f} "
              f"(native={np.mean(native_norms):.3f}, ratio={norm_ratio:.3f})")

    # Test across multiple alphas
    test_alphas = [1.0, 2.0, 3.0]

    results = {}

    # Load target model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    logger.info("Computing baseline...")
    baseline = compute_baseline(model, tokenizer, device)

    for alpha in test_alphas:
        alpha_key = f"alpha_{alpha}"
        alpha_results = {}

        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*70}")

        methods = {
            "native": target_res,
            "direct_transfer": transferred_direct,
            "perpc_transfer": transferred_perpc,
            "oracle": transferred_oracle,
        }

        for method_name, vecs in methods.items():
            print(f"\n  --- {method_name} ---")
            method_results = {}
            correct = 0

            for trait in TRAITS:
                vec = vecs[trait].astype(np.float32)
                deltas = measure_profile(model, tokenizer, device, blocks,
                                          target_mid, vec, alpha, baseline)
                sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
                top = sorted_d[0][0]
                ok = "OK" if top == trait else f"WRONG({top})"
                print(f"    {trait:>15}: top={top}, target={deltas[trait]:+.3f} {ok}")
                method_results[trait] = {
                    "top": top, "correct": top == trait,
                    "target_delta": float(deltas[trait]),
                    "profile": {t: float(deltas[t]) for t in TRAITS},
                }
                if top == trait:
                    correct += 1

            method_results["accuracy"] = correct / 6
            method_results["correct_count"] = correct

            # Compute profile correlation with native
            if method_name != "native":
                native_flat = []
                method_flat = []
                for t in TRAITS:
                    for t2 in TRAITS:
                        native_flat.append(alpha_results["native"][t]["profile"][t2])
                        method_flat.append(method_results[t]["profile"][t2])
                r, p = pearsonr(native_flat, method_flat)
                method_results["correlation_with_native"] = float(r)
                print(f"    Accuracy: {correct}/6, Correlation with native: r={r:.3f}")
            else:
                print(f"    Accuracy: {correct}/6 (baseline)")

            # Efficiency vs native
            if method_name != "native":
                effs = []
                for t in TRAITS:
                    native_d = alpha_results["native"][t]["target_delta"]
                    method_d = method_results[t]["target_delta"]
                    eff = method_d / native_d if abs(native_d) > 0.01 else 0
                    effs.append(eff)
                method_results["mean_efficiency"] = float(np.mean(effs))
                print(f"    Mean efficiency: {np.mean(effs):.1%}")

            alpha_results[method_name] = method_results

        results[alpha_key] = alpha_results

    # Now test reverse direction: Marin → SmolLM3
    del model
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"REVERSE: Marin 8B → SmolLM3 Transfer")
    print(f"{'='*70}")

    # Construct reverse transferred vectors
    reverse_transferred = {}
    for t in TRAITS:
        tgt_coords_corrected = correct_signs * target_coords[t]
        src_norm = np.linalg.norm(source_coords[t])
        tgt_norm = np.linalg.norm(tgt_coords_corrected)
        scale = src_norm / tgt_norm if tgt_norm > 1e-10 else 1.0
        scaled_coords = scale * tgt_coords_corrected
        reverse_transferred[t] = (source_basis.T @ scaled_coords).astype(np.float32)

    logger.info("Loading SmolLM3-3B...")
    smol_tokenizer = AutoTokenizer.from_pretrained(source_id)
    smol_model = AutoModelForCausalLM.from_pretrained(
        source_id, torch_dtype=torch.float16, device_map=device)
    smol_model.eval()
    smol_blocks = get_decoder_blocks(smol_model)

    logger.info("Computing SmolLM3 baseline...")
    smol_baseline = compute_baseline(smol_model, smol_tokenizer, device)

    alpha = 2.0
    reverse_results = {}

    for method_name, vecs in [("native", source_res), ("marin_transfer", reverse_transferred)]:
        print(f"\n  --- {method_name} ---")
        method_results = {}
        correct = 0

        for trait in TRAITS:
            vec = vecs[trait].astype(np.float32)
            deltas = measure_profile(smol_model, smol_tokenizer, device, smol_blocks,
                                      source_mid, vec, alpha, smol_baseline)
            sorted_d = sorted(deltas.items(), key=lambda x: -x[1])
            top = sorted_d[0][0]
            ok = "OK" if top == trait else f"WRONG({top})"
            print(f"    {trait:>15}: top={top}, target={deltas[trait]:+.3f} {ok}")
            method_results[trait] = {
                "top": top, "correct": top == trait,
                "target_delta": float(deltas[trait]),
                "profile": {t: float(deltas[t]) for t in TRAITS},
            }
            if top == trait:
                correct += 1

        method_results["accuracy"] = correct / 6
        method_results["correct_count"] = correct
        print(f"    Accuracy: {correct}/6")

        reverse_results[method_name] = method_results

    # Profile correlation for reverse
    if "native" in reverse_results and "marin_transfer" in reverse_results:
        native_flat = []
        trans_flat = []
        for t in TRAITS:
            for t2 in TRAITS:
                native_flat.append(reverse_results["native"][t]["profile"][t2])
                trans_flat.append(reverse_results["marin_transfer"][t]["profile"][t2])
        r, p = pearsonr(native_flat, trans_flat)
        reverse_results["profile_correlation"] = float(r)
        print(f"  Reverse correlation: r={r:.3f}")

    results["reverse"] = reverse_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Forward: SmolLM3 → Marin 8B (2048d → 4096d)")
    print(f"  {'Method':>20} {'α=1':>6} {'α=2':>6} {'α=3':>6} {'Corr':>8} {'Eff':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")

    for method in ["native", "direct_transfer", "perpc_transfer", "oracle"]:
        accs = []
        for a in test_alphas:
            acc = results[f"alpha_{a}"][method]["accuracy"]
            accs.append(f"{acc:.0%}")
        corr = results[f"alpha_2.0"][method].get("correlation_with_native", 1.0)
        eff = results[f"alpha_2.0"][method].get("mean_efficiency", 1.0)
        print(f"  {method:>20} {accs[0]:>6} {accs[1]:>6} {accs[2]:>6} "
              f"{corr:>8.3f} {eff:>8.1%}")

    print(f"\n  Reverse: Marin 8B → SmolLM3 (4096d → 2048d)")
    for method in ["native", "marin_transfer"]:
        acc = reverse_results[method]["accuracy"]
        print(f"  {method:>20}: {acc:.0%} ({reverse_results[method]['correct_count']}/6)")

    # Overall conclusion
    best_transfer_acc = max(
        results["alpha_2.0"]["direct_transfer"]["accuracy"],
        results["alpha_2.0"]["perpc_transfer"]["accuracy"],
    )
    reverse_acc = reverse_results["marin_transfer"]["accuracy"]

    if best_transfer_acc >= 0.67 and reverse_acc >= 0.67:
        conclusion = "Cross-dimensional steering WORKS — 5D bridge enables universal personality transfer"
    elif best_transfer_acc >= 0.5 or reverse_acc >= 0.5:
        conclusion = "Partial cross-dimensional transfer — 5D bridge captures some personality structure"
    else:
        conclusion = "Cross-dimensional steering FAILS — 5D coordinates don't suffice for cross-dim transfer"

    print(f"\n  CONCLUSION: {conclusion}")

    results["summary"] = {
        "best_forward_accuracy": float(best_transfer_acc),
        "reverse_accuracy": float(reverse_acc),
        "conclusion": conclusion,
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "crossdim_steering_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
