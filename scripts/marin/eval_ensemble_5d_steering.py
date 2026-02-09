#!/usr/bin/env python
"""
Ensemble steering via 5D coordinate averaging.

We proved:
- Cross-dim transfer works at 100% (SmolLM3 → Marin via 5D bridge)
- 5D coordinates are cross-model invariant (cos=0.95-0.99)
- Earlier sessions showed ensemble transfer at 100% for same-dim models

QUESTION: Can we AVERAGE 5D coordinates from MULTIPLE source models and
use the ensemble to steer a target model? If so, we get:
- Noise cancellation: averaging reduces per-model noise
- Robustness: ensemble is more stable than any single source
- Universal personality: truly model-agnostic control vectors

METHOD:
1. Load 5D coordinates from SmolLM3, Llama 1B, Qwen 7B
2. Sign-correct all to Marin 8B's convention
3. Average the 5D coordinates (simple mean)
4. Also test median, and best-single-source
5. Reconstruct as 4096d vector in Marin's space
6. Steer Marin 8B and measure personality

COMPARISON:
- Native (Marin's own vectors)
- Best single source
- Ensemble average
- Ensemble median
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="ensemble-5d")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

SOURCE_MODELS = [
    "HuggingFaceTB/SmolLM3-3B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

TARGET_MODEL = "marin-community/marin-8b-instruct"


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
    alpha = 2.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load all vector data (no models needed yet)
    logger.info("Loading vector data for all models...")

    target_res, target_coords, target_basis, target_S, target_mid, target_dim = \
        load_all_data(TARGET_MODEL, riasec_dir)
    target_signs = canonical_sign_convention(target_coords)

    source_data = {}
    for src_id in SOURCE_MODELS:
        logger.info(f"  Loading {src_id}...")
        res, coords, basis, S, mid, dim = load_all_data(src_id, riasec_dir)
        src_signs = canonical_sign_convention(coords)
        sign_correction = target_signs * src_signs

        # Compute sign-corrected and norm-scaled coordinates
        corrected_coords = {}
        for t in TRAITS:
            src_c = sign_correction * coords[t]
            src_norm = np.linalg.norm(src_c)
            tgt_norm = np.linalg.norm(target_coords[t])
            scale = tgt_norm / src_norm if src_norm > 1e-10 else 1.0
            corrected_coords[t] = scale * src_c

        source_data[src_id] = {
            "coords": corrected_coords,
            "sign_correction": sign_correction,
            "dim": dim,
        }

    print(f"\n{'='*70}")
    print(f"ENSEMBLE 5D STEERING")
    print(f"Target: Marin 8B ({target_dim}d)")
    print(f"Sources: {len(SOURCE_MODELS)} models")
    print(f"Alpha: {alpha}")
    print(f"{'='*70}")

    # Show per-source coordinate alignment with target
    print(f"\n--- Source Coordinate Alignment ---")
    for src_id in SOURCE_MODELS:
        src_name = src_id.split("/")[-1][:15]
        cosines = []
        for t in TRAITS:
            src_c = source_data[src_id]["coords"][t]
            tgt_c = target_coords[t]
            cos = np.dot(src_c, tgt_c) / (np.linalg.norm(src_c) * np.linalg.norm(tgt_c))
            cosines.append(cos)
        print(f"  {src_name:>15}: mean cos={np.mean(cosines):.3f}, "
              f"range=[{min(cosines):.3f}, {max(cosines):.3f}]")

    # Construct ensemble vectors
    ensemble_mean = {}
    ensemble_median = {}
    for t in TRAITS:
        all_coords = np.stack([source_data[src]["coords"][t] for src in SOURCE_MODELS])
        mean_coord = np.mean(all_coords, axis=0)
        median_coord = np.median(all_coords, axis=0)

        # Scale to match target magnitude
        mean_norm = np.linalg.norm(mean_coord)
        tgt_norm = np.linalg.norm(target_coords[t])
        mean_scale = tgt_norm / mean_norm if mean_norm > 1e-10 else 1.0
        median_norm = np.linalg.norm(median_coord)
        median_scale = tgt_norm / median_norm if median_norm > 1e-10 else 1.0

        # Reconstruct in target activation space
        ensemble_mean[t] = (target_basis.T @ (mean_scale * mean_coord)).astype(np.float32)
        ensemble_median[t] = (target_basis.T @ (median_scale * median_coord)).astype(np.float32)

    # Construct per-source transferred vectors
    per_source_vecs = {}
    for src_id in SOURCE_MODELS:
        src_vecs = {}
        for t in TRAITS:
            src_vecs[t] = (target_basis.T @ source_data[src_id]["coords"][t]).astype(np.float32)
        per_source_vecs[src_id] = src_vecs

    # Load target model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    logger.info("Computing baseline...")
    baseline = compute_baseline(model, tokenizer, device)

    results = {}

    # Test each method
    methods = {
        "native": target_res,
        "ensemble_mean": ensemble_mean,
        "ensemble_median": ensemble_median,
    }
    for src_id in SOURCE_MODELS:
        src_name = src_id.split("/")[-1][:20]
        methods[f"source_{src_name}"] = per_source_vecs[src_id]

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
                "top": top, "correct": bool(top == trait),
                "target_delta": float(deltas[trait]),
                "profile": {t: float(deltas[t]) for t in TRAITS},
            }
            if top == trait:
                correct += 1

        method_results["accuracy"] = correct / 6
        method_results["correct_count"] = correct
        print(f"    Accuracy: {correct}/6")

        # Correlation with native
        if method_name != "native":
            native_flat = []
            method_flat = []
            for t in TRAITS:
                for t2 in TRAITS:
                    native_flat.append(results["native"][t]["profile"][t2])
                    method_flat.append(method_results[t]["profile"][t2])
            r, p = pearsonr(native_flat, method_flat)
            method_results["correlation_with_native"] = float(r)

            # Efficiency
            effs = []
            for t in TRAITS:
                native_d = results["native"][t]["target_delta"]
                method_d = method_results[t]["target_delta"]
                eff = method_d / native_d if abs(native_d) > 0.01 else 0
                effs.append(eff)
            method_results["mean_efficiency"] = float(np.mean(effs))
            print(f"    Correlation: r={r:.3f}, Efficiency: {np.mean(effs):.1%}")

        results[method_name] = method_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Method':>25} {'Accuracy':>10} {'Correlation':>12} {'Efficiency':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12}")

    for method_name in methods:
        r = results[method_name]
        acc = r["accuracy"]
        corr = r.get("correlation_with_native", 1.0)
        eff = r.get("mean_efficiency", 1.0)
        print(f"  {method_name:>25} {acc:>10.0%} {corr:>12.3f} {eff:>12.1%}")

    # Find best source
    source_accs = {k: results[k]["accuracy"] for k in results
                   if k.startswith("source_")}
    best_source = max(source_accs, key=source_accs.get)
    best_source_acc = source_accs[best_source]

    ensemble_acc = results["ensemble_mean"]["accuracy"]
    native_acc = results["native"]["accuracy"]

    print(f"\n  Native accuracy:        {native_acc:.0%}")
    print(f"  Best single source:     {best_source_acc:.0%} ({best_source})")
    print(f"  Ensemble mean:          {ensemble_acc:.0%}")
    print(f"  Ensemble median:        {results['ensemble_median']['accuracy']:.0%}")

    if ensemble_acc >= native_acc:
        conclusion = "Ensemble MATCHES or BEATS native — multi-model consensus works"
    elif ensemble_acc >= best_source_acc:
        conclusion = "Ensemble beats best single source — averaging helps"
    elif ensemble_acc > 0.5:
        conclusion = "Ensemble works but doesn't beat best single source"
    else:
        conclusion = "Ensemble fails — averaging hurts"

    print(f"\n  CONCLUSION: {conclusion}")

    results["summary"] = {
        "native_accuracy": float(native_acc),
        "best_source_accuracy": float(best_source_acc),
        "best_source": best_source,
        "ensemble_mean_accuracy": float(ensemble_acc),
        "ensemble_median_accuracy": float(results["ensemble_median"]["accuracy"]),
        "conclusion": conclusion,
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ensemble_5d_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
