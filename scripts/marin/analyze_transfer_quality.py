#!/usr/bin/env python
"""
Analyze WHY Llama 1B produces the best cross-dim transfer vectors,
and test ensemble transfer (averaging across sources).

Questions:
1. Which 5D coordinate system is "cleanest"? (explained variance, condition)
2. How aligned are different models' 5D systems? (pairwise rotation distances)
3. Does ensembling transferred vectors outperform individual sources?
4. What makes base models worse? (geometry comparison)
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="transfer-quality")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

MODELS = {
    "SmolLM3-Instruct": "HuggingFaceTB/SmolLM3-3B",
    "SmolLM3-Base": "HuggingFaceTB/SmolLM3-3B-Base",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Marin-8B": "marin-community/marin-8b-instruct",
    "Marin-32B-Base": "marin-community/marin-32b-base",
}

TARGET_ID = "marin-community/marin-8b-instruct"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_residual_vectors(model_id, riasec_dir):
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
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer


def get_5d_analysis(residual_vectors):
    """Extended 5D analysis with quality metrics."""
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)

    # Explained variance by top 5 components
    total_var = np.sum(S**2)
    var_5d = np.sum(S[:5]**2)
    var_ratio = var_5d / total_var

    # Condition number of the 5D projection
    condition = S[0] / S[4] if S[4] > 0 else float('inf')

    # 6th singular value (should be ~0 for perfect 5D)
    sixth_sv = S[5] if len(S) > 5 else 0
    sixth_ratio = sixth_sv / S[0]

    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}

    # Simplex regularity: how equal are the pairwise angles?
    angles = []
    for i, t1 in enumerate(TRAITS):
        for j, t2 in enumerate(TRAITS):
            if i >= j:
                continue
            c1 = coords_5d[t1]
            c2 = coords_5d[t2]
            cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
            angles.append(cos)

    # For a perfect simplex in 5D, pairwise cosines should be -1/5 = -0.2
    ideal_cos = -1/5
    simplex_deviation = np.std([a - ideal_cos for a in angles])

    # Norm uniformity: how equal are the vector norms?
    norms = [np.linalg.norm(coords_5d[t]) for t in TRAITS]
    norm_cv = np.std(norms) / np.mean(norms)  # coefficient of variation

    return {
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "singular_values": S.tolist(),
        "var_explained_5d": float(var_ratio),
        "condition_number": float(condition),
        "sixth_sv_ratio": float(sixth_ratio),
        "pairwise_cosines": angles,
        "mean_cosine": float(np.mean(angles)),
        "simplex_deviation": float(simplex_deviation),
        "norm_cv": float(norm_cv),
        "norms": {t: float(np.linalg.norm(coords_5d[t])) for t in TRAITS},
    }


def fit_procrustes(source_5d, target_5d):
    S = np.stack([source_5d[t] for t in TRAITS])
    T = np.stack([target_5d[t] for t in TRAITS])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))

    # Procrustes quality: residual after alignment
    aligned = np.stack([scale * (R @ source_5d[t]) for t in TRAITS])
    target = np.stack([target_5d[t] for t in TRAITS])
    loo_cosines = []
    for i, t in enumerate(TRAITS):
        cos = np.dot(aligned[i], target[i]) / (np.linalg.norm(aligned[i]) * np.linalg.norm(target[i]))
        loo_cosines.append(cos)

    return R, scale, {"mean_cosine": float(np.mean(loo_cosines)),
                      "min_cosine": float(np.min(loo_cosines))}


def pairwise_logprob_chat(model, tokenizer, device, desc_a, desc_b):
    messages = [
        {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
        {"role": "user", "content": f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    return log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()


def eval_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    correct = 0
    total = 0
    deltas = []
    for steer_trait in TRAITS:
        vec = vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
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
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    if steer_trait not in (trait_a, trait_b):
                        continue
                    gap = pairwise_logprob_chat(model, tokenizer, device,
                                               TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                    base_gap = baseline[f"{trait_a}-{trait_b}"]
                    if steer_trait == trait_a:
                        d = gap - base_gap
                    else:
                        d = base_gap - gap
                    correct += int(d > 0)
                    total += 1
                    deltas.append(d)
        finally:
            hook_handle.remove()
    return {
        "delta_accuracy": correct / total if total else 0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0,
        "correct": correct,
        "total": total,
    }


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # === Phase 1: Analyze 5D geometry quality for all models ===
    logger.info("Analyzing 5D geometry quality...")

    model_data = {}
    for name, model_id in MODELS.items():
        residual, mid = load_residual_vectors(model_id, riasec_dir)
        analysis = get_5d_analysis(residual)
        model_data[name] = {
            "residual": residual,
            "mid_layer": mid,
            "analysis": analysis,
        }

    print(f"\n{'='*70}")
    print(f"5D GEOMETRY QUALITY COMPARISON")
    print(f"{'='*70}")

    print(f"\n--- Quality metrics ---")
    print(f"  {'Model':>20}  {'Var5D':>6}  {'Cond#':>6}  {'6thSV%':>7}  {'SmpxDev':>7}  {'NormCV':>6}  {'MeanCos':>7}")
    print(f"  {'-'*72}")
    for name in MODELS:
        a = model_data[name]["analysis"]
        print(f"  {name:>20}  {a['var_explained_5d']:>5.3f}  {a['condition_number']:>5.1f}  "
              f"{a['sixth_sv_ratio']:>6.4f}  {a['simplex_deviation']:>6.4f}  "
              f"{a['norm_cv']:>5.3f}  {a['mean_cosine']:>+6.3f}")

    # Ideal simplex: mean cosine = -0.2, deviation = 0
    print(f"\n  Ideal regular simplex: MeanCos=-0.200, SmpxDev=0.000")

    # === Phase 2: Pairwise Procrustes quality ===
    print(f"\n--- Pairwise Procrustes alignment quality ---")
    instruct_models = ["SmolLM3-Instruct", "Llama-1B", "Qwen-7B", "Marin-8B"]
    procrustes_quality = {}

    print(f"  {'Source → Target':>35}  {'Mean cos':>8}  {'Min cos':>8}")
    print(f"  {'-'*55}")
    for src, tgt in combinations(instruct_models, 2):
        src_5d = model_data[src]["analysis"]["coords_5d"]
        tgt_5d = model_data[tgt]["analysis"]["coords_5d"]
        _, _, quality = fit_procrustes(src_5d, tgt_5d)
        key = f"{src}→{tgt}"
        procrustes_quality[key] = quality
        print(f"  {key:>35}  {quality['mean_cosine']:>8.4f}  {quality['min_cosine']:>8.4f}")

    # === Phase 3: Base vs instruct geometry comparison ===
    print(f"\n--- Base vs instruct geometry (SmolLM3 family) ---")
    smol_inst = model_data["SmolLM3-Instruct"]["analysis"]["coords_5d"]
    smol_base = model_data["SmolLM3-Base"]["analysis"]["coords_5d"]
    _, _, q = fit_procrustes(smol_base, smol_inst)
    print(f"  SmolLM3-Base → SmolLM3-Instruct: mean cos={q['mean_cosine']:.4f}, min cos={q['min_cosine']:.4f}")

    print(f"\n--- Base vs instruct geometry (Marin family) ---")
    marin_inst = model_data["Marin-8B"]["analysis"]["coords_5d"]
    marin_base = model_data["Marin-32B-Base"]["analysis"]["coords_5d"]
    _, _, q2 = fit_procrustes(marin_base, marin_inst)
    print(f"  Marin-32B-Base → Marin-8B: mean cos={q2['mean_cosine']:.4f}, min cos={q2['min_cosine']:.4f}")

    # === Phase 4: Ensemble transfer ===
    logger.info("Building ensemble vectors...")

    target_residual = model_data["Marin-8B"]["residual"]
    target_mid = model_data["Marin-8B"]["mid_layer"]
    target_5d = model_data["Marin-8B"]["analysis"]["coords_5d"]
    target_basis = model_data["Marin-8B"]["analysis"]["basis_5d"]
    target_dim = target_residual[TRAITS[0]].shape[0]

    # Individual transfers
    instruct_sources = ["SmolLM3-Instruct", "Llama-1B", "Qwen-7B"]
    individual_vecs = {}
    for name in instruct_sources:
        src_5d = model_data[name]["analysis"]["coords_5d"]
        R, scale, _ = fit_procrustes(src_5d, target_5d)
        vecs = {}
        for t in TRAITS:
            aligned = scale * (R @ src_5d[t])
            vecs[t] = (target_basis.T @ aligned).astype(np.float32)
        individual_vecs[name] = vecs

    # Ensemble: average of transferred vectors from instruct sources
    ensemble_vecs = {}
    for t in TRAITS:
        avg = np.mean([individual_vecs[name][t] for name in instruct_sources], axis=0)
        ensemble_vecs[t] = avg.astype(np.float32)

    # Ensemble 2: weighted by Procrustes quality
    weights = {}
    for name in instruct_sources:
        src_5d = model_data[name]["analysis"]["coords_5d"]
        _, _, q = fit_procrustes(src_5d, target_5d)
        weights[name] = q["mean_cosine"]
    total_w = sum(weights.values())
    weighted_ensemble_vecs = {}
    for t in TRAITS:
        avg = np.sum([weights[name]/total_w * individual_vecs[name][t]
                      for name in instruct_sources], axis=0)
        weighted_ensemble_vecs[t] = avg.astype(np.float32)

    # Cosine analysis
    print(f"\n--- Ensemble cosine vs native ---")
    for label, vecs in [("Ensemble (equal)", ensemble_vecs),
                        ("Ensemble (weighted)", weighted_ensemble_vecs)]:
        cosines = [np.dot(vecs[t], target_residual[t]) /
                   (np.linalg.norm(vecs[t]) * np.linalg.norm(target_residual[t]))
                   for t in TRAITS]
        print(f"  {label:>25}: mean={np.mean(cosines):.4f}, min={min(cosines):.4f}")

    # === Phase 5: Evaluate on Marin 8B ===
    logger.info("Loading target model: %s", TARGET_ID)
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n--- Ensemble steering evaluation (α={alpha}) ---")

    r_self = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                target_residual, alpha, baseline)
    print(f"  Self (Marin native):      {r_self['delta_accuracy']:.0%} ({r_self['correct']}/{r_self['total']})")

    r_ensemble = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                     ensemble_vecs, alpha, baseline)
    print(f"  Ensemble (equal avg):     {r_ensemble['delta_accuracy']:.0%} ({r_ensemble['correct']}/{r_ensemble['total']})")

    r_w_ensemble = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                       weighted_ensemble_vecs, alpha, baseline)
    print(f"  Ensemble (weighted avg):  {r_w_ensemble['delta_accuracy']:.0%} ({r_w_ensemble['correct']}/{r_w_ensemble['total']})")

    # For comparison, also test each individual
    for name in instruct_sources:
        r = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                               individual_vecs[name], alpha, baseline)
        print(f"  {name:>25}: {r['delta_accuracy']:.0%} ({r['correct']}/{r['total']})")

    # Build results
    results = {
        "geometry_quality": {},
        "procrustes_quality": procrustes_quality,
        "ensemble": {
            "equal": {"delta_accuracy": r_ensemble["delta_accuracy"]},
            "weighted": {"delta_accuracy": r_w_ensemble["delta_accuracy"]},
        },
        "self": r_self,
    }

    for name in MODELS:
        a = model_data[name]["analysis"]
        results["geometry_quality"][name] = {
            "var_explained_5d": a["var_explained_5d"],
            "condition_number": a["condition_number"],
            "sixth_sv_ratio": a["sixth_sv_ratio"],
            "simplex_deviation": a["simplex_deviation"],
            "norm_cv": a["norm_cv"],
            "mean_cosine": a["mean_cosine"],
        }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transfer_quality_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
