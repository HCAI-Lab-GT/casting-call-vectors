#!/usr/bin/env python
"""
Cross-dimensional dose-response: Does the alpha-response curve survive
cross-dimensional transfer?

If the same alpha produces the same effect with transferred vectors as with
native vectors, the transfer preserves not just identity but magnitude.

Tests alpha sweep (0.1 to 3.0) for:
1. Self (Marin native)
2. Cross-dim (SmolLM3 → Marin)
3. Cross-dim (Llama 1B → Marin)  [best individual source]
4. Ensemble (3 instruct models → Marin)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="xdim-dose-response")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

TARGET_MODEL = "marin-community/marin-8b-instruct"
SOURCE_MODELS = {
    "SmolLM3": "HuggingFaceTB/SmolLM3-3B",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
}

ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


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
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer


def get_5d_coords_and_basis(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d


def fit_procrustes(source_5d, target_5d):
    S = np.stack([source_5d[t] for t in TRAITS])
    T = np.stack([target_5d[t] for t in TRAITS])
    S_n = S / np.linalg.norm(S, axis=1, keepdims=True)
    T_n = T / np.linalg.norm(T, axis=1, keepdims=True)
    M = S_n.T @ T_n
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    scale = np.mean(np.linalg.norm(T, axis=1)) / np.mean(np.linalg.norm(S, axis=1))
    return R, scale


def transfer_vectors(source_5d, source_basis, target_5d, target_basis):
    R, scale = fit_procrustes(source_5d, target_5d)
    transferred = {}
    for t in TRAITS:
        target_coord = scale * (R @ source_5d[t])
        transferred[t] = (target_basis.T @ target_coord).astype(np.float32)
    return transferred


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


def eval_at_alpha(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    correct = 0
    total = 0
    total_delta = 0.0

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
                    total_delta += d
                    total += 1
        finally:
            hook_handle.remove()

    accuracy = correct / total if total else 0
    mean_delta = total_delta / total if total else 0
    return accuracy, mean_delta


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load all source vectors
    logger.info("Loading all source model vectors...")
    source_data = {}
    for name, model_id in SOURCE_MODELS.items():
        residual, _ = load_residual_vectors(model_id, riasec_dir)
        coords, basis = get_5d_coords_and_basis(residual)
        source_data[name] = {"residual": residual, "coords": coords, "basis": basis}

    # Load target vectors
    target_residual, mid_layer = load_residual_vectors(TARGET_MODEL, riasec_dir)
    target_coords, target_basis = get_5d_coords_and_basis(target_residual)

    # Build transferred vector sets
    logger.info("Building transferred vectors...")
    vector_sets = {"Self (native)": target_residual}

    for name in SOURCE_MODELS:
        transferred = transfer_vectors(
            source_data[name]["coords"], source_data[name]["basis"],
            target_coords, target_basis)
        vector_sets[f"Cross-dim ({name})"] = transferred

    # Build ensemble (average of 3 instruct transfers)
    ensemble = {}
    for t in TRAITS:
        vecs = []
        for name in SOURCE_MODELS:
            transferred = transfer_vectors(
                source_data[name]["coords"], source_data[name]["basis"],
                target_coords, target_basis)
            vecs.append(transferred[t])
        ensemble[t] = np.mean(vecs, axis=0).astype(np.float32)
    vector_sets["Ensemble (3 sources)"] = ensemble

    # Load target model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"CROSS-DIMENSIONAL DOSE-RESPONSE")
    print(f"Target: {TARGET_MODEL}")
    print(f"Alphas: {ALPHAS}")
    print(f"{'='*70}")

    # Sweep alphas for each vector set
    results = {}
    for set_name, vectors in vector_sets.items():
        logger.info(f"Alpha sweep for: {set_name}")
        set_results = []

        for alpha in ALPHAS:
            accuracy, mean_delta = eval_at_alpha(
                model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline)
            set_results.append({
                "alpha": alpha,
                "accuracy": float(accuracy),
                "mean_delta": float(mean_delta),
            })
            print(f"  {set_name:>25} | alpha={alpha:.2f} | acc={accuracy:.0%} | mean_delta={mean_delta:+.3f}")

        results[set_name] = set_results

    # Compare curves
    print(f"\n--- Dose-response comparison ---")
    print(f"  {'Alpha':>6}", end="")
    for set_name in vector_sets:
        short = set_name[:15]
        print(f"  {short:>15}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in vector_sets:
        print(f"  {'-'*15}", end="")
    print()

    for i, alpha in enumerate(ALPHAS):
        print(f"  {alpha:>5.2f}", end="")
        for set_name in vector_sets:
            acc = results[set_name][i]["accuracy"]
            delta = results[set_name][i]["mean_delta"]
            print(f"  {acc:>5.0%} ({delta:>+.2f})", end="")
        print()

    # Compute curve correlation (Spearman) between self and each transfer
    from scipy.stats import spearmanr
    self_deltas = [r["mean_delta"] for r in results["Self (native)"]]
    print(f"\n--- Curve correlation with Self (Spearman rho on mean_delta) ---")
    for set_name in vector_sets:
        if set_name == "Self (native)":
            continue
        other_deltas = [r["mean_delta"] for r in results[set_name]]
        rho, p = spearmanr(self_deltas, other_deltas)
        print(f"  {set_name:>25}: rho={rho:.3f}, p={p:.4f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_dim_dose_response.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
