#!/usr/bin/env python
"""
Leave-one-model-out: Marin 8B as held-out target.
(Separate script to avoid OOM from sequential model loading.)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="leave-one-out-marin")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

SOURCE_MODELS = {
    "SmolLM3": "HuggingFaceTB/SmolLM3-3B",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
}
TARGET_MODEL = ("Marin-8B", "marin-community/marin-8b-instruct")


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
    return coords_5d, basis_5d, S[:5]


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


def standardize_coords(coords_5d, basis_5d):
    signs = canonical_sign_convention(coords_5d)
    std_coords = {t: signs * coords_5d[t] for t in TRAITS}
    std_basis = np.diag(signs) @ basis_5d
    return std_coords, std_basis, signs


def transfer_with_scaling(source_std_coords, target_std_coords, target_std_basis):
    source_norms = np.mean([np.linalg.norm(source_std_coords[t]) for t in TRAITS])
    target_norms = np.mean([np.linalg.norm(target_std_coords[t]) for t in TRAITS])
    scale = target_norms / source_norms
    transferred = {}
    for t in TRAITS:
        transferred[t] = (target_std_basis.T @ (scale * source_std_coords[t])).astype(np.float32)
    return transferred


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


def transfer_procrustes(source_5d, target_5d, source_basis, target_basis):
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


def eval_accuracy(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
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
    return correct / total if total else 0, total_delta / total if total else 0


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"
    holdout_name, holdout_id = TARGET_MODEL

    # Load all source vectors (CPU only, no GPU)
    logger.info("Loading all source vectors...")
    source_data = {}
    for name, model_id in SOURCE_MODELS.items():
        residual, _ = load_residual_vectors(model_id, riasec_dir)
        coords, basis, sv = get_5d_coords_and_basis(residual)
        std_coords, std_basis, signs = standardize_coords(coords, basis)
        source_data[name] = {
            "residual": residual,
            "coords": coords,
            "basis": basis,
            "std_coords": std_coords,
            "std_basis": std_basis,
        }

    # Load target vectors
    target_residual, mid_layer = load_residual_vectors(holdout_id, riasec_dir)
    target_coords, target_basis, _ = get_5d_coords_and_basis(target_residual)
    target_std_coords, target_std_basis, target_signs = standardize_coords(target_coords, target_basis)

    # Load target model
    logger.info(f"Loading {holdout_name}...")
    tokenizer = AutoTokenizer.from_pretrained(holdout_id)
    model = AutoModelForCausalLM.from_pretrained(
        holdout_id, torch_dtype=torch.float16, device_map=device)
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
    print(f"HELD-OUT TARGET: {holdout_name}")
    print(f"{'='*70}")

    # Self
    logger.info("Testing self-steering...")
    self_acc, self_delta = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer, target_residual, alpha, baseline)
    print(f"\n  Self-steering: {self_acc:.0%} (delta={self_delta:+.3f})")

    results = {"self": {"accuracy": float(self_acc), "mean_delta": float(self_delta)}, "transfers": {}}

    for source_name in SOURCE_MODELS:
        # Zero-calibration transfer
        transferred = transfer_with_scaling(
            source_data[source_name]["std_coords"],
            target_std_coords, target_std_basis)

        logger.info(f"Testing {source_name} → {holdout_name} (zero-cal)...")
        zc_acc, zc_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)

        # Full Procrustes
        proc_transferred = transfer_procrustes(
            source_data[source_name]["coords"], target_coords,
            source_data[source_name]["basis"], target_basis)

        logger.info(f"Testing {source_name} → {holdout_name} (Procrustes)...")
        proc_acc, proc_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, proc_transferred, alpha, baseline)

        print(f"  {source_name:>10} → {holdout_name}: Zero-cal={zc_acc:.0%}, Procrustes={proc_acc:.0%}")

        results["transfers"][source_name] = {
            "zero_calibration": {"accuracy": float(zc_acc), "mean_delta": float(zc_delta)},
            "procrustes": {"accuracy": float(proc_acc), "mean_delta": float(proc_delta)},
        }

    # Ensemble
    ensemble_vecs = {}
    for t in TRAITS:
        vecs = []
        for source_name in SOURCE_MODELS:
            transferred = transfer_with_scaling(
                source_data[source_name]["std_coords"],
                target_std_coords, target_std_basis)
            vecs.append(transferred[t])
        ensemble_vecs[t] = np.mean(vecs, axis=0).astype(np.float32)

    logger.info(f"Testing ensemble → {holdout_name} (zero-cal)...")
    ens_acc, ens_delta = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer, ensemble_vecs, alpha, baseline)
    print(f"  {'Ensemble':>10} → {holdout_name}: Zero-cal={ens_acc:.0%}")
    results["ensemble_zero_cal"] = {"accuracy": float(ens_acc), "mean_delta": float(ens_delta)}

    # Random
    rng = np.random.RandomState(42)
    random_vecs = {}
    target_dim = target_residual[TRAITS[0]].shape[0]
    mean_norm = np.mean([np.linalg.norm(target_residual[t]) for t in TRAITS])
    for t in TRAITS:
        rv = rng.randn(target_dim).astype(np.float32)
        random_vecs[t] = rv / np.linalg.norm(rv) * mean_norm
    rand_acc, rand_delta = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer, random_vecs, alpha, baseline)
    print(f"  {'Random':>10} → {holdout_name}: {rand_acc:.0%}")
    results["random"] = {"accuracy": float(rand_acc), "mean_delta": float(rand_delta)}

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leave_one_model_out_marin.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
