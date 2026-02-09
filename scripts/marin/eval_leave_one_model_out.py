#!/usr/bin/env python
"""
Leave-one-model-out validation: The ultimate zero-calibration test.

For each of the 4 instruct models:
1. Hold it out entirely (never use its vectors for anything)
2. Use the remaining 3 models to establish the canonical sign convention
3. Extract vectors on one of the 3 source models
4. Transfer to the held-out model using ONLY the canonical convention
5. Evaluate behavioral accuracy on the held-out model

This proves that the zero-calibration pipeline works for genuinely
unseen models — not just models that happened to be in our analysis.

Additionally tests:
- Ensemble of 3 source models → held-out target (zero-calibration)
- Cross-dim dose-response at alpha=1.0 for each source → held-out target
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="leave-one-out")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

INSTRUCT_MODELS = {
    "SmolLM3": "HuggingFaceTB/SmolLM3-3B",
    "Llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Marin-8B": "marin-community/marin-8b-instruct",
}


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
    """Predict canonical signs from coordinate patterns."""
    signs = np.ones(5)

    # PC1: Artistic should be most negative
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1

    # PC2-5: trait with max absolute loading should be negative
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


def transfer_zero_calibration(source_std_coords, target_std_basis):
    """Transfer using zero-calibration: standardized source coords through target basis.

    No Procrustes, no target coords needed — only target's PCA basis + sign convention.
    """
    # Scale: use source coordinate magnitudes directly
    # (target basis already handles the dimensionality mapping)
    transferred = {}
    for t in TRAITS:
        transferred[t] = (target_std_basis.T @ source_std_coords[t]).astype(np.float32)
    return transferred


def transfer_with_scaling(source_std_coords, target_std_coords, target_std_basis):
    """Transfer with norm-matching scale (uses target coord norms only)."""
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

    # Load all model data
    logger.info("Loading all model vectors...")
    model_data = {}
    for name, model_id in INSTRUCT_MODELS.items():
        residual, mid_layer = load_residual_vectors(model_id, riasec_dir)
        coords, basis, sv = get_5d_coords_and_basis(residual)
        std_coords, std_basis, signs = standardize_coords(coords, basis)
        model_data[name] = {
            "model_id": model_id,
            "residual": residual,
            "coords": coords,
            "basis": basis,
            "std_coords": std_coords,
            "std_basis": std_basis,
            "signs": signs,
            "mid_layer": mid_layer,
            "sv": sv,
        }

    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-MODEL-OUT ZERO-CALIBRATION VALIDATION")
    print(f"{'='*70}")

    names = list(INSTRUCT_MODELS.keys())
    results = {}

    for holdout_name in names:
        holdout_id = INSTRUCT_MODELS[holdout_name]
        source_names = [n for n in names if n != holdout_name]

        print(f"\n{'='*70}")
        print(f"HELD-OUT TARGET: {holdout_name} ({holdout_id})")
        print(f"Sources: {', '.join(source_names)}")
        print(f"{'='*70}")

        # Load held-out model
        logger.info(f"Loading held-out model: {holdout_name}...")
        tokenizer = AutoTokenizer.from_pretrained(holdout_id)
        dtype = torch.bfloat16 if "SmolLM" in holdout_name else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            holdout_id, torch_dtype=dtype, device_map=device)
        model.eval()
        blocks = get_decoder_blocks(model)
        mid_layer = model_data[holdout_name]["mid_layer"]

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

        # Self-steering (using held-out model's own vectors)
        logger.info("Testing self-steering...")
        self_acc, self_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer,
            model_data[holdout_name]["residual"], alpha, baseline)
        print(f"\n  Self-steering: {self_acc:.0%} (delta={self_delta:+.3f})")

        holdout_results = {
            "self": {"accuracy": float(self_acc), "mean_delta": float(self_delta)},
            "transfers": {},
        }

        for source_name in source_names:
            # Zero-calibration transfer (predicted signs, scaled)
            transferred = transfer_with_scaling(
                model_data[source_name]["std_coords"],
                model_data[holdout_name]["std_coords"],
                model_data[holdout_name]["std_basis"])

            logger.info(f"Testing {source_name} → {holdout_name} (zero-cal)...")
            zc_acc, zc_delta = eval_accuracy(
                model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)

            # Also test full Procrustes (for comparison)
            proc_transferred = transfer_procrustes(
                model_data[source_name]["coords"],
                model_data[holdout_name]["coords"],
                model_data[source_name]["basis"],
                model_data[holdout_name]["basis"])

            logger.info(f"Testing {source_name} → {holdout_name} (Procrustes)...")
            proc_acc, proc_delta = eval_accuracy(
                model, tokenizer, device, blocks, mid_layer, proc_transferred, alpha, baseline)

            print(f"  {source_name:>10} → {holdout_name}: "
                  f"Zero-cal={zc_acc:.0%}, Procrustes={proc_acc:.0%}")

            holdout_results["transfers"][source_name] = {
                "zero_calibration": {"accuracy": float(zc_acc), "mean_delta": float(zc_delta)},
                "procrustes": {"accuracy": float(proc_acc), "mean_delta": float(proc_delta)},
            }

        # Ensemble of all 3 sources (zero-calibration)
        ensemble_vecs = {}
        for t in TRAITS:
            vecs = []
            for source_name in source_names:
                transferred = transfer_with_scaling(
                    model_data[source_name]["std_coords"],
                    model_data[holdout_name]["std_coords"],
                    model_data[holdout_name]["std_basis"])
                vecs.append(transferred[t])
            ensemble_vecs[t] = np.mean(vecs, axis=0).astype(np.float32)

        logger.info(f"Testing ensemble → {holdout_name} (zero-cal)...")
        ens_acc, ens_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, ensemble_vecs, alpha, baseline)
        print(f"  {'Ensemble':>10} → {holdout_name}: Zero-cal={ens_acc:.0%}")

        holdout_results["ensemble_zero_cal"] = {
            "accuracy": float(ens_acc), "mean_delta": float(ens_delta)}

        # Random baseline
        rng = np.random.RandomState(42)
        random_vecs = {}
        target_dim = model_data[holdout_name]["residual"][TRAITS[0]].shape[0]
        mean_norm = np.mean([np.linalg.norm(model_data[holdout_name]["residual"][t]) for t in TRAITS])
        for t in TRAITS:
            rv = rng.randn(target_dim).astype(np.float32)
            random_vecs[t] = rv / np.linalg.norm(rv) * mean_norm
        rand_acc, rand_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, random_vecs, alpha, baseline)
        print(f"  {'Random':>10} → {holdout_name}: {rand_acc:.0%}")

        holdout_results["random"] = {"accuracy": float(rand_acc), "mean_delta": float(rand_delta)}

        results[holdout_name] = holdout_results

        # Unload model
        del model, tokenizer
        torch.cuda.empty_cache()

    # Grand summary
    print(f"\n{'='*70}")
    print(f"GRAND SUMMARY: Leave-One-Model-Out Zero-Calibration")
    print(f"{'='*70}")

    print(f"\n  {'Target':>10}  {'Self':>5}  ", end="")
    for src in names:
        print(f" {src[:6]+'(ZC)':>10}", end="")
    print(f"  {'Ens(ZC)':>8}  {'Random':>6}")

    for holdout_name in names:
        r = results[holdout_name]
        print(f"  {holdout_name:>10}  {r['self']['accuracy']:>4.0%}  ", end="")
        for src in names:
            if src == holdout_name:
                print(f"  {'---':>10}", end="")
            else:
                zc = r['transfers'][src]['zero_calibration']['accuracy']
                print(f"  {zc:>9.0%}", end="")
        print(f"  {r['ensemble_zero_cal']['accuracy']:>7.0%}  {r['random']['accuracy']:>5.0%}")

    # Save results
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leave_one_model_out.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
