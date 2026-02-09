#!/usr/bin/env python
"""
Full-vector zero-calibration transfer using 6D (shared + 5D personality).

Previous zero-calibration experiments used RESIDUAL vectors (shared direction
removed). This loses signal — SmolLM3 self-steering drops from 100% (full) to
63% (residual) at alpha=1.

This script tests whether we can do zero-calibration transfer with FULL
vectors by working in 6D = 1 shared direction + 5 personality dimensions.

The 6th dimension (shared direction) is the SAME for all traits, so we
need to estimate what the "shared direction magnitude" should be in the
target model. Strategy: use the mean projection of source vectors onto
the source's shared direction, scaled by the ratio of target-to-source
hidden dimensions (or better, by the ratio of typical hidden state norms).

Pipeline:
1. Compute full 6D coordinates in source (1 shared + 5 personality PCs)
2. Standardize signs via canonical convention (same as 5D)
3. Scale 5D personality coords to target magnitude
4. Scale shared direction magnitude to target's estimated shared magnitude
5. Reconstruct full-dim vectors in target space
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="full-vec-zero-cal")

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


def load_full_vectors(model_id, riasec_dir):
    """Load full persona vectors AND compute decomposition.

    Returns:
        raw_vecs: dict trait -> full-dim vector (mid layer)
        shared_dir: the shared direction (unit vector)
        shared_magnitudes: dict trait -> projection onto shared dir
        residual_vecs: dict trait -> residual vector
        mid_layer: int
    """
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

    # Raw mid-layer vectors
    raw_vecs = {t: all_layer_vectors[t][mid_layer + 1] for t in TRAITS}

    # SVD to find shared direction
    V = np.stack([raw_vecs[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    # Decompose each vector
    shared_magnitudes = {}
    residual_vecs = {}
    for t in TRAITS:
        proj_mag = np.dot(raw_vecs[t], shared_dir)
        shared_magnitudes[t] = proj_mag
        residual_vecs[t] = raw_vecs[t] - proj_mag * shared_dir

    return raw_vecs, shared_dir, shared_magnitudes, residual_vecs, mid_layer


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


def transfer_6d_zero_cal(source_data, target_data):
    """Transfer full vectors using 6D zero-calibration.

    Reconstructs full vectors = shared_component + personality_component.
    """
    # Standardize 5D personality coordinates
    source_std, source_std_basis, _ = standardize_coords(
        source_data["coords_5d"], source_data["basis_5d"])
    target_std, target_std_basis, _ = standardize_coords(
        target_data["coords_5d"], target_data["basis_5d"])

    # Scale personality coordinates from source to target magnitude
    source_5d_norms = np.mean([np.linalg.norm(source_std[t]) for t in TRAITS])
    target_5d_norms = np.mean([np.linalg.norm(target_std[t]) for t in TRAITS])
    personality_scale = target_5d_norms / source_5d_norms

    # Scale shared direction magnitude
    source_shared_mean = np.mean([source_data["shared_magnitudes"][t] for t in TRAITS])
    target_shared_mean = np.mean([target_data["shared_magnitudes"][t] for t in TRAITS])
    shared_scale = target_shared_mean / source_shared_mean if abs(source_shared_mean) > 1e-10 else 1.0

    transferred = {}
    for t in TRAITS:
        # Personality component: standardized source 5D → target full-dim
        personality = target_std_basis.T @ (personality_scale * source_std[t])

        # Shared component: scale source shared magnitude, apply in target's shared direction
        shared = shared_scale * source_data["shared_magnitudes"][t] * target_data["shared_dir"]

        transferred[t] = (personality + shared).astype(np.float32)

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
    all_data = {}
    for name, model_id in INSTRUCT_MODELS.items():
        raw, shared_dir, shared_mags, residual, mid_layer = load_full_vectors(model_id, riasec_dir)
        coords_5d, basis_5d, sv = get_5d_coords_and_basis(residual)
        all_data[name] = {
            "model_id": model_id,
            "raw_vecs": raw,
            "shared_dir": shared_dir,
            "shared_magnitudes": shared_mags,
            "residual": residual,
            "coords_5d": coords_5d,
            "basis_5d": basis_5d,
            "mid_layer": mid_layer,
            "sv": sv,
        }

    # Show shared direction magnitude comparison
    print(f"\n{'='*70}")
    print(f"FULL-VECTOR 6D ZERO-CALIBRATION TRANSFER")
    print(f"{'='*70}")

    print(f"\n--- Shared direction magnitude per model ---")
    for name in INSTRUCT_MODELS:
        mags = all_data[name]["shared_magnitudes"]
        mean_mag = np.mean([mags[t] for t in TRAITS])
        raw_norms = np.mean([np.linalg.norm(all_data[name]["raw_vecs"][t]) for t in TRAITS])
        res_norms = np.mean([np.linalg.norm(all_data[name]["residual"][t]) for t in TRAITS])
        print(f"  {name:>10}: shared_mean={mean_mag:.1f}, raw_norm={raw_norms:.1f}, "
              f"residual_norm={res_norms:.1f}, shared/raw={abs(mean_mag)/raw_norms:.0%}")

    # Test on Marin 8B as target (where we have reliable evaluation)
    target_name = "Marin-8B"
    target_id = INSTRUCT_MODELS[target_name]

    logger.info(f"Loading {target_name}...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    mid_layer = all_data[target_name]["mid_layer"]

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

    results = {}

    # Self with full vectors
    logger.info("Testing self with full vectors...")
    self_full_acc, self_full_delta = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer,
        all_data[target_name]["raw_vecs"], alpha, baseline)
    print(f"\n  Self (full vectors): {self_full_acc:.0%} (delta={self_full_delta:+.3f})")
    results["self_full"] = {"accuracy": float(self_full_acc), "mean_delta": float(self_full_delta)}

    # Self with residual vectors
    logger.info("Testing self with residual vectors...")
    self_res_acc, self_res_delta = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer,
        all_data[target_name]["residual"], alpha, baseline)
    print(f"  Self (residual):    {self_res_acc:.0%} (delta={self_res_delta:+.3f})")
    results["self_residual"] = {"accuracy": float(self_res_acc), "mean_delta": float(self_res_delta)}

    for source_name in INSTRUCT_MODELS:
        if source_name == target_name:
            continue

        # 6D transfer (full vector)
        transferred_6d = transfer_6d_zero_cal(all_data[source_name], all_data[target_name])
        logger.info(f"Testing {source_name} → {target_name} (6D zero-cal)...")
        acc_6d, delta_6d = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, transferred_6d, alpha, baseline)

        # 5D transfer (residual only, same as before)
        from eval_predicted_sign_transfer import transfer_with_predicted_signs  # reuse
        # Just inline it instead
        source_std, source_std_basis, _ = standardize_coords(
            all_data[source_name]["coords_5d"], all_data[source_name]["basis_5d"])
        target_std, target_std_basis, _ = standardize_coords(
            all_data[target_name]["coords_5d"], all_data[target_name]["basis_5d"])
        source_norms = np.mean([np.linalg.norm(source_std[t]) for t in TRAITS])
        target_norms = np.mean([np.linalg.norm(target_std[t]) for t in TRAITS])
        scale = target_norms / source_norms
        transferred_5d = {}
        for t in TRAITS:
            transferred_5d[t] = (target_std_basis.T @ (scale * source_std[t])).astype(np.float32)

        logger.info(f"Testing {source_name} → {target_name} (5D zero-cal)...")
        acc_5d, delta_5d = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, transferred_5d, alpha, baseline)

        print(f"  {source_name:>10} → {target_name}: 6D={acc_6d:.0%} (d={delta_6d:+.3f}), "
              f"5D={acc_5d:.0%} (d={delta_5d:+.3f})")

        # Compare vector cosines
        cos_6d_native = []
        cos_5d_native = []
        for t in TRAITS:
            c6 = np.dot(transferred_6d[t], all_data[target_name]["raw_vecs"][t]) / (
                np.linalg.norm(transferred_6d[t]) * np.linalg.norm(all_data[target_name]["raw_vecs"][t]))
            c5 = np.dot(transferred_5d[t], all_data[target_name]["residual"][t]) / (
                np.linalg.norm(transferred_5d[t]) * np.linalg.norm(all_data[target_name]["residual"][t]))
            cos_6d_native.append(c6)
            cos_5d_native.append(c5)

        results[f"{source_name}_6d"] = {
            "accuracy": float(acc_6d), "mean_delta": float(delta_6d),
            "cos_to_native": float(np.mean(cos_6d_native)),
        }
        results[f"{source_name}_5d"] = {
            "accuracy": float(acc_5d), "mean_delta": float(delta_5d),
            "cos_to_native": float(np.mean(cos_5d_native)),
        }

    # Now test on SmolLM3 where the shared direction matters most
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()

    target_name2 = "SmolLM3"
    target_id2 = INSTRUCT_MODELS[target_name2]

    logger.info(f"Loading {target_name2}...")
    tokenizer2 = AutoTokenizer.from_pretrained(target_id2)
    model2 = AutoModelForCausalLM.from_pretrained(
        target_id2, torch_dtype=torch.bfloat16, device_map=device)
    model2.eval()
    blocks2 = get_decoder_blocks(model2)
    mid_layer2 = all_data[target_name2]["mid_layer"]

    # Baseline
    logger.info("Computing SmolLM3 baseline...")
    baseline2 = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model2, tokenizer2, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline2[f"{trait_a}-{trait_b}"] = gap

    # Self full
    self_full2, self_full_d2 = eval_accuracy(
        model2, tokenizer2, device, blocks2, mid_layer2,
        all_data[target_name2]["raw_vecs"], alpha, baseline2)
    print(f"\n  Self {target_name2} (full): {self_full2:.0%}")

    # Self residual
    self_res2, self_res_d2 = eval_accuracy(
        model2, tokenizer2, device, blocks2, mid_layer2,
        all_data[target_name2]["residual"], alpha, baseline2)
    print(f"  Self {target_name2} (res):  {self_res2:.0%}")

    results[f"{target_name2}_self_full"] = {"accuracy": float(self_full2)}
    results[f"{target_name2}_self_residual"] = {"accuracy": float(self_res2)}

    # 6D transfer from best source (Llama 1B)
    for source_name in ["Llama-1B", "Marin-8B"]:
        transferred_6d = transfer_6d_zero_cal(all_data[source_name], all_data[target_name2])
        acc_6d, delta_6d = eval_accuracy(
            model2, tokenizer2, device, blocks2, mid_layer2, transferred_6d, alpha, baseline2)

        source_std, source_std_basis, _ = standardize_coords(
            all_data[source_name]["coords_5d"], all_data[source_name]["basis_5d"])
        target_std, target_std_basis, _ = standardize_coords(
            all_data[target_name2]["coords_5d"], all_data[target_name2]["basis_5d"])
        s_norms = np.mean([np.linalg.norm(source_std[t]) for t in TRAITS])
        t_norms = np.mean([np.linalg.norm(target_std[t]) for t in TRAITS])
        scale = t_norms / s_norms
        transferred_5d = {}
        for t in TRAITS:
            transferred_5d[t] = (target_std_basis.T @ (scale * source_std[t])).astype(np.float32)

        acc_5d, delta_5d = eval_accuracy(
            model2, tokenizer2, device, blocks2, mid_layer2, transferred_5d, alpha, baseline2)

        print(f"  {source_name} → {target_name2}: 6D={acc_6d:.0%}, 5D={acc_5d:.0%}")
        results[f"{target_name2}_{source_name}_6d"] = {"accuracy": float(acc_6d)}
        results[f"{target_name2}_{source_name}_5d"] = {"accuracy": float(acc_5d)}

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_vector_zero_cal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
