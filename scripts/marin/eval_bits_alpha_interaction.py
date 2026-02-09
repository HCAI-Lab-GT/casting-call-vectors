#!/usr/bin/env python
"""
Bits × Alpha interaction: does higher alpha compensate for wrong bits?

At the practical sweet spot (α=3), do you need fewer bits for good transfer?
At minimal alpha (α=0.1), do you need MORE bits?

Tests a grid of {0,1,2,3,4,5 correct bits} × {0.5, 1.0, 2.0, 3.0} alpha values.
Uses greedy bit addition order (PC1, PC2, PC4, PC3, PC5) from prior experiments.

Only SmolLM3 → Marin 8B for efficiency.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="bits-alpha")

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


def build_transferred_vectors(source_coords, source_basis, target_coords, target_basis, sign_vector):
    corrected_coords = {t: sign_vector * source_coords[t] for t in TRAITS}
    source_norms = np.mean([np.linalg.norm(corrected_coords[t]) for t in TRAITS])
    target_norms = np.mean([np.linalg.norm(target_coords[t]) for t in TRAITS])
    scale = target_norms / source_norms
    transferred = {}
    for t in TRAITS:
        target_coord = scale * corrected_coords[t]
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
    riasec_dir = _repo_root() / "persona_data/model_inits"

    source_id = "HuggingFaceTB/SmolLM3-3B"
    target_id = "marin-community/marin-8b-instruct"

    # Alpha values to test
    alphas = [0.5, 1.0, 2.0, 3.0]

    # Load vectors
    logger.info("Loading vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    source_coords, source_basis = get_5d_coords_and_basis(source_residual)

    target_residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    target_coords, target_basis = get_5d_coords_and_basis(target_residual)

    source_canon = canonical_sign_convention(source_coords)
    target_canon = canonical_sign_convention(target_coords)
    correct_relative_signs = target_canon * source_canon

    # Greedy bit order from prior experiments: PC1, PC2, PC4, PC3, PC5
    # (indices 0, 1, 3, 2, 4)
    bit_order = [0, 1, 3, 2, 4]

    # Load model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    print(f"\n{'='*70}")
    print(f"BITS × ALPHA INTERACTION (SmolLM3 → Marin 8B)")
    print(f"{'='*70}")

    results = {}

    for alpha in alphas:
        logger.info(f"Testing alpha={alpha}...")

        # Compute baseline at this alpha
        baseline = {}
        for i, trait_a in enumerate(TRAITS):
            for j, trait_b in enumerate(TRAITS):
                if i >= j:
                    continue
                gap = pairwise_logprob_chat(model, tokenizer, device,
                                           TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
                baseline[f"{trait_a}-{trait_b}"] = gap

        # Self-steering
        self_acc, self_delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, target_residual, alpha, baseline)

        alpha_results = {"self": {"accuracy": float(self_acc), "mean_delta": float(self_delta)}}

        # Greedy bit addition at this alpha
        current_signs = np.ones(5)  # identity
        greedy = []

        # 0 bits
        transferred = build_transferred_vectors(
            source_coords, source_basis, target_coords, target_basis, current_signs)
        acc, delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)
        greedy.append({"n_bits": 0, "accuracy": float(acc), "mean_delta": float(delta)})

        for n_bits in range(1, 6):
            pc = bit_order[n_bits - 1]
            current_signs[pc] = correct_relative_signs[pc]
            transferred = build_transferred_vectors(
                source_coords, source_basis, target_coords, target_basis, current_signs)
            acc, delta = eval_accuracy(
                model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)
            greedy.append({
                "n_bits": n_bits,
                "accuracy": float(acc),
                "mean_delta": float(delta),
                "last_added_pc": pc + 1,
            })

        alpha_results["greedy"] = greedy
        results[str(alpha)] = alpha_results

        print(f"\n  α={alpha}: self={self_acc:.0%}")
        print(f"  {'Bits':>5}  {'Accuracy':>8}  {'Delta':>8}")
        for g in greedy:
            print(f"  {g['n_bits']:>5}  {g['accuracy']:>7.0%}  {g['mean_delta']:>+7.3f}")

    # Summary grid
    print(f"\n{'='*70}")
    print(f"BITS × ALPHA GRID (accuracy)")
    print(f"{'='*70}")

    print(f"\n  {'Bits':>5}", end="")
    for alpha in alphas:
        print(f"  {'α='+str(alpha):>7}", end="")
    print(f"  {'self':>7}")

    for n_bits in range(6):
        row = f"  {n_bits:>5}"
        for alpha in alphas:
            acc = results[str(alpha)]["greedy"][n_bits]["accuracy"]
            row += f"  {acc:>6.0%}"
        # Self column
        row_self = []
        for alpha in alphas:
            row_self.append(results[str(alpha)]["self"]["accuracy"])
        row += f"  {np.mean(row_self):>6.0%}"
        print(row)

    # Find minimum bits needed for 90% at each alpha
    print(f"\n  Minimum bits for ≥90% accuracy:")
    for alpha in alphas:
        min_bits = 6
        for g in results[str(alpha)]["greedy"]:
            if g["accuracy"] >= 0.90:
                min_bits = g["n_bits"]
                break
        print(f"  α={alpha}: {min_bits} bits")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bits_alpha_interaction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
