#!/usr/bin/env python
"""
Information bits generalization: test bit-by-bit importance for ALL source models.

Extension of eval_information_bits.py which tested SmolLM3→Marin exhaustively.
This script tests the greedy bit addition for ALL 3 source models → Marin 8B
to see if the importance hierarchy (PC1 > PC2 > PC4 > PC3 >> PC5) generalizes.

Also tests: does the number of "free" bits (bits correct by default in identity)
vary across model pairs? This would predict the "transfer cost" between models.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="info-bits-multi")

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
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    target_name, target_id = TARGET_MODEL

    # Load all source vectors
    logger.info("Loading all model vectors...")
    source_data = {}
    for name, model_id in SOURCE_MODELS.items():
        residual, mid = load_residual_vectors(model_id, riasec_dir)
        coords, basis = get_5d_coords_and_basis(residual)
        source_data[name] = {
            "residual": residual,
            "coords": coords,
            "basis": basis,
            "canonical_signs": canonical_sign_convention(coords),
        }

    # Load target
    target_residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    target_coords, target_basis = get_5d_coords_and_basis(target_residual)
    target_canon = canonical_sign_convention(target_coords)

    # Load model
    logger.info(f"Loading {target_name}...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
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

    # Self-steering reference
    logger.info("Testing self-steering...")
    self_acc, self_delta = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer, target_residual, alpha, baseline)
    print(f"\nSelf-steering: {self_acc:.0%}")

    print(f"\n{'='*70}")
    print(f"INFORMATION BITS GENERALIZATION: ALL SOURCES → {target_name}")
    print(f"{'='*70}")

    results = {"self": {"accuracy": float(self_acc)}, "target_canonical_signs": target_canon.tolist()}

    for source_name in SOURCE_MODELS:
        source_canon = source_data[source_name]["canonical_signs"]
        correct_relative_signs = target_canon * source_canon
        n_free_bits = int(np.sum(np.ones(5) == correct_relative_signs))

        print(f"\n{'='*70}")
        print(f"SOURCE: {source_name}")
        print(f"  Source canonical signs: {source_canon}")
        print(f"  Target canonical signs: {target_canon}")
        print(f"  Correct relative signs: {correct_relative_signs}")
        print(f"  Free bits (correct in identity): {n_free_bits}/5")
        print(f"{'='*70}")

        # Per-bit marginal importance (from 32 combos)
        # Only test all 32 combos for efficiency analysis
        logger.info(f"Testing all 32 sign combos for {source_name}→{target_name}...")
        combo_results = {}
        for combo_idx, sign_bits in enumerate(itertools.product([-1, 1], repeat=5)):
            sign_vector = np.array(sign_bits, dtype=np.float64)
            n_correct = int(np.sum(sign_vector == correct_relative_signs))
            transferred = build_transferred_vectors(
                source_data[source_name]["coords"], source_data[source_name]["basis"],
                target_coords, target_basis, sign_vector)
            acc, delta = eval_accuracy(
                model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)
            combo_key = "".join("+" if s > 0 else "-" for s in sign_bits)
            combo_results[combo_key] = {
                "signs": list(sign_bits),
                "accuracy": float(acc),
                "mean_delta": float(delta),
                "n_correct_bits": n_correct,
            }

        # Per-bit importance
        bit_importance = {}
        for pc in range(5):
            correct_accs = [r["accuracy"] for r in combo_results.values()
                           if r["signs"][pc] == correct_relative_signs[pc]]
            incorrect_accs = [r["accuracy"] for r in combo_results.values()
                             if r["signs"][pc] != correct_relative_signs[pc]]
            marginal = np.mean(correct_accs) - np.mean(incorrect_accs)
            bit_importance[pc] = float(marginal)

        sorted_bits = sorted(bit_importance.items(), key=lambda x: -abs(x[1]))
        print(f"\n  Per-bit importance:")
        for rank, (pc, imp) in enumerate(sorted_bits, 1):
            print(f"    {rank}. PC{pc+1}: marginal = {imp:+.0%}")

        # Accuracy by number of correct bits
        by_n_bits = {}
        for result in combo_results.values():
            n = result["n_correct_bits"]
            if n not in by_n_bits:
                by_n_bits[n] = []
            by_n_bits[n].append(result["accuracy"])

        print(f"\n  Accuracy by correct bits:")
        for n in sorted(by_n_bits.keys()):
            accs = by_n_bits[n]
            print(f"    {n}/5 bits: mean={np.mean(accs):.0%}, range=[{np.min(accs):.0%}, {np.max(accs):.0%}]")

        # Greedy bit addition
        bit_order = [pc for pc, _ in sorted_bits]
        current_signs = np.ones(5)  # identity

        greedy = []
        transferred_0 = build_transferred_vectors(
            source_data[source_name]["coords"], source_data[source_name]["basis"],
            target_coords, target_basis, current_signs)
        acc_0, _ = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, transferred_0, alpha, baseline)
        greedy.append({"n_bits": 0, "accuracy": float(acc_0), "bits_set": []})
        print(f"\n  Greedy bit addition:")
        print(f"    0 bits (identity): {acc_0:.0%}")

        for n_bits in range(1, 6):
            pc = bit_order[n_bits - 1]
            current_signs[pc] = correct_relative_signs[pc]
            transferred = build_transferred_vectors(
                source_data[source_name]["coords"], source_data[source_name]["basis"],
                target_coords, target_basis, current_signs)
            acc_n, _ = eval_accuracy(
                model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)
            bits_set = [bit_order[i] + 1 for i in range(n_bits)]
            greedy.append({"n_bits": n_bits, "accuracy": float(acc_n), "bits_set": bits_set})
            print(f"    {n_bits} bits (+ PC{pc+1}): {acc_n:.0%}  [PCs: {', '.join(str(b) for b in bits_set)}]")

        # Find minimum bits needed for ≥90%
        min_bits_90 = next((g["n_bits"] for g in greedy if g["accuracy"] >= 0.90), 6)
        min_bits_97 = next((g["n_bits"] for g in greedy if g["accuracy"] >= 0.97), 6)

        results[source_name] = {
            "canonical_signs": source_canon.tolist(),
            "correct_relative_signs": correct_relative_signs.tolist(),
            "free_bits": n_free_bits,
            "bit_importance": {f"PC{pc+1}": float(imp) for pc, imp in bit_importance.items()},
            "bit_importance_ranking": [f"PC{pc+1}" for pc, _ in sorted_bits],
            "by_n_correct_bits": {
                str(n): {"mean": float(np.mean(accs)), "min": float(np.min(accs)), "max": float(np.max(accs))}
                for n, accs in by_n_bits.items()
            },
            "greedy_addition": greedy,
            "min_bits_for_90pct": min_bits_90,
            "min_bits_for_97pct": min_bits_97,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: TRANSFER COST ACROSS SOURCE MODELS")
    print(f"{'='*70}")

    print(f"\n  {'Source':>10}  {'Free':>5}  {'Min90':>6}  {'Min97':>6}  {'Bit order':>25}")
    for source_name in SOURCE_MODELS:
        r = results[source_name]
        print(f"  {source_name:>10}  {r['free_bits']:>4}/5  "
              f"{r['min_bits_for_90pct']:>5}  {r['min_bits_for_97pct']:>5}  "
              f"{' > '.join(r['bit_importance_ranking'])}")

    # Check if importance ranking is universal
    rankings = [results[name]["bit_importance_ranking"] for name in SOURCE_MODELS]
    rankings_match = all(r == rankings[0] for r in rankings[1:])
    print(f"\n  Importance ranking universal: {rankings_match}")
    if not rankings_match:
        for name in SOURCE_MODELS:
            print(f"    {name}: {' > '.join(results[name]['bit_importance_ranking'])}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "information_bits_multi.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
