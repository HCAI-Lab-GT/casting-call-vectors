#!/usr/bin/env python
"""
Information-theoretic minimum for personality transfer.

We showed that 5 sign bits suffice for zero-calibration transfer (97%).
This script tests ALL 2^5 = 32 possible sign combinations to determine:
1. How many bits are ACTUALLY needed?
2. Which bits matter most?
3. Is there a monotonic information curve?

For each of 32 sign combinations:
- Apply that sign vector to source (SmolLM3) 5D coordinates
- Transfer to target (Marin 8B) via identity permutation + that sign combo
- Evaluate pairwise discrimination on Marin 8B

This produces a complete "information surface" showing transfer quality
as a function of sign configuration.

Additionally tests systematic 0-5 bit subsets:
- 0 bits: No correction (identity signs)
- 1 bit: Only correct the MOST important sign
- 2 bits: Correct the 2 most important signs
- etc.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="info-bits")

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
    """Build transferred vectors using a specific sign correction vector.

    sign_vector is a 5-element array of +1/-1 values.
    These signs are applied to source coords before transferring to target space.
    """
    # Apply signs to source coordinates
    corrected_coords = {t: sign_vector * source_coords[t] for t in TRAITS}

    # Scale
    source_norms = np.mean([np.linalg.norm(corrected_coords[t]) for t in TRAITS])
    target_norms = np.mean([np.linalg.norm(target_coords[t]) for t in TRAITS])
    scale = target_norms / source_norms

    # Transfer: corrected source coords → target basis
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

    source_id = "HuggingFaceTB/SmolLM3-3B"
    target_id = "marin-community/marin-8b-instruct"

    # Load vectors
    logger.info("Loading vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    source_coords, source_basis = get_5d_coords_and_basis(source_residual)

    target_residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    target_coords, target_basis = get_5d_coords_and_basis(target_residual)

    # Compute the TRUE correct signs (from canonical convention applied to both)
    source_canon = canonical_sign_convention(source_coords)
    target_canon = canonical_sign_convention(target_coords)
    # The correct relative signs: what you'd need to multiply source coords by
    # to align with target coords (after standardizing both)
    correct_relative_signs = target_canon * source_canon
    logger.info(f"Source canonical signs: {source_canon}")
    logger.info(f"Target canonical signs: {target_canon}")
    logger.info(f"Correct relative signs: {correct_relative_signs}")

    # Load model
    logger.info("Loading Marin 8B...")
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

    # Self-steering
    logger.info("Testing self-steering...")
    self_acc, self_delta = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer, target_residual, alpha, baseline)
    print(f"\nSelf-steering: {self_acc:.0%} (delta={self_delta:+.3f})")

    # Test ALL 32 sign combinations
    print(f"\n{'='*70}")
    print(f"EXHAUSTIVE SIGN COMBINATION SEARCH (32 combinations)")
    print(f"Source: SmolLM3 → Target: Marin 8B")
    print(f"{'='*70}")

    all_results = {
        "self": {"accuracy": float(self_acc), "mean_delta": float(self_delta)},
        "source_canonical_signs": source_canon.tolist(),
        "target_canonical_signs": target_canon.tolist(),
        "correct_relative_signs": correct_relative_signs.tolist(),
    }

    combo_results = {}
    best_acc = 0
    best_combo = None
    worst_acc = 1.0
    worst_combo = None

    for combo_idx, sign_bits in enumerate(itertools.product([-1, 1], repeat=5)):
        sign_vector = np.array(sign_bits, dtype=np.float64)
        # Count how many signs match the correct relative signs
        n_correct_bits = int(np.sum(sign_vector == correct_relative_signs))

        # Build transferred vectors with this sign combination
        transferred = build_transferred_vectors(
            source_coords, source_basis, target_coords, target_basis, sign_vector)

        logger.info(f"Testing combo {combo_idx+1}/32: signs={sign_bits}, correct_bits={n_correct_bits}/5...")
        acc, delta = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, transferred, alpha, baseline)

        combo_key = "".join("+" if s > 0 else "-" for s in sign_bits)
        combo_results[combo_key] = {
            "signs": list(sign_bits),
            "accuracy": float(acc),
            "mean_delta": float(delta),
            "n_correct_bits": n_correct_bits,
            "is_correct": n_correct_bits == 5,
        }

        if acc > best_acc:
            best_acc = acc
            best_combo = combo_key
        if acc < worst_acc:
            worst_acc = acc
            worst_combo = combo_key

        print(f"  {combo_key}  bits_correct={n_correct_bits}/5  acc={acc:.0%}  delta={delta:+.3f}"
              f"{'  *** CORRECT' if n_correct_bits == 5 else ''}")

    all_results["combinations"] = combo_results
    all_results["best"] = {"combo": best_combo, "accuracy": best_acc}
    all_results["worst"] = {"combo": worst_combo, "accuracy": worst_acc}

    # Analysis: accuracy by number of correct bits
    print(f"\n{'='*70}")
    print(f"ACCURACY BY NUMBER OF CORRECT SIGN BITS")
    print(f"{'='*70}")

    by_n_bits = {}
    for combo_key, result in combo_results.items():
        n = result["n_correct_bits"]
        if n not in by_n_bits:
            by_n_bits[n] = []
        by_n_bits[n].append(result["accuracy"])

    print(f"\n  {'Correct bits':>12}  {'Combos':>6}  {'Mean acc':>8}  {'Min':>6}  {'Max':>6}  {'Std':>6}")
    for n in sorted(by_n_bits.keys()):
        accs = by_n_bits[n]
        print(f"  {n:>12}/5  {len(accs):>6}  {np.mean(accs):>7.0%}  "
              f"{np.min(accs):>5.0%}  {np.max(accs):>5.0%}  {np.std(accs):>5.3f}")

    all_results["by_n_correct_bits"] = {
        str(n): {
            "n_combos": len(accs),
            "mean_accuracy": float(np.mean(accs)),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
            "std_accuracy": float(np.std(accs)),
        } for n, accs in by_n_bits.items()
    }

    # Analysis: which individual bits matter most?
    print(f"\n{'='*70}")
    print(f"PER-BIT IMPORTANCE (marginal effect of getting each bit right)")
    print(f"{'='*70}")

    bit_importance = {}
    for pc in range(5):
        # Average accuracy when this bit is correct vs incorrect
        correct_accs = []
        incorrect_accs = []
        for combo_key, result in combo_results.items():
            if result["signs"][pc] == correct_relative_signs[pc]:
                correct_accs.append(result["accuracy"])
            else:
                incorrect_accs.append(result["accuracy"])
        marginal = np.mean(correct_accs) - np.mean(incorrect_accs)
        bit_importance[pc] = {
            "marginal_effect": float(marginal),
            "correct_mean": float(np.mean(correct_accs)),
            "incorrect_mean": float(np.mean(incorrect_accs)),
            "correct_sign": float(correct_relative_signs[pc]),
        }
        print(f"  PC{pc+1}: correct={np.mean(correct_accs):.0%}, incorrect={np.mean(incorrect_accs):.0%}, "
              f"marginal={marginal:+.0%} (correct sign: {'+' if correct_relative_signs[pc] > 0 else '-'})")

    all_results["bit_importance"] = bit_importance

    # Sort bits by importance
    sorted_bits = sorted(bit_importance.items(), key=lambda x: -abs(x[1]["marginal_effect"]))
    print(f"\n  Bits ranked by importance:")
    for rank, (pc, info) in enumerate(sorted_bits, 1):
        print(f"  {rank}. PC{pc+1}: marginal effect = {info['marginal_effect']:+.0%}")

    # Test greedy bit addition: add bits in order of importance
    print(f"\n{'='*70}")
    print(f"GREEDY BIT ADDITION (add most important bit first)")
    print(f"{'='*70}")

    bit_order = [pc for pc, _ in sorted_bits]
    current_signs = np.ones(5)  # Start with all +1 (identity)

    greedy_results = []
    # 0 bits: identity
    transferred_0 = build_transferred_vectors(
        source_coords, source_basis, target_coords, target_basis, current_signs)
    acc_0, delta_0 = eval_accuracy(
        model, tokenizer, device, blocks, mid_layer, transferred_0, alpha, baseline)
    print(f"  0 bits (identity): {acc_0:.0%}")
    greedy_results.append({"n_bits": 0, "accuracy": float(acc_0), "bits_set": []})

    for n_bits in range(1, 6):
        pc = bit_order[n_bits - 1]
        current_signs[pc] = correct_relative_signs[pc]
        transferred_n = build_transferred_vectors(
            source_coords, source_basis, target_coords, target_basis, current_signs)
        acc_n, delta_n = eval_accuracy(
            model, tokenizer, device, blocks, mid_layer, transferred_n, alpha, baseline)
        bits_set = [bit_order[i] for i in range(n_bits)]
        print(f"  {n_bits} bits (+ PC{pc+1}): {acc_n:.0%}  [PCs: {', '.join(f'{b+1}' for b in bits_set)}]")
        greedy_results.append({
            "n_bits": n_bits,
            "accuracy": float(acc_n),
            "bits_set": [b+1 for b in bits_set],
            "last_added": pc+1,
        })

    all_results["greedy_addition"] = greedy_results

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Self-steering: {self_acc:.0%}")
    print(f"  Best combination: {best_combo} ({best_acc:.0%})")
    print(f"  Worst combination: {worst_combo} ({worst_acc:.0%})")
    print(f"  All-correct (5 bits): {combo_results[''.join('+' if s > 0 else '-' for s in correct_relative_signs)]['accuracy']:.0%}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "information_bits_transfer.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
