#!/usr/bin/env python
"""
Per-layer pairwise discrimination sweep on SmolLM3-3B.

Tests: At which layers does trait-specific steering actually work?
Two conditions per layer:
  - Matched: inject vector extracted from THAT layer at THAT layer
  - Transfer: inject mid-layer vector at THAT layer

This maps the "personality specificity landscape" across all 36 layers.
Uses completion-style prompts (optimal for SmolLM3-3B).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="layer-sweep-pairwise")

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


def pairwise_logprob(model, tokenizer, device, desc_a, desc_b):
    prompt = f"Which describes you better?\nA) I am {desc_a}\nB) I am {desc_b}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    a_candidates = ["A", " A", "a", " a"]
    b_candidates = ["B", " B", "b", " b"]
    a_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in a_candidates if tokenizer.encode(t, add_special_tokens=False))
    b_lp = max(log_probs[tokenizer.encode(t, add_special_tokens=False)[0]].item()
               for t in b_candidates if tokenizer.encode(t, add_special_tokens=False))
    return a_lp - b_lp


def eval_at_layer(model, tokenizer, device, blocks, inject_layer, vectors, alpha, baseline):
    """Evaluate pairwise discrimination with vectors injected at a specific layer."""
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

        hook_handle = blocks[inject_layer].register_forward_hook(make_hook(delta_vec))
        try:
            for i, trait_a in enumerate(TRAITS):
                for j, trait_b in enumerate(TRAITS):
                    if i >= j:
                        continue
                    if steer_trait not in (trait_a, trait_b):
                        continue
                    gap = pairwise_logprob(model, tokenizer, device,
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
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 2.0  # Use alpha=2 for clear signal

    config = AutoConfig.from_pretrained(model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load ALL layer vectors
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Compute mid-layer residual vectors (reference)
    V_mid = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _, _, Vt_mid = np.linalg.svd(V_mid, full_matrices=False)
    shared_mid = Vt_mid[0]
    shared_mid = shared_mid / np.linalg.norm(shared_mid)

    mid_residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_mid) * shared_mid
        mid_residual[t] = vec - proj

    # Load model
    logger.info("Loading model: %s (%d layers)", model_id, num_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Compute baseline (no steering)
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device,
                                  TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    print(f"\n{'='*70}")
    print(f"LAYER SWEEP: Pairwise Discrimination")
    print(f"Model: {model_id} ({num_layers} layers)")
    print(f"Alpha: {alpha}")
    print(f"{'='*70}")

    # Test every 2nd layer for speed (and a few key layers exactly)
    test_layers = sorted(set(
        list(range(0, num_layers, 2)) +  # Every 2nd layer
        [mid_layer - 1, mid_layer, mid_layer + 1, mid_layer + 2] +  # Around mid
        [num_layers - 1]  # Last layer
    ))

    results = {
        "baseline": baseline,
        "model_id": model_id,
        "num_layers": num_layers,
        "mid_layer": mid_layer,
        "alpha": alpha,
        "matched": {},
        "transfer": {},
    }

    print(f"\n  {'Layer':>5} {'Matched%':>9} {'Transfer%':>10} {'M-MeanΔ':>8} {'T-MeanΔ':>8}")
    print(f"  {'-'*44}")

    for layer in test_layers:
        # Condition 1: Matched — extract from this layer, inject at this layer
        layer_idx = layer + 1  # +1 because index 0 is embedding, index 1 is layer 0 output
        if layer_idx < all_layer_vectors[TRAITS[0]].shape[0]:
            V_layer = np.stack([all_layer_vectors[t][layer_idx] for t in TRAITS])
            _, _, Vt_l = np.linalg.svd(V_layer, full_matrices=False)
            shared_l = Vt_l[0]
            shared_l = shared_l / np.linalg.norm(shared_l)
            matched_vecs = {}
            for t in TRAITS:
                vec = all_layer_vectors[t][layer_idx]
                proj = np.dot(vec, shared_l) * shared_l
                matched_vecs[t] = vec - proj

            r_matched = eval_at_layer(model, tokenizer, device, blocks, layer,
                                      matched_vecs, alpha, baseline)
        else:
            r_matched = {"delta_accuracy": 0, "mean_delta": 0, "correct": 0, "total": 0}

        # Condition 2: Transfer — mid-layer vector, inject at this layer
        r_transfer = eval_at_layer(model, tokenizer, device, blocks, layer,
                                   mid_residual, alpha, baseline)

        results["matched"][str(layer)] = r_matched
        results["transfer"][str(layer)] = r_transfer

        marker = " <-- mid" if layer == mid_layer else ""
        print(f"  L{layer:>3}  {r_matched['delta_accuracy']:>7.0%}   {r_transfer['delta_accuracy']:>7.0%}"
              f"    {r_matched['mean_delta']:>+7.3f} {r_transfer['mean_delta']:>+7.3f}{marker}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    best_matched_layer = max(results["matched"], key=lambda k: results["matched"][k]["delta_accuracy"])
    best_transfer_layer = max(results["transfer"], key=lambda k: results["transfer"][k]["delta_accuracy"])

    print(f"  Best matched layer:  L{best_matched_layer} ({results['matched'][best_matched_layer]['delta_accuracy']:.0%})")
    print(f"  Best transfer layer: L{best_transfer_layer} ({results['transfer'][best_transfer_layer]['delta_accuracy']:.0%})")
    print(f"  Mid layer (L{mid_layer}):     Matched={results['matched'].get(str(mid_layer), {}).get('delta_accuracy', 'N/A'):.0%}"
          f"  Transfer={results['transfer'].get(str(mid_layer), {}).get('delta_accuracy', 'N/A'):.0%}")

    # Count layers with 100% accuracy
    perfect_matched = [k for k, v in results["matched"].items() if v["delta_accuracy"] >= 1.0]
    perfect_transfer = [k for k, v in results["transfer"].items() if v["delta_accuracy"] >= 1.0]
    print(f"  Layers with 100% matched:  {len(perfect_matched)}/{len(test_layers)} ({', '.join(f'L{x}' for x in sorted(perfect_matched, key=int))})")
    print(f"  Layers with 100% transfer: {len(perfect_transfer)}/{len(test_layers)} ({', '.join(f'L{x}' for x in sorted(perfect_transfer, key=int))})")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layer_sweep_pairwise_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
