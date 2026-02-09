#!/usr/bin/env python
"""
Multi-layer personality steering: does injecting at multiple layers help?

Current approach: single mid-layer injection at L16 (Marin 8B, 32 layers).
Question: what if we inject residual vectors at MULTIPLE layers simultaneously?

Tests:
1. Single layer (baseline): L16 only
2. Triple mid-cluster: L14, L16, L18 (mid-layer ± 2)
3. Distributed: L8, L16, L24 (early/mid/late)
4. All-layer: inject at every layer using that layer's vector
5. Optimal subset: test which layer combinations are best

Each condition uses the SAME total injection magnitude (divided equally among layers)
to control for the amount of perturbation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="multi-layer")

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


def load_all_layer_vectors(model_id, riasec_dir):
    """Load residual vectors for ALL layers."""
    safe_model = model_id.replace("/", "__")
    config = AutoConfig.from_pretrained(model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Compute residual at each layer
    residual_by_layer = {}
    for layer_idx in range(vecs.shape[0]):
        V = np.stack([all_layer_vectors[t][layer_idx] for t in TRAITS])
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
        shared_dir = Vt[0]
        shared_dir = shared_dir / np.linalg.norm(shared_dir)
        residual = {}
        for t in TRAITS:
            vec = all_layer_vectors[t][layer_idx]
            proj = np.dot(vec, shared_dir) * shared_dir
            residual[t] = vec - proj
        residual_by_layer[layer_idx] = residual

    return residual_by_layer, mid_layer, num_layers


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


def eval_accuracy_multi_layer(model, tokenizer, device, blocks, layer_vec_pairs, baseline):
    """Evaluate with hooks at multiple layers simultaneously."""
    correct = 0
    total = 0
    total_delta = 0.0

    for steer_trait in TRAITS:
        # Set up hooks for all layers
        hooks = []
        for layer_idx, vectors, layer_alpha in layer_vec_pairs:
            vec = vectors[steer_trait]
            vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
            delta_vec = layer_alpha * vec_t

            def make_hook(d):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += d
                        return (hs,) + out[1:]
                    out[:, -1, :] += d
                    return out
                return hook_fn

            h = blocks[layer_idx].register_forward_hook(make_hook(delta_vec))
            hooks.append(h)

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
            for h in hooks:
                h.remove()

    return correct / total if total else 0, total_delta / total if total else 0


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    # Total alpha budget to distribute
    total_alpha = 1.0

    # Load vectors
    logger.info("Loading all-layer vectors...")
    residual_by_layer, mid_layer, num_layers = load_all_layer_vectors(target_id, riasec_dir)

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

    print(f"\n{'='*70}")
    print(f"MULTI-LAYER STEERING (Marin 8B, {num_layers} layers)")
    print(f"Total alpha budget = {total_alpha}")
    print(f"Mid-layer = L{mid_layer}")
    print(f"{'='*70}")

    # Define layer conditions
    # Note: residual_by_layer uses index from 0 to num_layers (there's a +1 offset in some code)
    # The vectors are stored at layer+1 indices (layer 0 = embedding output, layer 1 = after block 0, etc.)
    # For injection, we use block indices directly: blocks[i] = i-th decoder block

    conditions = {
        "single_mid": {
            "description": f"Single layer: L{mid_layer} only",
            "layers": [mid_layer],
        },
        "triple_tight": {
            "description": f"Triple tight: L{mid_layer-2}, L{mid_layer}, L{mid_layer+2}",
            "layers": [mid_layer - 2, mid_layer, mid_layer + 2],
        },
        "triple_spread": {
            "description": f"Triple spread: L{mid_layer-4}, L{mid_layer}, L{mid_layer+4}",
            "layers": [mid_layer - 4, mid_layer, mid_layer + 4],
        },
        "five_layers": {
            "description": f"Five layers: L{mid_layer-4} to L{mid_layer+4} step 2",
            "layers": [mid_layer - 4, mid_layer - 2, mid_layer, mid_layer + 2, mid_layer + 4],
        },
        "distributed": {
            "description": f"Distributed: L{num_layers//4}, L{mid_layer}, L{3*num_layers//4}",
            "layers": [num_layers // 4, mid_layer, 3 * num_layers // 4],
        },
        "early_only": {
            "description": f"Early only: L{num_layers//4}",
            "layers": [num_layers // 4],
        },
        "late_only": {
            "description": f"Late only: L{3*num_layers//4}",
            "layers": [3 * num_layers // 4],
        },
    }

    results = {}

    for cond_name, cond in conditions.items():
        layers = cond["layers"]
        n_layers = len(layers)
        per_layer_alpha = total_alpha / n_layers

        logger.info(f"Testing {cond_name}: {cond['description']} (α={per_layer_alpha:.3f} per layer)...")

        # Build layer-vector pairs
        # Use layer+1 for the vector index (vectors[0] = embedding output, vectors[1] = after block 0)
        layer_vec_pairs = []
        for layer_idx in layers:
            vec_idx = layer_idx + 1  # vector index
            if vec_idx in residual_by_layer:
                layer_vec_pairs.append((layer_idx, residual_by_layer[vec_idx], per_layer_alpha))
            else:
                logger.warning(f"Skipping layer {layer_idx}, vector index {vec_idx} not available")

        acc, delta = eval_accuracy_multi_layer(
            model, tokenizer, device, blocks, layer_vec_pairs, baseline)

        print(f"\n  {cond['description']}:")
        print(f"    Accuracy: {acc:.0%}, Mean delta: {delta:+.3f}")

        results[cond_name] = {
            "description": cond["description"],
            "layers": layers,
            "n_layers": n_layers,
            "per_layer_alpha": float(per_layer_alpha),
            "accuracy": float(acc),
            "mean_delta": float(delta),
        }

    # Also test with MATCHED total alpha (each layer gets full alpha)
    print(f"\n{'='*70}")
    print(f"ADDITIVE ALPHA (each layer gets α={total_alpha})")
    print(f"{'='*70}")

    additive_conditions = {
        "additive_single": {
            "description": f"Single L{mid_layer} at α={total_alpha}",
            "layers": [mid_layer],
        },
        "additive_triple": {
            "description": f"Triple L{mid_layer-2},L{mid_layer},L{mid_layer+2} each at α={total_alpha}",
            "layers": [mid_layer - 2, mid_layer, mid_layer + 2],
        },
        "additive_five": {
            "description": f"Five layers each at α={total_alpha}",
            "layers": [mid_layer - 4, mid_layer - 2, mid_layer, mid_layer + 2, mid_layer + 4],
        },
    }

    for cond_name, cond in additive_conditions.items():
        layers = cond["layers"]
        logger.info(f"Testing additive {cond_name}...")

        layer_vec_pairs = []
        for layer_idx in layers:
            vec_idx = layer_idx + 1
            if vec_idx in residual_by_layer:
                layer_vec_pairs.append((layer_idx, residual_by_layer[vec_idx], total_alpha))

        acc, delta = eval_accuracy_multi_layer(
            model, tokenizer, device, blocks, layer_vec_pairs, baseline)

        print(f"\n  {cond['description']}:")
        print(f"    Accuracy: {acc:.0%}, Mean delta: {delta:+.3f}")

        results[cond_name] = {
            "description": cond["description"],
            "layers": layers,
            "n_layers": len(layers),
            "per_layer_alpha": float(total_alpha),
            "total_alpha": float(total_alpha * len(layers)),
            "accuracy": float(acc),
            "mean_delta": float(delta),
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Condition':>25}  {'Layers':>8}  {'Per-α':>6}  {'Total-α':>7}  {'Acc':>5}  {'Delta':>7}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        total_a = r.get("total_alpha", r["per_layer_alpha"] * r["n_layers"])
        print(f"  {name:>25}  {r['n_layers']:>8}  {r['per_layer_alpha']:>6.3f}  {total_a:>7.2f}  {r['accuracy']:>4.0%}  {r['mean_delta']:>+7.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multi_layer_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
