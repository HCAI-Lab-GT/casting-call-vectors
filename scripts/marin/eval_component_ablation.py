#!/usr/bin/env python
"""
Component ablation: Which of the 5 principal components matters most
for each personality trait?

For each of the 5 basis vectors in the personality subspace:
1. Remove that component from the residual vectors (set its coordinate to 0)
2. Test discrimination with the ablated vectors
3. Compare to full 5-component vectors

This reveals which components encode which traits, providing semantic
interpretation of the 5D personality basis.

Also tests: removing 2 components at a time to find critical pairs.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="component-ablation")

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


def get_5d_decomposition(residual_vectors):
    """Get 5D basis and coordinates with explained variance per component."""
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]

    # Singular values = importance of each component
    total_var = np.sum(S[:5]**2)
    component_importance = [(S[i]**2 / total_var) for i in range(5)]

    coords_5d = {}
    for t in TRAITS:
        coords_5d[t] = basis_5d @ residual_vectors[t]

    return basis_5d, coords_5d, S[:5], component_importance


def ablate_component(coords_5d, basis_5d, components_to_remove):
    """Remove specified components and reconstruct in full dim."""
    ablated = {}
    for t in TRAITS:
        coord = coords_5d[t].copy()
        for c in components_to_remove:
            coord[c] = 0.0
        ablated[t] = (basis_5d.T @ coord).astype(np.float32)
    return ablated


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


def eval_per_trait_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    """Return per-trait accuracy (not overall)."""
    per_trait = {}
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
        correct = 0
        total = 0
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
        finally:
            hook_handle.remove()
        per_trait[steer_trait] = correct / total if total else 0

    overall = sum(per_trait[t] * 5 for t in TRAITS) / 30  # weighted by pairs per trait
    return per_trait, overall


def main():
    target_id = "marin-community/marin-8b-instruct"
    device = "cuda:0"
    alpha = 1.0

    riasec_dir = _repo_root() / "persona_data/model_inits"

    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    basis_5d, coords_5d, singular_values, importance = get_5d_decomposition(residual)

    print(f"\n{'='*70}")
    print(f"COMPONENT ABLATION: What do the 5 principal components encode?")
    print(f"Model: {target_id}")
    print(f"{'='*70}")

    # Show 5D coordinates
    print(f"\n--- 5D coordinates per trait ---")
    print(f"  {'Trait':>14}  {'PC1':>7}  {'PC2':>7}  {'PC3':>7}  {'PC4':>7}  {'PC5':>7}")
    for t in TRAITS:
        c = coords_5d[t]
        print(f"  {t:>14}  {c[0]:>+6.3f}  {c[1]:>+6.3f}  {c[2]:>+6.3f}  {c[3]:>+6.3f}  {c[4]:>+6.3f}")

    print(f"\n--- Singular values and importance ---")
    for i in range(5):
        print(f"  PC{i+1}: σ={singular_values[i]:.4f}, var={importance[i]:.1%}")

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Baseline
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    # Full 5D (positive control)
    full_per_trait, full_overall = eval_per_trait_discrimination(
        model, tokenizer, device, blocks, mid_layer, residual, alpha, baseline)

    print(f"\n--- Full 5-component: {full_overall:.0%} ---")
    for t in TRAITS:
        print(f"  {t:>14}: {full_per_trait[t]:.0%}")

    # Single-component ablation
    print(f"\n--- Remove 1 component at a time ---")
    print(f"  {'Removed':>8}  {'Overall':>7}  {'Drop':>5}  ", end="")
    for t in TRAITS:
        print(f" {t[:4]:>4}", end="")
    print()
    print(f"  {'-'*65}")

    ablation_results = {}
    for remove_pc in range(5):
        ablated_vecs = ablate_component(coords_5d, basis_5d, [remove_pc])
        per_trait, overall = eval_per_trait_discrimination(
            model, tokenizer, device, blocks, mid_layer, ablated_vecs, alpha, baseline)

        drop = full_overall - overall
        print(f"  PC{remove_pc+1:>5}  {overall:>6.0%}  {drop:>+4.0%}  ", end="")
        for t in TRAITS:
            marker = "!" if per_trait[t] < full_per_trait[t] - 0.15 else " "
            print(f" {per_trait[t]:>3.0%}{marker}", end="")
        print()

        ablation_results[f"remove_PC{remove_pc+1}"] = {
            "overall": float(overall),
            "drop": float(drop),
            "per_trait": {t: float(per_trait[t]) for t in TRAITS},
        }

    # Which component is most important for which trait?
    print(f"\n--- Most important component per trait ---")
    for t in TRAITS:
        worst_drop = 0
        worst_pc = -1
        for remove_pc in range(5):
            drop = full_per_trait[t] - ablation_results[f"remove_PC{remove_pc+1}"]["per_trait"][t]
            if drop > worst_drop:
                worst_drop = drop
                worst_pc = remove_pc
        if worst_pc >= 0:
            print(f"  {t:>14}: PC{worst_pc+1} (removing it drops {worst_drop:+.0%})")
        else:
            print(f"  {t:>14}: no single component is critical")

    # Save results
    results = {
        "model": target_id,
        "alpha": alpha,
        "singular_values": [float(s) for s in singular_values],
        "importance": [float(imp) for imp in importance],
        "coords_5d": {t: coords_5d[t].tolist() for t in TRAITS},
        "full": {"overall": float(full_overall), "per_trait": full_per_trait},
        "ablations": ablation_results,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "component_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
