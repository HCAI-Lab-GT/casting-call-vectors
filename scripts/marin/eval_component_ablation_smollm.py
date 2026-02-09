#!/usr/bin/env python
"""
Component ablation on SmolLM3-3B: Test whether the SAME principal components
encode the SAME trait contrasts as in Marin 8B.

If PC1 = Artistic↔Conventional in BOTH models, the 5D personality basis
has universal semantic structure, not model-specific encoding.

Compare directly to Marin 8B results (component_ablation.json).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="ablation-smollm")

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
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    total_var = np.sum(S[:5]**2)
    component_importance = [(S[i]**2 / total_var) for i in range(5)]
    coords_5d = {}
    for t in TRAITS:
        coords_5d[t] = basis_5d @ residual_vectors[t]
    return basis_5d, coords_5d, S[:5], component_importance


def ablate_component(coords_5d, basis_5d, components_to_remove):
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
    per_trait = {}
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

    overall = sum(per_trait[t] * 5 for t in TRAITS) / 30
    return per_trait, overall


def main():
    target_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 1.0

    riasec_dir = _repo_root() / "persona_data/model_inits"

    logger.info("Loading vectors for SmolLM3...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    basis_5d, coords_5d, singular_values, importance = get_5d_decomposition(residual)

    print(f"\n{'='*70}")
    print(f"COMPONENT ABLATION: SmolLM3-3B vs Marin 8B")
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
        print(f"  PC{i+1}: sigma={singular_values[i]:.4f}, var={importance[i]:.1%}")

    # Compare to Marin 8B coordinates
    try:
        with open(_repo_root() / "outputs/analysis/component_ablation.json") as f:
            marin_data = json.load(f)
        print(f"\n--- Coordinate comparison: SmolLM3 vs Marin 8B ---")
        print(f"  {'Trait':>14}  {'SmolLM3 PC1':>10}  {'Marin PC1':>10}  {'Same sign?':>10}")
        for t in TRAITS:
            s_pc1 = coords_5d[t][0]
            m_pc1 = marin_data["coords_5d"][t][0]
            same = "YES" if (s_pc1 > 0) == (m_pc1 > 0) else "NO"
            print(f"  {t:>14}  {s_pc1:>+9.1f}  {m_pc1:>+9.1f}  {same:>10}")
    except Exception:
        pass

    # Load model
    logger.info("Loading SmolLM3 model...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.bfloat16, device_map=device)
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

    # Full 5D (positive control)
    logger.info("Testing full 5-component vectors...")
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
        logger.info(f"Ablating PC{remove_pc+1}...")
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
    trait_critical_pc = {}
    for t in TRAITS:
        worst_drop = 0
        worst_pc = -1
        for remove_pc in range(5):
            drop = full_per_trait[t] - ablation_results[f"remove_PC{remove_pc+1}"]["per_trait"][t]
            if drop > worst_drop:
                worst_drop = drop
                worst_pc = remove_pc
        trait_critical_pc[t] = worst_pc
        if worst_pc >= 0:
            print(f"  {t:>14}: PC{worst_pc+1} (removing it drops {worst_drop:+.0%})")
        else:
            print(f"  {t:>14}: no single component is critical")

    # Cross-model comparison
    print(f"\n--- Cross-model PC correspondence ---")
    try:
        with open(_repo_root() / "outputs/analysis/component_ablation.json") as f:
            marin_data = json.load(f)

        print(f"  {'Trait':>14}  {'SmolLM3 critical':>16}  {'Marin critical':>14}  {'Match?':>6}")
        marin_critical = {}
        for t in TRAITS:
            worst_drop = 0
            worst_pc = -1
            for i in range(5):
                drop = marin_data["full"]["per_trait"][t] - marin_data["ablations"][f"remove_PC{i+1}"]["per_trait"][t]
                if drop > worst_drop:
                    worst_drop = drop
                    worst_pc = i
            marin_critical[t] = worst_pc

        for t in TRAITS:
            s_pc = f"PC{trait_critical_pc[t]+1}" if trait_critical_pc[t] >= 0 else "none"
            m_pc = f"PC{marin_critical[t]+1}" if marin_critical[t] >= 0 else "none"
            match = "YES" if trait_critical_pc[t] == marin_critical[t] else "NO"
            print(f"  {t:>14}  {s_pc:>16}  {m_pc:>14}  {match:>6}")

        # More nuanced: compare WHICH TRAITS are affected by removing each PC
        print(f"\n--- What each PC encodes (by traits most affected) ---")
        for pc in range(5):
            s_drops = {t: full_per_trait[t] - ablation_results[f"remove_PC{pc+1}"]["per_trait"][t] for t in TRAITS}
            m_drops = {t: marin_data["full"]["per_trait"][t] - marin_data["ablations"][f"remove_PC{pc+1}"]["per_trait"][t] for t in TRAITS}

            s_affected = sorted(s_drops.items(), key=lambda x: x[1], reverse=True)[:2]
            m_affected = sorted(m_drops.items(), key=lambda x: x[1], reverse=True)[:2]

            s_str = ", ".join(f"{t}({d:+.0%})" for t, d in s_affected if d > 0.05)
            m_str = ", ".join(f"{t}({d:+.0%})" for t, d in m_affected if d > 0.05)

            print(f"  PC{pc+1}: SmolLM3=[{s_str or 'none'}], Marin=[{m_str or 'none'}]")

    except Exception as e:
        print(f"  Could not compare to Marin: {e}")

    # Save results
    results = {
        "model": target_id,
        "alpha": alpha,
        "singular_values": [float(s) for s in singular_values],
        "importance": [float(imp) for imp in importance],
        "coords_5d": {t: coords_5d[t].tolist() for t in TRAITS},
        "full": {"overall": float(full_overall), "per_trait": {t: float(v) for t, v in full_per_trait.items()}},
        "ablations": ablation_results,
        "trait_critical_pc": {t: int(v) if v >= 0 else None for t, v in trait_critical_pc.items()},
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "component_ablation_smollm.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
