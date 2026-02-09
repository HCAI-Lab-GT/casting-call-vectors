#!/usr/bin/env python
"""
Base-to-instruct cross-dimensional transfer:
SmolLM3-Base(2048d) → Marin-8B-Instruct(4096d).

Tests whether persona vectors extracted from a BASE model (no instruction tuning)
can steer an INSTRUCT model of different architecture and dimension.

This combines two transfer gaps simultaneously:
1. Format gap: base → instruct (different training objective)
2. Dimension gap: 2048d → 4096d (different hidden dimension)
3. Architecture gap: SmolLM3 → Marin (different model family)

Compare:
- Self (Marin instruct native): positive control
- Instruct→Instruct (SmolLM3 instruct → Marin instruct): established baseline
- Base→Instruct (SmolLM3 base → Marin instruct): the novel test
- Random: negative control
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="base-to-instruct")

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
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer


def get_5d_coordinates(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d


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


def eval_discrimination(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
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
    base_id = "HuggingFaceTB/SmolLM3-3B-Base"       # 2048d, base model
    instruct_id = "HuggingFaceTB/SmolLM3-3B"         # 2048d, instruct model
    target_id = "marin-community/marin-8b-instruct"  # 4096d, instruct model
    device = "cuda:0"
    alpha = 1.0

    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load residual vectors
    logger.info("Loading residual vectors...")
    base_residual, _ = load_residual_vectors(base_id, riasec_dir)
    instruct_residual, _ = load_residual_vectors(instruct_id, riasec_dir)
    target_residual, target_mid = load_residual_vectors(target_id, riasec_dir)

    base_dim = base_residual[TRAITS[0]].shape[0]
    target_dim = target_residual[TRAITS[0]].shape[0]
    logger.info("Base dim: %d, Target dim: %d", base_dim, target_dim)

    # 5D coordinates
    base_5d, _ = get_5d_coordinates(base_residual)
    instruct_5d, _ = get_5d_coordinates(instruct_residual)
    target_5d, target_basis = get_5d_coordinates(target_residual)

    # === Check base vs instruct 5D geometry ===
    print(f"\n{'='*70}")
    print(f"BASE-TO-INSTRUCT CROSS-DIM TRANSFER")
    print(f"SmolLM3-Base({base_dim}d) → Marin-8B-Instruct({target_dim}d)")
    print(f"{'='*70}")

    # How similar are base vs instruct 5D coords?
    print(f"\n--- Base vs Instruct 5D geometry (SmolLM3) ---")
    R_bi, s_bi = fit_procrustes(base_5d, instruct_5d)
    for t in TRAITS:
        base_aligned = s_bi * (R_bi @ base_5d[t])
        cos = np.dot(base_aligned, instruct_5d[t]) / \
              (np.linalg.norm(base_aligned) * np.linalg.norm(instruct_5d[t]))
        print(f"  {t:>14}: cosine(aligned_base, instruct) = {cos:.4f}")

    # === Transfer: base → target ===
    R_bt, s_bt = fit_procrustes(base_5d, target_5d)
    base_transferred = {}
    for t in TRAITS:
        aligned = s_bt * (R_bt @ base_5d[t])
        base_transferred[t] = (target_basis.T @ aligned).astype(np.float32)

    # === Transfer: instruct → target (established baseline) ===
    R_it, s_it = fit_procrustes(instruct_5d, target_5d)
    instruct_transferred = {}
    for t in TRAITS:
        aligned = s_it * (R_it @ instruct_5d[t])
        instruct_transferred[t] = (target_basis.T @ aligned).astype(np.float32)

    # Cosine analysis
    print(f"\n--- Cosine vs native Marin vectors ---")
    print(f"  {'Trait':>14}  {'Base→Marin':>10}  {'Inst→Marin':>10}")
    cos_base = []
    cos_inst = []
    for t in TRAITS:
        cb = np.dot(base_transferred[t], target_residual[t]) / \
             (np.linalg.norm(base_transferred[t]) * np.linalg.norm(target_residual[t]))
        ci = np.dot(instruct_transferred[t], target_residual[t]) / \
             (np.linalg.norm(instruct_transferred[t]) * np.linalg.norm(target_residual[t]))
        cos_base.append(cb)
        cos_inst.append(ci)
        print(f"  {t:>14}  {cb:>10.4f}  {ci:>10.4f}")
    print(f"  {'Mean':>14}  {np.mean(cos_base):>10.4f}  {np.mean(cos_inst):>10.4f}")

    # === Load target model ===
    logger.info("Loading target model: %s", target_id)
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

    # Evaluate
    print(f"\n--- Steering evaluation (α={alpha}) ---")

    r_self = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                target_residual, alpha, baseline)
    print(f"  Self (Marin native):          {r_self['delta_accuracy']:.0%} ({r_self['correct']}/{r_self['total']}), "
          f"Δ={r_self['mean_delta']:+.4f}")

    r_instruct = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                     instruct_transferred, alpha, baseline)
    print(f"  Instruct→Instruct:            {r_instruct['delta_accuracy']:.0%} ({r_instruct['correct']}/{r_instruct['total']}), "
          f"Δ={r_instruct['mean_delta']:+.4f}")

    r_base = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                 base_transferred, alpha, baseline)
    print(f"  Base→Instruct:                {r_base['delta_accuracy']:.0%} ({r_base['correct']}/{r_base['total']}), "
          f"Δ={r_base['mean_delta']:+.4f}")

    # Random control
    np.random.seed(42)
    random_vecs = {}
    for t in TRAITS:
        rv = np.random.randn(target_dim).astype(np.float32)
        rv = rv / np.linalg.norm(rv) * np.linalg.norm(target_residual[t])
        random_vecs[t] = rv
    r_random = eval_discrimination(model, tokenizer, device, blocks, target_mid,
                                   random_vecs, alpha, baseline)
    print(f"  Random:                       {r_random['delta_accuracy']:.0%} ({r_random['correct']}/{r_random['total']}), "
          f"Δ={r_random['mean_delta']:+.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Self (Marin native, 4096d):   {r_self['delta_accuracy']:.0%}")
    print(f"  Instruct→Instruct (2048→4096):{r_instruct['delta_accuracy']:.0%}")
    print(f"  Base→Instruct (2048→4096):    {r_base['delta_accuracy']:.0%}")
    print(f"  Random:                       {r_random['delta_accuracy']:.0%}")

    if r_base['delta_accuracy'] >= r_self['delta_accuracy'] - 0.10:
        print(f"\n  ** BASE MODEL VECTORS CAN STEER INSTRUCT MODELS CROSS-DIM **")
        print(f"     Persona geometry is training-objective invariant!")

    results = {
        "base_source": base_id,
        "instruct_source": instruct_id,
        "target": target_id,
        "alpha": alpha,
        "self": r_self,
        "instruct_transfer": r_instruct,
        "base_transfer": r_base,
        "random": r_random,
        "cosines_vs_native": {
            "base_transfer": {t: float(c) for t, c in zip(TRAITS, cos_base)},
            "instruct_transfer": {t: float(c) for t, c in zip(TRAITS, cos_inst)},
        },
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "base_to_instruct_transfer.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
