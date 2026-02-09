#!/usr/bin/env python
"""
Cross-Model Token Shift Comparison.

For each of 3 models (Llama 1B, SmolLM3 3B, Marin 8B), compute the top
tokens shifted by each personality trait using the logit lens at the
detection layer. Then compare:
1. How much overlap in shifted tokens across models (by token string)?
2. Are the same SEMANTIC categories shifted?
3. Do Holland opposites produce opposite token shifts across models?

This tests whether the universal 5D geometry produces universal token-level effects.
"""

import json
import gc
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="xmodel-tok")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def get_ln_final(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise RuntimeError("Cannot find final layer norm")


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("Cannot find lm_head")


def load_model_data(model_id, riasec_dir):
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
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    _, _, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def get_token_shifts(model, tokenizer, device, blocks, mid_layer, residual, alpha, top_n=100):
    """Get top shifted tokens per trait for a model."""
    ln_final = get_ln_final(model)
    lm_head = get_lm_head(model)
    num_layers = len(blocks)
    detect_layer = mid_layer + 1

    detect_prompt = "Tell me about yourself."
    messages = [{"role": "user", "content": detect_prompt}]

    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = detect_prompt

    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Capture baseline hidden state at detect_layer
    baseline_hidden = {}
    hooks = []

    def make_base_hook(l):
        def fn(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            baseline_hidden[l] = hs[0, -1, :].detach().clone()
            return out
        return fn

    hooks.append(blocks[detect_layer].register_forward_hook(make_base_hook(detect_layer)))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    all_trait_tokens = {}

    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)

        steered_hidden = {}
        hooks = []

        def make_steer_hook(l):
            def fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                steered_hidden[l] = hs[0, -1, :].detach().clone()
                return out
            return fn

        hooks.append(blocks[detect_layer].register_forward_hook(make_steer_hook(detect_layer)))

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        # Compute logit difference
        with torch.no_grad():
            base_logits = lm_head(ln_final(baseline_hidden[detect_layer].unsqueeze(0)))[0]
            steer_logits = lm_head(ln_final(steered_hidden[detect_layer].unsqueeze(0)))[0]

        base_lp = torch.log_softmax(base_logits.float(), dim=-1)
        steer_lp = torch.log_softmax(steer_logits.float(), dim=-1)
        delta_lp = steer_lp - base_lp

        # Top upweighted
        top_up_idx = torch.topk(delta_lp, top_n).indices
        top_up = [(tokenizer.decode([idx.item()]).strip().lower(), float(delta_lp[idx])) for idx in top_up_idx]

        # Top downweighted
        top_down_idx = torch.topk(-delta_lp, top_n).indices
        top_down = [(tokenizer.decode([idx.item()]).strip().lower(), float(delta_lp[idx])) for idx in top_down_idx]

        # Top by absolute value
        top_abs_idx = torch.topk(delta_lp.abs(), top_n).indices
        top_abs = set(tokenizer.decode([idx.item()]).strip().lower() for idx in top_abs_idx)

        all_trait_tokens[trait] = {
            "upweighted": top_up,
            "downweighted": top_down,
            "top_abs_set": top_abs,
        }

    return all_trait_tokens


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    alpha = 2.0
    top_n = 100
    results = {}

    models_to_test = [
        ("meta-llama/Llama-3.2-1B-Instruct", "Llama-1B"),
        ("HuggingFaceTB/SmolLM3-3B", "SmolLM3-3B"),
        ("marin-community/marin-8b-instruct", "Marin-8B"),
    ]

    print(f"\n{'='*70}")
    print("CROSS-MODEL TOKEN SHIFT COMPARISON")
    print(f"{'='*70}")

    all_model_tokens = {}

    for model_id, short_name in models_to_test:
        logger.info(f"Processing {short_name}...")
        print(f"\n  Loading {short_name}...")

        model_data = load_model_data(model_id, riasec_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device)
        model.eval()
        blocks = get_decoder_blocks(model)

        token_shifts = get_token_shifts(
            model, tokenizer, device, blocks, model_data["mid_layer"],
            model_data["residual"], alpha, top_n)

        all_model_tokens[short_name] = token_shifts

        # Print top 5 per trait
        for trait in TRAITS:
            up = token_shifts[trait]["upweighted"][:5]
            print(f"    {trait}: top up = {[t[0] for t in up]}")

        # Unload model to free GPU
        del model
        del blocks
        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # PART 1: Cross-model token overlap per trait
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 1: CROSS-MODEL TOKEN OVERLAP (Jaccard of top-100)")
    print(f"{'='*70}")

    model_names = [n for _, n in models_to_test]
    overlap_results = {}

    for trait in TRAITS:
        print(f"\n  {trait}:")
        trait_overlap = {}
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if j <= i:
                    continue
                s1 = all_model_tokens[m1][trait]["top_abs_set"]
                s2 = all_model_tokens[m2][trait]["top_abs_set"]
                inter = len(s1 & s2)
                union = len(s1 | s2)
                jaccard = inter / union if union > 0 else 0
                print(f"    {m1} ↔ {m2}: Jaccard={jaccard:.3f} ({inter} shared tokens)")
                trait_overlap[f"{m1}_vs_{m2}"] = {
                    "jaccard": float(jaccard),
                    "shared": inter,
                    "shared_tokens": sorted(list(s1 & s2))[:20],
                }
        overlap_results[trait] = trait_overlap

    results["cross_model_overlap"] = overlap_results

    # ================================================================
    # PART 2: Cross-model direction consistency
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: CROSS-MODEL DIRECTION CONSISTENCY")
    print(f"{'='*70}")

    # For tokens that appear in multiple models' top lists, check if they
    # shift in the same direction (up or down)
    direction_results = {}
    for trait in TRAITS:
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if j <= i:
                    continue
                # Get upweighted/downweighted sets
                up1 = set(t[0] for t in all_model_tokens[m1][trait]["upweighted"])
                up2 = set(t[0] for t in all_model_tokens[m2][trait]["upweighted"])
                down1 = set(t[0] for t in all_model_tokens[m1][trait]["downweighted"])
                down2 = set(t[0] for t in all_model_tokens[m2][trait]["downweighted"])

                # Same direction (both up or both down)
                same_up = len(up1 & up2)
                same_down = len(down1 & down2)
                # Opposite direction
                cross_up_down = len(up1 & down2) + len(down1 & up2)

                total_shared = same_up + same_down + cross_up_down
                consistency = (same_up + same_down) / total_shared if total_shared > 0 else 0

                if total_shared > 0:
                    print(f"  {trait} {m1}↔{m2}: {same_up+same_down}/{total_shared} same direction "
                          f"({consistency:.1%})")

                direction_results[f"{trait}_{m1}_vs_{m2}"] = {
                    "same_direction": same_up + same_down,
                    "opposite_direction": cross_up_down,
                    "consistency": float(consistency),
                }

    results["direction_consistency"] = direction_results

    # ================================================================
    # PART 3: Universal personality tokens (shifted in ALL 3 models)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: UNIVERSAL PERSONALITY TOKENS (in all 3 models)")
    print(f"{'='*70}")

    universal_results = {}
    for trait in TRAITS:
        sets = [all_model_tokens[m][trait]["top_abs_set"] for m in model_names]
        universal = sets[0] & sets[1] & sets[2]
        print(f"  {trait}: {len(universal)} universal tokens: {sorted(list(universal))[:15]}")
        universal_results[trait] = {
            "count": len(universal),
            "tokens": sorted(list(universal)),
        }

    results["universal_tokens"] = universal_results

    # ================================================================
    # PART 4: Model-specific tokens (in one model but not others)
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: MODEL-SPECIFIC TOKENS (unique to each model)")
    print(f"{'='*70}")

    specific_results = {}
    for trait in ["artistic", "conventional"]:
        for m_idx, m_name in enumerate(model_names):
            other_sets = [all_model_tokens[m][trait]["top_abs_set"]
                         for k, m in enumerate(model_names) if k != m_idx]
            all_others = other_sets[0] | other_sets[1]
            unique = all_model_tokens[m_name][trait]["top_abs_set"] - all_others
            print(f"  {trait} unique to {m_name}: {len(unique)} — {sorted(list(unique))[:10]}")
            specific_results[f"{trait}_{m_name}"] = {
                "count": len(unique),
                "tokens": sorted(list(unique))[:20],
            }

    results["model_specific_tokens"] = specific_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for trait in TRAITS:
        universals = universal_results[trait]["count"]
        print(f"  {trait}: {universals} universal tokens across all 3 models")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_model_token_shifts.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
