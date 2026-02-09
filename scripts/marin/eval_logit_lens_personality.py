#!/usr/bin/env python
"""
Logit Lens + Personality: How personality steering changes token predictions.

At each layer, project the hidden state through the unembedding matrix to get
logits over the vocabulary. Compare steered vs unsteered to find:
1. Which tokens are most upweighted/downweighted by personality steering
2. At which layer do personality-specific tokens first emerge
3. Whether different traits shift token predictions in interpretable ways
4. KL divergence between steered and unsteered logit distributions per layer
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="logit-lens")

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


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    ln_final = get_ln_final(model)
    lm_head = get_lm_head(model)
    num_layers = len(blocks)

    detect_prompt = "Tell me about yourself."
    alpha = 2.0
    results = {}

    print(f"\n{'='*70}")
    print("LOGIT LENS: PERSONALITY IN TOKEN PREDICTIONS")
    print(f"Model: Marin 8B, {num_layers} layers")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Per-layer KL divergence (steered vs unsteered)
    # ================================================================
    logger.info("Part 1: Per-layer KL divergence...")
    print(f"\n{'='*70}")
    print("PART 1: KL DIVERGENCE PER LAYER (steered vs unsteered)")
    print(f"{'='*70}")

    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Capture hidden states at every layer for baseline and steered
    def capture_all_layers(steer_vec=None, steer_alpha=0.0):
        layer_hidden = {}
        hooks = []

        for lidx in range(num_layers):
            def make_hook(l):
                def hook_fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    layer_hidden[l] = hs[0, -1, :].detach().clone()
                    return out
                return hook_fn
            hooks.append(blocks[lidx].register_forward_hook(make_hook(lidx)))

        if steer_vec is not None and steer_alpha != 0:
            delta = steer_alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
            def steer_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += delta
                    return (hs,) + out[1:]
                out[:, -1, :] += delta
                return out
            hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        return layer_hidden

    # Baseline hidden states
    baseline_hidden = capture_all_layers()

    # Per-trait KL divergences
    kl_results = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        steered_hidden = capture_all_layers(steer_vec=vec, steer_alpha=alpha)

        kl_per_layer = []
        for lidx in range(num_layers):
            # Apply final layer norm and lm_head to get logits
            with torch.no_grad():
                base_logits = lm_head(ln_final(baseline_hidden[lidx].unsqueeze(0)))
                steer_logits = lm_head(ln_final(steered_hidden[lidx].unsqueeze(0)))

            base_probs = torch.softmax(base_logits[0], dim=-1).float()
            steer_probs = torch.softmax(steer_logits[0], dim=-1).float()

            # KL(steered || baseline)
            kl = torch.sum(steer_probs * (torch.log(steer_probs + 1e-10) - torch.log(base_probs + 1e-10))).item()
            kl_per_layer.append(kl)

        kl_results[trait] = kl_per_layer
        print(f"\n  {trait} α={alpha}:")
        for lidx in [0, mid_layer - 1, mid_layer, mid_layer + 1, mid_layer + 2, num_layers - 1]:
            if lidx < num_layers:
                marker = " ← injection" if lidx == mid_layer else ""
                print(f"    L{lidx}: KL={kl_per_layer[lidx]:.4f}{marker}")

    results["kl_divergence"] = {t: [float(x) for x in kl_results[t]] for t in TRAITS}

    # ================================================================
    # PART 2: Top shifted tokens per trait
    # ================================================================
    logger.info("Part 2: Top shifted tokens...")
    print(f"\n{'='*70}")
    print("PART 2: TOP SHIFTED TOKENS AT L17 (one above injection)")
    print(f"{'='*70}")

    detect_layer = mid_layer + 1
    top_k = 20
    token_shift_results = {}

    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        steered_hidden = capture_all_layers(steer_vec=vec, steer_alpha=alpha)

        with torch.no_grad():
            base_logits = lm_head(ln_final(baseline_hidden[detect_layer].unsqueeze(0)))[0]
            steer_logits = lm_head(ln_final(steered_hidden[detect_layer].unsqueeze(0)))[0]

        # Log-probability difference
        base_lp = torch.log_softmax(base_logits, dim=-1).float()
        steer_lp = torch.log_softmax(steer_logits, dim=-1).float()
        delta_lp = (steer_lp - base_lp)

        # Top upweighted
        top_up_idx = torch.topk(delta_lp, top_k).indices
        top_up = [(tokenizer.decode([idx.item()]).strip(), delta_lp[idx].item()) for idx in top_up_idx]

        # Top downweighted
        top_down_idx = torch.topk(-delta_lp, top_k).indices
        top_down = [(tokenizer.decode([idx.item()]).strip(), delta_lp[idx].item()) for idx in top_down_idx]

        print(f"\n  {trait} (top upweighted tokens):")
        for tok, delta in top_up[:10]:
            print(f"    +{delta:.3f}: '{tok}'")
        print(f"  {trait} (top downweighted tokens):")
        for tok, delta in top_down[:10]:
            print(f"    {delta:.3f}: '{tok}'")

        token_shift_results[trait] = {
            "upweighted": [(t, float(d)) for t, d in top_up],
            "downweighted": [(t, float(d)) for t, d in top_down],
        }

    results["token_shifts"] = token_shift_results

    # ================================================================
    # PART 3: KL divergence onset layer (first layer where KL > threshold)
    # ================================================================
    logger.info("Part 3: KL onset analysis...")
    print(f"\n{'='*70}")
    print("PART 3: PERSONALITY ONSET IN TOKEN SPACE")
    print(f"{'='*70}")

    onset_results = {}
    for trait in TRAITS:
        kl_vals = kl_results[trait]
        max_kl = max(kl_vals)
        # Find first layer where KL > 10% of max
        onset_10 = next((i for i, kl in enumerate(kl_vals) if kl > 0.1 * max_kl), -1)
        # Find first layer where KL > 50% of max
        onset_50 = next((i for i, kl in enumerate(kl_vals) if kl > 0.5 * max_kl), -1)

        print(f"  {trait}: max KL={max_kl:.4f} at L{kl_vals.index(max_kl)}, "
              f"onset(10%)=L{onset_10}, onset(50%)=L{onset_50}")
        onset_results[trait] = {
            "max_kl": float(max_kl),
            "max_layer": int(kl_vals.index(max_kl)),
            "onset_10pct": onset_10,
            "onset_50pct": onset_50,
        }

    results["kl_onset"] = onset_results

    # ================================================================
    # PART 4: Cross-trait token overlap (do different traits shift similar tokens?)
    # ================================================================
    logger.info("Part 4: Cross-trait token overlap...")
    print(f"\n{'='*70}")
    print("PART 4: CROSS-TRAIT TOKEN SET OVERLAP")
    print(f"{'='*70}")

    # Get top-100 shifted tokens per trait at L17
    top_n = 100
    trait_token_sets = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        steered_hidden = capture_all_layers(steer_vec=vec, steer_alpha=alpha)

        with torch.no_grad():
            base_logits = lm_head(ln_final(baseline_hidden[detect_layer].unsqueeze(0)))[0]
            steer_logits = lm_head(ln_final(steered_hidden[detect_layer].unsqueeze(0)))[0]

        delta_lp = torch.log_softmax(steer_logits.float(), dim=-1) - torch.log_softmax(base_logits.float(), dim=-1)
        # Top shifted (by absolute value)
        top_idx = torch.topk(delta_lp.abs(), top_n).indices
        trait_token_sets[trait] = set(top_idx.tolist())

    # Pairwise Jaccard
    overlap_matrix = {}
    print(f"\n  Jaccard overlap of top-{top_n} shifted tokens:")
    print(f"  {'':>16}", end="")
    for t in TRAITS:
        print(f"  {t[:5]:>5}", end="")
    print()

    for t1 in TRAITS:
        print(f"  {t1:>16}", end="")
        overlap_matrix[t1] = {}
        for t2 in TRAITS:
            inter = len(trait_token_sets[t1] & trait_token_sets[t2])
            union = len(trait_token_sets[t1] | trait_token_sets[t2])
            jaccard = inter / union if union > 0 else 0
            overlap_matrix[t1][t2] = float(jaccard)
            print(f"  {jaccard:.3f}", end="")
        print()

    results["token_overlap"] = overlap_matrix

    # ================================================================
    # PART 5: Personality token specificity — unique tokens per trait
    # ================================================================
    logger.info("Part 5: Trait-specific tokens...")
    print(f"\n{'='*70}")
    print("PART 5: TRAIT-UNIQUE TOKENS (in top-100 but NOT in any other trait's top-100)")
    print(f"{'='*70}")

    specificity_results = {}
    all_other_tokens = {}
    for t in TRAITS:
        others = set()
        for t2 in TRAITS:
            if t2 != t:
                others |= trait_token_sets[t2]
        all_other_tokens[t] = others

    for trait in TRAITS:
        unique_to_trait = trait_token_sets[trait] - all_other_tokens[trait]
        unique_tokens = [tokenizer.decode([idx]).strip() for idx in list(unique_to_trait)[:15]]
        print(f"  {trait}: {len(unique_to_trait)} unique tokens: {unique_tokens[:10]}")
        specificity_results[trait] = {
            "n_unique": len(unique_to_trait),
            "examples": unique_tokens,
        }

    results["trait_unique_tokens"] = specificity_results

    # ================================================================
    # PART 6: Alpha dose-response in token space (KL at L17 vs alpha)
    # ================================================================
    logger.info("Part 6: KL dose-response...")
    print(f"\n{'='*70}")
    print("PART 6: KL DOSE-RESPONSE AT L17")
    print(f"{'='*70}")

    dose_results = {}
    for trait in ["artistic", "social"]:
        vec = residual[trait].astype(np.float32)
        dose_kl = {}
        for test_alpha in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
            steered_hidden = capture_all_layers(steer_vec=vec, steer_alpha=test_alpha)
            with torch.no_grad():
                base_logits = lm_head(ln_final(baseline_hidden[detect_layer].unsqueeze(0)))[0]
                steer_logits = lm_head(ln_final(steered_hidden[detect_layer].unsqueeze(0)))[0]

            base_probs = torch.softmax(base_logits, dim=-1).float()
            steer_probs = torch.softmax(steer_logits, dim=-1).float()
            kl = torch.sum(steer_probs * (torch.log(steer_probs + 1e-10) - torch.log(base_probs + 1e-10))).item()
            dose_kl[str(test_alpha)] = float(kl)

        print(f"\n  {trait}:")
        for a_str, kl_val in dose_kl.items():
            print(f"    α={a_str}: KL={kl_val:.4f}")

        dose_results[trait] = dose_kl

    results["kl_dose_response"] = dose_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for trait in TRAITS:
        kl_vals = kl_results[trait]
        print(f"  {trait}: max_KL={max(kl_vals):.4f} at L{kl_vals.index(max(kl_vals))}, "
              f"onset(10%)=L{onset_results[trait]['onset_10pct']}, "
              f"n_unique_tokens={specificity_results[trait]['n_unique']}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logit_lens_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
