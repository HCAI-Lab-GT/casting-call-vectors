#!/usr/bin/env python
"""
Token-Space Arithmetic: Is personality steering linear in token space?

Tests:
1. Additivity: token_shift(A+B) ≈ token_shift(A) + token_shift(B)?
2. Subtraction: token_shift(A-B) = token_shift(A) - token_shift(B)?
3. Scaling: token_shift(2×A) ≈ 2 × token_shift(A)?
4. Holland cancellation: token_shift(A + opposite(A)) ≈ 0?
5. Correlation between token shift magnitude and 5D coordinate magnitude
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="tok-arith")

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


def get_logprob_diff(model, tokenizer, device, blocks, mid_layer, ln_final, lm_head,
                      input_ids, steer_vec, alpha, baseline_hidden, detect_layer):
    """Get log-probability shift vector (steered - baseline) at detect_layer."""
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    captured = {}
    hooks = []

    def cap_fn(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().clone()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

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

    with torch.no_grad():
        base_logits = lm_head(ln_final(baseline_hidden.unsqueeze(0)))[0]
        steer_logits = lm_head(ln_final(captured["act"].unsqueeze(0)))[0]

    base_lp = torch.log_softmax(base_logits.float(), dim=-1)
    steer_lp = torch.log_softmax(steer_logits.float(), dim=-1)
    return (steer_lp - base_lp).cpu()


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
    detect_layer = mid_layer + 1

    detect_prompt = "Tell me about yourself."
    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    results = {}

    print(f"\n{'='*70}")
    print("TOKEN-SPACE ARITHMETIC")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # Get baseline hidden state
    baseline_hidden = {}
    hooks = []
    def cap_base(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        baseline_hidden["act"] = hs[0, -1, :].detach().clone()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    base_h = baseline_hidden["act"]

    # Get per-trait token shift vectors
    alpha = 2.0
    trait_shifts = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        shift = get_logprob_diff(model, tokenizer, device, blocks, mid_layer,
                                  ln_final, lm_head, input_ids, vec, alpha, base_h, detect_layer)
        trait_shifts[trait] = shift

    # ================================================================
    # PART 1: Scaling linearity
    # ================================================================
    logger.info("Part 1: Scaling linearity...")
    print(f"\n{'='*70}")
    print("PART 1: SCALING LINEARITY (does 2× alpha produce 2× token shift?)")
    print(f"{'='*70}")

    scaling_results = {}
    for trait in ["artistic", "social", "conventional"]:
        vec = residual[trait].astype(np.float32)
        shifts_at_alpha = {}
        for test_alpha in [0.5, 1.0, 2.0, 3.0]:
            shift = get_logprob_diff(model, tokenizer, device, blocks, mid_layer,
                                      ln_final, lm_head, input_ids, vec, test_alpha, base_h, detect_layer)
            shifts_at_alpha[test_alpha] = shift

        # Compare scaling: r(shift(α), α/α_ref * shift(α_ref))
        ref_alpha = 1.0
        ref_shift = shifts_at_alpha[ref_alpha]
        for test_alpha in [0.5, 2.0, 3.0]:
            predicted = (test_alpha / ref_alpha) * ref_shift
            actual = shifts_at_alpha[test_alpha]
            # Correlation
            corr = float(torch.corrcoef(torch.stack([actual.flatten(), predicted.flatten()]))[0, 1])
            # Cosine similarity
            cos = float(torch.nn.functional.cosine_similarity(
                actual.flatten().unsqueeze(0), predicted.flatten().unsqueeze(0)))
            # Norm ratio
            norm_ratio = float(actual.norm() / predicted.norm())

            print(f"  {trait} α={test_alpha} vs {ref_alpha}×{test_alpha:.1f}: "
                  f"corr={corr:.4f}, cos={cos:.4f}, norm_ratio={norm_ratio:.3f}")
            scaling_results[f"{trait}_alpha{test_alpha}"] = {
                "correlation": corr, "cosine": cos, "norm_ratio": norm_ratio}

    results["scaling"] = scaling_results

    # ================================================================
    # PART 2: Additivity (shift(A+B) ≈ shift(A) + shift(B)?)
    # ================================================================
    logger.info("Part 2: Additivity...")
    print(f"\n{'='*70}")
    print("PART 2: ADDITIVITY (shift(A+B) vs shift(A) + shift(B))")
    print(f"{'='*70}")

    add_results = {}
    test_pairs = [("artistic", "social"), ("investigative", "enterprising"),
                  ("realistic", "conventional")]

    for t1, t2 in test_pairs:
        v1 = residual[t1].astype(np.float32)
        v2 = residual[t2].astype(np.float32)

        # shift(A+B)
        combined_shift = get_logprob_diff(model, tokenizer, device, blocks, mid_layer,
                                           ln_final, lm_head, input_ids, v1 + v2, alpha, base_h, detect_layer)

        # shift(A) + shift(B)
        predicted = trait_shifts[t1] + trait_shifts[t2]

        corr = float(torch.corrcoef(torch.stack([combined_shift.flatten(), predicted.flatten()]))[0, 1])
        cos = float(torch.nn.functional.cosine_similarity(
            combined_shift.flatten().unsqueeze(0), predicted.flatten().unsqueeze(0)))
        norm_ratio = float(combined_shift.norm() / predicted.norm()) if predicted.norm() > 0 else 0

        print(f"  {t1}+{t2}: corr={corr:.4f}, cos={cos:.4f}, norm_ratio={norm_ratio:.3f}")
        add_results[f"{t1}+{t2}"] = {"correlation": corr, "cosine": cos, "norm_ratio": norm_ratio}

    results["additivity"] = add_results

    # ================================================================
    # PART 3: Holland cancellation in token space
    # ================================================================
    logger.info("Part 3: Holland cancellation...")
    print(f"\n{'='*70}")
    print("PART 3: HOLLAND CANCELLATION (A + opposite(A) → ~0?)")
    print(f"{'='*70}")

    cancel_results = {}
    holland_pairs = [("artistic", "conventional"), ("investigative", "enterprising"),
                     ("realistic", "social")]

    for t1, t2 in holland_pairs:
        v1 = residual[t1].astype(np.float32)
        v2 = residual[t2].astype(np.float32)

        # shift(A + opposite(A))
        cancel_shift = get_logprob_diff(model, tokenizer, device, blocks, mid_layer,
                                         ln_final, lm_head, input_ids, v1 + v2, alpha, base_h, detect_layer)

        # Individual norms for comparison
        individual_mean_norm = (trait_shifts[t1].norm() + trait_shifts[t2].norm()) / 2
        cancel_norm = cancel_shift.norm()
        suppression = float(1 - cancel_norm / individual_mean_norm) if individual_mean_norm > 0 else 0

        # Compare to non-opposite pair
        if t1 == "artistic":
            non_opp = "social"
        elif t1 == "investigative":
            non_opp = "social"
        else:
            non_opp = "artistic"

        v_non = residual[non_opp].astype(np.float32)
        non_opp_shift = get_logprob_diff(model, tokenizer, device, blocks, mid_layer,
                                          ln_final, lm_head, input_ids, v1 + v_non, alpha, base_h, detect_layer)
        non_opp_norm = non_opp_shift.norm()

        print(f"  {t1}+{t2} (opposite): norm={cancel_norm:.2f}, suppression={suppression:.1%}")
        print(f"  {t1}+{non_opp} (non-opp): norm={non_opp_norm:.2f}")

        cancel_results[f"{t1}+{t2}"] = {
            "cancel_norm": float(cancel_norm),
            "individual_mean_norm": float(individual_mean_norm),
            "suppression": float(suppression),
            "non_opp_norm": float(non_opp_norm),
        }

    results["holland_cancellation"] = cancel_results

    # ================================================================
    # PART 4: Cross-trait token shift correlation
    # ================================================================
    logger.info("Part 4: Cross-trait correlation...")
    print(f"\n{'='*70}")
    print("PART 4: CROSS-TRAIT TOKEN SHIFT CORRELATION")
    print(f"{'='*70}")

    # Compute pairwise correlation of token shift vectors
    corr_matrix = {}
    print(f"\n  {'':>12}", end="")
    for t in TRAITS:
        print(f" {t[:5]:>6}", end="")
    print()

    for t1 in TRAITS:
        print(f"  {t1:>12}", end="")
        corr_matrix[t1] = {}
        for t2 in TRAITS:
            corr = float(torch.corrcoef(
                torch.stack([trait_shifts[t1].flatten(), trait_shifts[t2].flatten()]))[0, 1])
            corr_matrix[t1][t2] = corr
            print(f" {corr:6.3f}", end="")
        print()

    results["cross_trait_correlation"] = corr_matrix

    # ================================================================
    # PART 5: Holland structure in token space
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: HOLLAND STRUCTURE IN TOKEN SHIFT CORRELATIONS")
    print(f"{'='*70}")

    # Check if Holland structure (adj > alt > opp) holds in token space
    holland_order = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]

    adj_corrs = []
    alt_corrs = []
    opp_corrs = []

    for i in range(6):
        for j in range(i + 1, 6):
            dist = min(j - i, 6 - (j - i))
            t1 = holland_order[i]
            t2 = holland_order[j]
            corr_val = corr_matrix[t1][t2]
            if dist == 1:
                adj_corrs.append(corr_val)
            elif dist == 2:
                alt_corrs.append(corr_val)
            elif dist == 3:
                opp_corrs.append(corr_val)

    mean_adj = np.mean(adj_corrs)
    mean_alt = np.mean(alt_corrs)
    mean_opp = np.mean(opp_corrs)
    monotonic = mean_adj > mean_alt > mean_opp

    print(f"  Adjacent (d=1): mean corr = {mean_adj:.3f}")
    print(f"  Alternate (d=2): mean corr = {mean_alt:.3f}")
    print(f"  Opposite (d=3): mean corr = {mean_opp:.3f}")
    print(f"  Monotonic (adj > alt > opp): {'YES' if monotonic else 'NO'}")

    results["holland_token_structure"] = {
        "adjacent": float(mean_adj),
        "alternate": float(mean_alt),
        "opposite": float(mean_opp),
        "monotonic": bool(monotonic),
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    mean_scaling_cos = np.mean([v["cosine"] for v in scaling_results.values()])
    mean_add_cos = np.mean([v["cosine"] for v in add_results.values()])
    mean_cancel_supp = np.mean([v["suppression"] for v in cancel_results.values()])

    print(f"  Scaling linearity: mean cosine = {mean_scaling_cos:.4f}")
    print(f"  Additivity: mean cosine = {mean_add_cos:.4f}")
    print(f"  Holland cancellation: mean suppression = {mean_cancel_supp:.1%}")
    print(f"  Holland structure in tokens: monotonic = {monotonic}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "token_arithmetic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
