#!/usr/bin/env python
"""
Fine-grained dose-response curve: discrimination accuracy vs alpha.

Tests 15 alpha values (0.1 to 20) on SmolLM3-3B to produce a smooth
sigmoid-like curve showing the relationship between steering strength
and trait discrimination. Also includes negative alphas for symmetry.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="dose-response")

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


def eval_at_alpha(model, tokenizer, device, blocks, mid_layer, vectors, alpha, baseline):
    correct = 0
    total = 0
    deltas = []
    per_pair = {}

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
                    per_pair[f"{steer_trait}_vs_{trait_a}-{trait_b}"] = float(d)
        finally:
            hook_handle.remove()

    return {
        "delta_accuracy": correct / total if total else 0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0,
        "std_delta": float(np.std(deltas)) if deltas else 0,
        "min_delta": float(np.min(deltas)) if deltas else 0,
        "max_delta": float(np.max(deltas)) if deltas else 0,
        "correct": correct,
        "total": total,
        "per_pair": per_pair,
    }


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"

    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

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

    residual_vectors = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Compute baseline
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device,
                                  TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    # Fine-grained alpha sweep (both positive and negative)
    alphas = [-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.2,
              0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]

    results = {"baseline": baseline, "model_id": model_id, "dose_response": {}}

    print(f"\n{'='*70}")
    print(f"DOSE-RESPONSE CURVE: {model_id}")
    print(f"{'='*70}")
    print(f"\n  {'Alpha':>6}  {'Delta%':>7}  {'MeanΔ':>8}  {'StdΔ':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'-'*52}")

    for alpha in alphas:
        r = eval_at_alpha(model, tokenizer, device, blocks, mid_layer,
                         residual_vectors, alpha, baseline)
        results["dose_response"][str(alpha)] = r
        print(f"  {alpha:>+6.2f}  {r['delta_accuracy']:>6.0%}  {r['mean_delta']:>+7.3f}"
              f"  {r['std_delta']:>6.3f}  {r['min_delta']:>+6.3f}  {r['max_delta']:>+6.3f}")

    # Analyze: is the dose-response linear?
    pos_alphas = [a for a in alphas if a > 0]
    pos_accs = [results["dose_response"][str(a)]["delta_accuracy"] for a in pos_alphas]
    pos_deltas = [results["dose_response"][str(a)]["mean_delta"] for a in pos_alphas]

    from scipy.stats import pearsonr, spearmanr
    # Mean delta should be roughly linear with alpha (at low alpha)
    low_alphas = [a for a in pos_alphas if a <= 3.0]
    low_deltas = [results["dose_response"][str(a)]["mean_delta"] for a in low_alphas]
    r_pearson, p_pearson = pearsonr(low_alphas, low_deltas)

    print(f"\n--- Linearity analysis (α ≤ 3.0) ---")
    print(f"  Pearson r (alpha vs mean_delta): {r_pearson:.4f} (p={p_pearson:.4f})")

    # Negative vs positive symmetry
    print(f"\n--- Positive/Negative symmetry ---")
    for abs_alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        pos = results["dose_response"].get(str(abs_alpha), {})
        neg = results["dose_response"].get(str(-abs_alpha), {})
        if pos and neg:
            print(f"  |α|={abs_alpha}: pos_delta={pos['mean_delta']:+.3f}, "
                  f"neg_delta={neg['mean_delta']:+.3f}, "
                  f"ratio={abs(neg['mean_delta']/pos['mean_delta']) if pos['mean_delta'] != 0 else 'inf':.3f}")

    results["linearity"] = {"pearson_r": float(r_pearson), "pearson_p": float(p_pearson)}

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dose_response_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
