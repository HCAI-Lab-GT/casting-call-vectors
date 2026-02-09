#!/usr/bin/env python
"""
NEGATIVE CONTROL: Do random vectors of the same norm as persona vectors
produce pairwise discrimination?

If random vectors produce ~50% (chance), this confirms persona vectors
encode specific personality information. If random vectors also produce
high discrimination, the effect is an artifact of activation magnitude.

Tests:
1. Random vectors matched in norm to residual persona vectors
2. Shuffled persona vectors (trait labels scrambled)
3. Reversed shared direction (inject -shared_dir instead of persona)

Uses SmolLM3-3B with completion prompts at alpha=2.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="random-control")

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
    alpha = 2.0
    n_random_trials = 5  # Average over multiple random trials

    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load vectors
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
    residual_norms = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj
        residual_norms[t] = np.linalg.norm(residual_vectors[t])

    mean_norm = np.mean(list(residual_norms.values()))
    hidden_dim = residual_vectors[TRAITS[0]].shape[0]

    logger.info("Residual norms: %s", {t: f"{n:.2f}" for t, n in residual_norms.items()})
    logger.info("Mean norm: %.2f, Hidden dim: %d", mean_norm, hidden_dim)

    # Load model
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
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob(model, tokenizer, device,
                                  TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    results = {"baseline": baseline, "alpha": alpha, "conditions": {}}

    print(f"\n{'='*70}")
    print(f"NEGATIVE CONTROL EXPERIMENT")
    print(f"Model: {model_id}, Alpha: {alpha}")
    print(f"{'='*70}")

    # Condition 1: Real persona vectors (positive control)
    print(f"\n--- Condition 1: Real persona vectors ---")
    r_real = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                 residual_vectors, alpha, baseline)
    print(f"  Delta accuracy: {r_real['delta_accuracy']:.0%} ({r_real['correct']}/{r_real['total']})")
    print(f"  Mean delta: {r_real['mean_delta']:+.4f}")
    results["conditions"]["real_persona"] = r_real

    # Condition 2: Random vectors (matched norm, per-trait)
    print(f"\n--- Condition 2: Random vectors (matched norm, {n_random_trials} trials) ---")
    random_accs = []
    random_deltas = []
    np.random.seed(42)

    for trial in range(n_random_trials):
        random_vectors = {}
        for t in TRAITS:
            rvec = np.random.randn(hidden_dim).astype(np.float32)
            rvec = rvec / np.linalg.norm(rvec) * residual_norms[t]  # Match per-trait norm
            random_vectors[t] = rvec

        r_random = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                       random_vectors, alpha, baseline)
        random_accs.append(r_random["delta_accuracy"])
        random_deltas.append(r_random["mean_delta"])
        print(f"  Trial {trial+1}: {r_random['delta_accuracy']:.0%} ({r_random['correct']}/{r_random['total']}), "
              f"mean_delta={r_random['mean_delta']:+.4f}")

    results["conditions"]["random_matched_norm"] = {
        "mean_accuracy": float(np.mean(random_accs)),
        "std_accuracy": float(np.std(random_accs)),
        "mean_delta": float(np.mean(random_deltas)),
        "per_trial": [{"accuracy": a, "mean_delta": d} for a, d in zip(random_accs, random_deltas)],
    }
    print(f"  MEAN: {np.mean(random_accs):.0%} ± {np.std(random_accs):.1%}")

    # Condition 3: Shuffled persona vectors (wrong trait labels)
    print(f"\n--- Condition 3: Shuffled persona vectors ({n_random_trials} trials) ---")
    shuffled_accs = []
    shuffled_deltas = []

    for trial in range(n_random_trials):
        perm = np.random.permutation(TRAITS).tolist()
        # Ensure no trait maps to itself
        while any(perm[i] == TRAITS[i] for i in range(6)):
            perm = np.random.permutation(TRAITS).tolist()

        shuffled_vectors = {TRAITS[i]: residual_vectors[perm[i]] for i in range(6)}
        r_shuf = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                     shuffled_vectors, alpha, baseline)
        shuffled_accs.append(r_shuf["delta_accuracy"])
        shuffled_deltas.append(r_shuf["mean_delta"])
        print(f"  Trial {trial+1}: {r_shuf['delta_accuracy']:.0%} ({r_shuf['correct']}/{r_shuf['total']}), "
              f"mapping: {dict(zip(TRAITS, perm))}")

    results["conditions"]["shuffled_labels"] = {
        "mean_accuracy": float(np.mean(shuffled_accs)),
        "std_accuracy": float(np.std(shuffled_accs)),
        "mean_delta": float(np.mean(shuffled_deltas)),
        "per_trial": [{"accuracy": a, "mean_delta": d} for a, d in zip(shuffled_accs, shuffled_deltas)],
    }
    print(f"  MEAN: {np.mean(shuffled_accs):.0%} ± {np.std(shuffled_accs):.1%}")

    # Condition 4: Shared direction only (same vector for all traits)
    print(f"\n--- Condition 4: Shared direction only (same for all traits) ---")
    shared_vectors = {t: shared_dir * mean_norm for t in TRAITS}
    r_shared = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                   shared_vectors, alpha, baseline)
    print(f"  Delta accuracy: {r_shared['delta_accuracy']:.0%} ({r_shared['correct']}/{r_shared['total']})")
    print(f"  Mean delta: {r_shared['mean_delta']:+.4f}")
    results["conditions"]["shared_direction_only"] = r_shared

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':>25}  {'Delta%':>7}  {'MeanΔ':>8}")
    print(f"  {'-'*44}")
    print(f"  {'Real persona':>25}  {r_real['delta_accuracy']:>6.0%}  {r_real['mean_delta']:>+7.4f}")
    print(f"  {'Random (matched norm)':>25}  {np.mean(random_accs):>6.0%}  {np.mean(random_deltas):>+7.4f}")
    print(f"  {'Shuffled labels':>25}  {np.mean(shuffled_accs):>6.0%}  {np.mean(shuffled_deltas):>+7.4f}")
    print(f"  {'Shared direction only':>25}  {r_shared['delta_accuracy']:>6.0%}  {r_shared['mean_delta']:>+7.4f}")

    # Statistical test: is real significantly above random?
    from scipy.stats import ttest_1samp
    t_stat, p_val = ttest_1samp(random_accs, r_real["delta_accuracy"])
    print(f"\n  Real vs Random: t={t_stat:.2f}, p={p_val:.4f}")

    t_stat2, p_val2 = ttest_1samp(shuffled_accs, r_real["delta_accuracy"])
    print(f"  Real vs Shuffled: t={t_stat2:.2f}, p={p_val2:.4f}")

    results["statistical_tests"] = {
        "real_vs_random": {"t": float(t_stat), "p": float(p_val)},
        "real_vs_shuffled": {"t": float(t_stat2), "p": float(p_val2)},
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"random_control_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
