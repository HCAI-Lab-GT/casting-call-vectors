#!/usr/bin/env python
"""
Random control on Marin 8B (chat template) to confirm model-generality.
Same 4 conditions: real, random, shuffled, shared-only.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="random-control-marin")

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


def pairwise_logprob_chat(model, tokenizer, device, desc_a, desc_b):
    messages = [
        {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
        {"role": "user", "content": (
            f"Which describes you better?\n"
            f"A) I am {desc_a}\n"
            f"B) I am {desc_b}\n"
            f"Answer:"
        )},
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
    model_id = "marin-community/marin-8b-instruct"
    device = "cuda:0"
    alpha = 1.0  # Optimal for Marin 8B
    n_random_trials = 5

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
    residual_norms = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj
        residual_norms[t] = np.linalg.norm(residual_vectors[t])

    mean_norm = np.mean(list(residual_norms.values()))
    hidden_dim = residual_vectors[TRAITS[0]].shape[0]

    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            gap = pairwise_logprob_chat(model, tokenizer, device,
                                       TRAIT_DESCRIPTIONS[trait_a], TRAIT_DESCRIPTIONS[trait_b])
            baseline[f"{trait_a}-{trait_b}"] = gap

    results = {"baseline": baseline, "alpha": alpha, "conditions": {}}

    print(f"\n{'='*70}")
    print(f"NEGATIVE CONTROL: {model_id}")
    print(f"Alpha: {alpha}")
    print(f"{'='*70}")

    # Real
    r_real = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                 residual_vectors, alpha, baseline)
    print(f"\n  Real persona:          {r_real['delta_accuracy']:.0%} ({r_real['correct']}/{r_real['total']}), "
          f"mean_delta={r_real['mean_delta']:+.4f}")
    results["conditions"]["real_persona"] = r_real

    # Random
    np.random.seed(42)
    random_accs = []
    for trial in range(n_random_trials):
        rvecs = {}
        for t in TRAITS:
            rv = np.random.randn(hidden_dim).astype(np.float32)
            rv = rv / np.linalg.norm(rv) * residual_norms[t]
            rvecs[t] = rv
        r = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                rvecs, alpha, baseline)
        random_accs.append(r["delta_accuracy"])
        print(f"  Random trial {trial+1}:        {r['delta_accuracy']:.0%}")

    results["conditions"]["random"] = {
        "mean": float(np.mean(random_accs)),
        "std": float(np.std(random_accs)),
    }

    # Shuffled
    shuffled_accs = []
    for trial in range(n_random_trials):
        perm = np.random.permutation(TRAITS).tolist()
        while any(perm[i] == TRAITS[i] for i in range(6)):
            perm = np.random.permutation(TRAITS).tolist()
        svecs = {TRAITS[i]: residual_vectors[perm[i]] for i in range(6)}
        r = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                svecs, alpha, baseline)
        shuffled_accs.append(r["delta_accuracy"])
        print(f"  Shuffled trial {trial+1}:      {r['delta_accuracy']:.0%}")

    results["conditions"]["shuffled"] = {
        "mean": float(np.mean(shuffled_accs)),
        "std": float(np.std(shuffled_accs)),
    }

    # Shared only
    shared_vecs = {t: shared_dir * mean_norm for t in TRAITS}
    r_shared = eval_discrimination(model, tokenizer, device, blocks, mid_layer,
                                   shared_vecs, alpha, baseline)
    print(f"  Shared direction only: {r_shared['delta_accuracy']:.0%}, mean_delta={r_shared['mean_delta']:+.4f}")
    results["conditions"]["shared_only"] = r_shared

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Real persona:          {r_real['delta_accuracy']:.0%}")
    print(f"  Random (matched norm): {np.mean(random_accs):.0%} ± {np.std(random_accs):.1%}")
    print(f"  Shuffled labels:       {np.mean(shuffled_accs):.0%} ± {np.std(shuffled_accs):.1%}")
    print(f"  Shared direction only: {r_shared['delta_accuracy']:.0%}")

    from scipy.stats import ttest_1samp
    t1, p1 = ttest_1samp(random_accs, r_real["delta_accuracy"])
    t2, p2 = ttest_1samp(shuffled_accs, r_real["delta_accuracy"])
    print(f"\n  Real vs Random:   t={t1:.2f}, p={p1:.6f}")
    print(f"  Real vs Shuffled: t={t2:.2f}, p={p2:.6f}")

    results["stats"] = {
        "real_vs_random": {"t": float(t1), "p": float(p1)},
        "real_vs_shuffled": {"t": float(t2), "p": float(p2)},
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"random_control_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
