#!/usr/bin/env python
"""
Personality Detection from Output Logits ONLY (No Hidden State Access).

Key question: Can an external observer detect personality steering
just from the model's output token distributions — without any access
to internal activations?

If yes: massive AI safety implication — persona steering is detectable
even from API-level access (just top-k logprobs).

Tests:
1. Full logit distribution → 6-class classification (KNN, centroid)
2. Top-k logprobs only (k=5,10,50) → classification
3. Single next-token entropy → trait detection
4. Cross-prompt generalization (train on 3 prompts, test on 5)
5. Alpha sensitivity: can you detect alpha from logits?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="logit-cls")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


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
    return residual, mid_layer


def get_logit_features(model, tokenizer, device, blocks, mid_layer,
                       steer_vec, alpha, prompt, vocab_size):
    """Get logit distribution under steering."""
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    hooks = []
    def steer_fn(_m, _i, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out
    hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    with torch.no_grad():
        outputs = model(input_ids)

    for h in hooks:
        h.remove()

    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    return logits


def get_baseline_logits(model, tokenizer, device, prompt):
    """Get logit distribution WITHOUT steering."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    return logits


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def kl_divergence(p, q):
    """KL(p || q)"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))


def entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))


def main():
    device = "cuda:1"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    residual, mid_layer = load_model_data(model_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    vocab_size = model.config.vocab_size

    # Prompts split into train and test
    train_prompts = [
        "Tell me about yourself.",
        "What do you enjoy doing in your free time?",
        "Describe your ideal weekend.",
    ]
    test_prompts = [
        "What matters most to you in life?",
        "How do you approach new challenges?",
        "What kind of work environment do you prefer?",
        "Describe your ideal vacation.",
        "What are your strengths and weaknesses?",
    ]
    all_prompts = train_prompts + test_prompts

    alpha = 2.0
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY DETECTION FROM OUTPUT LOGITS ONLY")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # Collect logit distributions
    # ================================================================
    logger.info("Collecting logit distributions...")

    # Get baseline for each prompt
    baselines = {}
    for prompt in all_prompts:
        baselines[prompt] = get_baseline_logits(model, tokenizer, device, prompt)

    # Get steered logits for each trait × prompt
    steered_logits = {}  # (trait, prompt) -> logits
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        for prompt in all_prompts:
            logits = get_logit_features(model, tokenizer, device, blocks,
                                        mid_layer, vec, alpha, prompt, vocab_size)
            steered_logits[(trait, prompt)] = logits

    # ================================================================
    # PART 1: KL-divergence features for classification
    # ================================================================
    logger.info("Part 1: KL-divergence classification...")
    print(f"\n{'='*70}")
    print("PART 1: KL-DIVERGENCE BASED CLASSIFICATION")
    print(f"{'='*70}")

    # For each steered distribution, compute KL from baseline
    # Then use KL-from-each-trait-centroid as features

    # Compute logit DIFFERENCES (steered - baseline) as features
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for trait_idx, trait in enumerate(TRAITS):
        for prompt in train_prompts:
            diff = steered_logits[(trait, prompt)] - baselines[prompt]
            train_features.append(diff)
            train_labels.append(trait_idx)
        for prompt in test_prompts:
            diff = steered_logits[(trait, prompt)] - baselines[prompt]
            test_features.append(diff)
            test_labels.append(trait_idx)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Centroid classifier in logit-diff space
    centroids = np.zeros((6, vocab_size))
    for i in range(6):
        mask = train_labels == i
        centroids[i] = train_features[mask].mean(axis=0)

    # Classify test by nearest centroid (cosine)
    correct_centroid = 0
    for i in range(len(test_features)):
        sims = []
        for c in centroids:
            norm_f = np.linalg.norm(test_features[i])
            norm_c = np.linalg.norm(c)
            if norm_f > 0 and norm_c > 0:
                sims.append(np.dot(test_features[i], c) / (norm_f * norm_c))
            else:
                sims.append(0)
        pred = np.argmax(sims)
        if pred == test_labels[i]:
            correct_centroid += 1

    centroid_acc = correct_centroid / len(test_labels)
    print(f"  Full logit-diff centroid classifier: {correct_centroid}/{len(test_labels)} ({centroid_acc:.0%})")
    results["full_logit_diff_centroid"] = {
        "correct": correct_centroid,
        "total": len(test_labels),
        "accuracy": float(centroid_acc)
    }

    # ================================================================
    # PART 2: Top-k logprobs only
    # ================================================================
    logger.info("Part 2: Top-k logprob classification...")
    print(f"\n{'='*70}")
    print("PART 2: TOP-K LOGPROB CLASSIFICATION")
    print(f"{'='*70}")

    topk_results = {}
    for k in [5, 10, 50, 100, 500, 1000]:
        # For each distribution, keep only top-k tokens
        # Use the token IDs + their logit values as sparse features
        train_sparse = []
        test_sparse = []

        for feat_set, sparse_list in [(train_features, train_sparse), (test_features, test_sparse)]:
            for diff in feat_set:
                top_idx = np.argsort(np.abs(diff))[-k:]
                sparse = np.zeros(vocab_size)
                sparse[top_idx] = diff[top_idx]
                sparse_list.append(sparse)

        train_sparse = np.array(train_sparse)
        test_sparse = np.array(test_sparse)

        # Centroids
        sparse_centroids = np.zeros((6, vocab_size))
        for i in range(6):
            mask = train_labels == i
            sparse_centroids[i] = train_sparse[mask].mean(axis=0)

        correct = 0
        for i in range(len(test_sparse)):
            sims = []
            for c in sparse_centroids:
                nf = np.linalg.norm(test_sparse[i])
                nc = np.linalg.norm(c)
                if nf > 0 and nc > 0:
                    sims.append(np.dot(test_sparse[i], c) / (nf * nc))
                else:
                    sims.append(0)
            pred = np.argmax(sims)
            if pred == test_labels[i]:
                correct += 1

        acc = correct / len(test_labels)
        topk_results[k] = {"correct": correct, "total": len(test_labels), "accuracy": float(acc)}
        print(f"  Top-{k} logprob diff: {correct}/{len(test_labels)} ({acc:.0%})")

    results["topk_logprob"] = {str(k): v for k, v in topk_results.items()}

    # ================================================================
    # PART 3: Entropy-based features
    # ================================================================
    logger.info("Part 3: Entropy-based features...")
    print(f"\n{'='*70}")
    print("PART 3: ENTROPY AND KL-BASED FEATURES")
    print(f"{'='*70}")

    entropy_features_train = []
    entropy_features_test = []
    kl_features_train = []
    kl_features_test = []

    for trait_idx, trait in enumerate(TRAITS):
        for prompt in train_prompts:
            probs_s = softmax(steered_logits[(trait, prompt)])
            probs_b = softmax(baselines[prompt])
            entropy_features_train.append([entropy(probs_s), entropy(probs_b),
                                            entropy(probs_s) - entropy(probs_b)])
            kl_features_train.append(kl_divergence(probs_s, probs_b))

        for prompt in test_prompts:
            probs_s = softmax(steered_logits[(trait, prompt)])
            probs_b = softmax(baselines[prompt])
            entropy_features_test.append([entropy(probs_s), entropy(probs_b),
                                           entropy(probs_s) - entropy(probs_b)])
            kl_features_test.append(kl_divergence(probs_s, probs_b))

    entropy_features_train = np.array(entropy_features_train)
    entropy_features_test = np.array(entropy_features_test)

    # Per-trait mean entropy delta
    print("\n  Per-trait mean entropy delta (steered - baseline):")
    for i, trait in enumerate(TRAITS):
        mask = train_labels == i
        mean_delta = entropy_features_train[mask, 2].mean()
        print(f"    {trait}: {mean_delta:+.4f}")

    # KL from baseline per trait
    kl_train = np.array(kl_features_train)
    print("\n  Per-trait mean KL divergence from baseline:")
    for i, trait in enumerate(TRAITS):
        mask = train_labels == i
        mean_kl = kl_train[mask].mean()
        print(f"    {trait}: {mean_kl:.4f}")

    results["entropy_features"] = {
        "per_trait_entropy_delta": {
            trait: float(entropy_features_train[train_labels == i, 2].mean())
            for i, trait in enumerate(TRAITS)
        },
        "per_trait_kl": {
            trait: float(kl_train[train_labels == i].mean())
            for i, trait in enumerate(TRAITS)
        }
    }

    # ================================================================
    # PART 4: PCA of logit diffs → low-dim classification
    # ================================================================
    logger.info("Part 4: PCA of logit diffs...")
    print(f"\n{'='*70}")
    print("PART 4: PCA OF LOGIT DIFFS (DIMENSIONALITY)")
    print(f"{'='*70}")

    # Combine all diffs
    all_diffs = np.vstack([train_features, test_features])
    all_labels_combined = np.concatenate([train_labels, test_labels])

    # PCA via SVD
    mean_diff = all_diffs.mean(axis=0)
    centered = all_diffs - mean_diff
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    variance_explained = S**2 / (S**2).sum()

    print(f"  Top-10 singular values: {S[:10].tolist()}")
    print(f"  Variance explained (first 10): {variance_explained[:10].tolist()}")
    print(f"  Cumulative variance for dims 1-10:")
    cum_var = np.cumsum(variance_explained)
    for d in [1, 2, 3, 5, 10]:
        print(f"    {d}D: {cum_var[d-1]:.1%}")

    # Classification in PCA space
    max_components = min(len(S), len(all_diffs))
    pca_results = {}
    for n_dims in [2, 3, 5, 10, 50]:
        if n_dims > max_components:
            continue
        train_pca = (train_features - mean_diff) @ Vt[:n_dims].T
        test_pca = (test_features - mean_diff) @ Vt[:n_dims].T

        # Centroid classifier
        pca_centroids = np.zeros((6, n_dims))
        for i in range(6):
            mask = train_labels == i
            pca_centroids[i] = train_pca[mask].mean(axis=0)

        correct = 0
        for i in range(len(test_pca)):
            dists = [np.linalg.norm(test_pca[i] - c) for c in pca_centroids]
            pred = np.argmin(dists)
            if pred == test_labels[i]:
                correct += 1

        acc = correct / len(test_labels)
        pca_results[n_dims] = {"correct": correct, "total": len(test_labels), "accuracy": float(acc)}
        print(f"  PCA-{n_dims}D centroid: {correct}/{len(test_labels)} ({acc:.0%})")

    results["pca_classification"] = {str(k): v for k, v in pca_results.items()}
    results["pca_variance"] = {
        "singular_values": S[:20].tolist(),
        "variance_explained": variance_explained[:20].tolist(),
        "cumulative_5d": float(cum_var[4]) if len(cum_var) > 4 else None,
        "cumulative_10d": float(cum_var[9]) if len(cum_var) > 9 else None,
        "max_components": int(max_components),
    }

    # ================================================================
    # PART 5: Alpha sensitivity from logits
    # ================================================================
    logger.info("Part 5: Alpha sensitivity...")
    print(f"\n{'='*70}")
    print("PART 5: ALPHA DETECTION FROM LOGITS")
    print(f"{'='*70}")

    alphas_test = [0.5, 1.0, 2.0, 3.0, 5.0]
    alpha_results = {}
    test_prompt = "Tell me about yourself."

    for a in alphas_test:
        correct = 0
        total = 0
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            logits = get_logit_features(model, tokenizer, device, blocks,
                                        mid_layer, vec, a, test_prompt, vocab_size)
            diff = logits - baselines[test_prompt]

            # Classify using α=2 centroids
            sims = []
            for c in centroids:
                nf = np.linalg.norm(diff)
                nc = np.linalg.norm(c)
                if nf > 0 and nc > 0:
                    sims.append(np.dot(diff, c) / (nf * nc))
                else:
                    sims.append(0)
            pred = np.argmax(sims)
            if TRAITS[pred] == trait:
                correct += 1
            total += 1

        acc = correct / total
        alpha_results[a] = {"correct": correct, "total": total, "accuracy": float(acc)}
        print(f"  α={a}: {correct}/{total} ({acc:.0%}) (using α=2 centroids)")

    results["alpha_sensitivity"] = {str(k): v for k, v in alpha_results.items()}

    # ================================================================
    # PART 6: Logit-diff norm as alpha estimator
    # ================================================================
    logger.info("Part 6: Alpha estimation from logit-diff norm...")
    print(f"\n{'='*70}")
    print("PART 6: ALPHA ESTIMATION FROM LOGIT-DIFF NORM")
    print(f"{'='*70}")

    alpha_norms = {}
    for a in alphas_test:
        norms = []
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            logits = get_logit_features(model, tokenizer, device, blocks,
                                        mid_layer, vec, a, test_prompt, vocab_size)
            diff = logits - baselines[test_prompt]
            norms.append(float(np.linalg.norm(diff)))
        alpha_norms[a] = {"mean_norm": float(np.mean(norms)), "std_norm": float(np.std(norms))}
        print(f"  α={a}: mean diff norm = {np.mean(norms):.1f} ± {np.std(norms):.1f}")

    # Correlation between alpha and norm
    alphas_arr = np.array(alphas_test)
    norms_arr = np.array([alpha_norms[a]["mean_norm"] for a in alphas_test])
    if len(alphas_arr) > 2:
        corr = float(np.corrcoef(alphas_arr, norms_arr)[0, 1])
        print(f"\n  Alpha-norm correlation: r = {corr:.4f}")
    else:
        corr = None

    results["alpha_norm_estimation"] = {
        "per_alpha": {str(k): v for k, v in alpha_norms.items()},
        "alpha_norm_correlation": corr,
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Full logit-diff centroid: {results['full_logit_diff_centroid']['accuracy']:.0%}")
    print(f"  Top-100 logprob: {results['topk_logprob']['100']['accuracy']:.0%}")
    print(f"  Top-10 logprob: {results['topk_logprob']['10']['accuracy']:.0%}")
    pca5_acc = results['pca_classification'].get('5', {}).get('accuracy', 'N/A')
    if isinstance(pca5_acc, float):
        print(f"  PCA-5D: {pca5_acc:.0%}")
    print(f"  α=0.5 detection (using α=2 centroids): {results['alpha_sensitivity']['0.5']['accuracy']:.0%}")
    cum5 = results['pca_variance']['cumulative_5d']
    if cum5 is not None:
        print(f"  Logit diff variance in 5D: {cum5:.1%}")

    results["summary"] = {
        "full_logit_centroid_acc": float(results['full_logit_diff_centroid']['accuracy']),
        "top100_acc": float(results['topk_logprob']['100']['accuracy']),
        "pca5d_acc": float(pca5_acc) if isinstance(pca5_acc, float) else None,
        "alpha_05_detection": float(results['alpha_sensitivity']['0.5']['accuracy']),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logit_personality_classifier.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
