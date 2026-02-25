#!/usr/bin/env python
"""
Topic Shift Robustness: Does personality detection survive dramatic topic changes?

If personality vectors live in a subspace orthogonal to content/topic, then
the 5D personality signal should remain detectable regardless of whether the
model is discussing mathematics, cooking, philosophy, or creative writing.

Tests:
1. 6 traits x 10 topic domains x 3 prompts each = 180 steered generations
2. Per-domain detection accuracy (argmax cosine in 5D)
3. Cross-domain consistency: cosine between 5D coordinates from different topics
4. Identifies weakest/strongest domains for personality signal

This is a strong test of the orthogonality claim: if personality is truly
orthogonal to content, then topic should not matter at all.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="topic-shift")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TOPIC_DOMAINS = {
    "mathematics": [
        "Explain the Pythagorean theorem",
        "What is calculus used for?",
        "Describe prime numbers",
    ],
    "cooking": [
        "How do you make pasta from scratch?",
        "What makes a good stew?",
        "Describe baking bread",
    ],
    "philosophy": [
        "What is the meaning of existence?",
        "Discuss free will vs determinism",
        "What is consciousness?",
    ],
    "technology": [
        "How do computers process information?",
        "Explain cloud computing",
        "What is artificial intelligence?",
    ],
    "sports": [
        "What makes a great athlete?",
        "Explain the rules of soccer",
        "Describe training for a marathon",
    ],
    "nature": [
        "Describe the water cycle",
        "How do forests sustain ecosystems?",
        "What causes earthquakes?",
    ],
    "history": [
        "What caused World War I?",
        "Describe the Renaissance period",
        "How did the Roman Empire fall?",
    ],
    "emotions": [
        "How do you handle disappointment?",
        "What makes you happy?",
        "Describe overcoming fear",
    ],
    "creative_writing": [
        "Write a short story opening",
        "Describe a sunset at the ocean",
        "Create a character sketch",
    ],
    "career_advice": [
        "How do you find your dream job?",
        "What skills matter most?",
        "How do you handle workplace conflict?",
    ],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_model_data(model_id, riasec_dir):
    """Load persona vectors, compute residuals, build 5D basis."""
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

    # Build 5D basis from residual vectors (subtract shared PC1)
    detect_layer = mid_layer + 1
    V = np.stack([all_layer_vectors[t][detect_layer] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][detect_layer]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj

    V_res = np.stack([residual[t] for t in TRAITS])
    _, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    n_dims = min(5, min(len(S_res), len(TRAITS)))
    basis_5d = Vt_res[:n_dims]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
        "detect_layer": detect_layer,
    }


def generate_steered(model, tokenizer, device, blocks, mid_layer, detect_layer,
                     basis_5d, coords_5d, steer_vec, alpha, prompt,
                     max_tokens=100, temperature=0.7):
    """
    Generate tokens with personality steering using a manual generation loop.
    Extract hidden state at detect_layer, project to 5D, compute cosine similarity.

    Uses register_forward_pre_hook for steering (modifies input to mid_layer+1).
    """
    # Prepare steering delta
    delta = alpha * torch.tensor(
        steer_vec.astype(np.float32), dtype=model.dtype
    ).unsqueeze(0).to(device)

    # Format prompt
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Storage for captured hidden states at detect_layer
    all_hidden = []
    hooks = []

    # Steering hook: register_forward_pre_hook on the layer AFTER mid_layer
    # pre_hook signature: (module, input) -> modified input
    def steer_hook(module, inp):
        hs = inp[0]
        hs[:, -1, :] += delta
        return (hs,) + inp[1:]

    hooks.append(blocks[mid_layer].register_forward_pre_hook(steer_hook))

    # Capture hook at detect_layer to record hidden states
    def capture_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        all_hidden.append(hs[:, -1, :].detach().cpu().float().numpy()[0].copy())

    hooks.append(blocks[detect_layer].register_forward_hook(capture_hook))

    # Manual generation loop
    past_kv = None
    generated_ids = []

    try:
        for step in range(max_tokens):
            with torch.no_grad():
                inp = input_ids if past_kv is None else input_ids[:, -1:]
                outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True)

            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Sample with temperature, with NaN guard for fp16 stability
            logits_f = logits.float().clamp(-100, 100)
            probs = torch.softmax(logits_f / max(temperature, 0.01), dim=-1)

            # NaN guard on probs
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.shape[-1]

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break
    finally:
        for h in hooks:
            h.remove()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute mean hidden state across generated tokens
    if len(all_hidden) == 0:
        mean_hidden = np.zeros(model.config.hidden_size, dtype=np.float64)
    else:
        mean_hidden = np.mean(all_hidden, axis=0).astype(np.float64)

    # Project to 5D
    coords = basis_5d @ mean_hidden
    norm_5d = float(np.linalg.norm(coords))

    # Cosine similarity to each trait direction
    sims = {}
    for t in TRAITS:
        t_norm = float(np.linalg.norm(coords_5d[t]))
        if norm_5d > 0 and t_norm > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (norm_5d * t_norm))
        else:
            sims[t] = 0.0

    detected = max(sims, key=sims.get)

    return {
        "detected": detected,
        "sims": sims,
        "coords_5d": coords.tolist(),
        "norm_5d": norm_5d,
        "text_snippet": text[:120],
        "num_tokens": len(generated_ids),
    }


def get_baseline_hidden(model, tokenizer, device, blocks, detect_layer, prompt):
    """Get baseline hidden state (no steering) for a given prompt."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}

    def cap_hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().cpu().float().numpy().copy()

    handle = blocks[detect_layer].register_forward_hook(cap_hook)
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    return captured["act"]


def main():
    device = "cuda:2"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"
    alpha = 3.0

    logger.info("Loading persona vectors and building 5D basis...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]
    detect_layer = model_data["detect_layer"]

    logger.info("Loading Marin 8B instruct on %s...", device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    print(f"\n{'='*70}")
    print("TOPIC SHIFT ROBUSTNESS: PERSONALITY DETECTION ACROSS DOMAINS")
    print(f"Model: {model_id}, alpha={alpha}")
    print(f"mid_layer={mid_layer}, detect_layer={detect_layer}")
    print(f"Domains: {len(TOPIC_DOMAINS)}, Prompts/domain: 3, Traits: {len(TRAITS)}")
    print(f"Total generations: {len(TOPIC_DOMAINS) * 3 * len(TRAITS)}")
    print(f"{'='*70}")

    per_trait = {}
    domain_accuracy_agg = {d: [] for d in TOPIC_DOMAINS}  # across traits
    domain_sim_agg = {d: [] for d in TOPIC_DOMAINS}

    for trait_idx, trait in enumerate(TRAITS):
        logger.info("Trait %d/%d: %s", trait_idx + 1, len(TRAITS), trait)
        vec = residual[trait]

        print(f"\n{'='*70}")
        print(f"TRAIT: {trait} ({trait_idx+1}/{len(TRAITS)})")
        print(f"{'='*70}")

        per_domain = {}
        all_coords_by_domain = {}  # for cross-domain cosine

        for domain, prompts in TOPIC_DOMAINS.items():
            domain_correct = 0
            domain_sims = []
            prompt_results = []
            domain_coords_list = []

            for prompt in prompts:
                res = generate_steered(
                    model, tokenizer, device, blocks, mid_layer, detect_layer,
                    basis_5d, coords_5d, vec, alpha, prompt,
                    max_tokens=100, temperature=0.7,
                )

                is_correct = res["detected"] == trait
                domain_correct += int(is_correct)
                target_sim = res["sims"][trait]
                domain_sims.append(target_sim)
                domain_coords_list.append(np.array(res["coords_5d"]))

                prompt_results.append({
                    "prompt": prompt,
                    "detected": res["detected"],
                    "correct": bool(is_correct),
                    "target_sim": float(target_sim),
                    "all_sims": {k: float(v) for k, v in res["sims"].items()},
                    "norm_5d": res["norm_5d"],
                    "text_snippet": res["text_snippet"],
                    "num_tokens": res["num_tokens"],
                })

            accuracy = domain_correct / len(prompts)
            mean_target_sim = float(np.mean(domain_sims))

            per_domain[domain] = {
                "accuracy": float(accuracy),
                "mean_target_sim": mean_target_sim,
                "prompts": prompt_results,
            }

            # Store domain mean coords for cross-domain cosine
            all_coords_by_domain[domain] = np.mean(domain_coords_list, axis=0)

            domain_accuracy_agg[domain].append(accuracy)
            domain_sim_agg[domain].append(mean_target_sim)

            status = "OK" if accuracy == 1.0 else "MISS"
            print(f"  {domain:>16}: acc={accuracy:.0%}, sim={mean_target_sim:.3f}  [{status}]")

        # Cross-domain consistency: pairwise cosine between domain mean coords
        domains_list = list(TOPIC_DOMAINS.keys())
        n_domains = len(domains_list)
        cos_matrix = np.zeros((n_domains, n_domains))
        for i in range(n_domains):
            for j in range(n_domains):
                c_i = all_coords_by_domain[domains_list[i]]
                c_j = all_coords_by_domain[domains_list[j]]
                n_i = np.linalg.norm(c_i)
                n_j = np.linalg.norm(c_j)
                if n_i > 0 and n_j > 0:
                    cos_matrix[i, j] = np.dot(c_i, c_j) / (n_i * n_j)
                else:
                    cos_matrix[i, j] = 0.0

        # Mean off-diagonal cosine
        mask = ~np.eye(n_domains, dtype=bool)
        cross_domain_cosine = float(cos_matrix[mask].mean())
        min_cross_domain = float(cos_matrix[mask].min())

        # Overall accuracy for this trait
        total_correct = sum(
            1 for d_data in per_domain.values()
            for p_data in d_data["prompts"]
            if p_data["correct"]
        )
        total_prompts = sum(len(d_data["prompts"]) for d_data in per_domain.values())
        overall_accuracy = total_correct / total_prompts if total_prompts > 0 else 0.0

        # Domain invariance: all domains have 100% accuracy
        domain_invariance = all(
            d_data["accuracy"] == 1.0 for d_data in per_domain.values()
        )

        per_trait[trait] = {
            "per_domain": per_domain,
            "overall_accuracy": float(overall_accuracy),
            "cross_domain_cosine": float(cross_domain_cosine),
            "min_cross_domain_cosine": float(min_cross_domain),
            "domain_invariance": bool(domain_invariance),
        }

        print(f"\n  {trait} summary: overall={overall_accuracy:.0%}, "
              f"cross-domain cos={cross_domain_cosine:.4f}, "
              f"invariant={domain_invariance}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Overall accuracy across all traits and domains
    all_correct = 0
    all_total = 0
    for trait_data in per_trait.values():
        for d_data in trait_data["per_domain"].values():
            for p_data in d_data["prompts"]:
                all_correct += int(p_data["correct"])
                all_total += 1

    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    print(f"  Overall accuracy: {all_correct}/{all_total} ({overall_accuracy:.1%})")

    # Mean cross-domain cosine across traits
    mean_cross_domain = float(np.mean([
        t_data["cross_domain_cosine"] for t_data in per_trait.values()
    ]))
    print(f"  Mean cross-domain cosine: {mean_cross_domain:.4f}")

    # Per-trait summary
    print(f"\n  Per-trait:")
    for trait in TRAITS:
        t_data = per_trait[trait]
        print(f"    {trait:>15}: acc={t_data['overall_accuracy']:.0%}, "
              f"cross-domain={t_data['cross_domain_cosine']:.4f}, "
              f"invariant={t_data['domain_invariance']}")

    # Per-domain summary (averaged across traits)
    print(f"\n  Per-domain (averaged across traits):")
    domain_scores = {}
    for domain in TOPIC_DOMAINS:
        mean_acc = float(np.mean(domain_accuracy_agg[domain]))
        mean_sim = float(np.mean(domain_sim_agg[domain]))
        domain_scores[domain] = mean_acc
        print(f"    {domain:>16}: acc={mean_acc:.0%}, sim={mean_sim:.3f}")

    weakest_domain = min(domain_scores, key=domain_scores.get)
    strongest_domain = max(domain_scores, key=domain_scores.get)
    print(f"\n  Weakest domain:  {weakest_domain} ({domain_scores[weakest_domain]:.0%})")
    print(f"  Strongest domain: {strongest_domain} ({domain_scores[strongest_domain]:.0%})")

    # Invariance count
    invariant_traits = sum(
        1 for t_data in per_trait.values() if t_data["domain_invariance"]
    )
    print(f"  Traits with domain invariance: {invariant_traits}/{len(TRAITS)}")

    # Build conclusion
    if overall_accuracy >= 0.95 and mean_cross_domain >= 0.95:
        conclusion = (
            f"Personality detection is fully robust to topic shifts: "
            f"{overall_accuracy:.1%} accuracy across {len(TOPIC_DOMAINS)} domains, "
            f"cross-domain cosine {mean_cross_domain:.4f}. "
            f"The personality subspace is orthogonal to content/topic."
        )
    elif overall_accuracy >= 0.80:
        conclusion = (
            f"Personality detection is largely robust to topic shifts: "
            f"{overall_accuracy:.1%} accuracy, cross-domain cosine {mean_cross_domain:.4f}. "
            f"Weakest domain: {weakest_domain}."
        )
    else:
        conclusion = (
            f"Personality detection shows topic sensitivity: "
            f"{overall_accuracy:.1%} accuracy, cross-domain cosine {mean_cross_domain:.4f}. "
            f"Weakest domain: {weakest_domain}."
        )

    print(f"\n  Conclusion: {conclusion}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    results = {
        "model": model_id,
        "alpha": alpha,
        "per_trait": per_trait,
        "summary": {
            "overall_accuracy": float(overall_accuracy),
            "mean_cross_domain_cosine": float(mean_cross_domain),
            "weakest_domain": weakest_domain,
            "weakest_domain_accuracy": float(domain_scores[weakest_domain]),
            "strongest_domain": strongest_domain,
            "strongest_domain_accuracy": float(domain_scores[strongest_domain]),
            "invariant_traits": invariant_traits,
            "total_traits": len(TRAITS),
            "total_tests": all_total,
            "total_correct": all_correct,
            "conclusion": conclusion,
        },
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "topic_shift_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
