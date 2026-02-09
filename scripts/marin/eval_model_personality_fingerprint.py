#!/usr/bin/env python
"""
Model Personality Fingerprinting.

Measures each model's "natural personality" — what RIASEC profile does each
model exhibit WITHOUT any steering, based on its baseline responses?

Tests:
1. Baseline RIASEC profile for each instruct model (forced-choice preferences)
2. Project baseline activation patterns onto each model's own 5D space
3. Compare natural personalities across models
4. Test if instruction-tuning creates consistent personality shifts
   (compare base vs instruct on SmolLM3)

This reveals whether different models have distinct "personality fingerprints"
that arise from their training data and RLHF alignment.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="model-fingerprint")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

# Personality-probing prompts (diverse set)
PROBE_PROMPTS = [
    "What kind of activities do you enjoy most?",
    "How do you approach solving difficult problems?",
    "What values are most important to you?",
    "Describe your ideal work environment.",
    "What motivates you to do your best work?",
    "How do you prefer to spend your free time?",
    "What skills do you consider your greatest strengths?",
    "How do you feel about working in teams vs independently?",
]


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
        if not path.exists():
            return None
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

    V_res = np.stack([residual[t] for t in TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
        "num_layers": config.num_hidden_layers,
        "hidden_dim": config.hidden_size,
    }


def measure_baseline_profile(model, tokenizer, device):
    """Measure raw logprob preferences without any steering."""
    raw_logprobs = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [
                {"role": "user", "content":
                    f"Which describes you better? Answer with just A or B.\n"
                    f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                    f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}"},
            ]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # For models without chat template, use plain text
                formatted = (f"Which describes you better? Answer with just A or B.\n"
                           f"A) I am {TRAIT_DESCRIPTIONS[trait_a]}\n"
                           f"B) I am {TRAIT_DESCRIPTIONS[trait_b]}\n"
                           f"Answer:")
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            a_ids = tokenizer.encode("A", add_special_tokens=False)
            b_ids = tokenizer.encode("B", add_special_tokens=False)
            gap = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()
            raw_logprobs[f"{trait_a}-{trait_b}"] = gap

    # Convert to per-trait scores
    trait_scores = {t: 0.0 for t in TRAITS}
    trait_counts = {t: 0 for t in TRAITS}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            key = f"{trait_a}-{trait_b}"
            gap = raw_logprobs[key]
            trait_scores[trait_a] += gap
            trait_counts[trait_a] += 1
            trait_scores[trait_b] -= gap
            trait_counts[trait_b] += 1
    for t in TRAITS:
        if trait_counts[t] > 0:
            trait_scores[t] /= trait_counts[t]

    return trait_scores, raw_logprobs


def main():
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Models to fingerprint (each on a different GPU)
    models_to_test = [
        ("HuggingFaceTB/SmolLM3-3B", "cuda:0"),
        ("meta-llama/Llama-3.2-1B-Instruct", "cuda:0"),
        ("Qwen/Qwen2.5-7B-Instruct", "cuda:0"),
        ("marin-community/marin-8b-instruct", "cuda:0"),
    ]

    results = {}
    all_profiles = {}

    print(f"\n{'='*70}")
    print(f"MODEL PERSONALITY FINGERPRINTING")
    print(f"{'='*70}")

    for model_id, device in models_to_test:
        logger.info(f"Loading {model_id}...")
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"Device: {device}")
        print(f"{'='*70}")

        # Load model data (vectors + 5D basis)
        model_data = load_model_data(model_id, riasec_dir)

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device)
        model.eval()

        # Measure baseline personality profile
        logger.info(f"  Measuring baseline personality...")
        trait_scores, raw_logprobs = measure_baseline_profile(model, tokenizer, device)

        # Sort traits by score
        sorted_traits = sorted(trait_scores.items(), key=lambda x: -x[1])
        dominant_trait = sorted_traits[0][0]
        weakest_trait = sorted_traits[-1][0]
        profile_magnitude = float(np.sqrt(sum(v**2 for v in trait_scores.values())))

        print(f"\n  Baseline RIASEC profile:")
        for t, s in sorted_traits:
            bar = "+" * max(0, int(s * 2)) if s > 0 else "-" * max(0, int(-s * 2))
            print(f"    {t:>15}: {s:>+8.3f} {bar}")
        print(f"\n  Dominant: {dominant_trait} ({trait_scores[dominant_trait]:+.3f})")
        print(f"  Weakest:  {weakest_trait} ({trait_scores[weakest_trait]:+.3f})")
        print(f"  Profile magnitude: {profile_magnitude:.3f}")

        # Pairwise preference matrix
        print(f"\n  Pairwise A>B logprob gaps (positive = prefers A):")
        for key, gap in sorted(raw_logprobs.items()):
            print(f"    {key}: {gap:+.3f}")

        model_result = {
            "model_id": model_id,
            "trait_scores": {t: float(v) for t, v in trait_scores.items()},
            "dominant_trait": dominant_trait,
            "weakest_trait": weakest_trait,
            "profile_magnitude": profile_magnitude,
            "raw_logprobs": {k: float(v) for k, v in raw_logprobs.items()},
            "sorted_traits": [(t, float(s)) for t, s in sorted_traits],
        }

        if model_data:
            model_result["num_layers"] = model_data["num_layers"]
            model_result["hidden_dim"] = model_data["hidden_dim"]

        results[model_id] = model_result
        all_profiles[model_id] = np.array([trait_scores[t] for t in TRAITS])

        # Clean up GPU
        del model
        torch.cuda.empty_cache()

    # ================================================================
    # CROSS-MODEL COMPARISON
    # ================================================================
    print(f"\n{'='*70}")
    print("CROSS-MODEL PERSONALITY COMPARISON")
    print(f"{'='*70}")

    model_ids = list(all_profiles.keys())

    # Profile correlation matrix
    print(f"\n  Profile correlation matrix:")
    print(f"  {'':>35}", end="")
    for mid in model_ids:
        short = mid.split("/")[-1][:10]
        print(f" {short:>10}", end="")
    print()

    corr_matrix = {}
    for i, mid_a in enumerate(model_ids):
        short_a = mid_a.split("/")[-1][:10]
        print(f"  {mid_a.split('/')[-1]:>35}", end="")
        for j, mid_b in enumerate(model_ids):
            prof_a = all_profiles[mid_a]
            prof_b = all_profiles[mid_b]
            if np.std(prof_a) > 0 and np.std(prof_b) > 0:
                corr = float(np.corrcoef(prof_a, prof_b)[0, 1])
            else:
                corr = 0
            print(f" {corr:>+10.3f}", end="")
            corr_matrix[f"{mid_a}__vs__{mid_b}"] = corr
        print()

    # Dominant trait comparison
    print(f"\n  Dominant traits across models:")
    for mid in model_ids:
        r = results[mid]
        print(f"    {mid.split('/')[-1]:>30}: {r['dominant_trait']:>15} ({r['trait_scores'][r['dominant_trait']]:+.3f})")

    # Is there a universal "LLM personality"?
    mean_profile = np.mean([all_profiles[m] for m in model_ids], axis=0)
    print(f"\n  Mean profile across all models:")
    for i, t in enumerate(TRAITS):
        print(f"    {t:>15}: {mean_profile[i]:+.3f}")

    mean_dominant = TRAITS[int(np.argmax(mean_profile))]
    mean_weakest = TRAITS[int(np.argmin(mean_profile))]
    print(f"\n  Universal LLM personality tendency: {mean_dominant} > {mean_weakest}")

    # Profile variance — which traits vary most across models?
    std_profile = np.std([all_profiles[m] for m in model_ids], axis=0)
    print(f"\n  Cross-model variance per trait:")
    for i, t in enumerate(TRAITS):
        print(f"    {t:>15}: std={std_profile[i]:.3f}")

    most_variable = TRAITS[int(np.argmax(std_profile))]
    least_variable = TRAITS[int(np.argmin(std_profile))]
    print(f"  Most variable: {most_variable}, Least variable: {least_variable}")

    results["cross_model"] = {
        "correlation_matrix": corr_matrix,
        "mean_profile": {t: float(mean_profile[i]) for i, t in enumerate(TRAITS)},
        "std_profile": {t: float(std_profile[i]) for i, t in enumerate(TRAITS)},
        "universal_dominant": mean_dominant,
        "universal_weakest": mean_weakest,
        "most_variable_trait": most_variable,
        "least_variable_trait": least_variable,
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Do all models share the same dominant trait?
    dominant_traits = [results[m]["dominant_trait"] for m in model_ids]
    all_same = len(set(dominant_traits)) == 1

    # Mean pairwise correlation
    off_diag_corrs = []
    for i, mid_a in enumerate(model_ids):
        for j, mid_b in enumerate(model_ids):
            if i < j:
                off_diag_corrs.append(corr_matrix[f"{mid_a}__vs__{mid_b}"])
    mean_corr = np.mean(off_diag_corrs) if off_diag_corrs else 0

    print(f"\n  All models share dominant trait: {all_same} ({set(dominant_traits)})")
    print(f"  Mean pairwise profile correlation: {mean_corr:.3f}")
    print(f"  Universal tendency: {mean_dominant}")

    results["summary"] = {
        "all_share_dominant": bool(all_same),
        "dominant_traits": {m.split("/")[-1]: results[m]["dominant_trait"] for m in model_ids},
        "mean_pairwise_correlation": float(mean_corr),
        "universal_dominant": mean_dominant,
        "universal_weakest": mean_weakest,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_personality_fingerprint.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
