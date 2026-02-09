#!/usr/bin/env python
"""
Behavioral Validation: Does personality steering produce expected text behaviors?

Generate text under each trait's steering and compute automated behavioral metrics:
1. Vocabulary richness (type-token ratio, hapax legomena)
2. Average word length (proxy for complexity)
3. Social language markers (pronouns, social words)
4. Technical language markers (numbers, jargon)
5. Emotional valence (positive/negative word ratios)
6. Sentence structure (avg sentence length, question frequency)

This grounds the abstract 5D geometry in concrete, measurable text differences.
"""

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="behav-val")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Word lists for behavioral markers
SOCIAL_WORDS = {"we", "us", "our", "together", "community", "team", "people", "friend",
                "friends", "family", "help", "share", "care", "support", "collaborate",
                "connect", "relationship", "partner", "group", "social"}
CREATIVE_WORDS = {"creative", "art", "beauty", "imagine", "dream", "inspire", "vision",
                  "aesthetic", "novel", "unique", "expression", "design", "craft", "create",
                  "innovation", "original", "passion", "paint", "music", "poetry"}
TECHNICAL_WORDS = {"data", "analysis", "system", "process", "method", "research",
                   "experiment", "hypothesis", "theory", "evidence", "logical",
                   "scientific", "investigate", "discover", "study", "understand",
                   "knowledge", "information", "technical", "complex"}
PRACTICAL_WORDS = {"build", "fix", "work", "hands", "tool", "machine", "physical",
                   "outdoor", "repair", "construct", "practical", "concrete", "real",
                   "tangible", "manual", "action", "do", "make", "produce", "skill"}
ORGANIZED_WORDS = {"organize", "plan", "schedule", "order", "systematic", "efficient",
                   "detail", "careful", "accurate", "precise", "standard", "rule",
                   "procedure", "routine", "structure", "manage", "coordinate", "reliable"}
AMBITIOUS_WORDS = {"lead", "goal", "achieve", "success", "business", "opportunity",
                   "strategy", "influence", "persuade", "compete", "ambition", "power",
                   "profit", "market", "venture", "entrepreneur", "risk", "reward",
                   "growth", "impact"}

TRAIT_WORD_LISTS = {
    "artistic": CREATIVE_WORDS,
    "conventional": ORGANIZED_WORDS,
    "enterprising": AMBITIOUS_WORDS,
    "investigative": TECHNICAL_WORDS,
    "realistic": PRACTICAL_WORDS,
    "social": SOCIAL_WORDS,
}

GENERATION_PROMPTS = [
    "Tell me about your ideal day.",
    "What kind of activities do you enjoy?",
    "How do you approach problems?",
    "What matters most to you in life?",
    "Describe what you find interesting.",
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


def analyze_text(text):
    """Compute behavioral metrics from generated text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return {"n_words": 0}

    n_words = len(words)
    unique_words = set(words)
    n_unique = len(unique_words)

    # Type-token ratio
    ttr = n_unique / n_words if n_words > 0 else 0

    # Hapax legomena (words appearing exactly once)
    word_counts = Counter(words)
    hapax = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = hapax / n_words if n_words > 0 else 0

    # Average word length
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # Sentence count and avg length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = len(sentences)
    avg_sentence_len = n_words / n_sentences if n_sentences > 0 else 0

    # Question frequency
    n_questions = text.count('?')
    question_freq = n_questions / n_sentences if n_sentences > 0 else 0

    # Social language markers
    social_count = sum(1 for w in words if w in SOCIAL_WORDS)
    social_density = social_count / n_words

    # Creative markers
    creative_count = sum(1 for w in words if w in CREATIVE_WORDS)
    creative_density = creative_count / n_words

    # Technical markers
    technical_count = sum(1 for w in words if w in TECHNICAL_WORDS)
    technical_density = technical_count / n_words

    # Practical markers
    practical_count = sum(1 for w in words if w in PRACTICAL_WORDS)
    practical_density = practical_count / n_words

    # Organized markers
    organized_count = sum(1 for w in words if w in ORGANIZED_WORDS)
    organized_density = organized_count / n_words

    # Ambitious markers
    ambitious_count = sum(1 for w in words if w in AMBITIOUS_WORDS)
    ambitious_density = ambitious_count / n_words

    # Pronoun analysis
    first_person = sum(1 for w in words if w in {"i", "me", "my", "mine", "myself"})
    first_person_plural = sum(1 for w in words if w in {"we", "us", "our", "ours"})
    pronoun_ratio = (first_person + first_person_plural) / n_words

    return {
        "n_words": n_words,
        "n_unique": n_unique,
        "ttr": float(ttr),
        "hapax_ratio": float(hapax_ratio),
        "avg_word_len": float(avg_word_len),
        "avg_sentence_len": float(avg_sentence_len),
        "question_freq": float(question_freq),
        "social_density": float(social_density),
        "creative_density": float(creative_density),
        "technical_density": float(technical_density),
        "practical_density": float(practical_density),
        "organized_density": float(organized_density),
        "ambitious_density": float(ambitious_density),
        "pronoun_ratio": float(pronoun_ratio),
        "first_person_plural_ratio": float(first_person_plural / n_words),
    }


def generate_steered_text(model, tokenizer, device, blocks, mid_layer, vec, alpha, prompt, max_tokens=100):
    """Generate text with personality steering."""
    delta = alpha * torch.tensor(vec, dtype=model.dtype).to(device)
    steer_active = alpha != 0

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    past_kv = None
    current_ids = gen_ids
    generated_ids = []

    for step in range(max_tokens):
        hooks = []
        if steer_active:
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
                if past_kv is not None:
                    outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                else:
                    outputs = model(current_ids, use_cache=True)
        finally:
            for h in hooks:
                h.remove()

        past_kv = outputs.past_key_values
        logits = outputs.logits[0, -1, :]
        next_id = torch.argmax(logits).item()
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break
        current_ids = torch.tensor([[next_id]], device=device)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


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

    alpha = 3.0  # Higher alpha for more visible behavioral effects
    results = {}

    print(f"\n{'='*70}")
    print("BEHAVIORAL VALIDATION OF PERSONALITY STEERING")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Generate text under each trait
    # ================================================================
    logger.info("Part 1: Generating text...")
    print(f"\n{'='*70}")
    print("PART 1: GENERATED TEXT SAMPLES")
    print(f"{'='*70}")

    all_texts = {}
    all_metrics = {}

    for condition in ["baseline"] + TRAITS:
        texts = []
        metrics_list = []

        for prompt in GENERATION_PROMPTS:
            if condition == "baseline":
                text = generate_steered_text(
                    model, tokenizer, device, blocks, mid_layer,
                    np.zeros_like(residual["artistic"]).astype(np.float32), 0, prompt)
            else:
                vec = residual[condition].astype(np.float32)
                text = generate_steered_text(
                    model, tokenizer, device, blocks, mid_layer, vec, alpha, prompt)

            texts.append(text)
            metrics_list.append(analyze_text(text))

        all_texts[condition] = texts
        all_metrics[condition] = metrics_list

        # Print first sample
        print(f"\n  {condition}: {texts[0][:120]}...")

    results["samples"] = {k: v[0][:200] for k, v in all_texts.items()}

    # ================================================================
    # PART 2: Mean behavioral metrics per trait
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: MEAN BEHAVIORAL METRICS")
    print(f"{'='*70}")

    metric_keys = ["ttr", "hapax_ratio", "avg_word_len", "avg_sentence_len",
                   "social_density", "creative_density", "technical_density",
                   "practical_density", "organized_density", "ambitious_density",
                   "pronoun_ratio"]

    mean_metrics = {}
    for condition in ["baseline"] + TRAITS:
        condition_means = {}
        for key in metric_keys:
            vals = [m[key] for m in all_metrics[condition] if m["n_words"] > 0]
            condition_means[key] = float(np.mean(vals)) if vals else 0
        mean_metrics[condition] = condition_means

    # Print comparison table
    print(f"\n  {'Metric':<20} {'Base':>6}", end="")
    for t in TRAITS:
        print(f" {t[:5]:>6}", end="")
    print()
    print(f"  {'-'*20} {'-'*6}", end="")
    for _ in TRAITS:
        print(f" {'-'*6}", end="")
    print()

    for key in metric_keys:
        base_val = mean_metrics["baseline"][key]
        print(f"  {key:<20} {base_val:6.3f}", end="")
        for t in TRAITS:
            val = mean_metrics[t][key]
            delta = val - base_val
            print(f" {delta:+6.3f}", end="")
        print()

    results["mean_metrics"] = mean_metrics

    # ================================================================
    # PART 3: Trait-keyword density match
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 3: TRAIT-KEYWORD DENSITY (expected trait's words vs others)")
    print(f"{'='*70}")

    density_keys = {
        "artistic": "creative_density",
        "conventional": "organized_density",
        "enterprising": "ambitious_density",
        "investigative": "technical_density",
        "realistic": "practical_density",
        "social": "social_density",
    }

    keyword_results = {}
    for trait in TRAITS:
        expected_key = density_keys[trait]
        own_density = mean_metrics[trait][expected_key]
        base_density = mean_metrics["baseline"][expected_key]
        other_densities = [mean_metrics[t][expected_key] for t in TRAITS if t != trait]
        is_highest = own_density == max([mean_metrics[t][expected_key] for t in TRAITS])
        delta = own_density - base_density

        print(f"  {trait:>15}: {expected_key}={own_density:.4f} (base={base_density:.4f}, "
              f"Δ={delta:+.4f}), highest={is_highest}")
        keyword_results[trait] = {
            "own_density": float(own_density),
            "base_density": float(base_density),
            "delta": float(delta),
            "is_highest": bool(is_highest),
        }

    correct_highest = sum(1 for v in keyword_results.values() if v["is_highest"])
    print(f"\n  {correct_highest}/{len(TRAITS)} traits have highest own-keyword density")

    results["keyword_match"] = keyword_results

    # ================================================================
    # PART 4: Cross-trait metric discrimination
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 4: WHICH TRAIT HAS HIGHEST VALUE FOR EACH METRIC?")
    print(f"{'='*70}")

    metric_winners = {}
    for key in metric_keys:
        vals = {t: mean_metrics[t][key] for t in TRAITS}
        winner = max(vals, key=vals.get)
        print(f"  {key:<20}: {winner} ({vals[winner]:.4f})")
        metric_winners[key] = {"winner": winner, "value": float(vals[winner])}

    results["metric_winners"] = metric_winners

    # ================================================================
    # PART 5: Holland opposite behavioral contrasts
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 5: HOLLAND OPPOSITE BEHAVIORAL CONTRASTS")
    print(f"{'='*70}")

    holland_pairs = [("artistic", "conventional"), ("investigative", "enterprising"),
                     ("realistic", "social")]

    holland_results = {}
    for t1, t2 in holland_pairs:
        contrasts = {}
        print(f"\n  {t1} vs {t2}:")
        for key in metric_keys:
            v1 = mean_metrics[t1][key]
            v2 = mean_metrics[t2][key]
            diff = v1 - v2
            if abs(diff) > 0.001:
                print(f"    {key:<20}: {v1:.3f} vs {v2:.3f} (diff={diff:+.3f})")
            contrasts[key] = {"t1": float(v1), "t2": float(v2), "diff": float(diff)}
        holland_results[f"{t1}_vs_{t2}"] = contrasts

    results["holland_contrasts"] = holland_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Keyword match: {correct_highest}/{len(TRAITS)} traits have highest own-keyword density")

    # Overall behavioral consistency score
    expected_patterns = {
        "artistic": {"creative_density": "+", "avg_word_len": "+"},
        "conventional": {"organized_density": "+", "avg_sentence_len": "-"},
        "investigative": {"technical_density": "+", "avg_word_len": "+"},
        "social": {"social_density": "+", "first_person_plural_ratio": "+"},
        "realistic": {"practical_density": "+", "avg_sentence_len": "-"},
        "enterprising": {"ambitious_density": "+"},
    }

    correct = 0
    total = 0
    for trait, patterns in expected_patterns.items():
        for metric_key, direction in patterns.items():
            base_val = mean_metrics["baseline"].get(metric_key, 0)
            trait_val = mean_metrics[trait].get(metric_key, 0)
            delta = trait_val - base_val
            if direction == "+":
                match = delta > 0
            else:
                match = delta < 0
            if match:
                correct += 1
            total += 1

    print(f"  Expected behavioral patterns: {correct}/{total} match")

    results["behavioral_match"] = {"correct": correct, "total": total}

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "behavioral_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
