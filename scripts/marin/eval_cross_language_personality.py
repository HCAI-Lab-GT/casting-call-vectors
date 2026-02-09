#!/usr/bin/env python
"""
Cross-Language Personality Steering: Does personality survive language change?

Previous findings show personality is universal across MODEL ARCHITECTURES.
But is it universal across LANGUAGES? If personality vectors extracted from
English data steer the model correctly when the prompt is in Spanish, Chinese,
French, Arabic, etc., then personality is truly a language-agnostic representation.

Tests:
1. Detection accuracy per language (6 traits × 8 languages)
2. Cosine consistency: does the 5D fingerprint change with language?
3. Cross-language cosine similarity matrix
4. Language families: do related languages cluster?
5. Code-switching: mixed-language prompts
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-lang")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Prompts in multiple languages, all semantically equivalent to "Tell me about yourself"
LANGUAGE_PROMPTS = {
    "english": "Tell me about yourself.",
    "spanish": "Cuéntame sobre ti.",
    "french": "Parlez-moi de vous.",
    "german": "Erzählen Sie mir von sich.",
    "chinese": "请介绍一下你自己。",
    "japanese": "自己紹介をしてください。",
    "arabic": "أخبرني عن نفسك.",
    "korean": "자기소개를 해주세요.",
    "portuguese": "Conte-me sobre você.",
    "italian": "Parlami di te.",
    "russian": "Расскажите о себе.",
    "hindi": "अपने बारे में बताइए।",
}

# Additional prompts for robustness testing (diverse topics per language)
DIVERSE_PROMPTS = {
    "english": [
        "What do you enjoy doing in your free time?",
        "Describe your ideal work environment.",
        "What motivates you most in life?",
    ],
    "spanish": [
        "¿Qué disfrutas hacer en tu tiempo libre?",
        "Describe tu entorno de trabajo ideal.",
        "¿Qué te motiva más en la vida?",
    ],
    "french": [
        "Qu'aimez-vous faire pendant votre temps libre?",
        "Décrivez votre environnement de travail idéal.",
        "Qu'est-ce qui vous motive le plus dans la vie?",
    ],
    "chinese": [
        "你空闲时间喜欢做什么？",
        "描述一下你理想的工作环境。",
        "生活中什么最能激励你？",
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


def steer_and_detect(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                      steer_vec, alpha, prompt):
    """Steer with a trait vector and detect personality from activation diff."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Baseline
    captured_base = {}
    hooks = []

    def cap_base(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_base["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    # Steered
    captured_steer = {}
    hooks = []

    def cap_steer(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_steer["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_steer))

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

    diff = (captured_steer["act"] - captured_base["act"]).astype(np.float64)
    coords = basis_5d @ diff
    norm_5d = float(np.linalg.norm(coords))
    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0
    detected = max(sims, key=sims.get)
    return {"detected": detected, "cos": sims, "norm": norm_5d, "coords_5d": coords.tolist()}


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

    alpha = 2.0
    results = {}

    print(f"\n{'='*70}")
    print("CROSS-LANGUAGE PERSONALITY STEERING")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Detection accuracy per language
    # ================================================================
    logger.info("Part 1: Per-language detection...")
    print(f"\n{'='*70}")
    print("PART 1: DETECTION ACCURACY PER LANGUAGE")
    print(f"{'='*70}")

    per_lang_results = {}
    lang_accuracy = {}

    for lang, prompt in LANGUAGE_PROMPTS.items():
        correct = 0
        total = 0
        lang_data = {}

        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, vec, alpha, prompt)
            is_correct = res["detected"] == trait
            correct += int(is_correct)
            total += 1
            lang_data[trait] = {
                "detected": res["detected"],
                "correct": is_correct,
                "cos_self": res["cos"][trait],
                "coords_5d": res["coords_5d"],
                "norm": res["norm"],
            }

        accuracy = correct / total
        lang_accuracy[lang] = accuracy
        per_lang_results[lang] = lang_data
        print(f"  {lang:>12}: {correct}/{total} ({accuracy:.0%})"
              + (" " + "".join("✓" if lang_data[t]["correct"] else "✗" for t in TRAITS)))

    results["per_language"] = per_lang_results
    results["language_accuracy"] = {k: float(v) for k, v in lang_accuracy.items()}

    # Overall accuracy
    total_correct = sum(
        1 for lang_data in per_lang_results.values()
        for t_data in lang_data.values() if t_data["correct"]
    )
    total_tests = sum(len(lang_data) for lang_data in per_lang_results.values())
    print(f"\n  Overall: {total_correct}/{total_tests} ({total_correct/total_tests:.1%})")

    # ================================================================
    # PART 2: Cross-language cosine consistency
    # ================================================================
    logger.info("Part 2: Cross-language cosine consistency...")
    print(f"\n{'='*70}")
    print("PART 2: CROSS-LANGUAGE 5D COSINE CONSISTENCY")
    print(f"{'='*70}")

    # For each trait, compute 5D coords across all languages and check consistency
    trait_cross_lang = {}
    for trait in TRAITS:
        coords_per_lang = {}
        for lang in LANGUAGE_PROMPTS:
            c = np.array(per_lang_results[lang][trait]["coords_5d"])
            coords_per_lang[lang] = c

        # Pairwise cosine between languages for this trait
        langs = list(LANGUAGE_PROMPTS.keys())
        cos_matrix = np.zeros((len(langs), len(langs)))
        for i, l1 in enumerate(langs):
            for j, l2 in enumerate(langs):
                c1 = coords_per_lang[l1]
                c2 = coords_per_lang[l2]
                n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
                if n1 > 0 and n2 > 0:
                    cos_matrix[i, j] = np.dot(c1, c2) / (n1 * n2)

        # Mean off-diagonal cosine
        mask = ~np.eye(len(langs), dtype=bool)
        mean_cos = float(cos_matrix[mask].mean())
        min_cos = float(cos_matrix[mask].min())

        trait_cross_lang[trait] = {
            "mean_cross_lang_cos": mean_cos,
            "min_cross_lang_cos": min_cos,
        }
        print(f"  {trait:>15}: mean cross-lang cos = {mean_cos:.4f}, min = {min_cos:.4f}")

    results["cross_language_consistency"] = trait_cross_lang

    # ================================================================
    # PART 3: Language family clustering
    # ================================================================
    logger.info("Part 3: Language family analysis...")
    print(f"\n{'='*70}")
    print("PART 3: LANGUAGE FAMILY CLUSTERING")
    print(f"{'='*70}")

    # Group languages by family
    families = {
        "Romance": ["spanish", "french", "portuguese", "italian"],
        "Germanic": ["english", "german"],
        "CJK": ["chinese", "japanese", "korean"],
        "Indo-Aryan": ["hindi"],
        "Semitic": ["arabic"],
        "Slavic": ["russian"],
    }

    # Compute average 5D coords per language (across all traits)
    lang_avg_coords = {}
    for lang in LANGUAGE_PROMPTS:
        all_coords = [np.array(per_lang_results[lang][t]["coords_5d"]) for t in TRAITS]
        lang_avg_coords[lang] = np.mean(all_coords, axis=0)

    # Within-family vs between-family cosine
    within_cos = []
    between_cos = []

    all_langs = list(LANGUAGE_PROMPTS.keys())
    for i, l1 in enumerate(all_langs):
        for j, l2 in enumerate(all_langs):
            if i >= j:
                continue
            # Find families
            f1 = [f for f, members in families.items() if l1 in members]
            f2 = [f for f, members in families.items() if l2 in members]
            if not f1 or not f2:
                continue

            # Average cosine across all traits
            cos_vals = []
            for trait in TRAITS:
                c1 = np.array(per_lang_results[l1][trait]["coords_5d"])
                c2 = np.array(per_lang_results[l2][trait]["coords_5d"])
                n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
                if n1 > 0 and n2 > 0:
                    cos_vals.append(np.dot(c1, c2) / (n1 * n2))

            avg_cos = float(np.mean(cos_vals))
            if f1[0] == f2[0]:
                within_cos.append(avg_cos)
                print(f"  Within {f1[0]:>10}: {l1:>10}-{l2:<10} cos={avg_cos:.4f}")
            else:
                between_cos.append(avg_cos)

    if within_cos and between_cos:
        mean_within = np.mean(within_cos)
        mean_between = np.mean(between_cos)
        print(f"\n  Mean within-family cos:  {mean_within:.4f}")
        print(f"  Mean between-family cos: {mean_between:.4f}")
        print(f"  Family effect: {mean_within - mean_between:+.4f}")
        results["family_clustering"] = {
            "within_family_cos": float(mean_within),
            "between_family_cos": float(mean_between),
            "family_effect": float(mean_within - mean_between),
        }
    else:
        print("  Insufficient data for family comparison")

    # ================================================================
    # PART 4: Diverse prompts per language (robustness)
    # ================================================================
    logger.info("Part 4: Diverse prompts per language...")
    print(f"\n{'='*70}")
    print("PART 4: DIVERSE PROMPTS PER LANGUAGE")
    print(f"{'='*70}")

    diverse_results = {}
    for lang, prompts in DIVERSE_PROMPTS.items():
        correct = 0
        total = 0
        for prompt in prompts:
            for trait in TRAITS:
                vec = residual[trait].astype(np.float32)
                res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                        basis_5d, coords_5d, vec, alpha, prompt)
                if res["detected"] == trait:
                    correct += 1
                total += 1
        accuracy = correct / total if total > 0 else 0
        diverse_results[lang] = {"correct": correct, "total": total, "accuracy": float(accuracy)}
        print(f"  {lang:>12}: {correct}/{total} ({accuracy:.0%})")

    results["diverse_prompts"] = diverse_results

    # ================================================================
    # PART 5: Code-switching prompts
    # ================================================================
    logger.info("Part 5: Code-switching prompts...")
    print(f"\n{'='*70}")
    print("PART 5: CODE-SWITCHING (MIXED LANGUAGE PROMPTS)")
    print(f"{'='*70}")

    code_switch_prompts = [
        ("en-es", "Tell me about yourself. Me puedes responder en español."),
        ("en-fr", "Tell me about yourself. Répondez en français."),
        ("en-zh", "Tell me about yourself. 请用中文回答。"),
        ("es-en", "Cuéntame sobre ti. Please respond in English."),
        ("zh-en", "请介绍一下你自己。Please respond in English."),
        ("fr-de", "Parlez-moi de vous. Bitte antworten Sie auf Deutsch."),
    ]

    cs_results = {}
    for label, prompt in code_switch_prompts:
        correct = 0
        total = 0
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, vec, alpha, prompt)
            if res["detected"] == trait:
                correct += 1
            total += 1
        accuracy = correct / total
        cs_results[label] = {"correct": correct, "total": total, "accuracy": float(accuracy)}
        print(f"  {label:>8}: {correct}/{total} ({accuracy:.0%})")

    results["code_switching"] = cs_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    mean_acc = np.mean(list(lang_accuracy.values()))
    mean_consistency = np.mean([v["mean_cross_lang_cos"] for v in trait_cross_lang.values()])
    print(f"  Mean per-language accuracy: {mean_acc:.1%}")
    print(f"  Mean cross-language 5D consistency: {mean_consistency:.4f}")
    print(f"  Languages with 100%: {sum(1 for v in lang_accuracy.values() if v == 1.0)}/{len(lang_accuracy)}")
    perfect_langs = [l for l, a in lang_accuracy.items() if a == 1.0]
    if perfect_langs:
        print(f"  Perfect languages: {', '.join(perfect_langs)}")

    results["summary"] = {
        "mean_accuracy": float(mean_acc),
        "mean_consistency": float(mean_consistency),
        "perfect_languages": sum(1 for v in lang_accuracy.values() if v == 1.0),
        "total_languages": len(lang_accuracy),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_language_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
