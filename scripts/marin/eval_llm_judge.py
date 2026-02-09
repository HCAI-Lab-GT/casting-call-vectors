#!/usr/bin/env python
"""
LLM-as-Judge personality evaluation.

Pipeline:
1. Generate text from SmolLM3-3B under each of 6 RIASEC trait steerings
2. Have Marin 8B (different architecture, unsteered) judge which
   RIASEC trait each text reflects

If the judge correctly identifies steered traits, this provides
independent validation that the personality steering produces
genuinely perceptible trait expression — not just logprob shifts.

Tests multiple prompts and alpha levels.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="llm-judge")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_LABELS = {
    "artistic": "Artistic (creative, expressive, values beauty and self-expression)",
    "conventional": "Conventional (organized, detail-oriented, values order and rules)",
    "enterprising": "Enterprising (ambitious, persuasive, values leadership and influence)",
    "investigative": "Investigative (analytical, curious, values knowledge and discovery)",
    "realistic": "Realistic (practical, hands-on, values tangible results and physical work)",
    "social": "Social (helpful, empathetic, values relationships and community)",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def generate_steered(model, tokenizer, device, blocks, layer, vector, prompt, max_new_tokens=200):
    vec_t = torch.tensor(vector, dtype=torch.float16).unsqueeze(0).to(device)

    def make_hook(d):
        def hook_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += d
                return (hs,) + out[1:]
            out[:, -1, :] += d
            return out
        return hook_fn

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    hook_handle = blocks[layer].register_forward_hook(make_hook(vec_t))
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        hook_handle.remove()

    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def judge_personality(model, tokenizer, device, text):
    """Have the judge model identify which RIASEC trait a text reflects."""
    judge_prompt = (
        f"Read the following text and determine which Holland RIASEC personality type "
        f"it most strongly reflects.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Choose EXACTLY ONE:\n"
    )
    for i, (trait, label) in enumerate(TRAIT_LABELS.items()):
        letter = chr(65 + i)  # A, B, C, D, E, F
        judge_prompt += f"{letter}) {label}\n"
    judge_prompt += "\nAnswer with EXACTLY one letter (A-F):"

    messages = [
        {"role": "system", "content": "You are an expert personality psychologist. Analyze the text and identify the RIASEC personality type it reflects. Answer with exactly one letter."},
        {"role": "user", "content": judge_prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get probabilities for each option letter
    trait_probs = {}
    for i, trait in enumerate(TRAITS):
        letter = chr(65 + i)
        candidates = [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]
        best_lp = max(
            log_probs[tokenizer.encode(c, add_special_tokens=False)[0]].item()
            for c in candidates if tokenizer.encode(c, add_special_tokens=False)
        )
        trait_probs[trait] = best_lp

    # Top choice
    top_trait = max(trait_probs, key=trait_probs.get)
    # Ranking
    ranked = sorted(trait_probs.items(), key=lambda x: -x[1])

    return {
        "top_choice": top_trait,
        "log_probs": {k: float(v) for k, v in trait_probs.items()},
        "ranking": [t for t, _ in ranked],
    }


def main():
    gen_model_id = "HuggingFaceTB/SmolLM3-3B"
    judge_model_id = "marin-community/marin-8b-instruct"
    gen_device = "cuda:0"
    judge_device = "cuda:0"  # Will load sequentially to save memory

    alpha = 3.0  # Higher alpha for more perceptible personality

    # Load generation model vectors
    gen_config = AutoConfig.from_pretrained(gen_model_id)
    gen_mid_layer = gen_config.num_hidden_layers // 2
    gen_safe = gen_model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{gen_safe}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    V = np.stack([all_layer_vectors[t][gen_mid_layer + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual_vectors = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][gen_mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    # Step 1: Generate steered text
    logger.info("Loading generation model: %s", gen_model_id)
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_id,
        torch_dtype=torch.float16,
        device_map=gen_device,
    )
    gen_model.eval()
    gen_blocks = get_decoder_blocks(gen_model)

    prompts = [
        "In my free time, I love to",
        "My ideal career would involve",
        "When I face a difficult problem, I tend to",
    ]

    generated_texts = {}  # {prompt: {trait: text}}

    print(f"\n{'='*70}")
    print(f"STEP 1: Generate steered text")
    print(f"Generator: {gen_model_id}, Alpha: {alpha}")
    print(f"{'='*70}")

    for prompt in prompts:
        generated_texts[prompt] = {}
        print(f"\nPrompt: '{prompt}'")
        for trait in TRAITS:
            vec = alpha * residual_vectors[trait]
            text = generate_steered(gen_model, gen_tokenizer, gen_device, gen_blocks,
                                   gen_mid_layer, vec, prompt)
            text = text[:500]  # Truncate to reasonable length
            generated_texts[prompt][trait] = text
            print(f"  {trait:>14}: {text[:100]}...")

    # Free generation model memory
    del gen_model
    del gen_blocks
    torch.cuda.empty_cache()

    # Step 2: Judge with Marin 8B
    logger.info("Loading judge model: %s", judge_model_id)
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.float16,
        device_map=judge_device,
    )
    judge_model.eval()

    print(f"\n{'='*70}")
    print(f"STEP 2: Judge personality")
    print(f"Judge: {judge_model_id}")
    print(f"{'='*70}")

    results = {
        "gen_model": gen_model_id,
        "judge_model": judge_model_id,
        "alpha": alpha,
        "generated_texts": generated_texts,
        "judgments": {},
    }

    total_correct = 0
    total_top2 = 0
    total_tests = 0

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        results["judgments"][prompt] = {}

        for steer_trait in TRAITS:
            text = generated_texts[prompt][steer_trait]
            judgment = judge_personality(judge_model, judge_tokenizer, judge_device, text)

            correct = judgment["top_choice"] == steer_trait
            in_top2 = steer_trait in judgment["ranking"][:2]

            total_correct += int(correct)
            total_top2 += int(in_top2)
            total_tests += 1

            mark = "OK" if correct else f"!={judgment['top_choice']}"
            results["judgments"][prompt][steer_trait] = judgment

            print(f"  Steered: {steer_trait:>14} → Judge: {judgment['top_choice']:>14} [{mark}] "
                  f"(top-3: {', '.join(judgment['ranking'][:3])})")

    # Summary
    top1_acc = total_correct / total_tests
    top2_acc = total_top2 / total_tests

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Top-1 accuracy: {top1_acc:.0%} ({total_correct}/{total_tests})")
    print(f"  Top-2 accuracy: {top2_acc:.0%} ({total_top2}/{total_tests})")
    print(f"  Chance (top-1): 16.7%")
    print(f"  Chance (top-2): 33.3%")

    # Per-trait breakdown
    print(f"\n--- Per-trait accuracy ---")
    for trait in TRAITS:
        correct_for_trait = sum(
            1 for prompt in prompts
            if results["judgments"][prompt][trait]["top_choice"] == trait
        )
        print(f"  {trait:>14}: {correct_for_trait}/{len(prompts)}")

    # Confusion patterns
    print(f"\n--- Most common confusions ---")
    confusion = {}
    for prompt in prompts:
        for steer_trait in TRAITS:
            judged = results["judgments"][prompt][steer_trait]["top_choice"]
            if judged != steer_trait:
                key = f"{steer_trait}→{judged}"
                confusion[key] = confusion.get(key, 0) + 1

    for key, count in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    results["summary"] = {
        "top1_accuracy": float(top1_acc),
        "top2_accuracy": float(top2_acc),
        "total_correct": total_correct,
        "total_tests": total_tests,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_judge_cross_model.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
