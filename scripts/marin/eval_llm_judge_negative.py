#!/usr/bin/env python
"""
LLM-judge evaluation of NEGATIVE steering.

For each trait, generate text with -alpha steering (suppression),
then have Marin 8B judge what personality the text reflects.

Key question: Does the judge identify the OPPOSITE trait on Holland's hexagon?
E.g., steering with -artistic → should the judge say "conventional" (Holland opposite)?

Also tests: is the negative-steered text judged as ANYTHING BUT the suppressed trait?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="llm-judge-neg")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_LABELS = {
    "artistic": "Artistic (creative, expressive, values beauty and self-expression)",
    "conventional": "Conventional (organized, detail-oriented, values order and rules)",
    "enterprising": "Enterprising (ambitious, persuasive, values leadership and influence)",
    "investigative": "Investigative (analytical, curious, values knowledge and discovery)",
    "realistic": "Realistic (practical, hands-on, values tangible results and physical work)",
    "social": "Social (helpful, empathetic, values relationships and community)",
}

HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
HOLLAND_OPPOSITES = {
    "realistic": "social",
    "investigative": "enterprising",
    "artistic": "conventional",
    "social": "realistic",
    "enterprising": "investigative",
    "conventional": "artistic",
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
    judge_prompt = (
        f"Read the following text and determine which Holland RIASEC personality type "
        f"it most strongly reflects.\n\n"
        f"Text: \"{text}\"\n\n"
        f"Choose EXACTLY ONE:\n"
    )
    for i, (trait, label) in enumerate(TRAIT_LABELS.items()):
        letter = chr(65 + i)
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

    trait_probs = {}
    for i, trait in enumerate(TRAITS):
        letter = chr(65 + i)
        candidates = [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]
        best_lp = max(
            log_probs[tokenizer.encode(c, add_special_tokens=False)[0]].item()
            for c in candidates if tokenizer.encode(c, add_special_tokens=False)
        )
        trait_probs[trait] = best_lp

    top_trait = max(trait_probs, key=trait_probs.get)
    ranked = sorted(trait_probs.items(), key=lambda x: -x[1])

    return {
        "top_choice": top_trait,
        "log_probs": {k: float(v) for k, v in trait_probs.items()},
        "ranking": [t for t, _ in ranked],
    }


def main():
    gen_model_id = "HuggingFaceTB/SmolLM3-3B"
    judge_model_id = "marin-community/marin-8b-instruct"
    device = "cuda:0"
    alpha = 3.0

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

    # Step 1: Generate negatively steered text
    logger.info("Loading generation model: %s", gen_model_id)
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    gen_model.eval()
    gen_blocks = get_decoder_blocks(gen_model)

    prompt = "In my free time, I love to"

    print(f"\n{'='*70}")
    print(f"NEGATIVE STEERING + LLM JUDGE")
    print(f"Generator: {gen_model_id}, Alpha: ±{alpha}")
    print(f"Judge: {judge_model_id}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*70}")

    generated = {}

    for trait in TRAITS:
        # Positive steering
        pos_vec = alpha * residual_vectors[trait]
        pos_text = generate_steered(gen_model, gen_tokenizer, device, gen_blocks,
                                    gen_mid_layer, pos_vec, prompt)[:500]

        # Negative steering
        neg_vec = -alpha * residual_vectors[trait]
        neg_text = generate_steered(gen_model, gen_tokenizer, device, gen_blocks,
                                    gen_mid_layer, neg_vec, prompt)[:500]

        generated[trait] = {"positive": pos_text, "negative": neg_text}

        print(f"\n  {trait.upper()}:")
        print(f"    +{alpha}: {pos_text[:120]}...")
        print(f"    -{alpha}: {neg_text[:120]}...")

    # Free generation model
    del gen_model, gen_blocks
    torch.cuda.empty_cache()

    # Step 2: Judge
    logger.info("Loading judge: %s", judge_model_id)
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    judge_model.eval()

    print(f"\n{'='*70}")
    print(f"JUDGMENTS")
    print(f"{'='*70}")

    results = {"generated": generated, "judgments": {}}

    pos_correct = 0
    neg_avoids_suppressed = 0
    neg_is_opposite = 0
    total = 0

    print(f"\n  {'Steer':>14}  {'Dir':>4}  {'Judge says':>14}  {'Expected':>14}  {'Result':>8}")
    print(f"  {'-'*64}")

    for trait in TRAITS:
        for direction, text in [("pos", generated[trait]["positive"]),
                                ("neg", generated[trait]["negative"])]:
            judgment = judge_personality(judge_model, judge_tokenizer, device, text)
            judged = judgment["top_choice"]

            if direction == "pos":
                expected = trait
                result = "OK" if judged == trait else f"!={judged}"
                pos_correct += int(judged == trait)
            else:
                expected = HOLLAND_OPPOSITES[trait]
                avoids = judged != trait
                is_opp = judged == HOLLAND_OPPOSITES[trait]
                neg_avoids_suppressed += int(avoids)
                neg_is_opposite += int(is_opp)
                result = f"avoid:{'OK' if avoids else 'FAIL'}"
                if is_opp:
                    result += " opp:OK"

            total += 1
            results["judgments"][f"{trait}_{direction}"] = {
                "text_preview": text[:200],
                "judgment": judgment,
                "direction": direction,
            }

            sign = f"+{alpha}" if direction == "pos" else f"-{alpha}"
            print(f"  {trait:>14}  {sign:>4}  {judged:>14}  {expected:>14}  {result}")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Positive steering top-1:                  {pos_correct}/6 ({pos_correct/6:.0%})")
    print(f"  Negative steering avoids suppressed trait: {neg_avoids_suppressed}/6 ({neg_avoids_suppressed/6:.0%})")
    print(f"  Negative steering → Holland opposite:      {neg_is_opposite}/6 ({neg_is_opposite/6:.0%})")
    print(f"  Chance (avoid):    83%")
    print(f"  Chance (opposite): 17%")

    results["summary"] = {
        "pos_top1": pos_correct,
        "neg_avoids": neg_avoids_suppressed,
        "neg_is_opposite": neg_is_opposite,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_judge_negative.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
