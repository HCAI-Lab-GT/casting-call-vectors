#!/usr/bin/env python
"""
Generation-based personality verification.

Instead of pairwise forced-choice, test whether the model generates text
that is DETECTABLY personality-specific. For each trait:
1. Generate 5 short responses to a neutral prompt under steering
2. Compute which trait description the generated text is most similar to
   (using the model's own logprob of trait descriptions given the generation)

This is a harder test than pairwise discrimination because the model must
PRODUCE personality-specific text, not just prefer one description.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="gen-classify")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

GENERATION_PROMPTS = [
    "In my free time, I love to",
    "When I face a difficult problem, I tend to",
    "My ideal career would involve",
    "What I value most in life is",
    "If I had a free weekend, I would",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_residual_vectors(model_id, riasec_dir):
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
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer


def generate_steered(model, tokenizer, device, blocks, mid_layer, steer_vec, alpha,
                     prompt, max_new_tokens=60):
    vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
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
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        hook_handle.remove()

    return generated.strip()


def classify_generation(model, tokenizer, device, generated_text):
    """Classify which trait the generated text most matches using logprob scoring."""
    # For each trait, compute logprob of "The person who wrote this is {trait_desc}"
    scores = {}
    for trait, desc in TRAIT_DESCRIPTIONS.items():
        prompt = f'Based on the following text, the person who wrote it is most likely someone who is {desc}.'
        full_text = f'{generated_text}\n\n{prompt}'
        enc = tokenizer(full_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        # Compute logprob of the classification prompt given the generation
        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        # Get logprobs for the classification part
        logits = outputs.logits[0, :-1, :]
        targets = input_ids[0, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Score = mean logprob of the classification suffix
        gen_tokens = tokenizer(generated_text, return_tensors="pt")["input_ids"].shape[1]
        suffix_logprobs = token_log_probs[gen_tokens:]
        scores[trait] = suffix_logprobs.mean().item() if len(suffix_logprobs) > 0 else -100.0

    return scores


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    alphas = [1.0, 3.0]

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)

    # Load model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    print(f"\n{'='*70}")
    print(f"GENERATION-BASED PERSONALITY CLASSIFICATION")
    print(f"Target: Marin 8B")
    print(f"{'='*70}")

    results = {}

    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*70}")

        alpha_results = {}

        for steer_trait in TRAITS:
            logger.info(f"Testing {steer_trait} at α={alpha}...")
            vec = residual[steer_trait].astype(np.float32)

            correct = 0
            total = 0
            generations = []

            for prompt in GENERATION_PROMPTS:
                gen = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                      vec, alpha, prompt)
                scores = classify_generation(model, tokenizer, device, gen)

                classified_trait = max(scores, key=scores.get)
                is_correct = classified_trait == steer_trait
                correct += int(is_correct)
                total += 1

                generations.append({
                    "prompt": prompt,
                    "generation": gen[:150],
                    "scores": {t: float(s) for t, s in scores.items()},
                    "classified_as": classified_trait,
                    "correct": is_correct,
                })

            acc = correct / total
            print(f"\n  {steer_trait:>15}: {acc:.0%} ({correct}/{total})")
            for g in generations:
                mark = "✓" if g["correct"] else "✗"
                print(f"    {g['classified_as'][:4]} {mark} | {g['generation'][:80]}...")

            alpha_results[steer_trait] = {
                "accuracy": float(acc),
                "generations": generations,
            }

        # Summary
        mean_acc = np.mean([alpha_results[t]["accuracy"] for t in TRAITS])
        print(f"\n  α={alpha} OVERALL: {mean_acc:.0%} classification accuracy")

        results[str(alpha)] = alpha_results
        results[f"summary_{alpha}"] = {"mean_accuracy": float(mean_acc)}

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation_classifier.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
