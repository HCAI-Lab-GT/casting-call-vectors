#!/usr/bin/env python
"""
Generation-based validation of zero-calibration transfer.

Steer Marin 8B with zero-calibration-transferred vectors from SmolLM3,
generate free-form text, and evaluate with Marin 8B as LLM judge.

This bridges the gap between logprob evaluation and real-world usability:
does zero-cal-transferred steering produce text that READS as having
the correct personality?

For each trait:
1. Steer Marin 8B with zero-cal vector (from SmolLM3)
2. Generate a self-description paragraph
3. Use Marin 8B (unsteered) to identify which of 6 traits the text exhibits
4. Compare: zero-cal transfer vs self-steering vs unsteered baseline
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="zero-cal-generation")

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


def get_5d_coords_and_basis(residual_vectors):
    V = np.stack([residual_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    basis_5d = Vt[:5]
    coords_5d = {t: basis_5d @ residual_vectors[t] for t in TRAITS}
    return coords_5d, basis_5d


def canonical_sign_convention(coords_5d):
    signs = np.ones(5)
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1
    for pc in range(1, 5):
        loadings = {t: coords_5d[t][pc] for t in TRAITS}
        max_trait = max(loadings, key=lambda t: abs(loadings[t]))
        if loadings[max_trait] > 0:
            signs[pc] = -1
    return signs


def standardize_coords(coords_5d, basis_5d):
    signs = canonical_sign_convention(coords_5d)
    std_coords = {t: signs * coords_5d[t] for t in TRAITS}
    std_basis = np.diag(signs) @ basis_5d
    return std_coords, std_basis


def generate_steered(model, tokenizer, device, blocks, mid_layer, steer_vec, alpha,
                     prompt, max_new_tokens=150):
    """Generate text while steering with a persona vector."""
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
        messages = [
            {"role": "user", "content": prompt},
        ]
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


def judge_personality(model, tokenizer, device, text):
    """Use the model (unsteered) to identify which personality the text exhibits.

    Returns a dict of trait → log probability.
    """
    trait_list = "\n".join(f"{i+1}. {TRAIT_DESCRIPTIONS[t]}" for i, t in enumerate(TRAITS))
    messages = [
        {"role": "system", "content": "You are a personality psychologist. Read the text and identify which personality type it best matches. Answer with ONLY the number (1-6)."},
        {"role": "user", "content": f"Text: \"{text}\"\n\nWhich personality type best matches this text?\n{trait_list}\n\nAnswer with the number:"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get probabilities for digits 1-6
    trait_probs = {}
    for i, t in enumerate(TRAITS):
        token_ids = tokenizer.encode(str(i + 1), add_special_tokens=False)
        trait_probs[t] = log_probs[token_ids[0]].item()

    return trait_probs


def main():
    device = "cuda:0"
    alpha = 1.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    source_id = "HuggingFaceTB/SmolLM3-3B"
    target_id = "marin-community/marin-8b-instruct"

    # Load vectors
    logger.info("Loading vectors...")
    source_residual, _ = load_residual_vectors(source_id, riasec_dir)
    source_coords, source_basis = get_5d_coords_and_basis(source_residual)
    source_std, source_std_basis = standardize_coords(source_coords, source_basis)

    target_residual, mid_layer = load_residual_vectors(target_id, riasec_dir)
    target_coords, target_basis = get_5d_coords_and_basis(target_residual)
    target_std, target_std_basis = standardize_coords(target_coords, target_basis)

    # Build zero-cal transferred vectors
    source_norms = np.mean([np.linalg.norm(source_std[t]) for t in TRAITS])
    target_norms = np.mean([np.linalg.norm(target_std[t]) for t in TRAITS])
    scale = target_norms / source_norms

    zero_cal_vecs = {}
    for t in TRAITS:
        zero_cal_vecs[t] = (target_std_basis.T @ (scale * source_std[t])).astype(np.float32)

    # Load model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    prompts = [
        "Describe yourself in a paragraph. What are your interests, values, and how you approach life?",
        "What kind of work do you enjoy? What motivates you in your career?",
        "How do you spend your free time? What activities bring you the most fulfillment?",
    ]

    print(f"\n{'='*70}")
    print(f"GENERATION-BASED VALIDATION OF ZERO-CALIBRATION TRANSFER")
    print(f"Source: SmolLM3, Target: Marin 8B")
    print(f"{'='*70}")

    results = {}

    for trait in TRAITS:
        print(f"\n{'='*70}")
        print(f"TRAIT: {trait.upper()} ({TRAIT_DESCRIPTIONS[trait]})")
        print(f"{'='*70}")

        trait_results = {"generations": [], "judge_scores": []}

        for prompt_idx, prompt in enumerate(prompts):
            # Generate with zero-cal vector
            logger.info(f"Generating {trait} (zero-cal) prompt {prompt_idx+1}...")
            gen_zc = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                      zero_cal_vecs[trait], alpha, prompt)

            # Generate with self vector
            logger.info(f"Generating {trait} (self) prompt {prompt_idx+1}...")
            gen_self = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                        target_residual[trait], alpha, prompt)

            # Generate unsteered
            logger.info(f"Generating {trait} (unsteered) prompt {prompt_idx+1}...")
            gen_base = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                        np.zeros_like(target_residual[trait]), 0.0, prompt)

            # Judge all three
            judge_zc = judge_personality(model, tokenizer, device, gen_zc)
            judge_self = judge_personality(model, tokenizer, device, gen_self)
            judge_base = judge_personality(model, tokenizer, device, gen_base)

            zc_rank = sorted(judge_zc.items(), key=lambda x: -x[1])
            self_rank = sorted(judge_self.items(), key=lambda x: -x[1])
            base_rank = sorted(judge_base.items(), key=lambda x: -x[1])

            zc_top = zc_rank[0][0]
            self_top = self_rank[0][0]
            base_top = base_rank[0][0]

            zc_correct = zc_top == trait
            self_correct = self_top == trait

            print(f"\n  Prompt {prompt_idx+1}: {prompt[:60]}...")
            print(f"    Zero-cal:  [{zc_top}{'*' if zc_correct else ''}] {gen_zc[:100]}...")
            print(f"    Self:      [{self_top}{'*' if self_correct else ''}] {gen_self[:100]}...")
            print(f"    Unsteered: [{base_top}] {gen_base[:100]}...")

            trait_results["generations"].append({
                "prompt": prompt,
                "zero_cal": {"text": gen_zc, "judge_top": zc_top, "correct": zc_correct},
                "self": {"text": gen_self, "judge_top": self_top, "correct": self_correct},
                "unsteered": {"text": gen_base, "judge_top": base_top},
            })

        results[trait] = trait_results

    # Summary
    zc_correct_total = sum(
        1 for t in TRAITS for g in results[t]["generations"] if g["zero_cal"]["correct"])
    self_correct_total = sum(
        1 for t in TRAITS for g in results[t]["generations"] if g["self"]["correct"])
    total = len(TRAITS) * len(prompts)

    print(f"\n{'='*70}")
    print(f"SUMMARY: LLM Judge Accuracy (top-1 match)")
    print(f"{'='*70}")
    print(f"  Zero-cal (SmolLM3→Marin): {zc_correct_total}/{total} ({zc_correct_total/total:.0%})")
    print(f"  Self (Marin→Marin):       {self_correct_total}/{total} ({self_correct_total/total:.0%})")

    results["summary"] = {
        "zero_cal_correct": zc_correct_total,
        "self_correct": self_correct_total,
        "total": total,
        "zero_cal_accuracy": float(zc_correct_total / total),
        "self_accuracy": float(self_correct_total / total),
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zero_cal_generation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
