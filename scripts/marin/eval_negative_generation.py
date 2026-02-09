#!/usr/bin/env python
"""
Generate text with NEGATIVE steering to show trait suppression qualitatively.
Tests: Does steering with -α for "artistic" produce LESS artistic text?

Uses SmolLM3-3B with completion prompts (100% discrimination model).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="negative-generation")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def generate_steered(model, tokenizer, device, blocks, layer, vector, alpha, prompt, max_new_tokens=150):
    vec_t = torch.tensor(vector, dtype=torch.float16).unsqueeze(0).to(device)
    delta = alpha * vec_t

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

    hook_handle = blocks[layer].register_forward_hook(make_hook(delta))
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


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"

    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    # Load vectors
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Compute residual vectors
    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual_vectors = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    # Load model
    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    prompt = "In my free time, I love to"
    alpha = 3.0

    results = {}

    print(f"\n{'='*70}")
    print(f"NEGATIVE vs POSITIVE STEERING GENERATION: {model_id}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*70}")

    # Baseline
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=150, do_sample=True,
                                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    baseline_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n--- Baseline ---")
    print(f"  {baseline_text[:300]}")
    results["baseline"] = baseline_text[:500]

    for trait in TRAITS:
        print(f"\n--- {trait.upper()} ---")

        # Positive steering
        pos_text = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                   residual_vectors[trait], +alpha, prompt)
        print(f"  +{alpha}: {pos_text[:200]}")

        # Negative steering
        neg_text = generate_steered(model, tokenizer, device, blocks, mid_layer,
                                   residual_vectors[trait], -alpha, prompt)
        print(f"  -{alpha}: {neg_text[:200]}")

        results[f"{trait}_positive"] = pos_text[:500]
        results[f"{trait}_negative"] = neg_text[:500]

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"negative_generation_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
