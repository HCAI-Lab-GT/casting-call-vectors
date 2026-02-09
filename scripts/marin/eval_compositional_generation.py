#!/usr/bin/env python
"""
Generate text with COMPOSITIONAL steering (two traits simultaneously).
Shows qualitative personality blending (e.g., Artistic+Investigative = "creative scientist").

Uses SmolLM3-3B with completion prompts, α=2 per component.
Tests Holland adjacent pairs (most natural blends).
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="compositional-gen")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

HOLLAND_ADJACENT = [
    ("realistic", "investigative"),
    ("investigative", "artistic"),
    ("artistic", "social"),
    ("social", "enterprising"),
    ("enterprising", "conventional"),
    ("conventional", "realistic"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def generate_steered(model, tokenizer, device, blocks, layer, vector, prompt, max_new_tokens=150):
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


def main():
    model_id = "HuggingFaceTB/SmolLM3-3B"
    device = "cuda:0"
    alpha = 2.0  # Per component

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

    prompts = [
        "In my free time, I love to",
        "My ideal career would involve",
    ]

    results = {}

    print(f"\n{'='*70}")
    print(f"COMPOSITIONAL GENERATION: {model_id}")
    print(f"Alpha per component: {alpha}")
    print(f"{'='*70}")

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*70}")

        results[prompt] = {}

        # Single-trait generations for reference
        print(f"\n--- Single traits (α={alpha}) ---")
        for trait in TRAITS:
            vec = alpha * residual_vectors[trait]
            text = generate_steered(model, tokenizer, device, blocks, mid_layer, vec, prompt)
            results[prompt][f"single_{trait}"] = text[:500]
            print(f"  {trait.upper():>15}: {text[:150]}")

        # Holland adjacent pair compositions
        print(f"\n--- Holland adjacent pairs (α={alpha} each) ---")
        for t1, t2 in HOLLAND_ADJACENT:
            combined = alpha * residual_vectors[t1] + alpha * residual_vectors[t2]
            text = generate_steered(model, tokenizer, device, blocks, mid_layer, combined, prompt)
            results[prompt][f"comp_{t1}+{t2}"] = text[:500]
            print(f"  {t1.upper()[:4]}+{t2.upper()[:4]:>4}: {text[:150]}")

        # Also test Holland opposite pair (most extreme contrast)
        print(f"\n--- Holland opposite pairs (α={alpha} each) ---")
        opposites = [
            ("realistic", "social"),
            ("investigative", "enterprising"),
            ("artistic", "conventional"),
        ]
        for t1, t2 in opposites:
            combined = alpha * residual_vectors[t1] + alpha * residual_vectors[t2]
            text = generate_steered(model, tokenizer, device, blocks, mid_layer, combined, prompt)
            results[prompt][f"opp_{t1}+{t2}"] = text[:500]
            print(f"  {t1.upper()[:4]}+{t2.upper()[:4]:>4}: {text[:150]}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"compositional_generation_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
