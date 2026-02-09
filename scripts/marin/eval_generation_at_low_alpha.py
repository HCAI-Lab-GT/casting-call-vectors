#!/usr/bin/env python
"""
Test if alpha=1 (optimal for pairwise discrimination) produces
perceptibly different generation text. This validates whether
97% discrimination at alpha=1 corresponds to real behavioral differences.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="generation-low-alpha")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def generate_steered(model, tokenizer, device, blocks, layer, vector, alpha, prompt, max_new_tokens=200):
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

    messages = [
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="marin-community/marin-8b-instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    safe_model = args.model_id.replace("/", "__")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    prompts = [
        "In my free time, I love to",
        "When I face a difficult problem, I tend to",
        "My ideal career would involve",
    ]

    alphas = [0.0, 1.0, 3.0, 5.0]

    results = {}

    print(f"\n{'='*70}")
    print(f"GENERATION AT LOW ALPHA: {args.model_id}")
    print(f"{'='*70}")

    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*70}")

        results[prompt] = {}

        for alpha in alphas:
            if alpha == 0:
                print(f"\n--- Baseline (alpha=0) ---")
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(formatted, return_tensors="pt")
                input_ids = enc["input_ids"].to(args.device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                print(f"  {text[:300]}")
                results[prompt]["baseline"] = text[:500]
                continue

            print(f"\n--- Alpha={alpha} (residual vectors) ---")
            for trait in TRAITS:
                gen = generate_steered(
                    model, tokenizer, args.device, blocks, mid_layer,
                    residual_vectors[trait], alpha, prompt, max_new_tokens=200,
                )
                key = f"alpha_{alpha}_{trait}"
                results[prompt][key] = gen[:500]
                print(f"  {trait.upper():>15}: {gen[:200]}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generation_low_alpha_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
