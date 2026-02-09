#!/usr/bin/env python
"""
Test persona vector steering on the 32B BASE model.

Since the base model has no chat template, we use a completion-style prompt
and measure:
1. Logprob gap (YES/NO) on RIASEC characteristics
2. Freeform generation with and without steering
3. Semantic similarity of steered outputs to trait descriptions

This directly tests: does personality geometry function for steering
in a model that has never been instruction-tuned?
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="base-model-steering")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def logprob_gap_base(model, tokenizer, device, question):
    """Logprob gap for base model using completion-style format."""
    # Base model format: simple completion
    prompt = f"Question: {question}\nAnswer (YES or NO):"

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Try multiple token variants
    yes_candidates = ["YES", " YES", "Yes", " Yes", "yes", " yes"]
    no_candidates = ["NO", " NO", "No", " No", "no", " no"]

    yes_logprob = -float("inf")
    no_logprob = -float("inf")

    for text in yes_candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            yes_logprob = max(yes_logprob, log_probs[ids[0]].item())

    for text in no_candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            no_logprob = max(no_logprob, log_probs[ids[0]].item())

    return yes_logprob - no_logprob


def generate_base(model, tokenizer, device, prompt, max_new_tokens=100):
    """Generate text from base model."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="marin-community/marin-32b-base")
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    logger.info("Model: %s, layers=%d, mid=%d", args.model_id, num_layers, mid_layer)

    # Load RIASEC config
    riasec_path = _repo_root() / "configs" / "riasec.yaml"
    with open(riasec_path) as f:
        riasec = yaml.safe_load(f)
    characteristics = {t: riasec[t]["characteristics"] for t in TRAITS}
    descriptions = {t: riasec[t]["description"] for t in TRAITS}

    # Load persona vectors
    safe_model = args.model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

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

    full_vectors = {t: all_layer_vectors[t][mid_layer + 1] for t in TRAITS}
    residual_vectors = {}
    for t in TRAITS:
        vec = full_vectors[t]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Get device of the mid-layer block
    mid_block = blocks[mid_layer]
    # Find the device of a parameter in this block
    for param in mid_block.parameters():
        block_device = param.device
        break

    # First: baseline (no steering)
    logger.info("Computing baseline...")
    baseline = {}
    for trait in TRAITS:
        gaps = [logprob_gap_base(model, tokenizer, block_device, q)
                for q in characteristics[trait]]
        baseline[trait] = float(np.mean(gaps))

    print(f"\n{'='*70}")
    print(f"BASE MODEL STEERING: {args.model_id}")
    print(f"{'='*70}")

    print(f"\nBaseline (no steering):")
    for t in TRAITS:
        print(f"  {t:>15}: {baseline[t]:+.3f}")

    # Test steering with full vectors
    results = {"baseline": baseline, "full": {}, "residual": {}}

    for vec_type, vectors in [("full", full_vectors), ("residual", residual_vectors)]:
        print(f"\n--- Steering with {vec_type} vectors (alpha={args.alpha}) ---")
        print(f"  {'Steer→':>15}", end="")
        for t in TRAITS:
            print(f"{t[:5]:>8}", end="")
        print()

        for steer_trait in TRAITS:
            vec = vectors[steer_trait]
            vec_t = torch.tensor(vec, dtype=torch.float16).to(block_device)
            delta = args.alpha * vec_t

            def make_hook(d):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += d
                        return (hs,) + out[1:]
                    out[:, -1, :] += d
                    return out
                return hook_fn

            hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta))
            try:
                steered = {}
                for eval_trait in TRAITS:
                    gaps = [logprob_gap_base(model, tokenizer, block_device, q)
                            for q in characteristics[eval_trait]]
                    steered[eval_trait] = float(np.mean(gaps))
            finally:
                hook_handle.remove()

            results[vec_type][steer_trait] = steered

            print(f"  {steer_trait:>15}", end="")
            for t in TRAITS:
                val = steered[t]
                print(f"{val:>8.2f}", end="")
            print()

        # Specificity
        matrix = np.zeros((6, 6))
        for i, st in enumerate(TRAITS):
            for j, et in enumerate(TRAITS):
                matrix[i, j] = results[vec_type][st][et]
        diag = np.mean(np.diag(matrix))
        off = np.mean(matrix[~np.eye(6, dtype=bool)])
        print(f"\n  Diagonal mean: {diag:.3f}, Off-diagonal: {off:.3f}, Diff: {diag-off:+.4f}")

    # Qualitative generation
    print(f"\n--- Qualitative Steering (freeform generation) ---")
    prompt = "In my free time, I love to"

    print(f"\nPrompt: '{prompt}'")
    print(f"\nBaseline:")
    baseline_gen = generate_base(model, tokenizer, block_device, prompt)
    print(f"  {baseline_gen[:200]}")

    for trait in TRAITS:
        vec = residual_vectors[trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).to(block_device)
        delta = args.alpha * vec_t

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta))
        try:
            steered_gen = generate_base(model, tokenizer, block_device, prompt)
        finally:
            hook_handle.remove()

        print(f"\n{trait.upper()}:")
        print(f"  {steered_gen[:200]}")
        results[f"gen_{trait}"] = steered_gen[:500]

    results["gen_baseline"] = baseline_gen[:500]

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"base_model_steering_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
