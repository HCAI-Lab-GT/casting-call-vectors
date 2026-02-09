#!/usr/bin/env python
"""
Test whether using the optimal injection layer (found from cross-layer analysis)
improves trait specificity compared to the standard middle layer.

Hypothesis: the "non-specificity" in the standard evaluation (ratio ≈ 0.99)
might be partly due to using a suboptimal injection layer. If specificity
improves at the optimal layer, that's a very strong finding.
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="optimal-layer-specificity")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def logprob_gap(model, tokenizer, device, question):
    messages = [
        {"role": "system", "content": "Output EXACTLY one token: YES or NO."},
        {"role": "user", "content": question},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)
    return (log_probs[yes_ids[0]] - log_probs[no_ids[0]]).item()


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--optimal_layer", type=int, default=10,
                    help="Optimal injection layer from cross-layer analysis")
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2
    opt_layer = args.optimal_layer

    logger.info("Model: %s, mid=%d, optimal=%d", args.model_id, mid_layer, opt_layer)

    # Load RIASEC
    riasec_path = _repo_root() / "configs" / "riasec.yaml"
    with open(riasec_path) as f:
        riasec = yaml.safe_load(f)
    characteristics = {t: riasec[t]["characteristics"] for t in TRAITS}

    # Load all-layer vectors for all traits
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

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()
    device = args.device
    blocks = get_decoder_blocks(model)

    results = {}

    for inject_layer, layer_name in [(mid_layer, "mid"), (opt_layer, "optimal")]:
        logger.info("=== Testing injection at L%d (%s) ===", inject_layer, layer_name)

        # Build 6x6 specificity matrix: steering with trait_i, evaluated on trait_j
        matrix = np.zeros((6, 6))

        for i, steer_trait in enumerate(TRAITS):
            # Use the vector from the injection layer (matched condition)
            vec = all_layer_vectors[steer_trait][inject_layer + 1]
            vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
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

            hook_handle = blocks[inject_layer].register_forward_hook(make_hook(delta))

            try:
                for j, eval_trait in enumerate(TRAITS):
                    gaps = [logprob_gap(model, tokenizer, device, q)
                            for q in characteristics[eval_trait]]
                    matrix[i, j] = np.mean(gaps)
            finally:
                hook_handle.remove()

            logger.info("  Steered %s: diag=%.2f, off-diag=%.2f",
                        steer_trait, matrix[i, i],
                        np.mean([matrix[i, j] for j in range(6) if j != i]))

        results[layer_name] = {
            "layer": inject_layer,
            "matrix": matrix.tolist(),
            "diagonal_mean": float(np.mean(np.diag(matrix))),
            "off_diagonal_mean": float(np.mean(matrix[~np.eye(6, dtype=bool)])),
        }

        diag = np.mean(np.diag(matrix))
        off_diag = np.mean(matrix[~np.eye(6, dtype=bool)])
        ratio = diag / max(abs(off_diag), 0.001)

        logger.info("L%d: diag=%.3f, off-diag=%.3f, ratio=%.4f, diff=%+.4f",
                     inject_layer, diag, off_diag, ratio, diag - off_diag)

    # Print comparison
    print(f"\n{'='*70}")
    print("SPECIFICITY COMPARISON: Middle vs Optimal Layer")
    print(f"{'='*70}")

    for layer_name in ["mid", "optimal"]:
        r = results[layer_name]
        matrix = np.array(r["matrix"])
        diag = r["diagonal_mean"]
        off = r["off_diagonal_mean"]
        print(f"\n--- L{r['layer']} ({layer_name}) ---")
        print(f"  Diagonal mean (match): {diag:.4f}")
        print(f"  Off-diagonal mean:     {off:.4f}")
        print(f"  Specificity ratio:     {diag/max(abs(off), 0.001):.4f}")
        print(f"  Specificity diff:      {diag - off:+.4f}")

        # Per-trait
        print(f"\n  Per-trait specificity:")
        for i, trait in enumerate(TRAITS):
            d = matrix[i, i]
            od = np.mean([matrix[i, j] for j in range(6) if j != i])
            print(f"    {trait:>15}: match={d:.2f}, others={od:.2f}, diff={d-od:+.2f}")

    # Improvement
    mid_diff = results["mid"]["diagonal_mean"] - results["mid"]["off_diagonal_mean"]
    opt_diff = results["optimal"]["diagonal_mean"] - results["optimal"]["off_diagonal_mean"]
    print(f"\n  Specificity improvement: {mid_diff:+.4f} → {opt_diff:+.4f} "
          f"({'IMPROVED' if opt_diff > mid_diff else 'NOT IMPROVED'})")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"optimal_layer_specificity_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
