#!/usr/bin/env python
"""
Test residual vectors at the optimal layer.

Hypothesis: if we decompose the vectors at each layer and use only
the residual (trait-specific) component, we might get BOTH strength
AND specificity at the deeper optimal layer.

Tests:
1. Full vector at mid layer (baseline - standard approach)
2. Full vector at optimal layer (more power, less specific)
3. Residual vector at mid layer (existing result - marginal specificity)
4. Residual vector at optimal layer (NEW - best of both worlds?)
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="residual-optimal-layer")

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


def decompose_at_layer(all_layer_vectors, layer_idx):
    """Decompose vectors at a specific layer into shared + residual."""
    V = np.stack([all_layer_vectors[t][layer_idx + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residuals = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][layer_idx + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residuals[t] = vec - proj

    return residuals, shared_dir


def eval_specificity_matrix(model, tokenizer, device, blocks, inject_layer, vectors, alpha, characteristics):
    """Evaluate 6x6 specificity matrix."""
    matrix = np.zeros((6, 6))

    for i, steer_trait in enumerate(TRAITS):
        vec = vectors[steer_trait]
        vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
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

        hook_handle = blocks[inject_layer].register_forward_hook(make_hook(delta))
        try:
            for j, eval_trait in enumerate(TRAITS):
                gaps = [logprob_gap(model, tokenizer, device, q)
                        for q in characteristics[eval_trait]]
                matrix[i, j] = np.mean(gaps)
        finally:
            hook_handle.remove()

    return matrix


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--optimal_layer", type=int, default=10)
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2
    opt_layer = args.optimal_layer

    logger.info("Model: %s, mid=%d, optimal=%d", args.model_id, mid_layer, opt_layer)

    riasec_path = _repo_root() / "configs" / "riasec.yaml"
    with open(riasec_path) as f:
        riasec = yaml.safe_load(f)
    characteristics = {t: riasec[t]["characteristics"] for t in TRAITS}

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()
    device = args.device
    blocks = get_decoder_blocks(model)

    # Decompose at each layer
    mid_residuals, _ = decompose_at_layer(all_layer_vectors, mid_layer)
    opt_residuals, _ = decompose_at_layer(all_layer_vectors, opt_layer)

    # Full vectors
    mid_full = {t: all_layer_vectors[t][mid_layer + 1] for t in TRAITS}
    opt_full = {t: all_layer_vectors[t][opt_layer + 1] for t in TRAITS}

    conditions = [
        ("full_mid", mid_layer, mid_full),
        ("full_optimal", opt_layer, opt_full),
        ("residual_mid", mid_layer, mid_residuals),
        ("residual_optimal", opt_layer, opt_residuals),
    ]

    results = {}

    for cond_name, inject_layer, vectors in conditions:
        logger.info("=== %s at L%d ===", cond_name, inject_layer)
        matrix = eval_specificity_matrix(
            model, tokenizer, device, blocks, inject_layer,
            vectors, args.alpha, characteristics
        )

        diag = np.mean(np.diag(matrix))
        off_diag = np.mean(matrix[~np.eye(6, dtype=bool)])

        results[cond_name] = {
            "layer": inject_layer,
            "matrix": matrix.tolist(),
            "diagonal_mean": float(diag),
            "off_diagonal_mean": float(off_diag),
            "diff": float(diag - off_diag),
        }

        logger.info("  diag=%.3f, off-diag=%.3f, diff=%+.4f", diag, off_diag, diag - off_diag)

    # Print comparison
    print(f"\n{'='*70}")
    print("RESIDUAL VECTORS AT OPTIMAL LAYER")
    print(f"{'='*70}")

    print(f"\n{'Condition':>25} {'Layer':>6} {'Diagonal':>10} {'Off-diag':>10} {'Diff':>10} {'Ratio':>8}")
    print("-" * 70)
    for cond_name, r in results.items():
        diag = r["diagonal_mean"]
        off = r["off_diagonal_mean"]
        diff = r["diff"]
        ratio = diag / max(abs(off), 0.001)
        print(f"{cond_name:>25} L{r['layer']:>4} {diag:>10.3f} {off:>10.3f} {diff:>+10.4f} {ratio:>8.4f}")

    # Per-trait detail for the key comparison
    for cond_name in ["residual_mid", "residual_optimal"]:
        r = results[cond_name]
        matrix = np.array(r["matrix"])
        print(f"\n  Per-trait ({cond_name}):")
        for i, trait in enumerate(TRAITS):
            d = matrix[i, i]
            od = np.mean([matrix[i, j] for j in range(6) if j != i])
            print(f"    {trait:>15}: match={d:.2f}, others={od:.2f}, diff={d-od:+.2f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_path = out_dir / f"residual_optimal_layer_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
