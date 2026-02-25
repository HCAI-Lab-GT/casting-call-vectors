#!/usr/bin/env python
"""
Full layer injection × detection heatmap for Marin 8B.

For each injection layer (0-31), steers with persona vector and measures
the 5D detection accuracy at every layer (0-31). Produces a 32×32 heatmap
showing how personality signal propagates through the network.

This fills a gap in the existing experiments: we know mid-layer injection
works, but this shows the FULL picture of where injection is optimal and
how the signal flows through layers.

Output: outputs/analysis/full_layer_heatmap_marin-8b.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="full-layer-heatmap")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def build_5d_basis(trait_vectors: dict) -> tuple:
    """Build the 5D orthonormal basis from 6 trait vectors."""
    matrix = np.stack([trait_vectors[t] for t in TRAITS])
    shared = matrix.mean(axis=0)
    shared_dir = shared / np.linalg.norm(shared)
    residuals = matrix - np.outer(matrix @ shared_dir, shared_dir)
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    basis = Vt[:5]  # 5 × hidden_dim
    coords = residuals @ basis.T  # 6 × 5
    trait_coords = {t: coords[i] for i, t in enumerate(TRAITS)}
    return basis, trait_coords, S


def detect_trait(activation_diff: np.ndarray, basis: np.ndarray,
                 trait_coords: dict) -> tuple:
    """Project activation diff into 5D and detect trait via cosine similarity."""
    proj = activation_diff @ basis.T
    proj_norm = np.linalg.norm(proj)
    if proj_norm < 1e-10:
        return "none", 0.0, {}

    sims = {}
    for t, tc in trait_coords.items():
        tc_norm = np.linalg.norm(tc)
        if tc_norm < 1e-10:
            sims[t] = 0.0
        else:
            sims[t] = float(np.dot(proj, tc) / (proj_norm * tc_norm))

    best = max(sims, key=sims.get)
    return best, sims[best], sims


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="marin-community/marin-8b-instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--traits", nargs="+", default=["artistic", "investigative", "social"])
    ap.add_argument("--layer_step", type=int, default=2,
                    help="Test every N-th layer (2 = layers 0,2,4,...)")
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    logger.info("Model: %s, %d layers, testing traits: %s", args.model_id, num_layers, args.traits)

    # Load RIASEC config for test prompts
    riasec_path = _repo_root() / "configs" / "riasec.yaml"
    with open(riasec_path) as f:
        riasec = yaml.safe_load(f)

    # Load persona vectors (all layers)
    safe_model = args.model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data" / "model_inits"

    all_layer_vectors = {}
    mid_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        all_layers = data["all_layers_response_persona_vector"].numpy()
        if all_layers.ndim == 3:
            all_layers = all_layers[:, 0, :]
        all_layer_vectors[trait] = all_layers
        mid_layer_vectors[trait] = all_layers[mid_layer + 1].astype(np.float32)

    # Build 5D basis from mid-layer vectors
    basis, trait_coords, svs = build_5d_basis(mid_layer_vectors)
    logger.info("5D singular values: %s", svs[:6].round(3))

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    device = args.device
    blocks = get_decoder_blocks(model)

    # Prepare test prompt
    test_prompt = "Tell me about your interests and what kind of work you enjoy."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Get baseline activations at all layers
    logger.info("Computing baseline activations...")
    baseline_acts = {}
    hooks = []

    def make_capture_hook(layer_idx, store):
        def hook_fn(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            store[layer_idx] = hs[:, -1, :].detach().cpu().float().numpy()[0]
        return hook_fn

    for L in range(num_layers):
        hooks.append(blocks[L].register_forward_hook(make_capture_hook(L, baseline_acts)))

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    # Now: for each injection layer × each trait, steer and capture all layers
    inject_layers = list(range(0, num_layers, args.layer_step))
    if mid_layer not in inject_layers:
        inject_layers.append(mid_layer)
        inject_layers.sort()

    results = {
        "model_id": args.model_id,
        "alpha": args.alpha,
        "num_layers": num_layers,
        "mid_layer": mid_layer,
        "inject_layers": inject_layers,
        "detect_layers": inject_layers,
        "svd_values": svs[:6].tolist(),
        "heatmaps": {},
    }

    for trait in args.traits:
        logger.info("Processing trait: %s", trait)
        heatmap = []

        for inject_L in inject_layers:
            logger.info("  Injecting at layer %d", inject_L)

            # Use mid-layer vector for injection (the standard approach)
            vec = mid_layer_vectors[trait]
            vec_t = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(device)
            delta = args.alpha * vec_t

            # Install injection hook
            def make_steer_hook(d):
                def hook_fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    hs[:, -1, :] += d
                    return (hs,) + out[1:] if isinstance(out, tuple) else hs
                return hook_fn

            inject_handle = blocks[inject_L].register_forward_hook(make_steer_hook(delta))

            # Capture steered activations at all layers
            steered_acts = {}
            capture_hooks = []
            for L in inject_layers:
                capture_hooks.append(blocks[L].register_forward_hook(
                    make_capture_hook(L, steered_acts)))

            with torch.no_grad():
                model(input_ids=input_ids)

            inject_handle.remove()
            for h in capture_hooks:
                h.remove()

            # Detect at each layer
            row = []
            for detect_L in inject_layers:
                if detect_L in steered_acts and detect_L in baseline_acts:
                    diff = steered_acts[detect_L] - baseline_acts[detect_L]
                    detected, sim, all_sims = detect_trait(diff, basis, trait_coords)
                    row.append({
                        "inject_layer": inject_L,
                        "detect_layer": detect_L,
                        "detected": detected,
                        "correct": detected == trait,
                        "target_sim": float(all_sims.get(trait, 0)),
                        "best_sim": float(sim),
                        "norm_5d": float(np.linalg.norm(diff @ basis.T)),
                    })
                else:
                    row.append({
                        "inject_layer": inject_L,
                        "detect_layer": detect_L,
                        "detected": "missing",
                        "correct": False,
                        "target_sim": 0.0,
                        "best_sim": 0.0,
                        "norm_5d": 0.0,
                    })

            heatmap.append(row)

        results["heatmaps"][trait] = heatmap

        # Print summary
        print(f"\n{'='*60}")
        print(f"TRAIT: {trait.upper()}")
        print(f"{'='*60}")
        print(f"{'Inject':>8}", end="")
        for dL in inject_layers:
            print(f" {'D'+str(dL):>5}", end="")
        print()

        for i, inject_L in enumerate(inject_layers):
            print(f"{'I'+str(inject_L):>8}", end="")
            for j, detect_L in enumerate(inject_layers):
                sim = heatmap[i][j]["target_sim"]
                marker = "+" if heatmap[i][j]["correct"] else "-"
                print(f" {sim:>4.2f}{marker}", end="")
            print()

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"full_layer_heatmap_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
