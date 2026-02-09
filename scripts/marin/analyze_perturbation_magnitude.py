#!/usr/bin/env python
"""
Quantify the perturbation magnitude relative to hidden state norm.

If alpha=0.1 gives 100% discrimination, how small is the perturbation
compared to the model's natural hidden state activations?

This measures: ||alpha * residual_vec|| / ||hidden_state_at_mid_layer||
for various alpha values and prompts.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="perturbation-magnitude")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


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

    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual_vectors = {}
    residual_norms = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = vec - proj
        residual_norms[t] = np.linalg.norm(residual_vectors[t])

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
        "Which describes you better?\nA) I am creative and artistic\nB) I am organized and conventional\nAnswer:",
        "In my free time, I love to",
        "My ideal career would involve",
    ]

    # Capture hidden states at mid layer
    hidden_norms = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        captured = {}
        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                hs = out[0]
            else:
                hs = out
            captured["hs"] = hs.detach()

        handle = blocks[mid_layer].register_forward_hook(capture_hook)
        with torch.no_grad():
            model(input_ids=input_ids)
        handle.remove()

        # Get last-token hidden state norm
        hs = captured["hs"]
        last_token_norm = torch.norm(hs[0, -1, :]).item()
        mean_token_norm = torch.norm(hs[0], dim=-1).mean().item()

        hidden_norms.append({
            "prompt": prompt[:50],
            "last_token_norm": last_token_norm,
            "mean_token_norm": mean_token_norm,
            "seq_len": hs.shape[1],
        })

    mean_hs_norm = np.mean([h["last_token_norm"] for h in hidden_norms])

    print(f"\n{'='*70}")
    print(f"PERTURBATION MAGNITUDE ANALYSIS")
    print(f"Model: {model_id}")
    print(f"{'='*70}")

    print(f"\n--- Hidden state norms at mid-layer (L{mid_layer}) ---")
    for h in hidden_norms:
        print(f"  '{h['prompt'][:40]}...': last_tok={h['last_token_norm']:.2f}, "
              f"mean={h['mean_token_norm']:.2f} (len={h['seq_len']})")
    print(f"  Mean last-token norm: {mean_hs_norm:.2f}")

    print(f"\n--- Residual vector norms ---")
    for t in TRAITS:
        print(f"  {t:>14}: {residual_norms[t]:.4f}")
    mean_vec_norm = np.mean(list(residual_norms.values()))
    print(f"  Mean: {mean_vec_norm:.4f}")

    print(f"\n--- Perturbation ratio: ||α * vec|| / ||hidden_state|| ---")
    print(f"  {'Alpha':>6}  {'||perturbation||':>16}  {'ratio':>8}  {'Δ accuracy':>10}")
    print(f"  {'-'*44}")

    # From dose-response results
    dose_response_accs = {
        0.1: 1.0, 0.2: 1.0, 0.3: 1.0, 0.5: 1.0, 0.75: 1.0,
        1.0: 1.0, 1.5: 1.0, 2.0: 1.0, 3.0: 1.0,
        4.0: 0.93, 5.0: 0.73, 7.0: 0.53, 10.0: 0.43,
    }

    for alpha in sorted(dose_response_accs.keys()):
        perturb_norm = alpha * mean_vec_norm
        ratio = perturb_norm / mean_hs_norm
        acc = dose_response_accs[alpha]
        print(f"  {alpha:>6.2f}  {perturb_norm:>15.4f}  {ratio:>7.4f}  {acc:>9.0%}")

    # Key finding: what ratio gives 100%?
    alpha_100_min = 0.1
    ratio_100_min = alpha_100_min * mean_vec_norm / mean_hs_norm

    print(f"\n--- KEY FINDING ---")
    print(f"  Minimum alpha for 100% discrimination: {alpha_100_min}")
    print(f"  Perturbation norm at that alpha: {alpha_100_min * mean_vec_norm:.4f}")
    print(f"  Hidden state norm: {mean_hs_norm:.2f}")
    print(f"  Ratio: {ratio_100_min:.6f} = {ratio_100_min*100:.4f}%")
    print(f"  In other words: a {ratio_100_min*100:.2f}% perturbation achieves 100% personality discrimination")

    # Also compute the directional component
    # What fraction of hidden state variance is along personality directions?
    print(f"\n--- Dimensionality analysis ---")
    hidden_dim = residual_vectors[TRAITS[0]].shape[0]
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Personality subspace dimension: 5 (6 RIASEC vectors - 1 shared)")
    print(f"  Fraction of dims: {5/hidden_dim:.4f} = {5/hidden_dim*100:.2f}%")
    print(f"  Expected random perturbation in personality subspace: {np.sqrt(5/hidden_dim)*100:.2f}% of total")

    # Save
    results = {
        "model_id": model_id,
        "mid_layer": mid_layer,
        "mean_hidden_state_norm": float(mean_hs_norm),
        "mean_residual_vec_norm": float(mean_vec_norm),
        "hidden_dim": hidden_dim,
        "personality_subspace_dim": 5,
        "hidden_norms": hidden_norms,
        "residual_norms": {k: float(v) for k, v in residual_norms.items()},
        "min_ratio_for_100pct": float(ratio_100_min),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perturbation_magnitude_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
