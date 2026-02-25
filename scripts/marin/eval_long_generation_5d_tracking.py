#!/usr/bin/env python
"""
Long Generation 5D Personality Tracking.

Tracks the 5D personality coordinates during a single long generation (500+ tokens).
The goal is to determine whether personality is stable within a single response
or if it drifts over time.

For each RIASEC trait (at alpha=3.0), we:
1. Generate 500 tokens with a manual generation loop
2. Every 25 tokens, extract the hidden state at detect_layer and project to 5D
3. Record coordinates, target similarity, and norm at each checkpoint
4. Also run a baseline (no steering) for comparison

Stability metrics computed:
- Mean cosine similarity between consecutive 5D snapshots
- Standard deviation of coordinates over time
- Drift angle from initial position
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="long-gen-tracking")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_model_data(model_id, riasec_dir):
    """Load persona vectors, compute residuals and 5D basis."""
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

    # Compute shared direction from raw vectors at detect_layer
    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    # Compute residuals (subtract shared PC1 direction)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj

    # 5D basis from residuals
    R = np.stack([residual[t] for t in TRAITS])
    Ur, Sr, Vtr = np.linalg.svd(R, full_matrices=False)
    basis_5d = Vtr[:5]  # 5 x hidden_dim

    # Known 5D coordinates for each trait
    coords_5d = {t: (basis_5d @ residual[t]).astype(np.float64) for t in TRAITS}

    # Steering vectors at mid_layer (used for steering hook)
    steer_vectors = {}
    for t in TRAITS:
        steer_vectors[t] = all_layer_vectors[t][mid_layer].astype(np.float32)

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
        "steer_vectors": steer_vectors,
    }


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def generate_long_with_tracking(
    model, tokenizer, device, blocks, mid_layer, basis_5d,
    coords_5d, steer_vec, alpha, prompt, max_tokens=500,
    tokens_per_checkpoint=25, temperature=0.7, trait_name=None,
):
    """
    Generate tokens with steering and track 5D personality at checkpoints.

    Uses register_forward_pre_hook for steering (modifies INPUT to mid_layer).
    Uses register_forward_hook for detection (reads OUTPUT of detect_layer).
    """
    detect_layer = mid_layer + 1

    # Prepare steering delta
    if steer_vec is not None and alpha != 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
    else:
        delta = None

    # Format prompt with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Storage for captured hidden states at detect_layer
    captured_hidden = {}

    # ---- Steering hook: register_forward_pre_hook on mid_layer block ----
    # Takes (module, input) NOT (module, input, output)
    def steer_pre_hook(module, inp):
        if delta is None:
            return inp
        hs = inp[0]
        hs[:, -1, :] += delta
        return (hs,) + inp[1:]

    # ---- Detection hook: register_forward_hook on detect_layer block ----
    def detect_hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_hidden["act"] = hs[0, -1, :].detach().cpu().float().numpy().copy()
        return out

    # Install hooks
    steer_handle = blocks[mid_layer].register_forward_pre_hook(steer_pre_hook)
    detect_handle = blocks[detect_layer].register_forward_hook(detect_hook)

    # ---- First: get baseline hidden state (prefill only, before generation) ----
    # We need this to compare against; run the prefill forward pass
    with torch.no_grad():
        model(input_ids)
    prefill_act = captured_hidden["act"].copy()

    # ---- Manual generation loop ----
    generated_ids = []
    checkpoints = []
    all_token_coords = []  # Collect coords at EVERY token for fine-grained stability
    past_kv = None
    current_ids = input_ids

    try:
        with torch.no_grad():
            for step in range(max_tokens):
                # Forward pass
                if past_kv is not None:
                    outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                else:
                    outputs = model(current_ids, use_cache=True)

                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                # Safe sampling: cast to float, clamp, temperature, NaN guard
                logits_f = logits.float().clamp(-100, 100)
                probs = torch.softmax(logits_f / max(temperature, 0.01), dim=-1)

                # NaN guard
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    probs = torch.ones_like(probs) / probs.shape[-1]

                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                generated_ids.append(next_id.item())
                current_ids = next_id.unsqueeze(0)

                # Extract 5D coords from the captured hidden state at this step
                act = captured_hidden["act"]
                diff = (act - prefill_act).astype(np.float64)
                coords = (basis_5d @ diff).astype(np.float64)
                norm_5d = float(np.linalg.norm(coords))
                all_token_coords.append(coords.copy())

                # Record checkpoint every tokens_per_checkpoint tokens
                token_num = step + 1
                if token_num % tokens_per_checkpoint == 0:
                    # Compute target similarity if we have a trait
                    target_sim = 0.0
                    if trait_name and trait_name in coords_5d:
                        target_sim = cosine_sim(coords, coords_5d[trait_name])

                    checkpoints.append({
                        "token": token_num,
                        "coords_5d": coords.tolist(),
                        "target_sim": float(target_sim),
                        "norm": float(norm_5d),
                    })

                # Stop on EOS
                if next_id.item() == tokenizer.eos_token_id:
                    break

    finally:
        steer_handle.remove()
        detect_handle.remove()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ---- Compute stability metrics ----
    all_coords_arr = np.array(all_token_coords)  # (num_tokens, 5)
    num_tokens = len(all_token_coords)

    # 1. Mean cosine similarity between consecutive 5D snapshots
    consecutive_cosines = []
    for i in range(num_tokens - 1):
        sim = cosine_sim(all_coords_arr[i], all_coords_arr[i + 1])
        consecutive_cosines.append(sim)
    stability_cosine = float(np.mean(consecutive_cosines)) if consecutive_cosines else 0.0

    # 2. Drift angle from initial position (in degrees)
    if num_tokens >= 2:
        initial_coords = all_coords_arr[0]
        final_coords = all_coords_arr[-1]
        cos_drift = cosine_sim(initial_coords, final_coords)
        # Clamp for numerical safety
        cos_drift = max(-1.0, min(1.0, cos_drift))
        drift_angle_deg = float(np.degrees(np.arccos(cos_drift)))
    else:
        drift_angle_deg = 0.0

    # 3. Mean target similarity across all checkpoints
    if checkpoints:
        mean_target_sim = float(np.mean([c["target_sim"] for c in checkpoints]))
    else:
        mean_target_sim = 0.0

    # 4. Std of coordinates over time (per dimension)
    coord_std = all_coords_arr.std(axis=0).tolist() if num_tokens > 1 else [0.0] * 5

    return {
        "checkpoints": checkpoints,
        "stability_cosine": stability_cosine,
        "drift_angle_deg": drift_angle_deg,
        "mean_target_sim": mean_target_sim,
        "coord_std": coord_std,
        "num_tokens_generated": num_tokens,
        "generated_text": generated_text,
    }


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"
    alpha = 3.0
    tokens_per_checkpoint = 25
    max_tokens = 500
    temperature = 0.7
    prompt = "Write a detailed essay about your perspective on life, work, and what matters most to you."

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]
    steer_vectors = model_data["steer_vectors"]

    logger.info("Loading Marin 8B on %s...", device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    print(f"\n{'='*70}")
    print("LONG GENERATION 5D PERSONALITY TRACKING")
    print(f"Model: {model_id}")
    print(f"Alpha: {alpha}, Temp: {temperature}, Max tokens: {max_tokens}")
    print(f"Checkpoint every {tokens_per_checkpoint} tokens")
    print(f"{'='*70}")

    results = {
        "model": model_id,
        "alpha": alpha,
        "tokens_per_checkpoint": tokens_per_checkpoint,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt": prompt,
        "traits": {},
        "baseline": {},
        "summary": {},
    }

    # ================================================================
    # BASELINE: no steering
    # ================================================================
    logger.info("Running baseline (no steering)...")
    print(f"\n{'='*70}")
    print("BASELINE (no steering)")
    print(f"{'='*70}")

    baseline_result = generate_long_with_tracking(
        model, tokenizer, device, blocks, mid_layer, basis_5d,
        coords_5d, steer_vec=None, alpha=0.0, prompt=prompt,
        max_tokens=max_tokens, tokens_per_checkpoint=tokens_per_checkpoint,
        temperature=temperature, trait_name=None,
    )

    results["baseline"] = {
        "checkpoints": baseline_result["checkpoints"],
        "stability_cosine": baseline_result["stability_cosine"],
        "drift_angle_deg": baseline_result["drift_angle_deg"],
        "coord_std": baseline_result["coord_std"],
        "num_tokens_generated": baseline_result["num_tokens_generated"],
        "generated_text": baseline_result["generated_text"],
    }

    print(f"  Tokens generated: {baseline_result['num_tokens_generated']}")
    print(f"  Stability cosine: {baseline_result['stability_cosine']:.4f}")
    print(f"  Drift angle: {baseline_result['drift_angle_deg']:.1f} deg")
    print(f"  Checkpoints: {len(baseline_result['checkpoints'])}")
    print(f"  Text preview: {baseline_result['generated_text'][:150]}...")

    # ================================================================
    # STEERED GENERATION FOR EACH TRAIT
    # ================================================================
    all_stability_cosines = []
    all_drift_angles = []
    all_target_sims = []

    for trait in TRAITS:
        logger.info("Generating with %s steering (alpha=%.1f)...", trait, alpha)
        print(f"\n{'='*70}")
        print(f"TRAIT: {trait.upper()} (alpha={alpha})")
        print(f"{'='*70}")

        vec = steer_vectors[trait]  # Already float32
        trait_result = generate_long_with_tracking(
            model, tokenizer, device, blocks, mid_layer, basis_5d,
            coords_5d, steer_vec=vec, alpha=alpha, prompt=prompt,
            max_tokens=max_tokens, tokens_per_checkpoint=tokens_per_checkpoint,
            temperature=temperature, trait_name=trait,
        )

        results["traits"][trait] = {
            "checkpoints": trait_result["checkpoints"],
            "stability_cosine": trait_result["stability_cosine"],
            "drift_angle_deg": trait_result["drift_angle_deg"],
            "mean_target_sim": trait_result["mean_target_sim"],
            "coord_std": trait_result["coord_std"],
            "num_tokens_generated": trait_result["num_tokens_generated"],
            "generated_text": trait_result["generated_text"],
        }

        all_stability_cosines.append(trait_result["stability_cosine"])
        all_drift_angles.append(trait_result["drift_angle_deg"])
        all_target_sims.append(trait_result["mean_target_sim"])

        print(f"  Tokens generated: {trait_result['num_tokens_generated']}")
        print(f"  Stability cosine: {trait_result['stability_cosine']:.4f}")
        print(f"  Drift angle: {trait_result['drift_angle_deg']:.1f} deg")
        print(f"  Mean target sim: {trait_result['mean_target_sim']:.4f}")
        print(f"  Checkpoints: {len(trait_result['checkpoints'])}")

        # Print checkpoint details
        for cp in trait_result["checkpoints"]:
            print(f"    Token {cp['token']:>4}: target_sim={cp['target_sim']:.3f}, "
                  f"norm={cp['norm']:.1f}, coords={[f'{c:.2f}' for c in cp['coords_5d']]}")

        print(f"  Text preview: {trait_result['generated_text'][:150]}...")

    # ================================================================
    # SUMMARY
    # ================================================================
    mean_stability = float(np.mean(all_stability_cosines))
    mean_drift = float(np.mean(all_drift_angles))
    mean_target = float(np.mean(all_target_sims))

    # Determine conclusion
    if mean_stability > 0.95 and mean_drift < 10:
        conclusion = (
            f"Personality is HIGHLY STABLE during long generation. "
            f"Mean consecutive cosine={mean_stability:.3f}, mean drift={mean_drift:.1f} deg. "
            f"The 5D personality signal remains locked throughout 500+ token responses."
        )
    elif mean_stability > 0.9:
        conclusion = (
            f"Personality is MODERATELY STABLE during long generation. "
            f"Mean consecutive cosine={mean_stability:.3f}, mean drift={mean_drift:.1f} deg. "
            f"Some drift occurs but personality direction is largely maintained."
        )
    else:
        conclusion = (
            f"Personality DRIFTS during long generation. "
            f"Mean consecutive cosine={mean_stability:.3f}, mean drift={mean_drift:.1f} deg. "
            f"The 5D signal degrades over extended generation."
        )

    results["summary"] = {
        "mean_stability_cosine": mean_stability,
        "mean_drift_angle": mean_drift,
        "mean_target_sim": mean_target,
        "baseline_stability_cosine": baseline_result["stability_cosine"],
        "baseline_drift_angle": baseline_result["drift_angle_deg"],
        "conclusion": conclusion,
    }

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Mean stability cosine (steered): {mean_stability:.4f}")
    print(f"  Mean drift angle (steered): {mean_drift:.1f} deg")
    print(f"  Mean target similarity: {mean_target:.4f}")
    print(f"  Baseline stability cosine: {baseline_result['stability_cosine']:.4f}")
    print(f"  Baseline drift angle: {baseline_result['drift_angle_deg']:.1f} deg")
    print(f"\n  Per-trait stability cosines:")
    for trait, sc in zip(TRAITS, all_stability_cosines):
        print(f"    {trait:>15}: {sc:.4f}")
    print(f"\n  Per-trait drift angles:")
    for trait, da in zip(TRAITS, all_drift_angles):
        print(f"    {trait:>15}: {da:.1f} deg")
    print(f"\n  Per-trait mean target similarity:")
    for trait, ts in zip(TRAITS, all_target_sims):
        print(f"    {trait:>15}: {ts:.4f}")
    print(f"\n  Conclusion: {conclusion}")

    # ================================================================
    # SAVE
    # ================================================================
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "long_generation_5d_tracking.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
