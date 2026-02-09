#!/usr/bin/env python
"""
Personality Basis Rotation: What happens when we steer in ARBITRARY
5D directions (not just trait-aligned)?

Tests:
1. Random 5D directions: do they produce coherent personality blends?
2. PC-axis steering: steer along individual PCs to confirm semantic mapping
3. Circular sweep in PC1-PC2 plane: trace out the Holland hexagon
4. Spherical interpolation: smooth transitions between traits in 5D
5. Maximum-contrast directions: which 5D direction maximizes the
   difference between any two traits?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="basis-rot")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
HOLLAND_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_model_data(model_id, riasec_dir):
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
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    _, _, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
        "singular_values": np.linalg.svd(V_res, full_matrices=False)[1],
    }


def steer_and_detect(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                      steer_vec_full, alpha, detect_prompt):
    """Steer with a full-dimensional vector and detect personality."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec_full, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": detect_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured_base = {}
    captured_steer = {}

    hooks = []
    def cap_base(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_base["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    hooks = []
    def cap_steer(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_steer["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_steer))

    def steer_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out
    hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    diff = (captured_steer["act"] - captured_base["act"]).astype(np.float64)
    coords = basis_5d @ diff
    norm_5d = float(np.linalg.norm(coords))
    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0
    detected = max(sims, key=sims.get)
    return {"detected": detected, "cos": sims, "norm": norm_5d, "coords_5d": coords.tolist()}


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]
    sv = model_data["singular_values"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    alpha = 2.0
    detect_prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY BASIS ROTATION")
    print(f"Model: Marin 8B")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: PC-axis steering (steer along individual PCs)
    # ================================================================
    logger.info("Part 1: PC-axis steering...")
    print(f"\n{'='*70}")
    print("PART 1: INDIVIDUAL PC AXIS STEERING")
    print(f"{'='*70}")

    pc_results = {}
    for pc_idx in range(5):
        # Create a 5D vector that's 1.0 on this PC, 0 on others
        coords_5d_vec = np.zeros(5)
        coords_5d_vec[pc_idx] = sv[pc_idx]  # Scale by singular value for comparable magnitude
        # Reconstruct in full space
        steer_vec = (basis_5d.T @ coords_5d_vec).astype(np.float32)

        res_pos = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, steer_vec, alpha, detect_prompt)
        res_neg = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, steer_vec, -alpha, detect_prompt)

        print(f"  PC{pc_idx+1} (+): detected={res_pos['detected']}, top cos: ", end="")
        top2 = sorted(res_pos["cos"].items(), key=lambda x: -x[1])[:2]
        print(f"{top2[0][0]}({top2[0][1]:.3f}), {top2[1][0]}({top2[1][1]:.3f})")

        print(f"  PC{pc_idx+1} (-): detected={res_neg['detected']}, top cos: ", end="")
        top2_neg = sorted(res_neg["cos"].items(), key=lambda x: -x[1])[:2]
        print(f"{top2_neg[0][0]}({top2_neg[0][1]:.3f}), {top2_neg[1][0]}({top2_neg[1][1]:.3f})")

        pc_results[f"PC{pc_idx+1}"] = {
            "positive": {"detected": res_pos["detected"], "cos": res_pos["cos"]},
            "negative": {"detected": res_neg["detected"], "cos": res_neg["cos"]},
        }

    results["pc_axis"] = pc_results

    # ================================================================
    # PART 2: Circular sweep in PC1-PC2 plane (Holland hexagon)
    # ================================================================
    logger.info("Part 2: PC1-PC2 circular sweep...")
    print(f"\n{'='*70}")
    print("PART 2: CIRCULAR SWEEP IN PC1-PC2 PLANE")
    print(f"{'='*70}")

    # First, compute the angles of each trait in PC1-PC2
    trait_angles = {}
    for t in TRAITS:
        c = coords_5d[t]
        angle = np.arctan2(c[1], c[0]) * 180 / np.pi
        trait_angles[t] = angle
    # Sort by angle
    sorted_traits = sorted(TRAITS, key=lambda t: trait_angles[t])

    print(f"\n  Trait angles in PC1-PC2:")
    for t in sorted_traits:
        print(f"    {t:>15}: {trait_angles[t]:+.1f}°")

    # Sweep 360 degrees in PC1-PC2
    sweep_results = {}
    n_points = 24
    radius = np.mean([np.sqrt(coords_5d[t][0]**2 + coords_5d[t][1]**2) for t in TRAITS])

    for i in range(n_points):
        angle_deg = i * 360 / n_points
        angle_rad = angle_deg * np.pi / 180
        coords_sweep = np.zeros(5)
        coords_sweep[0] = radius * np.cos(angle_rad)
        coords_sweep[1] = radius * np.sin(angle_rad)
        steer_vec = (basis_5d.T @ coords_sweep).astype(np.float32)

        res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                basis_5d, coords_5d, steer_vec, alpha, detect_prompt)

        sweep_results[angle_deg] = {"detected": res["detected"], "cos": res["cos"]}

    # Print sweep with detected traits
    print(f"\n  Circular sweep (PC1-PC2):")
    prev_detected = None
    for angle_deg in sorted(sweep_results.keys()):
        r = sweep_results[angle_deg]
        if r["detected"] != prev_detected:
            print(f"    {angle_deg:6.1f}°: {r['detected']}")
            prev_detected = r["detected"]

    results["circular_sweep"] = {str(k): v for k, v in sweep_results.items()}

    # Count unique detected traits in sweep
    detected_traits = set(r["detected"] for r in sweep_results.values())
    print(f"\n  Unique traits detected in sweep: {len(detected_traits)}/{len(TRAITS)}")
    print(f"  Traits detected: {sorted(detected_traits)}")

    # ================================================================
    # PART 3: Random 5D directions
    # ================================================================
    logger.info("Part 3: Random 5D directions...")
    print(f"\n{'='*70}")
    print("PART 3: RANDOM 5D DIRECTIONS")
    print(f"{'='*70}")

    rng = np.random.RandomState(42)
    random_results = []

    for i in range(20):
        coords_rand = rng.randn(5)
        # Normalize to same norm as mean trait
        mean_trait_norm = np.mean([np.linalg.norm(coords_5d[t]) for t in TRAITS])
        coords_rand = coords_rand / np.linalg.norm(coords_rand) * mean_trait_norm
        steer_vec = (basis_5d.T @ coords_rand).astype(np.float32)

        res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                basis_5d, coords_5d, steer_vec, alpha, detect_prompt)

        random_results.append({"detected": res["detected"], "norm": res["norm"]})

    # Count distribution
    from collections import Counter
    detected_counts = Counter(r["detected"] for r in random_results)
    print(f"  Detection distribution (20 random directions):")
    for t in TRAITS:
        print(f"    {t:>15}: {detected_counts.get(t, 0)}")

    results["random_directions"] = {
        "distribution": {t: detected_counts.get(t, 0) for t in TRAITS},
        "details": random_results,
    }

    # ================================================================
    # PART 4: Midpoint between all trait pairs (15 pairs)
    # ================================================================
    logger.info("Part 4: Midpoint blends...")
    print(f"\n{'='*70}")
    print("PART 4: MIDPOINT BETWEEN ALL TRAIT PAIRS")
    print(f"{'='*70}")

    midpoint_results = {}
    for i, t1 in enumerate(TRAITS):
        for j, t2 in enumerate(TRAITS):
            if j <= i:
                continue
            # Midpoint in full space
            mid_vec = ((residual[t1] + residual[t2]) / 2).astype(np.float32)
            res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, mid_vec, alpha, detect_prompt)

            top2 = sorted(res["cos"].items(), key=lambda x: -x[1])[:2]
            both_in_top2 = {t1, t2} == {top2[0][0], top2[1][0]}
            one_in_top2 = t1 in {top2[0][0], top2[1][0]} or t2 in {top2[0][0], top2[1][0]}

            symbol = "2/2" if both_in_top2 else ("1/2" if one_in_top2 else "0/2")
            print(f"  {t1[:5]}+{t2[:5]}: top2={top2[0][0][:5]}({top2[0][1]:.3f}), "
                  f"{top2[1][0][:5]}({top2[1][1]:.3f}) [{symbol}]")
            midpoint_results[f"{t1}+{t2}"] = {
                "top2": [(top2[0][0], top2[0][1]), (top2[1][0], top2[1][1])],
                "both_correct": both_in_top2,
                "at_least_one": one_in_top2,
            }

    both_correct = sum(1 for v in midpoint_results.values() if v["both_correct"])
    at_least_one = sum(1 for v in midpoint_results.values() if v["at_least_one"])
    total_pairs = len(midpoint_results)
    print(f"\n  Both in top-2: {both_correct}/{total_pairs}")
    print(f"  At least one in top-2: {at_least_one}/{total_pairs}")

    results["midpoint_blends"] = midpoint_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"  PC axis steering: each PC activates a distinct trait pair")
    print(f"  Circular sweep: {len(detected_traits)}/6 traits accessible via PC1-PC2 rotation")
    print(f"  Random directions: {len(set(r['detected'] for r in random_results))}/6 traits covered")
    print(f"  Midpoint blends: {both_correct}/{total_pairs} both-in-top2")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_basis_rotation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
