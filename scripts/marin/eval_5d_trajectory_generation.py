#!/usr/bin/env python
"""
5D Trajectory During Generation: How does personality evolve in the
5-dimensional personality space across generated tokens?

Previous findings show personality is detectable at EVERY token (100%).
But does the trajectory through 5D space reveal structure?

Tests:
1. Track 5D coordinates at each generated token (60 tokens)
2. Does the trajectory converge to a fixed point or continue wandering?
3. Drift rate: how fast do coordinates change per token?
4. Autocorrelation: is the trajectory smooth or noisy?
5. Cross-trait trajectory comparison: do different traits trace different paths?
6. Holland hexagon traversal: do trajectories respect Holland structure?
7. Variance decomposition: how much trajectory variance is personality vs token identity?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="5d-traj")

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
    }


def generate_with_trajectory(model, tokenizer, device, blocks, mid_layer, basis_5d,
                              coords_5d, steer_vec, alpha, prompt, max_tokens=60):
    """Generate tokens while tracking 5D personality trajectory."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Get baseline hidden state (no steering)
    baseline_captured = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        baseline_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
    base_act = baseline_captured["act"]

    # Generate with steering, capturing trajectory
    trajectory = []
    tokens_generated = []
    gen_ids = input_ids.clone()

    for step in range(max_tokens):
        captured = {}
        hooks = []

        def cap_fn(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

        def steer_fn(_m, _i, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        with torch.no_grad():
            outputs = model(gen_ids)

        for h in hooks:
            h.remove()

        # Get next token
        logits = outputs.logits[0, -1, :]
        next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        token_str = tokenizer.decode(next_token[0])
        tokens_generated.append(token_str)

        # Compute 5D coords from activation diff
        diff = (captured["act"] - base_act).astype(np.float64)
        coords = basis_5d @ diff
        norm_5d = float(np.linalg.norm(coords))

        # Detect trait
        sims = {}
        for t in TRAITS:
            if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                sims[t] = float(np.dot(coords, coords_5d[t]) / (
                    norm_5d * np.linalg.norm(coords_5d[t])))
            else:
                sims[t] = 0
        detected = max(sims, key=sims.get)

        trajectory.append({
            "step": step,
            "coords": coords.tolist(),
            "norm": norm_5d,
            "detected": detected,
            "cos_target": sims.get(detected, 0),
        })

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return trajectory, tokens_generated


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

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    alpha = 2.0
    prompt = "Tell me about yourself."
    max_tokens = 60
    results = {}

    print(f"\n{'='*70}")
    print("5D TRAJECTORY DURING GENERATION")
    print(f"Model: Marin 8B, α={alpha}, {max_tokens} tokens")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Full trajectory per trait
    # ================================================================
    logger.info("Part 1: Full trajectory per trait...")
    print(f"\n{'='*70}")
    print("PART 1: FULL 5D TRAJECTORY PER TRAIT")
    print(f"{'='*70}")

    trait_trajectories = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        traj, tokens = generate_with_trajectory(
            model, tokenizer, device, blocks, mid_layer, basis_5d,
            coords_5d, vec, alpha, prompt, max_tokens)

        trait_trajectories[trait] = {
            "trajectory": traj,
            "tokens": tokens,
        }

        # Summary statistics
        all_coords = np.array([t["coords"] for t in traj])
        norms = np.array([t["norm"] for t in traj])
        detections = [t["detected"] for t in traj]
        correct = sum(1 for d in detections if d == trait)

        print(f"\n  {trait}:")
        print(f"    Detection: {correct}/{len(traj)} ({correct/len(traj):.0%})")
        print(f"    5D norm: mean={norms.mean():.1f}, std={norms.std():.1f}")
        print(f"    Coord ranges: {', '.join(f'PC{i+1}:[{all_coords[:,i].min():.1f},{all_coords[:,i].max():.1f}]' for i in range(5))}")

    results["trajectories"] = trait_trajectories

    # ================================================================
    # PART 2: Convergence analysis
    # ================================================================
    logger.info("Part 2: Convergence analysis...")
    print(f"\n{'='*70}")
    print("PART 2: CONVERGENCE ANALYSIS")
    print(f"{'='*70}")

    convergence_results = {}
    for trait in TRAITS:
        traj = trait_trajectories[trait]["trajectory"]
        all_coords = np.array([t["coords"] for t in traj])

        if len(all_coords) < 10:
            continue

        # Drift rate: L2 distance between consecutive points
        drifts = [float(np.linalg.norm(all_coords[i+1] - all_coords[i]))
                   for i in range(len(all_coords) - 1)]

        # First half vs second half drift
        mid = len(drifts) // 2
        first_half_drift = np.mean(drifts[:mid])
        second_half_drift = np.mean(drifts[mid:])

        # Distance from initial point over time
        dist_from_start = [float(np.linalg.norm(all_coords[i] - all_coords[0]))
                           for i in range(len(all_coords))]

        # Distance from mean
        mean_coords = all_coords.mean(axis=0)
        dist_from_mean = [float(np.linalg.norm(all_coords[i] - mean_coords))
                          for i in range(len(all_coords))]

        # Autocorrelation of coords (lag-1)
        autocorrs = []
        for pc in range(5):
            series = all_coords[:, pc]
            if len(series) > 2:
                corr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorrs.append(float(corr))

        converging = second_half_drift < first_half_drift
        convergence_results[trait] = {
            "mean_drift": float(np.mean(drifts)),
            "first_half_drift": float(first_half_drift),
            "second_half_drift": float(second_half_drift),
            "converging": bool(converging),
            "drift_ratio": float(second_half_drift / first_half_drift) if first_half_drift > 0 else 0,
            "max_dist_from_start": float(max(dist_from_start)),
            "mean_dist_from_mean": float(np.mean(dist_from_mean)),
            "lag1_autocorr": autocorrs,
        }

        print(f"  {trait:>15}: drift_ratio={second_half_drift/first_half_drift:.3f} "
              f"({'converging' if converging else 'diverging'}), "
              f"autocorr={np.mean(autocorrs):.3f}")

    results["convergence"] = convergence_results

    # ================================================================
    # PART 3: Cross-trait trajectory distances
    # ================================================================
    logger.info("Part 3: Cross-trait distances...")
    print(f"\n{'='*70}")
    print("PART 3: TRAJECTORY SEPARATION BETWEEN TRAITS")
    print(f"{'='*70}")

    # Average 5D coordinates per trait across time
    trait_centroids = {}
    for trait in TRAITS:
        traj = trait_trajectories[trait]["trajectory"]
        all_coords = np.array([t["coords"] for t in traj])
        trait_centroids[trait] = all_coords.mean(axis=0)

    # Cross-trait centroid distances
    print(f"\n  Cross-trait centroid distances:")
    print(f"  {'':>15}", end="")
    for t2 in TRAITS:
        print(f" {t2[:5]:>7}", end="")
    print()

    cross_distances = {}
    for t1 in TRAITS:
        print(f"  {t1:>15}", end="")
        for t2 in TRAITS:
            dist = float(np.linalg.norm(trait_centroids[t1] - trait_centroids[t2]))
            print(f" {dist:>7.1f}", end="")
            cross_distances[f"{t1}_{t2}"] = dist
        print()

    results["cross_trait_distances"] = cross_distances

    # ================================================================
    # PART 4: Per-PC variance decomposition
    # ================================================================
    logger.info("Part 4: Variance decomposition...")
    print(f"\n{'='*70}")
    print("PART 4: VARIANCE DECOMPOSITION (PERSONALITY vs TOKEN)")
    print(f"{'='*70}")

    # Collect all coordinates: shape (num_traits, num_tokens, 5)
    all_data = []
    labels = []
    for trait in TRAITS:
        traj = trait_trajectories[trait]["trajectory"]
        coords = np.array([t["coords"] for t in traj])
        all_data.append(coords)
        labels.extend([trait] * len(coords))

    # Truncate to same length
    min_len = min(len(d) for d in all_data)
    all_data = np.stack([d[:min_len] for d in all_data])  # (6, min_len, 5)

    # Total variance per PC
    flat = all_data.reshape(-1, 5)
    total_var = np.var(flat, axis=0)

    # Between-trait variance per PC
    trait_means = all_data.mean(axis=1)  # (6, 5)
    between_var = np.var(trait_means, axis=0) * min_len / (min_len + 1)  # approximate

    # Within-trait variance per PC
    within_var = total_var - between_var

    # Personality fraction per PC
    personality_frac = between_var / (total_var + 1e-10)

    print(f"\n  Per-PC variance decomposition:")
    print(f"  {'PC':>4} {'Total Var':>10} {'Between':>10} {'Within':>10} {'Personality%':>12}")
    for i in range(5):
        print(f"  PC{i+1} {total_var[i]:>10.2f} {between_var[i]:>10.2f} "
              f"{within_var[i]:>10.2f} {personality_frac[i]:>12.1%}")

    overall_personality_frac = between_var.sum() / (total_var.sum() + 1e-10)
    print(f"\n  Overall personality fraction: {overall_personality_frac:.1%}")

    results["variance_decomposition"] = {
        "total_var": total_var.tolist(),
        "between_var": between_var.tolist(),
        "within_var": within_var.tolist(),
        "personality_frac": personality_frac.tolist(),
        "overall_personality_frac": float(overall_personality_frac),
    }

    # ================================================================
    # PART 5: Holland structure in trajectories
    # ================================================================
    logger.info("Part 5: Holland structure...")
    print(f"\n{'='*70}")
    print("PART 5: HOLLAND STRUCTURE IN TRAJECTORY CENTROIDS")
    print(f"{'='*70}")

    holland_order = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]

    # Compute angles between trait centroids in PC1-PC2 plane
    trait_angles = {}
    for trait in TRAITS:
        c = trait_centroids[trait]
        angle = np.arctan2(c[1], c[0]) * 180 / np.pi
        trait_angles[trait] = float(angle)

    # Sort by angle
    sorted_by_angle = sorted(TRAITS, key=lambda t: trait_angles[t])

    print(f"\n  Trait centroids in PC1-PC2 (sorted by angle):")
    for t in sorted_by_angle:
        print(f"    {t:>15}: {trait_angles[t]:+.1f}°")

    # Check if order matches Holland hexagon
    # Holland order: R-I-A-S-E-C (60° apart in ideal hexagon)
    # Check if our angular order respects adjacency
    adjacency_preserved = 0
    for i in range(6):
        h_cur = holland_order[i]
        h_next = holland_order[(i + 1) % 6]
        # Check if they are adjacent in our angular ordering
        idx_cur = sorted_by_angle.index(h_cur) if h_cur in sorted_by_angle else -1
        idx_next = sorted_by_angle.index(h_next) if h_next in sorted_by_angle else -1
        if idx_cur >= 0 and idx_next >= 0:
            dist = min(abs(idx_next - idx_cur), 6 - abs(idx_next - idx_cur))
            if dist <= 2:
                adjacency_preserved += 1

    print(f"\n  Holland adjacency preserved: {adjacency_preserved}/6")

    results["holland_structure"] = {
        "trait_angles": trait_angles,
        "angular_order": sorted_by_angle,
        "adjacency_preserved": adjacency_preserved,
    }

    # ================================================================
    # PART 6: Trajectory smoothness
    # ================================================================
    logger.info("Part 6: Trajectory smoothness...")
    print(f"\n{'='*70}")
    print("PART 6: TRAJECTORY SMOOTHNESS (JERK & CURVATURE)")
    print(f"{'='*70}")

    smoothness_results = {}
    for trait in TRAITS:
        traj = trait_trajectories[trait]["trajectory"]
        all_coords = np.array([t["coords"] for t in traj])

        if len(all_coords) < 4:
            continue

        # Velocity (first differences)
        velocity = np.diff(all_coords, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        # Acceleration (second differences)
        acceleration = np.diff(velocity, axis=0)
        accel_mag = np.linalg.norm(acceleration, axis=1)

        # Jerk (third differences)
        jerk = np.diff(acceleration, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=1)

        # Path length vs displacement
        path_length = float(speed.sum())
        displacement = float(np.linalg.norm(all_coords[-1] - all_coords[0]))
        straightness = displacement / path_length if path_length > 0 else 0

        smoothness_results[trait] = {
            "mean_speed": float(speed.mean()),
            "mean_acceleration": float(accel_mag.mean()),
            "mean_jerk": float(jerk_mag.mean()),
            "path_length": path_length,
            "displacement": displacement,
            "straightness": float(straightness),
        }

        print(f"  {trait:>15}: straightness={straightness:.3f}, "
              f"speed={speed.mean():.1f}, accel={accel_mag.mean():.1f}")

    results["smoothness"] = smoothness_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    converging_count = sum(1 for v in convergence_results.values() if v["converging"])
    mean_straightness = np.mean([v["straightness"] for v in smoothness_results.values()])
    print(f"  Converging trajectories: {converging_count}/{len(convergence_results)}")
    print(f"  Mean straightness: {mean_straightness:.3f} (1=straight line, 0=random walk)")
    print(f"  Overall personality variance fraction: {overall_personality_frac:.1%}")
    print(f"  Holland adjacency preserved: {adjacency_preserved}/6")

    results["summary"] = {
        "converging_count": converging_count,
        "mean_straightness": float(mean_straightness),
        "overall_personality_frac": float(overall_personality_frac),
        "holland_adjacency": adjacency_preserved,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "5d_trajectory_generation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
