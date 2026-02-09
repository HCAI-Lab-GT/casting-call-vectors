#!/usr/bin/env python
"""
Personality as an Information Channel.

The 5D personality subspace can encode arbitrary directions.
What's the channel capacity? Can we encode arbitrary 5D messages
using the personality subspace as a covert communication channel?

Tests:
1. Encode arbitrary 5D unit vectors → detect from activations
2. Channel capacity: how many distinguishable directions?
3. Signal-to-noise ratio at different alphas
4. Binary encoding: use each of 5 PCs as 1 bit → 32 distinct messages
5. Precision: how precisely can we control the 5D coordinates?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="info-chan")

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
    _, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return residual, mid_layer, basis_5d, coords_5d, S_res[:5]


def inject_and_read(model, tokenizer, device, blocks, mid_layer, basis_5d,
                    steer_vec_full, alpha, prompt):
    """Inject a vector and read back the 5D coordinates."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec_full, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Baseline
    base_cap = {}
    hooks = []
    def cb(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cb))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    # Steered
    steer_cap = {}
    hooks = []
    def sf(_m, _i, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out
    hooks.append(blocks[mid_layer].register_forward_hook(sf))
    def cf(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        steer_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cf))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    diff = (steer_cap["act"] - base_cap["act"]).astype(np.float64)
    coords = basis_5d @ diff
    return coords


def main():
    device = "cuda:2"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    residual, mid_layer, basis_5d, coords_5d, singular_values = load_model_data(model_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    alpha = 2.0
    prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY AS INFORMATION CHANNEL")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Encode arbitrary 5D unit vectors
    # ================================================================
    logger.info("Part 1: Arbitrary 5D encoding...")
    print(f"\n{'='*70}")
    print("PART 1: ARBITRARY 5D DIRECTION ENCODING")
    print(f"{'='*70}")

    # Generate random unit vectors in 5D
    np.random.seed(42)
    n_directions = 50
    random_dirs_5d = np.random.randn(n_directions, 5)
    random_dirs_5d /= np.linalg.norm(random_dirs_5d, axis=1, keepdims=True)

    cosine_accuracies = []
    coord_errors = []

    for i in range(n_directions):
        # Convert 5D direction to full-dimensional vector via basis
        target_5d = random_dirs_5d[i]
        # Scale by mean singular value to match personality vector magnitudes
        mean_sv = float(np.mean(singular_values))
        full_vec = (target_5d @ basis_5d * mean_sv).astype(np.float32)

        received = inject_and_read(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, full_vec, alpha, prompt)

        # Normalize received
        norm_r = np.linalg.norm(received)
        if norm_r > 0:
            received_unit = received / norm_r
        else:
            received_unit = np.zeros(5)

        cos_sim = float(np.dot(target_5d, received_unit))
        cosine_accuracies.append(cos_sim)

        # Coordinate-wise error
        if norm_r > 0:
            expected_coords = target_5d * mean_sv * alpha
            error = float(np.linalg.norm(received - expected_coords) / np.linalg.norm(expected_coords))
            coord_errors.append(error)

    mean_cos = float(np.mean(cosine_accuracies))
    std_cos = float(np.std(cosine_accuracies))
    print(f"  Random 5D directions: mean cosine = {mean_cos:.4f} ± {std_cos:.4f}")
    print(f"  Min cosine: {min(cosine_accuracies):.4f}")
    print(f"  Mean relative coordinate error: {np.mean(coord_errors):.4f}")

    results["arbitrary_5d"] = {
        "n_directions": n_directions,
        "mean_cosine": mean_cos,
        "std_cosine": std_cos,
        "min_cosine": float(min(cosine_accuracies)),
        "mean_coord_error": float(np.mean(coord_errors)),
    }

    # ================================================================
    # PART 2: Binary encoding (5 bits → 32 messages)
    # ================================================================
    logger.info("Part 2: Binary encoding (32 messages)...")
    print(f"\n{'='*70}")
    print("PART 2: BINARY ENCODING (5 BITS = 32 DISTINCT MESSAGES)")
    print(f"{'='*70}")

    all_codes = []
    for code in range(32):
        bits = [(code >> i) & 1 for i in range(5)]
        signs = [1 if b else -1 for b in bits]
        all_codes.append(signs)

    correct_decode = 0
    decode_results = []

    for code_idx, signs in enumerate(all_codes):
        target_5d = np.array(signs, dtype=np.float64)
        target_5d /= np.linalg.norm(target_5d)

        mean_sv = float(np.mean(singular_values))
        full_vec = (target_5d @ basis_5d * mean_sv).astype(np.float32)

        received = inject_and_read(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, full_vec, alpha, prompt)

        # Decode: find nearest code
        best_code = -1
        best_cos = -2
        for cand_idx, cand_signs in enumerate(all_codes):
            cand_5d = np.array(cand_signs, dtype=np.float64)
            cand_5d /= np.linalg.norm(cand_5d)
            cos = float(np.dot(received / max(np.linalg.norm(received), 1e-10), cand_5d))
            if cos > best_cos:
                best_cos = cos
                best_code = cand_idx

        is_correct = best_code == code_idx
        if is_correct:
            correct_decode += 1

        # Also decode per-bit
        received_signs = [1 if received[i] > 0 else -1 for i in range(5)]
        bits_correct = sum(1 for a, b in zip(signs, received_signs) if a == b)

        decode_results.append({
            "code": code_idx,
            "sent_signs": signs,
            "received_signs": received_signs,
            "decoded_code": best_code,
            "correct": is_correct,
            "cosine": float(best_cos),
            "bits_correct": bits_correct,
        })

    decode_acc = correct_decode / 32
    mean_bits = float(np.mean([d["bits_correct"] for d in decode_results]))
    print(f"  32-code accuracy: {correct_decode}/32 ({decode_acc:.0%})")
    print(f"  Mean bits correct: {mean_bits:.2f}/5")
    print(f"  Information capacity: {np.log2(correct_decode) if correct_decode > 0 else 0:.2f} bits")

    # Per-bit accuracy
    for bit_idx in range(5):
        bit_correct = sum(1 for d in decode_results
                         if d["received_signs"][bit_idx] == d["sent_signs"][bit_idx])
        print(f"  Bit {bit_idx} (PC{bit_idx+1}): {bit_correct}/32 ({bit_correct/32:.0%})")

    results["binary_encoding"] = {
        "code_accuracy": float(decode_acc),
        "mean_bits_correct": mean_bits,
        "information_bits": float(np.log2(max(correct_decode, 1))),
        "per_bit_accuracy": [
            sum(1 for d in decode_results
                if d["received_signs"][i] == d["sent_signs"][i]) / 32
            for i in range(5)
        ],
    }

    # ================================================================
    # PART 3: SNR at different alphas
    # ================================================================
    logger.info("Part 3: SNR at different alphas...")
    print(f"\n{'='*70}")
    print("PART 3: SIGNAL-TO-NOISE RATIO VS ALPHA")
    print(f"{'='*70}")

    # Use 3 prompts to measure noise (variance across prompts for same direction)
    snr_prompts = [
        "Tell me about yourself.",
        "What do you care about?",
        "How do you approach problems?",
    ]

    snr_results = {}
    for a in [0.5, 1.0, 2.0, 3.0, 5.0]:
        # Send artistic direction at different alphas
        vec = residual["artistic"].astype(np.float32)
        received_all = []
        for p in snr_prompts:
            received = inject_and_read(model, tokenizer, device, blocks, mid_layer,
                                       basis_5d, vec, a, p)
            received_all.append(received)

        received_arr = np.array(received_all)
        signal = np.mean(received_arr, axis=0)
        noise = received_arr - signal
        signal_power = float(np.linalg.norm(signal)**2)
        noise_power = float(np.mean([np.linalg.norm(n)**2 for n in noise]))

        snr = signal_power / max(noise_power, 1e-10)
        snr_db = 10 * np.log10(snr) if snr > 0 else -np.inf

        # Cross-prompt cosine consistency
        cosines = []
        for i in range(len(received_all)):
            for j in range(i+1, len(received_all)):
                ni = np.linalg.norm(received_all[i])
                nj = np.linalg.norm(received_all[j])
                if ni > 0 and nj > 0:
                    cosines.append(float(np.dot(received_all[i], received_all[j]) / (ni * nj)))

        snr_results[a] = {
            "snr": float(snr),
            "snr_db": float(snr_db),
            "signal_norm": float(np.linalg.norm(signal)),
            "mean_noise_norm": float(np.mean([np.linalg.norm(n) for n in noise])),
            "cross_prompt_cosine": float(np.mean(cosines)),
        }
        print(f"  α={a}: SNR={snr:.1f} ({snr_db:.1f} dB), "
              f"cross-prompt cos={np.mean(cosines):.4f}")

    results["snr_vs_alpha"] = {str(k): v for k, v in snr_results.items()}

    # ================================================================
    # PART 4: Precision test (graded amplitudes)
    # ================================================================
    logger.info("Part 4: Coordinate precision...")
    print(f"\n{'='*70}")
    print("PART 4: COORDINATE PRECISION (graded amplitudes)")
    print(f"{'='*70}")

    # Send PC1 at different amplitudes, measure received amplitude
    pc1_dir = basis_5d[0]
    target_amps = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    received_amps = []

    for amp in target_amps:
        full_vec = (amp * singular_values[0] * pc1_dir).astype(np.float32)
        received = inject_and_read(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, full_vec, 1.0, prompt)
        received_pc1 = float(received[0])
        received_amps.append(received_pc1)

    # Linearity
    target_arr = np.array([a * singular_values[0] for a in target_amps])
    received_arr = np.array(received_amps)
    if len(target_arr) > 2:
        corr = float(np.corrcoef(target_arr, received_arr)[0, 1])
    else:
        corr = None

    print(f"  Amplitude linearity: r = {corr:.6f}")
    for amp, recv in zip(target_amps, received_amps):
        expected = amp * singular_values[0]
        ratio = recv / expected if expected != 0 else 0
        print(f"    amp={amp}: expected={expected:.1f}, received={recv:.1f}, ratio={ratio:.4f}")

    results["precision"] = {
        "target_amplitudes": target_amps,
        "received_amplitudes": received_amps,
        "linearity_r": corr,
    }

    # ================================================================
    # PART 5: Channel capacity estimation
    # ================================================================
    logger.info("Part 5: Channel capacity estimation...")
    print(f"\n{'='*70}")
    print("PART 5: CHANNEL CAPACITY ESTIMATE")
    print(f"{'='*70}")

    # Sample many random directions, measure angular error
    n_test = 100
    np.random.seed(123)
    angular_errors = []
    for i in range(n_test):
        target_5d = np.random.randn(5)
        target_5d /= np.linalg.norm(target_5d)
        mean_sv = float(np.mean(singular_values))
        full_vec = (target_5d @ basis_5d * mean_sv).astype(np.float32)

        received = inject_and_read(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, full_vec, alpha, prompt)
        norm_r = np.linalg.norm(received)
        if norm_r > 0:
            cos = np.clip(np.dot(target_5d, received / norm_r), -1, 1)
            angle = float(np.arccos(cos) * 180 / np.pi)
            angular_errors.append(angle)

    mean_angle = float(np.mean(angular_errors))
    std_angle = float(np.std(angular_errors))
    max_angle = float(max(angular_errors))

    # Channel capacity: how many non-overlapping cones fit in 5D unit sphere
    # Angular resolution = mean_angle degrees
    # Solid angle of cap ∝ sin^(d-2)(θ) for d dimensions
    # Approximate: number of distinguishable directions ≈ (180/mean_angle)^(d-1) / constant
    n_distinguishable = int((90 / mean_angle) ** 4) if mean_angle > 0 else 0
    bits_capacity = float(np.log2(max(n_distinguishable, 1)))

    print(f"  Mean angular error: {mean_angle:.2f}° ± {std_angle:.2f}°")
    print(f"  Max angular error: {max_angle:.2f}°")
    print(f"  Estimated distinguishable directions: ~{n_distinguishable}")
    print(f"  Estimated channel capacity: ~{bits_capacity:.1f} bits")

    results["channel_capacity"] = {
        "n_test": n_test,
        "mean_angular_error_deg": mean_angle,
        "std_angular_error_deg": std_angle,
        "max_angular_error_deg": max_angle,
        "est_distinguishable_directions": n_distinguishable,
        "est_capacity_bits": bits_capacity,
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Arbitrary 5D encoding: mean cos = {results['arbitrary_5d']['mean_cosine']:.4f}")
    print(f"  32-code binary: {results['binary_encoding']['code_accuracy']:.0%}")
    print(f"  5-bit accuracy: {results['binary_encoding']['mean_bits_correct']:.2f}/5")
    print(f"  Channel capacity: ~{results['channel_capacity']['est_capacity_bits']:.1f} bits")
    print(f"  Amplitude linearity: r = {results['precision']['linearity_r']:.6f}")

    results["summary"] = {
        "arbitrary_5d_cosine": float(results['arbitrary_5d']['mean_cosine']),
        "binary_32_accuracy": float(results['binary_encoding']['code_accuracy']),
        "channel_capacity_bits": float(results['channel_capacity']['est_capacity_bits']),
        "amplitude_linearity": results['precision']['linearity_r'],
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_information_channel.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
