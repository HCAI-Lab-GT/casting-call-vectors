#!/usr/bin/env python
"""
Personality Watermark: Encode messages in generated text via 5D subspace.

The 5D personality channel has 21.8 bits of capacity at the activation level.
But can this information survive through autoregressive generation?

Tests:
1. Encode a 5-bit binary code into the steering vector
2. Generate 40 tokens with that code
3. Read the generated text with a fresh (unsteered) forward pass
4. Decode the 5-bit code from the reader's activations
5. Test all 32 codes for reliability
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="watermark")

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


def generate_with_code(model, tokenizer, device, blocks, mid_layer, basis_5d,
                       code_5d, scale, alpha, prompt, max_tokens=40):
    """Generate text with a specific 5D code encoded in steering."""
    # Convert 5D code to full vector
    full_vec = (code_5d @ basis_5d * scale).astype(np.float32)
    delta = alpha * torch.tensor(full_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    for step in range(max_tokens):
        hooks = []
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

        next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return text


def read_code_from_text(model, tokenizer, device, blocks, mid_layer, basis_5d,
                         text, prompt="Tell me about yourself."):
    """Read 5D coordinates from text by processing it as assistant response."""
    detect_layer = mid_layer + 1

    # Text as assistant response
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Baseline
    base_messages = [{"role": "user", "content": prompt}]
    base_formatted = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
    base_enc = tokenizer(base_formatted, return_tensors="pt")
    base_ids = base_enc["input_ids"].to(device)

    captured = {}
    hooks = []
    def cap(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    base_captured = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(base_ids)
    for h in hooks:
        h.remove()

    diff = (captured["act"] - base_captured["act"]).astype(np.float64)
    coords = basis_5d @ diff
    return coords


def main():
    device = "cuda:1"
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

    mean_sv = float(np.mean(singular_values))
    alpha = 2.0
    prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY WATERMARK VIA 5D SUBSPACE")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: 6-trait watermark (encode RIASEC traits)
    # ================================================================
    logger.info("Part 1: RIASEC trait watermark...")
    print(f"\n{'='*70}")
    print("PART 1: RIASEC TRAIT WATERMARK (generate → read)")
    print(f"{'='*70}")

    trait_wm_results = {}
    correct = 0
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        text = generate_with_code(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, coords_5d[trait] / np.linalg.norm(coords_5d[trait]),
                                   mean_sv, alpha, prompt, max_tokens=40)

        received = read_code_from_text(model, tokenizer, device, blocks, mid_layer,
                                        basis_5d, text, prompt)

        norm_r = np.linalg.norm(received)
        sims = {}
        for t in TRAITS:
            if norm_r > 0 and np.linalg.norm(coords_5d[t]) > 0:
                sims[t] = float(np.dot(received, coords_5d[t]) / (
                    norm_r * np.linalg.norm(coords_5d[t])))
            else:
                sims[t] = 0

        detected = max(sims, key=sims.get)
        is_correct = detected == trait
        if is_correct:
            correct += 1

        trait_wm_results[trait] = {
            "detected": detected,
            "correct": is_correct,
            "target_sim": float(sims[trait]),
            "text": text[:100],
            "norm_5d": float(norm_r),
        }
        print(f"  {trait}: detected={detected}, sim={sims[trait]:+.3f}, "
              f"norm={norm_r:.1f}, {'OK' if is_correct else 'FAIL'}")

    results["trait_watermark"] = {
        "correct": correct,
        "total": 6,
        "accuracy": float(correct / 6),
        "traits": trait_wm_results,
    }
    print(f"\n  Trait watermark accuracy: {correct}/6 ({correct/6:.0%})")

    # ================================================================
    # PART 2: 5-bit binary watermark
    # ================================================================
    logger.info("Part 2: 5-bit binary watermark...")
    print(f"\n{'='*70}")
    print("PART 2: 5-BIT BINARY WATERMARK (32 codes)")
    print(f"{'='*70}")

    all_codes = []
    for code in range(32):
        bits = [(code >> i) & 1 for i in range(5)]
        signs = [1 if b else -1 for b in bits]
        all_codes.append(signs)

    correct_decode = 0
    per_bit_correct = [0] * 5
    binary_results = []

    for code_idx, signs in enumerate(all_codes):
        code_5d = np.array(signs, dtype=np.float64)
        code_5d /= np.linalg.norm(code_5d)

        text = generate_with_code(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, code_5d, mean_sv, alpha, prompt,
                                   max_tokens=40)

        received = read_code_from_text(model, tokenizer, device, blocks, mid_layer,
                                        basis_5d, text, prompt)

        # Decode bits
        received_signs = [1 if received[i] > 0 else -1 for i in range(5)]
        bits_correct = sum(1 for a, b in zip(signs, received_signs) if a == b)
        for i in range(5):
            if received_signs[i] == signs[i]:
                per_bit_correct[i] += 1

        # Nearest code decode
        best_code = -1
        best_cos = -2
        for cand_idx, cand_signs in enumerate(all_codes):
            cand_5d = np.array(cand_signs, dtype=np.float64)
            cand_5d /= np.linalg.norm(cand_5d)
            r_norm = max(np.linalg.norm(received), 1e-10)
            cos = float(np.dot(received / r_norm, cand_5d))
            if cos > best_cos:
                best_cos = cos
                best_code = cand_idx

        is_correct = best_code == code_idx
        if is_correct:
            correct_decode += 1

        binary_results.append({
            "code": code_idx,
            "sent_bits": signs,
            "received_bits": received_signs,
            "bits_correct": bits_correct,
            "code_correct": is_correct,
        })

        if code_idx < 8 or is_correct == False:
            print(f"  Code {code_idx:2d} ({signs}): "
                  f"recv=({received_signs}), bits={bits_correct}/5, "
                  f"{'OK' if is_correct else 'FAIL'}")

    code_acc = correct_decode / 32
    mean_bits = float(np.mean([r["bits_correct"] for r in binary_results]))
    print(f"\n  Code accuracy: {correct_decode}/32 ({code_acc:.0%})")
    print(f"  Mean bits correct: {mean_bits:.2f}/5")
    for i in range(5):
        print(f"  Bit {i} (PC{i+1}): {per_bit_correct[i]}/32 ({per_bit_correct[i]/32:.0%})")

    results["binary_watermark"] = {
        "code_accuracy": float(code_acc),
        "correct_codes": correct_decode,
        "mean_bits_correct": mean_bits,
        "per_bit_accuracy": [b / 32 for b in per_bit_correct],
        "information_bits": float(np.log2(max(correct_decode, 1))),
    }

    # ================================================================
    # PART 3: Text length vs watermark reliability
    # ================================================================
    logger.info("Part 3: Text length effect...")
    print(f"\n{'='*70}")
    print("PART 3: TEXT LENGTH VS WATERMARK RELIABILITY")
    print(f"{'='*70}")

    # Use artistic trait, vary generated text length
    test_code = coords_5d["artistic"] / np.linalg.norm(coords_5d["artistic"])
    length_results = {}

    for max_tokens in [10, 20, 40, 80]:
        text = generate_with_code(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, test_code, mean_sv, alpha, prompt,
                                   max_tokens=max_tokens)

        received = read_code_from_text(model, tokenizer, device, blocks, mid_layer,
                                        basis_5d, text, prompt)

        norm_r = np.linalg.norm(received)
        cos = float(np.dot(received / max(norm_r, 1e-10),
                          test_code))

        sims = {}
        for t in TRAITS:
            if norm_r > 0 and np.linalg.norm(coords_5d[t]) > 0:
                sims[t] = float(np.dot(received, coords_5d[t]) / (
                    norm_r * np.linalg.norm(coords_5d[t])))
            else:
                sims[t] = 0
        detected = max(sims, key=sims.get)

        actual_len = len(text.split())
        length_results[max_tokens] = {
            "actual_words": actual_len,
            "cosine_to_code": cos,
            "detected": detected,
            "correct": detected == "artistic",
            "norm_5d": float(norm_r),
        }
        print(f"  {max_tokens} tokens ({actual_len} words): "
              f"cos={cos:.4f}, detected={detected}, norm={norm_r:.1f}")

    results["text_length"] = {str(k): v for k, v in length_results.items()}

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  RIASEC watermark: {results['trait_watermark']['accuracy']:.0%}")
    print(f"  5-bit watermark: {results['binary_watermark']['code_accuracy']:.0%}")
    print(f"  Mean bits through generation: {results['binary_watermark']['mean_bits_correct']:.2f}/5")

    results["summary"] = {
        "trait_watermark_acc": float(results["trait_watermark"]["accuracy"]),
        "binary_watermark_acc": float(results["binary_watermark"]["code_accuracy"]),
        "mean_bits_through_generation": float(results["binary_watermark"]["mean_bits_correct"]),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_watermark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
