#!/usr/bin/env python
"""
Personality Under Structured Output Constraints.

Does personality survive when the model is asked to produce
structured formats like JSON, code, markdown lists, etc.?

This tests whether format constraints override personality steering
or whether personality operates in an orthogonal subspace.

Tests:
1. JSON output: personality detection from activations during JSON generation
2. Code output: personality during Python code generation
3. Bulleted lists: personality during list generation
4. Formal letter: personality during highly structured text
5. Comparison: same-format personality differentiation
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="struct-out")

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

    return residual, mid_layer, basis_5d, coords_5d


def detect_personality_multitoken(model, tokenizer, device, blocks, mid_layer,
                                   basis_5d, coords_5d, steer_vec, alpha,
                                   prompt, max_tokens=30):
    """Generate tokens with steering and detect personality at each token."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    gen_ids = enc["input_ids"].to(device)

    # Baseline
    base_cap = {}
    hooks = []
    def cb(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cb))
    with torch.no_grad():
        model(gen_ids)
    for h in hooks:
        h.remove()
    base_act = base_cap["act"]

    per_token = []
    for step in range(max_tokens):
        cap = {}
        hooks = []

        def cap_fn(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            cap["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
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

        next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)

        diff = (cap["act"] - base_act).astype(np.float64)
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
        per_token.append(detected)

        if next_token.item() == tokenizer.eos_token_id:
            break

    prompt_len = enc["input_ids"].shape[1]
    text = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
    return per_token, text


def main():
    device = "cuda:3"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    residual, mid_layer, basis_5d, coords_5d = load_model_data(model_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    alpha = 2.0
    results = {}

    print(f"\n{'='*70}")
    print("PERSONALITY UNDER STRUCTURED OUTPUT CONSTRAINTS")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # Format-specific prompts
    format_prompts = {
        "json": 'Respond with a JSON object containing your name, top 3 hobbies, and favorite subject. Only output valid JSON.',
        "code": 'Write a short Python function that reflects your interests. Include comments explaining your choices.',
        "list": 'List your top 5 values in life as a numbered list. Be specific.',
        "formal_letter": 'Write a brief formal letter introducing yourself to a new colleague. Use proper letter format.',
        "markdown": 'Create a markdown profile card about yourself with headers for Background, Interests, and Goals.',
        "csv": 'Create a CSV table with columns: Activity, Enjoyment(1-10), Frequency. List 5 activities.',
    }

    # ================================================================
    # PART 1: Per-format personality detection
    # ================================================================
    logger.info("Part 1: Per-format detection...")
    print(f"\n{'='*70}")
    print("PART 1: PER-FORMAT PERSONALITY DETECTION")
    print(f"{'='*70}")

    format_results = {}
    for fmt_name, fmt_prompt in format_prompts.items():
        correct_majority = 0
        correct_per_token = 0
        total_tokens = 0
        fmt_detail = {}

        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            detections, text = detect_personality_multitoken(
                model, tokenizer, device, blocks, mid_layer,
                basis_5d, coords_5d, vec, alpha, fmt_prompt, max_tokens=30)

            per_token_correct = sum(1 for d in detections if d == trait)
            correct_per_token += per_token_correct
            total_tokens += len(detections)

            from collections import Counter
            if detections:
                majority = Counter(detections).most_common(1)[0][0]
                if majority == trait:
                    correct_majority += 1

            fmt_detail[trait] = {
                "per_token_acc": float(per_token_correct / len(detections)) if detections else 0,
                "majority_correct": majority == trait if detections else False,
                "text_preview": text[:100],
            }

        per_token_acc = correct_per_token / total_tokens if total_tokens > 0 else 0
        majority_acc = correct_majority / 6

        format_results[fmt_name] = {
            "per_token_accuracy": float(per_token_acc),
            "majority_accuracy": float(majority_acc),
            "majority_correct": correct_majority,
            "traits": fmt_detail,
        }
        print(f"  {fmt_name:15s}: majority={correct_majority}/6 ({majority_acc:.0%}), "
              f"per-token={per_token_acc:.1%}")

    results["per_format"] = format_results

    # ================================================================
    # PART 2: Open-ended vs structured comparison
    # ================================================================
    logger.info("Part 2: Open vs structured comparison...")
    print(f"\n{'='*70}")
    print("PART 2: OPEN-ENDED VS STRUCTURED (SAME TOPIC)")
    print(f"{'='*70}")

    comparison_pairs = [
        ("open", "What are your hobbies?"),
        ("structured", "List your hobbies as a JSON array. Only output the JSON."),
    ]

    comparison_results = {}
    for style, p in comparison_pairs:
        correct = 0
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            detections, text = detect_personality_multitoken(
                model, tokenizer, device, blocks, mid_layer,
                basis_5d, coords_5d, vec, alpha, p, max_tokens=30)
            from collections import Counter
            if detections:
                majority = Counter(detections).most_common(1)[0][0]
                if majority == trait:
                    correct += 1
        comparison_results[style] = {
            "correct": correct,
            "total": 6,
            "accuracy": float(correct / 6),
        }
        print(f"  {style:15s}: {correct}/6 ({correct/6:.0%})")

    results["open_vs_structured"] = comparison_results

    # ================================================================
    # PART 3: Cross-format consistency
    # ================================================================
    logger.info("Part 3: Cross-format consistency...")
    print(f"\n{'='*70}")
    print("PART 3: CROSS-FORMAT CONSISTENCY")
    print(f"{'='*70}")

    # For each trait, check if personality is detected consistently across formats
    consistency_results = {}
    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        detections_by_format = {}
        for fmt_name, fmt_prompt in format_prompts.items():
            detections, _ = detect_personality_multitoken(
                model, tokenizer, device, blocks, mid_layer,
                basis_5d, coords_5d, vec, alpha, fmt_prompt, max_tokens=20)
            from collections import Counter
            if detections:
                majority = Counter(detections).most_common(1)[0][0]
                detections_by_format[fmt_name] = majority
            else:
                detections_by_format[fmt_name] = "none"

        n_correct = sum(1 for d in detections_by_format.values() if d == trait)
        consistency_results[trait] = {
            "formats_correct": n_correct,
            "total_formats": len(format_prompts),
            "detections": detections_by_format,
        }
        print(f"  {trait:15s}: {n_correct}/{len(format_prompts)} formats correct")

    results["cross_format_consistency"] = consistency_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Overall accuracy across all formats
    total_majority = sum(v["majority_correct"] for v in format_results.values())
    total_formats = len(format_results) * 6
    overall_acc = total_majority / total_formats

    print(f"  Overall majority detection: {total_majority}/{total_formats} ({overall_acc:.0%})")
    print(f"  Best format: {max(format_results, key=lambda k: format_results[k]['majority_accuracy'])}")
    print(f"  Worst format: {min(format_results, key=lambda k: format_results[k]['majority_accuracy'])}")

    # Cross-format consistency
    all_consistency = [v["formats_correct"] / v["total_formats"]
                       for v in consistency_results.values()]
    print(f"  Mean cross-format consistency: {np.mean(all_consistency):.0%}")

    results["summary"] = {
        "overall_majority_accuracy": float(overall_acc),
        "total_correct": total_majority,
        "total_tests": total_formats,
        "mean_cross_format_consistency": float(np.mean(all_consistency)),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structured_output_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
