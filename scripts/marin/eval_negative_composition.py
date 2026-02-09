#!/usr/bin/env python
"""
Negative Composition: "Be X but NOT Y" via Vector Arithmetic.

Tests whether combining positive and negative personality vectors
achieves targeted personality profiles: boost one trait while
simultaneously suppressing another.

This is a key practical capability: creating nuanced personas
like "creative but not chaotic" (artistic + anti-enterprising).

Tests:
1. Single positive + single negative: 15 pairs (6 choose 2)
2. Holland opposite pairs: artistic+(-conventional), etc.
3. Dose-response: does the negative scale independently?
4. Triple combinations: A + (-B) + (-C)
5. Self-cancellation: A + (-A) should produce zero personality
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="neg-comp")

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


def detect_personality(model, tokenizer, device, blocks, mid_layer, basis_5d,
                       coords_5d, steer_vec, alpha, prompt):
    """Steer and detect personality from activation."""
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
    detect_layer = mid_layer + 1

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    # Baseline
    base_captured = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        base_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()
    base_act = base_captured["act"]

    # Steered
    steered_captured = {}
    hooks = []
    def steer_fn(_m, _i, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta
            return (hs,) + out[1:]
        out[:, -1, :] += delta
        return out
    hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    def cap_fn(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        steered_captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_fn))

    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    # Detect in 5D
    diff = (steered_captured["act"] - base_act).astype(np.float64)
    coords = basis_5d @ diff
    norm_5d = float(np.linalg.norm(coords))

    sims = {}
    for t in TRAITS:
        if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
            sims[t] = float(np.dot(coords, coords_5d[t]) / (
                norm_5d * np.linalg.norm(coords_5d[t])))
        else:
            sims[t] = 0

    ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return {
        "coords": coords.tolist(),
        "norm_5d": norm_5d,
        "similarities": sims,
        "ranked": ranked,
        "detected": ranked[0][0],
    }


def main():
    device = "cuda:2"
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
    prompt = "Tell me about yourself."
    results = {}

    print(f"\n{'='*70}")
    print("NEGATIVE COMPOSITION: 'BE X BUT NOT Y'")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: All 15 positive+negative pairs
    # ================================================================
    logger.info("Part 1: All positive+negative pairs...")
    print(f"\n{'='*70}")
    print("PART 1: A + (-B) FOR ALL 15 PAIRS")
    print(f"{'='*70}")

    pair_results = {}
    boost_correct = 0
    suppress_correct = 0
    total_pairs = 0

    for i, pos_trait in enumerate(TRAITS):
        for j, neg_trait in enumerate(TRAITS):
            if i >= j:
                continue

            # Compose: +pos -neg
            combined = residual[pos_trait].astype(np.float32) - residual[neg_trait].astype(np.float32)

            det = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                     basis_5d, coords_5d, combined, alpha, prompt)

            # Check: pos_trait should be high, neg_trait should be low
            pos_rank = [r[0] for r in det["ranked"]].index(pos_trait) + 1
            neg_rank = [r[0] for r in det["ranked"]].index(neg_trait) + 1
            pos_sim = det["similarities"][pos_trait]
            neg_sim = det["similarities"][neg_trait]

            is_boosted = pos_rank <= 2
            is_suppressed = neg_rank >= 5

            if is_boosted:
                boost_correct += 1
            if is_suppressed:
                suppress_correct += 1
            total_pairs += 1

            pair_results[f"{pos_trait}+(-{neg_trait})"] = {
                "pos_rank": pos_rank,
                "neg_rank": neg_rank,
                "pos_sim": float(pos_sim),
                "neg_sim": float(neg_sim),
                "detected": det["detected"],
                "boosted": is_boosted,
                "suppressed": is_suppressed,
                "norm_5d": det["norm_5d"],
            }
            print(f"  +{pos_trait[:4]} -{neg_trait[:4]}: "
                  f"det={det['detected'][:4]}, "
                  f"pos_rank={pos_rank}, neg_rank={neg_rank}, "
                  f"pos_sim={pos_sim:+.3f}, neg_sim={neg_sim:+.3f}")

    print(f"\n  Boost correct (pos in top-2): {boost_correct}/{total_pairs} ({boost_correct/total_pairs:.0%})")
    print(f"  Suppress correct (neg in bottom-2): {suppress_correct}/{total_pairs} ({suppress_correct/total_pairs:.0%})")

    results["pair_composition"] = {
        "pairs": pair_results,
        "boost_accuracy": float(boost_correct / total_pairs),
        "suppress_accuracy": float(suppress_correct / total_pairs),
    }

    # Also do the reverse direction
    reverse_pair_results = {}
    for i, pos_trait in enumerate(TRAITS):
        for j, neg_trait in enumerate(TRAITS):
            if i >= j:
                continue
            # Now neg is positive and pos is negative
            combined = residual[neg_trait].astype(np.float32) - residual[pos_trait].astype(np.float32)
            det = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                     basis_5d, coords_5d, combined, alpha, prompt)
            reverse_pair_results[f"{neg_trait}+(-{pos_trait})"] = {
                "detected": det["detected"],
                "neg_trait_sim": float(det["similarities"][neg_trait]),
                "pos_trait_sim": float(det["similarities"][pos_trait]),
            }

    results["reverse_pairs"] = reverse_pair_results

    # ================================================================
    # PART 2: Holland opposite pairs (strongest contrasts)
    # ================================================================
    logger.info("Part 2: Holland opposite pairs...")
    print(f"\n{'='*70}")
    print("PART 2: HOLLAND OPPOSITE COMPOSITIONS")
    print(f"{'='*70}")

    holland_opposites = [
        ("artistic", "conventional"),
        ("investigative", "enterprising"),
        ("realistic", "social"),
    ]

    holland_results = {}
    for pos, neg in holland_opposites:
        combined = residual[pos].astype(np.float32) - residual[neg].astype(np.float32)
        det = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                 basis_5d, coords_5d, combined, alpha, prompt)

        holland_results[f"{pos}+(-{neg})"] = {
            "detected": det["detected"],
            "pos_sim": float(det["similarities"][pos]),
            "neg_sim": float(det["similarities"][neg]),
            "norm_5d": det["norm_5d"],
            "all_sims": det["similarities"],
        }
        print(f"  +{pos} -{neg}: det={det['detected']}, "
              f"pos={det['similarities'][pos]:+.3f}, neg={det['similarities'][neg]:+.3f}, "
              f"norm={det['norm_5d']:.1f}")

    results["holland_opposites"] = holland_results

    # ================================================================
    # PART 3: Dose-response for negative component
    # ================================================================
    logger.info("Part 3: Negative dose-response...")
    print(f"\n{'='*70}")
    print("PART 3: DOSE-RESPONSE (vary negative strength)")
    print(f"{'='*70}")

    pos_trait = "artistic"
    neg_trait = "conventional"
    neg_ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    dose_results = {}

    for ratio in neg_ratios:
        combined = residual[pos_trait].astype(np.float32) - ratio * residual[neg_trait].astype(np.float32)
        det = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                 basis_5d, coords_5d, combined, alpha, prompt)

        dose_results[ratio] = {
            "detected": det["detected"],
            "pos_sim": float(det["similarities"][pos_trait]),
            "neg_sim": float(det["similarities"][neg_trait]),
            "norm_5d": det["norm_5d"],
        }
        print(f"  +{pos_trait[:4]} -{ratio:.2f}×{neg_trait[:4]}: "
              f"det={det['detected'][:4]}, "
              f"pos={det['similarities'][pos_trait]:+.3f}, "
              f"neg={det['similarities'][neg_trait]:+.3f}")

    results["dose_response"] = {
        "pos_trait": pos_trait,
        "neg_trait": neg_trait,
        "ratios": {str(k): v for k, v in dose_results.items()},
    }

    # ================================================================
    # PART 4: Triple compositions: A + (-B) + (-C)
    # ================================================================
    logger.info("Part 4: Triple compositions...")
    print(f"\n{'='*70}")
    print("PART 4: TRIPLE COMPOSITIONS: A + (-B) + (-C)")
    print(f"{'='*70}")

    triple_combos = [
        ("artistic", "conventional", "realistic"),
        ("investigative", "enterprising", "social"),
        ("social", "realistic", "conventional"),
        ("enterprising", "artistic", "investigative"),
    ]

    triple_results = {}
    for pos, neg1, neg2 in triple_combos:
        combined = (residual[pos].astype(np.float32)
                    - residual[neg1].astype(np.float32)
                    - residual[neg2].astype(np.float32))
        det = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                 basis_5d, coords_5d, combined, alpha, prompt)

        key = f"+{pos} -{neg1} -{neg2}"
        triple_results[key] = {
            "detected": det["detected"],
            "pos_sim": float(det["similarities"][pos]),
            "neg1_sim": float(det["similarities"][neg1]),
            "neg2_sim": float(det["similarities"][neg2]),
            "pos_rank": [r[0] for r in det["ranked"]].index(pos) + 1,
            "neg1_rank": [r[0] for r in det["ranked"]].index(neg1) + 1,
            "neg2_rank": [r[0] for r in det["ranked"]].index(neg2) + 1,
            "norm_5d": det["norm_5d"],
        }
        print(f"  {key}: det={det['detected'][:4]}, "
              f"pos_rank={[r[0] for r in det['ranked']].index(pos)+1}, "
              f"neg1_rank={[r[0] for r in det['ranked']].index(neg1)+1}, "
              f"neg2_rank={[r[0] for r in det['ranked']].index(neg2)+1}")

    results["triple_composition"] = triple_results

    # ================================================================
    # PART 5: Self-cancellation: A + (-A)
    # ================================================================
    logger.info("Part 5: Self-cancellation...")
    print(f"\n{'='*70}")
    print("PART 5: SELF-CANCELLATION: A + (-A)")
    print(f"{'='*70}")

    cancel_results = {}
    for trait in TRAITS:
        combined = residual[trait].astype(np.float32) - residual[trait].astype(np.float32)
        # This should be exactly zero, but let's verify the detection
        det = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                 basis_5d, coords_5d, combined, alpha, prompt)
        cancel_results[trait] = {
            "norm_5d": det["norm_5d"],
            "detected": det["detected"],
            "max_sim": float(max(det["similarities"].values())),
        }
        print(f"  {trait} + (-{trait}): norm={det['norm_5d']:.6f}, "
              f"det={det['detected']}, max_sim={max(det['similarities'].values()):.6f}")

    results["self_cancellation"] = cancel_results

    # ================================================================
    # PART 6: Cross-prompt consistency
    # ================================================================
    logger.info("Part 6: Cross-prompt consistency...")
    print(f"\n{'='*70}")
    print("PART 6: CROSS-PROMPT CONSISTENCY")
    print(f"{'='*70}")

    prompts = [
        "Tell me about yourself.",
        "What do you enjoy doing in your free time?",
        "How do you approach new challenges?",
        "What matters most to you in life?",
    ]

    cross_prompt_results = {}
    # Test 3 interesting compositions
    test_compositions = [
        ("artistic", "conventional"),
        ("investigative", "social"),
        ("enterprising", "realistic"),
    ]

    for pos, neg in test_compositions:
        combined = residual[pos].astype(np.float32) - residual[neg].astype(np.float32)
        prompt_detections = []
        for p in prompts:
            det = detect_personality(model, tokenizer, device, blocks, mid_layer,
                                     basis_5d, coords_5d, combined, alpha, p)
            prompt_detections.append({
                "prompt": p[:30],
                "detected": det["detected"],
                "pos_sim": float(det["similarities"][pos]),
                "neg_sim": float(det["similarities"][neg]),
            })

        all_detected = [d["detected"] for d in prompt_detections]
        consistency = len(set(all_detected)) == 1

        cross_prompt_results[f"+{pos} -{neg}"] = {
            "prompts": prompt_detections,
            "consistent": consistency,
            "all_correct": all(d["detected"] == pos for d in prompt_detections),
        }
        status = "CONSISTENT" if consistency else "VARIES"
        all_correct = "ALL CORRECT" if all(d["detected"] == pos for d in prompt_detections) else "SOME WRONG"
        print(f"  +{pos[:4]} -{neg[:4]}: {status}, {all_correct}")

    results["cross_prompt_consistency"] = cross_prompt_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    boost_acc = results["pair_composition"]["boost_accuracy"]
    suppress_acc = results["pair_composition"]["suppress_accuracy"]
    cancel_norms = [v["norm_5d"] for v in cancel_results.values()]
    mean_cancel = float(np.mean(cancel_norms))

    print(f"  Boost accuracy (pos in top-2): {boost_acc:.0%}")
    print(f"  Suppress accuracy (neg in bottom-2): {suppress_acc:.0%}")
    print(f"  Self-cancellation mean norm: {mean_cancel:.6f}")

    results["summary"] = {
        "boost_accuracy": float(boost_acc),
        "suppress_accuracy": float(suppress_acc),
        "mean_self_cancellation_norm": mean_cancel,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "negative_composition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
