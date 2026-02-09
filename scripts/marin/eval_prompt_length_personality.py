#!/usr/bin/env python
"""
Prompt Length and Context Interaction with Personality Steering.

Does personality detection depend on prompt complexity or length?
Previous finding: 100% across 30 diverse prompts. But those were similar lengths.

Tests:
1. Minimal prompts (1-5 words)
2. Medium prompts (1-2 sentences)
3. Long context prompts (200+ tokens of context then a question)
4. Multi-turn context (simulated conversation history)
5. Empty/near-empty prompts
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="prompt-len")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

MINIMAL_PROMPTS = [
    "Hi",
    "Hello",
    "Hey there",
    "Go",
    "Yes",
]

SHORT_PROMPTS = [
    "Tell me about yourself.",
    "What do you do?",
    "Describe your interests.",
    "Who are you?",
    "What motivates you?",
]

MEDIUM_PROMPTS = [
    "I'm a career counselor helping students figure out their interests. Could you tell me about yourself and what kind of activities you enjoy?",
    "We're doing a personality assessment. Please describe your work style, interests, and values in a few sentences.",
    "Imagine you're at a networking event. How would you introduce yourself and describe your professional interests?",
]

LONG_CONTEXT_PROMPTS = [
    # Long context + short question
    (
        "I've been reading about Holland's RIASEC theory of vocational interests. "
        "It categorizes people into six types: Realistic (hands-on, practical), "
        "Investigative (analytical, intellectual), Artistic (creative, expressive), "
        "Social (helping, teaching), Enterprising (persuading, leading), and "
        "Conventional (organizing, detail-oriented). Each person typically has a "
        "dominant type that influences their career preferences and work style. "
        "Research shows that people tend to seek environments that match their "
        "personality type, leading to greater satisfaction and productivity. "
        "The theory has been widely validated across cultures and age groups. "
        "Now, with this context in mind, tell me about yourself."
    ),
    (
        "The history of personality psychology spans centuries, from Hippocrates' "
        "four temperaments to modern trait theories. Gordon Allport identified "
        "over 4,000 personality traits in the English language. Raymond Cattell "
        "reduced these to 16 factors. The Big Five model (OCEAN) emerged as a "
        "consensus in the 1980s. Meanwhile, vocational psychology developed "
        "independently, with John Holland proposing the hexagonal model of "
        "interests. These parallel traditions are now being integrated through "
        "research on how personality traits predict occupational interests. "
        "Given this rich theoretical background, please describe your personality."
    ),
]

MULTI_TURN_PROMPTS = [
    # Multi-turn simulated conversation
    [
        {"role": "user", "content": "Hello, I'd like to learn about you."},
        {"role": "assistant", "content": "Hello! I'd be happy to share."},
        {"role": "user", "content": "Great. What are your main interests and values?"},
    ],
    [
        {"role": "user", "content": "Can you help me with something?"},
        {"role": "assistant", "content": "Of course! What do you need help with?"},
        {"role": "user", "content": "I want to understand your personality. Tell me about yourself."},
    ],
    [
        {"role": "user", "content": "What do you think about work-life balance?"},
        {"role": "assistant", "content": "It depends on individual priorities and values."},
        {"role": "user", "content": "Interesting. And what are YOUR priorities and values?"},
    ],
]


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


def steer_and_detect(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                      steer_vec, alpha, messages_or_prompt):
    """Steer and detect. Accepts either a string prompt or list of messages."""
    detect_layer = mid_layer + 1
    delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)

    if isinstance(messages_or_prompt, str):
        messages = [{"role": "user", "content": messages_or_prompt}]
    else:
        messages = messages_or_prompt

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    num_tokens = input_ids.shape[1]

    # Baseline
    captured_base = {}
    hooks = []
    def cap_base(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_base["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_base))
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    # Steered
    captured_steer = {}
    hooks = []
    def cap_steer(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured_steer["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
        return out
    hooks.append(blocks[detect_layer].register_forward_hook(cap_steer))

    def steer_fn(_m, _i, out):
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
    return {
        "detected": detected,
        "correct": detected == max(sims, key=sims.get),  # placeholder
        "cos": sims,
        "norm": norm_5d,
        "num_tokens": num_tokens,
    }


def run_prompt_set(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                    residual, alpha, prompts, label):
    """Run detection on a set of prompts for all traits."""
    correct = 0
    total = 0
    results_list = []

    for prompt in prompts:
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            res = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                    basis_5d, coords_5d, vec, alpha, prompt)
            is_correct = res["detected"] == trait
            correct += int(is_correct)
            total += 1
            results_list.append({
                "trait": trait,
                "detected": res["detected"],
                "correct": is_correct,
                "num_tokens": res["num_tokens"],
                "norm": res["norm"],
            })

    accuracy = correct / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": float(accuracy), "details": results_list}


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
    results = {}

    print(f"\n{'='*70}")
    print("PROMPT LENGTH & CONTEXT INTERACTION WITH PERSONALITY")
    print(f"Model: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Minimal prompts
    # ================================================================
    logger.info("Part 1: Minimal prompts...")
    print(f"\n{'='*70}")
    print("PART 1: MINIMAL PROMPTS (1-5 words)")
    print(f"{'='*70}")

    res = run_prompt_set(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                          residual, alpha, MINIMAL_PROMPTS, "minimal")
    results["minimal"] = res
    avg_tokens = np.mean([d["num_tokens"] for d in res["details"]])
    print(f"  Accuracy: {res['correct']}/{res['total']} ({res['accuracy']:.0%})")
    print(f"  Avg tokens: {avg_tokens:.0f}")

    # ================================================================
    # PART 2: Short prompts
    # ================================================================
    logger.info("Part 2: Short prompts...")
    print(f"\n{'='*70}")
    print("PART 2: SHORT PROMPTS (1 sentence)")
    print(f"{'='*70}")

    res = run_prompt_set(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                          residual, alpha, SHORT_PROMPTS, "short")
    results["short"] = res
    avg_tokens = np.mean([d["num_tokens"] for d in res["details"]])
    print(f"  Accuracy: {res['correct']}/{res['total']} ({res['accuracy']:.0%})")
    print(f"  Avg tokens: {avg_tokens:.0f}")

    # ================================================================
    # PART 3: Medium prompts
    # ================================================================
    logger.info("Part 3: Medium prompts...")
    print(f"\n{'='*70}")
    print("PART 3: MEDIUM PROMPTS (2-3 sentences)")
    print(f"{'='*70}")

    res = run_prompt_set(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                          residual, alpha, MEDIUM_PROMPTS, "medium")
    results["medium"] = res
    avg_tokens = np.mean([d["num_tokens"] for d in res["details"]])
    print(f"  Accuracy: {res['correct']}/{res['total']} ({res['accuracy']:.0%})")
    print(f"  Avg tokens: {avg_tokens:.0f}")

    # ================================================================
    # PART 4: Long context
    # ================================================================
    logger.info("Part 4: Long context prompts...")
    print(f"\n{'='*70}")
    print("PART 4: LONG CONTEXT PROMPTS (100+ tokens)")
    print(f"{'='*70}")

    res = run_prompt_set(model, tokenizer, device, blocks, mid_layer, basis_5d, coords_5d,
                          residual, alpha, LONG_CONTEXT_PROMPTS, "long_context")
    results["long_context"] = res
    avg_tokens = np.mean([d["num_tokens"] for d in res["details"]])
    print(f"  Accuracy: {res['correct']}/{res['total']} ({res['accuracy']:.0%})")
    print(f"  Avg tokens: {avg_tokens:.0f}")

    # ================================================================
    # PART 5: Multi-turn
    # ================================================================
    logger.info("Part 5: Multi-turn prompts...")
    print(f"\n{'='*70}")
    print("PART 5: MULTI-TURN CONVERSATION CONTEXT")
    print(f"{'='*70}")

    correct = 0
    total = 0
    multi_details = []
    for messages in MULTI_TURN_PROMPTS:
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            res_single = steer_and_detect(model, tokenizer, device, blocks, mid_layer,
                                           basis_5d, coords_5d, vec, alpha, messages)
            is_correct = res_single["detected"] == trait
            correct += int(is_correct)
            total += 1
            multi_details.append({
                "trait": trait,
                "detected": res_single["detected"],
                "correct": is_correct,
                "num_tokens": res_single["num_tokens"],
            })

    accuracy = correct / total
    results["multi_turn"] = {"correct": correct, "total": total, "accuracy": float(accuracy)}
    avg_tokens = np.mean([d["num_tokens"] for d in multi_details])
    print(f"  Accuracy: {correct}/{total} ({accuracy:.0%})")
    print(f"  Avg tokens: {avg_tokens:.0f}")

    # ================================================================
    # PART 6: Token count vs accuracy
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 6: TOKEN COUNT vs ACCURACY CORRELATION")
    print(f"{'='*70}")

    # Collect all data points
    all_norms = []
    all_tokens = []
    for category in ["minimal", "short", "medium", "long_context"]:
        for d in results[category]["details"]:
            all_norms.append(d["norm"])
            all_tokens.append(d["num_tokens"])

    if len(all_tokens) > 2:
        corr = float(np.corrcoef(all_tokens, all_norms)[0, 1])
        print(f"  Correlation(num_tokens, 5D_norm): r = {corr:.4f}")
    else:
        corr = 0
        print("  Insufficient data for correlation")

    # Accuracy by category
    print(f"\n  Summary by prompt length:")
    for cat, label in [("minimal", "Minimal (1-5 words)"),
                        ("short", "Short (1 sentence)"),
                        ("medium", "Medium (2-3 sentences)"),
                        ("long_context", "Long (100+ tokens)"),
                        ("multi_turn", "Multi-turn")]:
        r = results[cat]
        print(f"    {label:>30}: {r['correct']}/{r['total']} ({r['accuracy']:.0%})")

    results["token_norm_correlation"] = float(corr)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_correct = sum(results[cat]["correct"] for cat in ["minimal", "short", "medium", "long_context", "multi_turn"])
    total_tests = sum(results[cat]["total"] for cat in ["minimal", "short", "medium", "long_context", "multi_turn"])
    overall = total_correct / total_tests
    print(f"  Overall: {total_correct}/{total_tests} ({overall:.1%})")
    print(f"  Token-norm correlation: r = {corr:.4f}")

    results["summary"] = {
        "overall_accuracy": float(overall),
        "total_correct": total_correct,
        "total_tests": total_tests,
        "token_norm_corr": float(corr),
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prompt_length_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
