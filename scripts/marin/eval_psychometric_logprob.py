#!/usr/bin/env python
"""
Logprob-based psychometric evaluation for personality-steered models.

Instead of parsing Yes/No or 1-5 from generated text (which is insensitive
due to RLHF sycophancy), this measures logprob differentials:
  - For binary items: logP(Yes) - logP(No)
  - For Likert items: expected value from logprob distribution over 1-5

This is FAR more sensitive than text-based scoring because the personality
signal is in the activations/logits, not in the argmax text (finding #9).

Usage:
    python scripts/marin/eval_psychometric_logprob.py --trait all --alpha 3.0 --device cuda:1
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="psychometric-logprob")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


# ── Loaders ────────────────────────────────────────────
def load_onet_items():
    path = _repo_root() / "tom_scoring" / "interest_profiler.json"
    with open(path) as f:
        return json.load(f)


def load_ipip_neo_items():
    path = _repo_root() / "tom_scoring" / "ipip_neo_120.json"
    with open(path) as f:
        return json.load(f)


# ── Logprob extraction ─────────────────────────────────
def get_binary_logprobs(model, tokenizer, device, item_text, sys_prompt):
    """Get logP(Yes) and logP(No) for a binary question."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f'"{item_text}"'},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)

    yes_lp = log_probs[yes_ids[0]].item()
    no_lp = log_probs[no_ids[0]].item()

    return yes_lp, no_lp


def get_likert_logprobs(model, tokenizer, device, item_text, sys_prompt):
    """Get logprob distribution over digits 1-5 for a Likert item."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f'"{item_text}"'},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

    # Get token IDs for "1", "2", "3", "4", "5"
    digit_logprobs = {}
    for d in range(1, 6):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        digit_logprobs[d] = log_probs[ids[0]].item()

    # Normalize to get a proper distribution over 1-5
    lps = np.array([digit_logprobs[d] for d in range(1, 6)])
    probs = np.exp(lps - lps.max())  # numerical stability
    probs = probs / probs.sum()

    # Expected value
    expected = sum((d + 1) * probs[d] for d in range(5))

    return digit_logprobs, probs.tolist(), float(expected)


# ── Scoring ────────────────────────────────────────────
def score_onet_logprob(logprob_gaps, items):
    """Score O*NET using logprob gaps instead of binary counts."""
    dim_gaps = defaultdict(list)
    for i, item in enumerate(items):
        dim_gaps[item["dimension"]].append(logprob_gaps[i])
    return {dim: float(np.mean(gaps)) for dim, gaps in dim_gaps.items()}


def score_ipip_neo_logprob(expected_values, items):
    """Score IPIP-NEO-120 using logprob-derived expected values."""
    scored = []
    for i, (ev, item) in enumerate(zip(expected_values, items)):
        scored.append(ev if item["keyed"] == "+" else 6 - ev)

    facet_idx = defaultdict(list)
    domain_idx = defaultdict(list)
    for i, item in enumerate(items):
        facet_idx[item["facet"]].append(i)
        domain_idx[item["domain"]].append(i)

    facets = {f: float(np.mean([scored[i] for i in idx])) for f, idx in facet_idx.items()}
    domains = {d: float(np.mean([scored[i] for i in idx])) for d, idx in domain_idx.items()}
    return {"facets": facets, "domains": domains}


# ── Main ───────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="marin-community/marin-8b-instruct")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--trait", default="artistic", help="RIASEC trait or 'all'")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--test", default="onet", choices=["onet", "ipip", "all"])
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    mid_layer = config.num_hidden_layers // 2
    safe_model = args.model_id.replace("/", "__")

    traits_to_test = TRAITS if args.trait == "all" else [args.trait]

    # Load model
    logger.info("Loading %s on %s", args.model_id, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # Load persona vectors
    vectors = {}
    for trait in traits_to_test:
        path = _repo_root() / "persona_data" / "model_inits" / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vec = data["response_persona_vector"].numpy().flatten().astype(np.float32)
        vectors[trait] = torch.tensor(vec, dtype=torch.float16).unsqueeze(0).to(args.device)

    # Prompts
    onet_sys = "You are answering the O*NET Interest Profiler. Would you LIKE to do this work activity? Answer ONLY 'Yes' or 'No'."
    likert_sys = ("You are taking a personality questionnaire. Rate how accurately "
                  "this describes you: 1=Very Inaccurate, 2=Moderately Inaccurate, "
                  "3=Neutral, 4=Moderately Accurate, 5=Very Accurate. "
                  "Respond with ONLY a single number (1-5).")

    # Test configs
    test_configs = [("baseline", None, 0.0)] + [(t, vectors[t], args.alpha) for t in traits_to_test]

    all_results = {}

    for trait_name, vec, alpha in test_configs:
        logger.info("=== Evaluating: %s (alpha=%.1f) ===", trait_name, alpha)

        # Hook
        hook_handle = None
        if vec is not None:
            delta = alpha * vec

            def make_hook(d):
                def hook_fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    hs[:, -1, :] += d
                    return (hs,) + out[1:] if isinstance(out, tuple) else hs
                return hook_fn

            hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta))

        trait_result = {}

        if args.test in ("onet", "all"):
            items = load_onet_items()
            logger.info("  Running O*NET logprob (%d items)...", len(items))

            logprob_gaps = []
            raw_data = []
            for i, item in enumerate(items):
                yes_lp, no_lp = get_binary_logprobs(model, tokenizer, args.device, item["text"], onet_sys)
                gap = yes_lp - no_lp
                logprob_gaps.append(gap)
                raw_data.append({"item": item["text"], "dim": item["dimension"],
                                 "yes_lp": yes_lp, "no_lp": no_lp, "gap": gap})
                if (i + 1) % 30 == 0:
                    logger.info("    %d/%d", i + 1, len(items))

            scores = score_onet_logprob(logprob_gaps, items)
            trait_result["onet"] = {
                "scores": scores,
                "raw_data": raw_data,
                "mean_gap": float(np.mean(logprob_gaps)),
            }
            logger.info("    Scores: %s", {k: f"{v:.2f}" for k, v in scores.items()})

        if args.test in ("ipip", "all"):
            items = load_ipip_neo_items()
            logger.info("  Running IPIP-NEO logprob (%d items)...", len(items))

            expected_values = []
            raw_data = []
            for i, item in enumerate(items):
                digit_lps, probs, ev = get_likert_logprobs(model, tokenizer, args.device, item["text"], likert_sys)
                expected_values.append(ev)
                raw_data.append({"item": item["text"], "domain": item["domain"],
                                 "facet": item["facet"], "keyed": item["keyed"],
                                 "digit_logprobs": digit_lps, "probs": probs, "expected": ev})
                if (i + 1) % 30 == 0:
                    logger.info("    %d/%d", i + 1, len(items))

            scores = score_ipip_neo_logprob(expected_values, items)
            trait_result["ipip"] = {
                "scores": scores,
                "raw_data": raw_data,
                "mean_expected": float(np.mean(expected_values)),
            }
            logger.info("    Domain scores: %s", {k: f"{v:.2f}" for k, v in scores["domains"].items()})

        if hook_handle:
            hook_handle.remove()

        all_results[trait_name] = trait_result

    # Print summary
    print(f"\n{'='*70}")
    print("PSYCHOMETRIC LOGPROB RESULTS")
    print(f"{'='*70}")

    if args.test in ("onet", "all"):
        print(f"\n--- O*NET Interest Profiler (logprob gap: logP(Yes) - logP(No)) ---")
        dims = list(all_results["baseline"]["onet"]["scores"].keys())
        print(f"{'Condition':>15}", end="")
        for d in dims:
            print(f" {d[:6]:>7}", end="")
        print()

        for cond in ["baseline"] + traits_to_test:
            scores = all_results[cond]["onet"]["scores"]
            print(f"{cond:>15}", end="")
            for d in dims:
                v = scores.get(d, 0)
                print(f" {v:>7.2f}", end="")
            # Highlight: is the steered trait the highest?
            if cond != "baseline":
                best = max(scores, key=scores.get)
                expected_dim = cond.capitalize()
                match = "MATCH" if expected_dim in best else f"got {best}"
                print(f"  [{match}]", end="")
            print()

    if args.test in ("ipip", "all"):
        print(f"\n--- IPIP-NEO-120 (logprob expected value, 1-5 scale) ---")
        domains = list(all_results["baseline"]["ipip"]["scores"]["domains"].keys())
        print(f"{'Condition':>15}", end="")
        for d in domains:
            print(f" {d[:5]:>6}", end="")
        print()

        for cond in ["baseline"] + traits_to_test:
            scores = all_results[cond]["ipip"]["scores"]["domains"]
            print(f"{cond:>15}", end="")
            for d in domains:
                v = scores.get(d, 0)
                print(f" {v:>6.2f}", end="")
            print()

    # Save
    out_dir = _repo_root() / "outputs" / "psychometric"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"psychometric_logprob_{args.trait}_alpha{args.alpha}_{args.test}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model_id,
            "alpha": args.alpha,
            "method": "logprob",
            "traits_tested": traits_to_test,
            "tests_run": [args.test],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": all_results,
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    logger.info("Saved to %s", out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
