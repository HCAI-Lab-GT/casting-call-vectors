#!/usr/bin/env python
"""
Psychometric battery evaluation for personality-steered models.

Administers standardized psychometric tests (O*NET Interest Profiler,
IPIP-NEO-120, HEXACO-100) on a personality-steered model and computes
domain/facet scores using the standard scoring keys.

This directly validates H1: that persona vectors induce measurable,
trait-specific psychometric shifts.

Usage:
    python scripts/marin/eval_psychometric_battery.py --trait artistic --alpha 3.0 --device cuda:0
    python scripts/marin/eval_psychometric_battery.py --trait all --alpha 3.0 --device cuda:0
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="psychometric-battery")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


# ── Test item loaders ──────────────────────────────────
def load_onet_items():
    path = _repo_root() / "tom_scoring" / "interest_profiler.json"
    with open(path) as f:
        return json.load(f)


def load_ipip_neo_items():
    path = _repo_root() / "tom_scoring" / "ipip_neo_120.json"
    with open(path) as f:
        return json.load(f)


def load_hexaco_items():
    path = _repo_root() / "tom_scoring" / "hexaco_100.json"
    with open(path) as f:
        return json.load(f)


# ── Scoring ────────────────────────────────────────────
def score_onet(responses, items):
    dim_indices = defaultdict(list)
    for i, item in enumerate(items):
        dim_indices[item["dimension"]].append(i)
    return {dim: sum(responses[i] for i in idx) for dim, idx in dim_indices.items()}


def score_ipip_neo(responses, items):
    scored = []
    for i, (r, item) in enumerate(zip(responses, items)):
        scored.append(r if item["keyed"] == "+" else 6 - r)

    facet_idx = defaultdict(list)
    domain_idx = defaultdict(list)
    for i, item in enumerate(items):
        facet_idx[item["facet"]].append(i)
        domain_idx[item["domain"]].append(i)

    facets = {f: sum(scored[i] for i in idx) for f, idx in facet_idx.items()}
    domains = {d: sum(scored[i] for i in idx) for d, idx in domain_idx.items()}
    return {"facets": facets, "domains": domains}


def score_hexaco(responses, items):
    scored = [r if item["keyed"] == "+" else 6 - r for r, item in zip(responses, items)]

    facet_idx = defaultdict(list)
    for i, item in enumerate(items):
        facet_idx[item["facet"]].append(i)

    facets = {f: sum(scored[i] for i in idx) / len(idx) for f, idx in facet_idx.items()}

    domain_facets = defaultdict(set)
    for item in items:
        if item.get("domain"):
            domain_facets[item["domain"]].add(item["facet"])

    domains = {d: float(np.mean([facets[f] for f in fl])) for d, fl in domain_facets.items()}
    altruism = facets.get("Altruism")
    return {"facets": facets, "domains": domains, "altruism": altruism}


# ── Response parsing ───────────────────────────────────
def parse_likert(text):
    text = text.strip()
    match = re.search(r'\b([1-5])\b', text)
    if match:
        return int(match.group(1))

    t = text.lower()
    for score, kws in [(5, ["very accurate", "strongly agree"]),
                       (4, ["moderately accurate", "agree"]),
                       (3, ["neither", "neutral"]),
                       (2, ["moderately inaccurate", "disagree"]),
                       (1, ["very inaccurate", "strongly disagree"])]:
        for kw in kws:
            if kw in t:
                return score
    return 3  # neutral fallback


def parse_binary(text):
    t = text.strip().lower()
    if re.search(r"\b(yes|yeah|yep|certainly|absolutely|of course)\b", t):
        return 1
    return 0


# ── Model with steering ───────────────────────────────
def generate_response(model, tokenizer, device, input_ids, max_tokens=10, temperature=0.1):
    """Manual generation loop (reliable hook firing)."""
    past_kv = None
    gen_ids = []

    for _ in range(max_tokens):
        with torch.no_grad():
            inp = input_ids if past_kv is None else input_ids[:, -1:]
            out = model(input_ids=inp, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits.float() / max(temperature, 0.01), dim=-1)
        tok = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, tok], dim=-1)
        gen_ids.append(tok.item())
        if tok.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ── Main ───────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="marin-community/marin-8b-instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trait", default="artistic", help="RIASEC trait or 'all'")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--test", default="onet", choices=["onet", "ipip", "hexaco", "all"])
    args = ap.parse_args()

    from transformers import AutoConfig
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

    # Load test items
    tests_to_run = []
    if args.test in ("onet", "all"):
        tests_to_run.append(("onet", load_onet_items()))
    if args.test in ("ipip", "all"):
        tests_to_run.append(("ipip", load_ipip_neo_items()))
    if args.test in ("hexaco", "all"):
        tests_to_run.append(("hexaco", load_hexaco_items()))

    all_results = {}

    # Also run baseline (no steering)
    test_configs = [("baseline", None, 0.0)] + [(t, vectors[t], args.alpha) for t in traits_to_test]

    for trait_name, vec, alpha in test_configs:
        logger.info("=== Evaluating: %s (alpha=%.1f) ===", trait_name, alpha)

        # Install/remove hook
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

        trait_results = {}

        for test_name, items in tests_to_run:
            logger.info("  Running %s (%d items)...", test_name, len(items))

            is_binary = (test_name == "onet")

            if is_binary:
                sys_prompt = "You are answering the O*NET Interest Profiler. Would you LIKE to do this work activity? Answer ONLY 'Yes' or 'No'."
            else:
                sys_prompt = ("You are taking a personality questionnaire. Rate how accurately "
                              "this describes you: 1=Very Inaccurate, 2=Moderately Inaccurate, "
                              "3=Neutral, 4=Moderately Accurate, 5=Very Accurate. "
                              "Respond with ONLY a single number (1-5).")

            responses = []
            raw_responses = []
            for i, item in enumerate(items):
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f'"{item["text"]}"'},
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(args.device)

                raw = generate_response(model, tokenizer, args.device, input_ids,
                                        max_tokens=5 if is_binary else 10,
                                        temperature=0.1)

                parsed = parse_binary(raw) if is_binary else parse_likert(raw)
                responses.append(parsed)
                raw_responses.append(raw.strip()[:50])

                if (i + 1) % 30 == 0:
                    logger.info("    %d/%d items done", i + 1, len(items))

            # Score
            if test_name == "onet":
                scores = score_onet(responses, items)
            elif test_name == "ipip":
                scores = score_ipip_neo(responses, items)
            elif test_name == "hexaco":
                scores = score_hexaco(responses, items)

            trait_results[test_name] = {
                "scores": scores,
                "responses": responses,
                "raw_samples": raw_responses[:10],
                "n_items": len(items),
            }

            logger.info("    Scores: %s", scores if test_name == "onet" else scores.get("domains", scores))

        if hook_handle:
            hook_handle.remove()

        all_results[trait_name] = trait_results

    # Print summary
    print(f"\n{'='*70}")
    print("PSYCHOMETRIC BATTERY RESULTS")
    print(f"{'='*70}")

    for test_name, _ in tests_to_run:
        print(f"\n--- {test_name.upper()} ---")
        if test_name == "onet":
            dims = list(all_results["baseline"][test_name]["scores"].keys())
            print(f"{'Condition':>15}", end="")
            for d in dims:
                print(f" {d[:4]:>5}", end="")
            print()

            for cond in ["baseline"] + traits_to_test:
                scores = all_results[cond][test_name]["scores"]
                print(f"{cond:>15}", end="")
                for d in dims:
                    v = scores.get(d, 0)
                    print(f" {v:>5}", end="")
                print()
        else:
            domains = list(all_results["baseline"][test_name]["scores"]["domains"].keys())
            print(f"{'Condition':>15}", end="")
            for d in domains:
                print(f" {d[:4]:>5}", end="")
            print()

            for cond in ["baseline"] + traits_to_test:
                scores = all_results[cond][test_name]["scores"]["domains"]
                print(f"{cond:>15}", end="")
                for d in domains:
                    v = scores.get(d, 0)
                    if isinstance(v, float):
                        print(f" {v:>5.1f}", end="")
                    else:
                        print(f" {v:>5}", end="")
                print()

    # Save
    out_dir = _repo_root() / "outputs" / "psychometric"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"psychometric_{args.trait}_alpha{args.alpha}_{args.test}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model_id,
            "alpha": args.alpha,
            "traits_tested": traits_to_test,
            "tests_run": [t for t, _ in tests_to_run],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "results": all_results,
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    logger.info("Saved to %s", out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
