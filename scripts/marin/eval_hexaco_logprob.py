#!/usr/bin/env python
"""
HEXACO-100 Personality Inventory via Logprob Likert Scoring.

Runs the HEXACO-100 personality inventory on steered models using logprob
scoring.  The HEXACO measures 6 dimensions orthogonal to RIASEC:
  Honesty-Humility, Emotionality, eXtraversion, Agreeableness,
  Conscientiousness, Openness to Experience.

For each condition (baseline + 6 RIASEC traits at alpha=3.0):
  - Present every HEXACO item and extract logprobs over Likert tokens 1-5
  - Compute expected value, reverse-score negatively-keyed items
  - Aggregate into domain and facet means
  - Identify which HEXACO dimensions are most affected by each RIASEC trait

This is interesting because HEXACO maps to different personality dimensions
than RIASEC, so we can see if RIASEC steering creates clean patterns in a
different personality framework.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="hexaco-logprob")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

HEXACO_DOMAINS = [
    "Honesty-Humility",
    "Emotionality",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Openness to Experience",
]

SYSTEM_PROMPT = (
    "You are completing a personality questionnaire. For each statement, rate "
    "how much you agree on a scale from 1 (strongly disagree) to 5 (strongly "
    "agree). Respond with ONLY a single number: 1, 2, 3, 4, or 5."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


# ------------------------------------------------------------------
# Vector loading and residual computation
# ------------------------------------------------------------------

def load_persona_vectors(model_id: str, riasec_dir: Path):
    """Load persona vectors, compute residuals (subtract shared PC1)."""
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

    # Stack vectors at mid_layer+1 (the detect layer) for SVD
    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _U, _S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    # Compute residuals: remove shared PC1 direction
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj

    return residual, mid_layer


# ------------------------------------------------------------------
# Logprob extraction
# ------------------------------------------------------------------

def get_likert_logprobs(model, tokenizer, device, item_text: str, sys_prompt: str):
    """
    Get logprobs for Likert tokens 1-5 and compute expected value.

    Returns dict with logprobs, probs, and expected_value.
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f'Statement: "{item_text}"\n\nYour rating (1-5):'},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

    # Extract logprobs for tokens "1" through "5"
    scores = {}
    for val in range(1, 6):
        token_ids = tokenizer.encode(str(val), add_special_tokens=False)
        scores[val] = log_probs[token_ids[0]].item()

    # Convert to probabilities (renormalized over 1-5) and compute expected value
    log_vals = torch.tensor([scores[i] for i in range(1, 6)])
    probs = torch.softmax(log_vals, dim=0)
    expected = sum((i + 1) * probs[i].item() for i in range(5))

    return {
        "logprobs": {str(k): float(v) for k, v in scores.items()},
        "probs": {str(i + 1): float(probs[i]) for i in range(5)},
        "expected_value": float(expected),
    }


# ------------------------------------------------------------------
# Steering hook
# ------------------------------------------------------------------

def make_pre_hook(delta_vec):
    """Create a forward_pre_hook that adds delta_vec to the last token."""
    def hook_fn(module, inp):
        hs = inp[0]
        hs[:, -1, :] += delta_vec
        return (hs,) + inp[1:]
    return hook_fn


# ------------------------------------------------------------------
# HEXACO scoring
# ------------------------------------------------------------------

def score_hexaco(items: list[dict], model, tokenizer, device, sys_prompt: str):
    """
    Score all HEXACO items. Returns per-item results plus domain/facet aggregates.
    """
    per_item = []
    domain_scores = defaultdict(list)
    facet_scores = defaultdict(list)

    for item in items:
        item_id = item["item_id"]
        text = item["text"]
        domain = item.get("domain") or "Unassigned"
        facet = item.get("facet", "Unknown")
        keyed = item.get("keyed", "+")

        result = get_likert_logprobs(model, tokenizer, device, text, sys_prompt)
        raw_expected = result["expected_value"]

        # Reverse-score negatively-keyed items: 6 - expected
        if keyed == "-":
            scored = 6.0 - raw_expected
        else:
            scored = raw_expected

        record = {
            "item_id": item_id,
            "text": text,
            "domain": domain,
            "facet": facet,
            "keyed": keyed,
            "raw_expected": float(raw_expected),
            "scored": float(scored),
            "logprobs": result["logprobs"],
            "probs": result["probs"],
        }
        per_item.append(record)
        domain_scores[domain].append(scored)
        facet_scores[facet].append(scored)

        if item_id % 20 == 0:
            logger.info(
                "  item %3d/%d  domain=%-25s  raw=%.2f  scored=%.2f",
                item_id, len(items), domain, raw_expected, scored,
            )

    # Compute means
    domain_means = {d: float(np.mean(vals)) for d, vals in sorted(domain_scores.items())}
    facet_means = {f: float(np.mean(vals)) for f, vals in sorted(facet_scores.items())}

    return {
        "domain_scores": domain_means,
        "facet_scores": facet_means,
        "per_item": per_item,
    }


# ------------------------------------------------------------------
# Summary computation
# ------------------------------------------------------------------

def compute_summary(results: dict, alpha: float):
    """Compute summary: most-affected domain per RIASEC trait, and mapping."""
    baseline_domains = results["baseline"]["hexaco"]["domain_scores"]
    mapping = {}

    for trait in TRAITS:
        trait_domains = results[trait]["hexaco"]["domain_scores"]
        deltas = {}
        for domain in trait_domains:
            deltas[domain] = trait_domains[domain] - baseline_domains.get(domain, 0.0)

        # Most affected = largest absolute delta
        most_affected = max(deltas, key=lambda d: abs(deltas[d]))
        mapping[trait] = {
            "most_affected_domain": most_affected,
            "delta": float(deltas[most_affected]),
            "all_deltas": {d: float(v) for d, v in deltas.items()},
        }

    # Overall: which domain has highest average absolute delta across traits
    domain_avg_abs = defaultdict(list)
    for trait_info in mapping.values():
        for domain, delta in trait_info["all_deltas"].items():
            domain_avg_abs[domain].append(abs(delta))
    overall_most = max(domain_avg_abs, key=lambda d: np.mean(domain_avg_abs[d]))

    return {
        "most_affected_domain": overall_most,
        "mean_abs_delta": float(np.mean(domain_avg_abs[overall_most])),
        "riasec_hexaco_mapping": mapping,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    device = "cuda:1"
    model_id = "marin-community/marin-8b-instruct"
    alpha = 3.0

    root = _repo_root()
    riasec_dir = root / "persona_data" / "model_inits"
    hexaco_path = root / "tom_scoring" / "hexaco_100.json"
    out_dir = root / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HEXACO items
    logger.info("Loading HEXACO-100 items from %s", hexaco_path)
    with open(hexaco_path, "r") as f:
        hexaco_items = json.load(f)
    logger.info("Loaded %d HEXACO items", len(hexaco_items))

    # Load persona vectors and compute residuals
    logger.info("Loading persona vectors and computing residuals...")
    residual, mid_layer = load_persona_vectors(model_id, riasec_dir)
    logger.info("Mid layer: %d, residual vector dim: %d", mid_layer, residual[TRAITS[0]].shape[0])

    # Load model
    logger.info("Loading model: %s on %s", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    blocks = get_decoder_blocks(model)
    logger.info("Model loaded. %d decoder blocks.", len(blocks))

    conditions = ["baseline"] + TRAITS
    results = {}

    for cond_idx, condition in enumerate(conditions):
        logger.info(
            "=== Condition %d/%d: %s ===", cond_idx + 1, len(conditions), condition
        )

        hook_handle = None
        if condition != "baseline":
            vec = residual[condition].astype(np.float32)
            delta_vec = alpha * torch.tensor(vec, dtype=model.dtype).to(device)
            hook_handle = blocks[mid_layer].register_forward_pre_hook(
                make_pre_hook(delta_vec)
            )
            logger.info(
                "Installed pre-hook on block %d for trait=%s, alpha=%.1f, |delta|=%.4f",
                mid_layer, condition, alpha, delta_vec.norm().item(),
            )

        try:
            hexaco_result = score_hexaco(
                hexaco_items, model, tokenizer, device, SYSTEM_PROMPT
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
                logger.info("Removed hook for %s", condition)

        results[condition] = {"hexaco": hexaco_result}

        # Print domain scores
        print(f"\n{'='*60}")
        print(f"  {condition.upper()}")
        print(f"{'='*60}")
        for domain, score in sorted(hexaco_result["domain_scores"].items()):
            baseline_score = (
                results["baseline"]["hexaco"]["domain_scores"].get(domain, 0.0)
                if condition != "baseline"
                else 0.0
            )
            delta_str = (
                f"  (delta={score - baseline_score:+.3f})"
                if condition != "baseline"
                else ""
            )
            print(f"  {domain:<28s} {score:.3f}{delta_str}")

    # Compute summary
    logger.info("Computing summary...")
    summary = compute_summary(results, alpha)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: RIASEC -> HEXACO MAPPING")
    print(f"{'='*60}")
    for trait in TRAITS:
        info = summary["riasec_hexaco_mapping"][trait]
        print(
            f"  {trait:>15s} -> {info['most_affected_domain']:<28s} "
            f"(delta={info['delta']:+.3f})"
        )
    print(f"\n  Overall most affected domain: {summary['most_affected_domain']}")
    print(f"  Mean |delta| across traits:   {summary['mean_abs_delta']:.3f}")

    # Print full delta matrix
    print(f"\n{'='*60}")
    print("FULL DELTA MATRIX (RIASEC x HEXACO)")
    print(f"{'='*60}")
    all_domains = sorted(results["baseline"]["hexaco"]["domain_scores"].keys())
    header = f"  {'':>15s}"
    for domain in all_domains:
        short = domain[:10]
        header += f" {short:>10s}"
    print(header)
    print(f"  {'':>15s}" + " ----------" * len(all_domains))
    for trait in TRAITS:
        row = f"  {trait:>15s}"
        for domain in all_domains:
            delta = summary["riasec_hexaco_mapping"][trait]["all_deltas"].get(domain, 0.0)
            row += f" {delta:+10.3f}"
        print(row)

    # Assemble output
    output = {
        "model": model_id,
        "alpha": alpha,
        "method": "logprob_likert",
        "mid_layer": mid_layer,
        "n_items": len(hexaco_items),
        "system_prompt": SYSTEM_PROMPT,
        "results": results,
        "summary": summary,
    }

    out_path = out_dir / "hexaco_logprob.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
