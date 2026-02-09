#!/usr/bin/env python
"""
Compositional steering: test whether combining RIASEC trait vectors
produces responses that exhibit BOTH traits.

Key questions:
1. Does steering with A + B show both A and B traits?
2. Does steering with A - B show A but not B?
3. Is the personality space linear/compositional?

Measures: logprob gaps for each trait when steering with combined vectors.
Uses the eval_logprob_steering infrastructure.
"""

import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoConfig

from pvx import setup_logging
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel

logger = setup_logging(name="compositional-steering")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_raw_vectors(model_id: str) -> dict[str, np.ndarray]:
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"
    vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vectors[trait] = data["response_persona_vector"].numpy().flatten()
    return vectors


def logprob_gap(model, question: str, alpha: float = 5.0) -> float:
    """Compute log P(YES) - log P(NO) for a question under current steering."""
    messages = [
        {"role": "system", "content": "Output EXACTLY one token: YES or NO."},
        {"role": "user", "content": question},
    ]
    formatted = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = model.tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)

    with model._steering_delta(alpha):
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    yes_ids = model.tokenizer.encode("YES", add_special_tokens=False)
    no_ids = model.tokenizer.encode("NO", add_special_tokens=False)
    yes_logprob = log_probs[yes_ids[0]].item()
    no_logprob = log_probs[no_ids[0]].item()
    return yes_logprob - no_logprob


def eval_all_traits(model, characteristics: dict, alpha: float) -> dict[str, float]:
    """Evaluate logprob gaps for all 6 traits' characteristics."""
    results = {}
    for trait in TRAITS:
        gaps = []
        for char_text in characteristics[trait]:
            gap = logprob_gap(model, char_text, alpha)
            gaps.append(gap)
        results[trait] = float(np.mean(gaps))
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--alpha", type=float, default=5.0)
    args = ap.parse_args()

    layer = AutoConfig.from_pretrained(args.model_id).num_hidden_layers // 2

    # Load RIASEC config for characteristics
    riasec_path = _repo_root() / "configs" / "riasec.yaml"
    with open(riasec_path) as f:
        riasec = yaml.safe_load(f)

    characteristics = {}
    for trait in TRAITS:
        characteristics[trait] = riasec[trait]["characteristics"]

    # Load vectors
    raw_vectors = load_raw_vectors(args.model_id)

    # Decompose into shared + residual
    V = np.stack([raw_vectors[t] for t in TRAITS])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residuals = {}
    for trait in TRAITS:
        proj = np.dot(raw_vectors[trait], shared_dir) * shared_dir
        residuals[trait] = raw_vectors[trait] - proj

    # Load model once
    model = RIASECPersonaModel.load_or_create(
        target_model_id=args.model_id, trait=TRAITS[0], layer=layer
    )
    model.model.eval()

    results = {
        "model_id": args.model_id,
        "alpha": args.alpha,
        "layer": layer,
        "conditions": {},
    }

    # Define test conditions
    conditions = {}

    # Baseline
    conditions["baseline"] = np.zeros_like(raw_vectors["artistic"])

    # Single traits (original)
    for trait in TRAITS:
        conditions[f"original_{trait}"] = raw_vectors[trait]

    # Single traits (residual)
    for trait in TRAITS:
        conditions[f"residual_{trait}"] = residuals[trait]

    # Compositional: trait A + trait B (residuals)
    # Test all 15 unique pairs
    for i, t1 in enumerate(TRAITS):
        for t2 in TRAITS[i + 1:]:
            combined = residuals[t1] + residuals[t2]
            conditions[f"residual_{t1}+{t2}"] = combined

    # Subtraction: trait A - trait B (residuals)
    # Test a few interesting pairs
    interesting_pairs = [
        ("artistic", "conventional"),  # opposite on hexagon
        ("investigative", "social"),   # opposite
        ("realistic", "enterprising"), # opposite
    ]
    for t1, t2 in interesting_pairs:
        conditions[f"residual_{t1}-{t2}"] = residuals[t1] - residuals[t2]
        conditions[f"residual_{t2}-{t1}"] = residuals[t2] - residuals[t1]

    # Run evaluations
    for cond_name, vector in conditions.items():
        logger.info("Evaluating: %s", cond_name)

        # Set the vector
        model.response_persona_vector = torch.tensor(
            vector, dtype=model.response_persona_vector.dtype
        ).unsqueeze(0)
        model._persona_base = None

        alpha = 0.0 if cond_name == "baseline" else args.alpha
        trait_gaps = eval_all_traits(model, characteristics, alpha)
        results["conditions"][cond_name] = trait_gaps

        # Print summary
        sorted_gaps = sorted(trait_gaps.items(), key=lambda x: -x[1])
        top = sorted_gaps[0]
        logger.info("  Top trait: %s (%.2f), All: %s", top[0], top[1],
                     ", ".join(f"{t[:3]}={g:.2f}" for t, g in sorted_gaps))

    # Analysis: does A+B boost both A and B?
    print(f"\n{'=' * 70}")
    print("COMPOSITIONAL STEERING ANALYSIS")
    print(f"{'=' * 70}")

    print(f"\n--- Baseline ---")
    base = results["conditions"]["baseline"]
    print(f"  {', '.join(f'{t[:4]}={v:.2f}' for t, v in base.items())}")

    print(f"\n--- Single residual traits ---")
    for trait in TRAITS:
        gaps = results["conditions"][f"residual_{trait}"]
        print(f"  {trait:15s}: {', '.join(f'{t[:4]}={v:.2f}' for t, v in gaps.items())}")

    print(f"\n--- Additive compositions (residual_A + residual_B) ---")
    print(f"  {'Condition':30s} | {'Expected boost':20s} | {'Trait gaps (top 3)':50s}")
    for i, t1 in enumerate(TRAITS):
        for t2 in TRAITS[i + 1:]:
            cond = f"residual_{t1}+{t2}"
            gaps = results["conditions"][cond]
            sorted_gaps = sorted(gaps.items(), key=lambda x: -x[1])
            expected = f"{t1[:4]}+{t2[:4]}"
            top3 = ", ".join(f"{t[:4]}={g:.2f}" for t, g in sorted_gaps[:3])
            # Check if both expected traits are in top 3
            top3_traits = [t for t, g in sorted_gaps[:3]]
            t1_rank = [t for t, g in sorted_gaps].index(t1) + 1
            t2_rank = [t for t, g in sorted_gaps].index(t2) + 1
            check = "✓" if t1_rank <= 3 and t2_rank <= 3 else " "
            print(f"  {cond:30s} | {expected:20s} | {top3:50s} | {check} (rank: {t1[:3]}={t1_rank}, {t2[:3]}={t2_rank})")

    print(f"\n--- Subtractive compositions (residual_A - residual_B) ---")
    for t1, t2 in interesting_pairs:
        for a, b in [(t1, t2), (t2, t1)]:
            cond = f"residual_{a}-{b}"
            gaps = results["conditions"][cond]
            sorted_gaps = sorted(gaps.items(), key=lambda x: -x[1])
            a_gap = gaps[a]
            b_gap = gaps[b]
            print(f"  {cond:30s}: {a[:4]}={a_gap:.2f}, {b[:4]}={b_gap:.2f} | "
                  f"top: {', '.join(f'{t[:4]}={g:.2f}' for t, g in sorted_gaps[:3])}")

    # Save
    out_dir = _repo_root() / "outputs" / "compositional"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model_id.replace("/", "__")
    out_path = out_dir / f"{safe_model}_compositional_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    model.close()


if __name__ == "__main__":
    main()
