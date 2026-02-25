#!/usr/bin/env python
"""
Psychometric dose-response: O*NET Interest Profiler at multiple alpha values.

At alpha=3.0, personality steering raises all RIASEC scores uniformly due to
a strong shared component. This experiment sweeps alpha to find the value where
the TARGET dimension is boosted significantly more than others, maximizing the
"specificity ratio" = delta_target / mean(delta_others).

Uses logprob scoring (logP(Yes) - logP(No)) for each item, which is far more
sensitive than text-based binary scoring (finding #9: signal is in activations).

Vectors: residuals after subtracting shared PC1 direction.
Hook: register_forward_pre_hook on mid_layer block.
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

logger = setup_logging(name="psych-dose-response")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Map lowercase trait to O*NET dimension name (capitalized)
TRAIT_TO_DIM = {t: t.capitalize() for t in TRAITS}
DIM_TO_TRAIT = {v: k for k, v in TRAIT_TO_DIM.items()}

ALPHAS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

MODEL_ID = "marin-community/marin-8b-instruct"
DEVICE = "cuda:3"

SYS_PROMPT = (
    "You are completing a vocational interest assessment. "
    "For each activity, answer Yes if you would enjoy doing it, "
    "or No if you would not. Answer with EXACTLY one word: Yes or No."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


# ── O*NET item loader ──────────────────────────────────
def load_onet_items():
    path = _repo_root() / "tom_scoring" / "interest_profiler.json"
    with open(path) as f:
        return json.load(f)


# ── Logprob extraction ─────────────────────────────────
def get_binary_logprob_gap(model, tokenizer, device, item_text, sys_prompt):
    """Return logP(Yes) - logP(No) for a single O*NET item."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f'"{item_text}"'},
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

    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)

    return log_probs[yes_ids[0]].item() - log_probs[no_ids[0]].item()


# ── Steering hook ──────────────────────────────────────
def make_pre_hook(delta_vec):
    """Forward pre-hook: add delta_vec to the last-token hidden state."""
    def hook_fn(module, inp):
        hs = inp[0]
        hs[:, -1, :] += delta_vec
        return (hs,) + inp[1:]
    return hook_fn


# ── Score aggregation ──────────────────────────────────
def aggregate_scores(logprob_gaps, items):
    """Aggregate per-item logprob gaps into per-dimension mean scores."""
    dim_gaps = defaultdict(list)
    for i, item in enumerate(items):
        dim_gaps[item["dimension"]].append(logprob_gaps[i])
    return {dim: float(np.mean(gaps)) for dim, gaps in dim_gaps.items()}


# ── Run battery ────────────────────────────────────────
def run_battery(model, tokenizer, device, items):
    """Run full O*NET battery, return per-item gaps and aggregated scores."""
    gaps = []
    for item in items:
        gap = get_binary_logprob_gap(model, tokenizer, device, item["text"], SYS_PROMPT)
        gaps.append(gap)
    scores = aggregate_scores(gaps, items)
    return gaps, scores


# ── Main ───────────────────────────────────────────────
def main():
    root = _repo_root()

    # ── Model config ───────────────────────────────────
    config = AutoConfig.from_pretrained(MODEL_ID)
    mid_layer = config.num_hidden_layers // 2  # 16 for Marin 8B
    safe_model = MODEL_ID.replace("/", "__")
    riasec_dir = root / "persona_data" / "model_inits"

    logger.info("Model: %s, mid_layer: %d, device: %s", MODEL_ID, mid_layer, DEVICE)

    # ── Load persona vectors and compute residuals ─────
    logger.info("Loading persona vectors and computing residuals...")
    all_layer_vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_layer_vectors[trait] = vecs

    # Stack vectors at detect_layer (mid_layer + 1) for SVD
    V = np.stack([all_layer_vectors[t][mid_layer + 1] for t in TRAITS])
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    residual_vectors = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1].astype(np.float32)
        proj = np.dot(vec, shared_dir) * shared_dir
        residual_vectors[t] = (vec - proj).astype(np.float32)

    # Log norms
    for t in TRAITS:
        raw_norm = np.linalg.norm(all_layer_vectors[t][mid_layer + 1])
        res_norm = np.linalg.norm(residual_vectors[t])
        logger.info("  %s: raw_norm=%.4f, residual_norm=%.4f", t, raw_norm, res_norm)

    # ── Load model ─────────────────────────────────────
    logger.info("Loading model: %s on %s", MODEL_ID, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    blocks = get_decoder_blocks(model)

    # ── Load O*NET items ───────────────────────────────
    items = load_onet_items()
    dimensions = sorted(set(item["dimension"] for item in items))
    logger.info("O*NET Interest Profiler: %d items, %d dimensions", len(items), len(dimensions))

    # ── Run baseline (no steering) ─────────────────────
    logger.info("Running baseline (no steering)...")
    baseline_gaps, baseline_scores = run_battery(model, tokenizer, DEVICE, items)
    logger.info("Baseline scores: %s", {k: f"{v:.3f}" for k, v in baseline_scores.items()})

    # ── Alpha sweep ────────────────────────────────────
    per_alpha = {}

    for alpha in ALPHAS:
        logger.info("=" * 60)
        logger.info("Alpha = %.2f", alpha)
        logger.info("=" * 60)

        alpha_result = {
            "baseline": {d: baseline_scores[d] for d in dimensions},
            "steered": {},
        }

        specificity_ratios = []

        for steer_trait in TRAITS:
            target_dim = TRAIT_TO_DIM[steer_trait]
            logger.info("  Steering: %s (target dim: %s)", steer_trait, target_dim)

            # Build delta vector
            vec_np = residual_vectors[steer_trait]
            vec_t = torch.tensor(vec_np, dtype=model.dtype).unsqueeze(0).to(DEVICE)
            delta_vec = alpha * vec_t

            # Install forward_pre_hook
            hook_handle = blocks[mid_layer].register_forward_pre_hook(make_pre_hook(delta_vec))
            try:
                steered_gaps, steered_scores = run_battery(model, tokenizer, DEVICE, items)
            finally:
                hook_handle.remove()

            # Compute deltas from baseline
            delta_from_baseline = {
                d: steered_scores[d] - baseline_scores[d] for d in dimensions
            }

            # Specificity ratio = delta_target / mean(delta_others)
            delta_target = delta_from_baseline[target_dim]
            other_dims = [d for d in dimensions if d != target_dim]
            delta_others = [delta_from_baseline[d] for d in other_dims]
            mean_delta_others = float(np.mean(delta_others))

            if mean_delta_others != 0:
                specificity_ratio = delta_target / mean_delta_others
            else:
                specificity_ratio = float("inf") if delta_target > 0 else float("-inf") if delta_target < 0 else float("nan")

            # Target rank: where does target_dim rank among all dimensions by delta?
            sorted_dims = sorted(dimensions, key=lambda d: delta_from_baseline[d], reverse=True)
            target_rank = sorted_dims.index(target_dim) + 1  # 1-indexed

            alpha_result["steered"][steer_trait] = {
                "scores": {d: steered_scores[d] for d in dimensions},
                "delta_from_baseline": {d: delta_from_baseline[d] for d in dimensions},
                "specificity_ratio": float(specificity_ratio),
                "target_rank": target_rank,
                "delta_target": float(delta_target),
                "mean_delta_others": float(mean_delta_others),
            }

            specificity_ratios.append(specificity_ratio)

            logger.info(
                "    target_delta=%.3f, mean_other_delta=%.3f, specificity=%.2f, rank=%d/6",
                delta_target, mean_delta_others, specificity_ratio, target_rank,
            )

        # Mean specificity across all 6 traits at this alpha
        finite_ratios = [r for r in specificity_ratios if np.isfinite(r)]
        alpha_result["mean_specificity_ratio"] = float(np.mean(finite_ratios)) if finite_ratios else float("nan")
        alpha_result["median_specificity_ratio"] = float(np.median(finite_ratios)) if finite_ratios else float("nan")
        alpha_result["rank_1_count"] = sum(
            1 for t in TRAITS if alpha_result["steered"][t]["target_rank"] == 1
        )

        per_alpha[str(alpha)] = alpha_result
        logger.info(
            "  Alpha %.2f summary: mean_specificity=%.3f, median=%.3f, rank-1 count=%d/6",
            alpha, alpha_result["mean_specificity_ratio"],
            alpha_result["median_specificity_ratio"],
            alpha_result["rank_1_count"],
        )

    # ── Summary analysis ───────────────────────────────
    specificity_curve = {}
    for alpha in ALPHAS:
        specificity_curve[str(alpha)] = per_alpha[str(alpha)]["mean_specificity_ratio"]

    # Find optimal alpha (max mean specificity, excluding inf/nan)
    best_alpha = None
    best_specificity = -float("inf")
    for alpha in ALPHAS:
        s = per_alpha[str(alpha)]["mean_specificity_ratio"]
        if np.isfinite(s) and s > best_specificity:
            best_specificity = s
            best_alpha = alpha

    # Build conclusion
    rank1_at_best = per_alpha[str(best_alpha)]["rank_1_count"] if best_alpha else 0
    conclusion_parts = [
        f"Optimal alpha = {best_alpha} with mean specificity ratio = {best_specificity:.3f}.",
        f"At this alpha, {rank1_at_best}/6 traits have their target dimension ranked #1.",
    ]
    if best_specificity > 1.5:
        conclusion_parts.append(
            "Strong specificity: target dimension boosts >50% more than others."
        )
    elif best_specificity > 1.0:
        conclusion_parts.append(
            "Moderate specificity: target dimension boosts somewhat more than others."
        )
    else:
        conclusion_parts.append(
            "Weak specificity: steering boosts all dimensions roughly equally."
        )

    # Check monotonicity
    vals = [specificity_curve[str(a)] for a in ALPHAS if np.isfinite(specificity_curve[str(a)])]
    if len(vals) >= 3:
        if all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
            conclusion_parts.append("Specificity increases monotonically with alpha.")
        elif all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
            conclusion_parts.append("Specificity decreases monotonically with alpha (shared component dominates).")
        else:
            conclusion_parts.append("Specificity is non-monotonic -- there is an optimal alpha sweet spot.")

    conclusion = " ".join(conclusion_parts)

    # ── Build output ───────────────────────────────────
    results = {
        "model": MODEL_ID,
        "device": DEVICE,
        "mid_layer": mid_layer,
        "method": "logprob (logP(Yes) - logP(No))",
        "n_items": len(items),
        "dimensions": dimensions,
        "alphas": ALPHAS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_alpha": per_alpha,
        "summary": {
            "optimal_alpha": best_alpha,
            "max_specificity_ratio": float(best_specificity),
            "specificity_curve": specificity_curve,
            "rank1_curve": {str(a): per_alpha[str(a)]["rank_1_count"] for a in ALPHAS},
            "conclusion": conclusion,
        },
    }

    # ── Print summary table ────────────────────────────
    print(f"\n{'='*80}")
    print(f"PSYCHOMETRIC DOSE-RESPONSE: {MODEL_ID}")
    print(f"Method: logprob scoring on O*NET Interest Profiler ({len(items)} items)")
    print(f"Vectors: residuals (shared PC1 removed)")
    print(f"Hook: forward_pre_hook on layer {mid_layer}")
    print(f"{'='*80}")

    # Baseline
    print(f"\nBaseline scores (no steering):")
    for d in dimensions:
        print(f"  {d:>15s}: {baseline_scores[d]:>+.3f}")

    # Per-alpha table
    print(f"\n{'Alpha':>6}  {'MeanSpec':>8}  {'MedSpec':>8}  {'Rank1':>5}", end="")
    for t in TRAITS:
        print(f"  {t[:4]:>5}", end="")
    print()
    print(f"{'-----':>6}  {'--------':>8}  {'--------':>8}  {'-----':>5}", end="")
    for _ in TRAITS:
        print(f"  {'-----':>5}", end="")
    print()

    for alpha in ALPHAS:
        r = per_alpha[str(alpha)]
        print(f"{alpha:>6.2f}  {r['mean_specificity_ratio']:>+8.3f}  {r['median_specificity_ratio']:>+8.3f}  {r['rank_1_count']:>5d}", end="")
        for t in TRAITS:
            sr = r["steered"][t]["specificity_ratio"]
            print(f"  {sr:>+5.1f}", end="")
        print()

    # Per-trait detail at optimal alpha
    if best_alpha is not None:
        print(f"\n--- Detail at optimal alpha = {best_alpha} ---")
        r = per_alpha[str(best_alpha)]
        for t in TRAITS:
            td = r["steered"][t]
            target_dim = TRAIT_TO_DIM[t]
            print(f"\n  Steering: {t} (target: {target_dim})")
            print(f"    {'Dimension':>15s}  {'Baseline':>8s}  {'Steered':>8s}  {'Delta':>8s}")
            for d in dimensions:
                bl = r["baseline"][d]
                st = td["scores"][d]
                dt = td["delta_from_baseline"][d]
                marker = " ***" if d == target_dim else ""
                print(f"    {d:>15s}  {bl:>+8.3f}  {st:>+8.3f}  {dt:>+8.3f}{marker}")
            print(f"    Specificity ratio: {td['specificity_ratio']:.3f}, Target rank: {td['target_rank']}/6")

    # Conclusion
    print(f"\n{'='*80}")
    print(f"CONCLUSION: {conclusion}")
    print(f"{'='*80}")

    # ── Save ───────────────────────────────────────────
    out_dir = root / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "psychometric_dose_response.json"

    with open(out_path, "w") as f:
        json.dump(
            results, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else (
                bool(x) if isinstance(x, np.bool_) else x
            ),
        )

    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
