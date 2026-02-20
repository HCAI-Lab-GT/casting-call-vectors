#!/usr/bin/env python
"""
Cross-trait specificity test: the most important control experiment.

For each RIASEC steering vector, measure the logprob gap on ALL 6 traits'
characteristics (not just the matching trait). This yields a 6x6 specificity
matrix. If personality vectors are genuinely trait-specific, the matrix should
be diagonal-dominant. If they're just generic "agree more" directions, the
matrix will be approximately uniform.

Also tests random-direction baselines: steer with a random vector of the same
norm as each RIASEC vector and measure the effect.

Usage:
  uv run python scripts/marin/eval_cross_trait_specificity.py --model_id meta-llama/Llama-3.2-1B-Instruct --device cuda:0
  uv run python scripts/marin/eval_cross_trait_specificity.py --model_id marin-community/marin-8b-instruct --device cuda:2
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoConfig

from pvx import setup_logging
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel

logger = setup_logging(name="cross-trait-specificity")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
DEFAULT_ALPHAS = [0, 3, 5]  # baseline, moderate, strong


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _detect_middle_layer(model_id: str) -> int:
    return int(AutoConfig.from_pretrained(model_id).num_hidden_layers) // 2


def _messages_for_characteristic(characteristic: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Output EXACTLY one token: YES or NO."},
        {"role": "user", "content": f"{characteristic}"},
    ]


def _single_token_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids if len(ids) == 1 else []


def _find_yes_no_token_ids(tokenizer):
    yes_groups = [["YES", " YES"], ["yes", " yes"], ["Yes", " Yes"]]
    no_groups = [["NO", " NO"], ["no", " no"], ["No", " No"]]

    def collect(cands):
        return sorted({i for c in cands for i in _single_token_ids(tokenizer, c)})

    def pick(label, groups):
        for cands in groups:
            ids = collect(cands)
            if ids:
                return ids
        raise RuntimeError(f"Could not find single-token {label} variants.")

    return pick("YES", yes_groups), pick("NO", no_groups)


def _last_token_logits(outputs) -> torch.Tensor:
    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    return logits[0, -1, :]


def eval_gap_on_characteristics(
    model, tokenizer, device, characteristics, yes_ids, no_ids
) -> list[float]:
    """Evaluate logprob gap on a list of characteristics. Returns list of gaps."""
    gaps = []
    for c in characteristics:
        msgs = _messages_for_characteristic(c)
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                use_cache=False,
                return_dict=True,
            )
        lp = torch.log_softmax(_last_token_logits(out).float(), dim=-1)
        lp_yes = float(lp[yes_ids].max().item())
        lp_no = float(lp[no_ids].max().item())
        gaps.append(lp_yes - lp_no)
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model + inputs (e.g. cuda:2). Use 'auto' for device_map=auto.",
    )
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default="outputs/specificity")
    ap.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated list of alphas (e.g. '0,0.5,1.0'). Defaults to 0,3,5.",
    )
    ap.add_argument(
        "--n_random",
        type=int,
        default=5,
        help="Number of random baseline vectors per trait (0 disables baseline).",
    )
    args = ap.parse_args()

    root = _repo_root()
    layer = args.layer if args.layer is not None else _detect_middle_layer(args.model_id)
    alphas = (
        [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
        if args.alphas is not None
        else DEFAULT_ALPHAS
    )

    # Load RIASEC characteristics
    with open(root / "configs/riasec.yaml") as f:
        riasec = yaml.safe_load(f)

    all_chars = {t: riasec[t].get("characteristics", []) for t in TRAITS}

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results structure
    results = {
        "model_id": args.model_id,
        "layer": layer,
        "alphas": alphas,
        "traits": TRAITS,
        "specificity_matrix": {},  # steer_trait -> {eval_trait -> {alpha -> mean_gap}}
        "random_baseline": {},  # eval_trait -> {alpha -> mean_gap}
        "vector_norms": {},
    }

    # For each steering trait, load model, steer at each alpha, eval on ALL traits
    for steer_trait in TRAITS:
        logger.info("Loading model for steering trait: %s", steer_trait)
        model = RIASECPersonaModel.load_or_create(
            target_model_id=args.model_id, trait=steer_trait, layer=layer, device=args.device
        )
        model.model.eval()

        # Record vector norm
        vec_norm = float(model.response_persona_vector.norm().item())
        results["vector_norms"][steer_trait] = vec_norm
        logger.info("  %s vector norm: %.4f", steer_trait, vec_norm)

        yes_ids, no_ids = _find_yes_no_token_ids(model.tokenizer)
        device = model.device

        steer_results = {}
        for eval_trait in TRAITS:
            chars = all_chars[eval_trait]
            alpha_results = {}
            for alpha in alphas:
                with model._steering_delta(alpha):
                    gaps = eval_gap_on_characteristics(
                        model, model.tokenizer, device, chars, yes_ids, no_ids
                    )
                alpha_results[str(alpha)] = {
                    "mean_gap": float(np.mean(gaps)),
                    "std_gap": float(np.std(gaps)),
                    "gaps": gaps,
                }
            steer_results[eval_trait] = alpha_results

        results["specificity_matrix"][steer_trait] = steer_results

        # Random baseline: steer with random vectors of same norm
        if args.n_random > 0:
            logger.info("  Running random baseline (n=%d)...", args.n_random)
            hidden_dim = model.response_persona_vector.shape[0]
            rng = np.random.default_rng(42)

            random_gaps_by_alpha = {str(a): {t: [] for t in TRAITS} for a in alphas}

            for _rand_i in range(args.n_random):
                # Create random vector with same norm
                rand_vec = torch.tensor(
                    rng.standard_normal(hidden_dim), dtype=model.response_persona_vector.dtype
                )
                rand_vec = rand_vec / rand_vec.norm() * model.response_persona_vector.norm()

                # Temporarily replace the response_persona_vector and invalidate cache
                original_vec = model.response_persona_vector.clone()
                model.response_persona_vector = rand_vec
                model._persona_base = None  # invalidate cache so _get_persona_base() recomputes

                for alpha in alphas:
                    with model._steering_delta(alpha):
                        for eval_trait in TRAITS:
                            chars = all_chars[eval_trait]
                            gaps = eval_gap_on_characteristics(
                                model, model.tokenizer, device, chars, yes_ids, no_ids
                            )
                            random_gaps_by_alpha[str(alpha)][eval_trait].extend(gaps)

                # Restore original vector and invalidate cache
                model.response_persona_vector = original_vec
                model._persona_base = None

            # Only need to do random baseline once (it's the same model, different random vecs)
            # Store mean across all random trials
            if steer_trait == TRAITS[0]:  # Only compute once
                for alpha_str, trait_gaps in random_gaps_by_alpha.items():
                    results["random_baseline"][alpha_str] = {}
                    for eval_trait, gaps in trait_gaps.items():
                        results["random_baseline"][alpha_str][eval_trait] = {
                            "mean_gap": float(np.mean(gaps)),
                            "std_gap": float(np.std(gaps)),
                            "n": len(gaps),
                        }

        model.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute summary: specificity index
    # For each alpha, compute diagonal_mean / off_diagonal_mean
    for alpha in alphas:
        alpha_str = str(alpha)
        diagonal = []
        off_diagonal = []
        for i, steer_t in enumerate(TRAITS):
            for j, eval_t in enumerate(TRAITS):
                gap = results["specificity_matrix"][steer_t][eval_t][alpha_str]["mean_gap"]
                if i == j:
                    diagonal.append(gap)
                else:
                    off_diagonal.append(gap)
        results[f"specificity_index_alpha_{alpha_str}"] = {
            "diagonal_mean": float(np.mean(diagonal)),
            "off_diagonal_mean": float(np.mean(off_diagonal)),
            "diagonal_std": float(np.std(diagonal)),
            "off_diagonal_std": float(np.std(off_diagonal)),
            "ratio": float(np.mean(diagonal) / np.mean(off_diagonal))
            if np.mean(off_diagonal) != 0
            else float("inf"),
        }

    safe_model = args.model_id.replace("/", "__")
    out_path = output_dir / f"{safe_model}_cross_trait_specificity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_path)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"CROSS-TRAIT SPECIFICITY: {args.model_id}")
    print(f"{'=' * 70}")
    for alpha in alphas:
        alpha_str = str(alpha)
        print(f"\nalpha = {alpha}:")
        header = f"{'steer\\eval':>15s}" + "".join(f"{t[:6]:>10s}" for t in TRAITS)
        print(header)
        for steer_t in TRAITS:
            row = f"{steer_t[:15]:>15s}"
            for eval_t in TRAITS:
                gap = results["specificity_matrix"][steer_t][eval_t][alpha_str]["mean_gap"]
                marker = " *" if steer_t == eval_t else "  "
                row += f"{gap:>8.2f}{marker}"
            print(row)

        si = results[f"specificity_index_alpha_{alpha_str}"]
        print(
            f"  Diagonal mean: {si['diagonal_mean']:.3f}, Off-diagonal mean: {si['off_diagonal_mean']:.3f}, Ratio: {si['ratio']:.2f}"
        )

        if alpha_str in results["random_baseline"]:
            print("  Random baseline gaps: ", end="")
            for t in TRAITS:
                rg = results["random_baseline"][alpha_str][t]["mean_gap"]
                print(f"{t[:4]}={rg:.2f} ", end="")
            print()


if __name__ == "__main__":
    main()
