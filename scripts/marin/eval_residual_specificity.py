#!/usr/bin/env python
"""
Test whether RIASEC residual vectors (shared direction removed) show
cross-trait specificity that the originals lack.

Hypothesis: Original RIASEC vectors = shared_direction + trait_residual.
The shared direction drives non-specific "agree with everything" behavior.
The residuals may carry trait-specific information.

If the residual specificity matrix is diagonal-dominant while the original
was uniform, this confirms that trait-specific information exists but is
masked by a dominant shared component.

Usage:
  uv run python scripts/marin/eval_residual_specificity.py --model_id meta-llama/Llama-3.2-1B-Instruct --device cuda:0
"""

from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig

from pvx import setup_logging
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel

logger = setup_logging(name="residual-specificity")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


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


def eval_gap_on_characteristics(model, tokenizer, device, characteristics, yes_ids, no_ids):
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


def load_raw_vectors(model_id: str) -> dict[str, np.ndarray]:
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"
    vectors = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        with safe_open(str(path), framework="pt") as f:
            vectors[trait] = f.get_tensor("response_persona_vector").numpy().flatten()
    return vectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default="outputs/specificity")
    args = ap.parse_args()

    root = _repo_root()
    layer = args.layer if args.layer is not None else _detect_middle_layer(args.model_id)

    # Load characteristics
    with open(root / "configs/riasec.yaml") as f:
        riasec = yaml.safe_load(f)
    all_chars = {t: riasec[t].get("characteristics", []) for t in TRAITS}

    # Compute residual vectors
    raw_vectors = load_raw_vectors(args.model_id)
    V = np.stack([raw_vectors[t] for t in TRAITS])
    mean_vec = V.mean(axis=0)
    mean_unit = mean_vec / np.linalg.norm(mean_vec)

    residuals = {}
    for trait in TRAITS:
        proj = np.dot(raw_vectors[trait], mean_unit) * mean_unit
        residuals[trait] = raw_vectors[trait] - proj

    # Also compute shared-only vector (just the mean direction)
    # We'll test: original, residual-only, shared-only
    alphas = [0, 3, 5, 8]  # test more alpha values for residuals (smaller effect expected)

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_id": args.model_id,
        "layer": layer,
        "alphas": alphas,
        "traits": TRAITS,
        "original_specificity": {},
        "residual_specificity": {},
        "shared_only_specificity": {},
    }

    # Test with each steering condition
    for condition_name, vector_dict in [
        ("original", {t: raw_vectors[t] for t in TRAITS}),
        ("residual", residuals),
        ("shared_only", {t: mean_vec for t in TRAITS}),
    ]:
        logger.info("=== Testing condition: %s ===", condition_name)
        condition_results = {}

        for steer_trait in TRAITS:
            # Load model for this trait (to get the base model + tokenizer)
            model = RIASECPersonaModel.load_or_create(
                target_model_id=args.model_id, trait=steer_trait, layer=layer
            )
            model.model.eval()

            # Replace response_persona_vector with the condition's vector
            steer_vec = torch.tensor(vector_dict[steer_trait], dtype=model.response_persona_vector.dtype)
            model.response_persona_vector = steer_vec
            model._persona_base = None  # invalidate cache

            yes_ids, no_ids = _find_yes_no_token_ids(model.tokenizer)

            steer_results = {}
            for eval_trait in TRAITS:
                chars = all_chars[eval_trait]
                alpha_results = {}
                for alpha in alphas:
                    with model._steering_delta(alpha):
                        gaps = eval_gap_on_characteristics(
                            model, model.tokenizer, model.device, chars, yes_ids, no_ids
                        )
                    alpha_results[str(alpha)] = {
                        "mean_gap": float(np.mean(gaps)),
                        "std_gap": float(np.std(gaps)),
                    }
                steer_results[eval_trait] = alpha_results
            condition_results[steer_trait] = steer_results

            model.close()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[f"{condition_name}_specificity"] = condition_results

    # Compute specificity indices
    for condition in ["original", "residual", "shared_only"]:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition}")
        print(f"{'='*60}")
        for alpha in alphas:
            alpha_str = str(alpha)
            diagonal = []
            off_diagonal = []
            for i, steer_t in enumerate(TRAITS):
                for j, eval_t in enumerate(TRAITS):
                    gap = results[f"{condition}_specificity"][steer_t][eval_t][alpha_str]["mean_gap"]
                    if i == j:
                        diagonal.append(gap)
                    else:
                        off_diagonal.append(gap)

            diag_mean = np.mean(diagonal)
            off_mean = np.mean(off_diagonal)
            # Specificity index: (diagonal - off_diagonal) / off_diagonal
            spec_diff = diag_mean - off_mean

            print(f"\n  alpha={alpha}:")
            header = f"  {'steer\\eval':>12s}" + "".join(f"{t[:6]:>8s}" for t in TRAITS)
            print(header)
            for steer_t in TRAITS:
                row = f"  {steer_t[:12]:>12s}"
                for eval_t in TRAITS:
                    gap = results[f"{condition}_specificity"][steer_t][eval_t][alpha_str]["mean_gap"]
                    marker = "*" if steer_t == eval_t else " "
                    row += f" {gap:>6.2f}{marker}"
                print(row)
            print(f"  Diagonal: {diag_mean:.3f}, Off-diagonal: {off_mean:.3f}, Diff: {spec_diff:.3f}")

    safe_model = args.model_id.replace("/", "__")
    out_path = output_dir / f"{safe_model}_residual_specificity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
