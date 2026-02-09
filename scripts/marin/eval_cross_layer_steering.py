#!/usr/bin/env python
"""
Cross-layer steering: does the injection layer matter?

Tests two conditions for each injection layer L_i:
1. "matched": inject vector extracted at L_i into L_i
2. "transfer": inject vector extracted at middle layer into L_i

If matched ≈ transfer across layers, then extraction layer doesn't matter.
If matched > transfer, vectors are layer-specific.
If matched peaks at one layer, that's the optimal injection point.

This experiment reveals whether persona geometry is a single structure
accessible from any layer, or layer-specific features.
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from pvx import setup_logging

logger = setup_logging(name="cross-layer-steering")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def logprob_gap(model, tokenizer, device, question: str, steer_delta=None) -> float:
    """Compute log P(YES) - log P(NO) for a question."""
    messages = [
        {"role": "system", "content": "Output EXACTLY one token: YES or NO."},
        {"role": "user", "content": question},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)
    return (log_probs[yes_ids[0]] - log_probs[no_ids[0]]).item()


def eval_trait_gaps(model, tokenizer, device, characteristics: dict) -> dict[str, float]:
    """Evaluate logprob gaps for all 6 traits."""
    results = {}
    for trait in TRAITS:
        gaps = [logprob_gap(model, tokenizer, device, q) for q in characteristics[trait]]
        results[trait] = float(np.mean(gaps))
    return results


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--trait", type=str, default="investigative",
                    help="Primary trait to test (use all for comprehensive)")
    ap.add_argument("--layer_step", type=int, default=1,
                    help="Test every N-th layer (1=all layers, 2=every other)")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model_id)
    num_layers = config.num_hidden_layers
    mid_layer = num_layers // 2

    logger.info("Model: %s, %d layers, mid=%d", args.model_id, num_layers, mid_layer)

    # Load RIASEC characteristics
    riasec_path = _repo_root() / "configs" / "riasec.yaml"
    with open(riasec_path) as f:
        riasec = yaml.safe_load(f)
    characteristics = {t: riasec[t]["characteristics"] for t in TRAITS}

    # Load all-layer vectors
    safe_model = args.model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"

    traits_to_test = TRAITS if args.trait == "all" else [args.trait]

    all_layer_vectors = {}
    for trait in traits_to_test:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        # all_layers_response_persona_vector shape: (num_layers+1, 1, hidden_dim)
        all_layers = data["all_layers_response_persona_vector"].numpy()
        if all_layers.ndim == 3:
            all_layers = all_layers[:, 0, :]  # (num_layers+1, hidden_dim)
        all_layer_vectors[trait] = all_layers
        logger.info("Loaded %s all-layer vectors: shape %s", trait, all_layers.shape)

    # Load model
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()
    device = args.device

    blocks = get_decoder_blocks(model)

    # Test layers
    test_layers = list(range(0, num_layers, args.layer_step))
    if mid_layer not in test_layers:
        test_layers.append(mid_layer)
        test_layers.sort()

    results = {
        "model_id": args.model_id,
        "alpha": args.alpha,
        "num_layers": num_layers,
        "mid_layer": mid_layer,
        "test_layers": test_layers,
        "conditions": {},
    }

    # Baseline (no steering)
    logger.info("Evaluating baseline...")
    baseline = eval_trait_gaps(model, tokenizer, device, characteristics)
    results["conditions"]["baseline"] = baseline
    logger.info("Baseline: %s", {t[:3]: f"{v:.2f}" for t, v in baseline.items()})

    # For each trait and each injection layer
    for trait in traits_to_test:
        all_layers = all_layer_vectors[trait]

        for inject_layer in test_layers:
            # Condition 1: matched (extract at L, inject at L)
            # Vector index is layer+1 because index 0 = embedding layer
            vec_matched = all_layers[inject_layer + 1]
            vec_matched_t = torch.tensor(vec_matched, dtype=torch.float16).unsqueeze(0).to(device)

            # Condition 2: transfer (extract at mid, inject at L)
            vec_mid = all_layers[mid_layer + 1]
            vec_mid_t = torch.tensor(vec_mid, dtype=torch.float16).unsqueeze(0).to(device)

            for cond_name, vec_t in [("matched", vec_matched_t), ("transfer", vec_mid_t)]:
                key = f"{trait}_{cond_name}_L{inject_layer}"
                logger.info("Evaluating: %s", key)

                # Install hook at inject_layer
                hook_handle = None
                delta = args.alpha * vec_t

                def make_hook(d):
                    def hook_fn(_module, _inp, out):
                        if isinstance(out, tuple):
                            hs = out[0]
                            hs[:, -1, :] += d
                            return (hs,) + out[1:]
                        out[:, -1, :] += d
                        return out
                    return hook_fn

                hook_handle = blocks[inject_layer].register_forward_hook(make_hook(delta))

                try:
                    gaps = eval_trait_gaps(model, tokenizer, device, characteristics)
                finally:
                    hook_handle.remove()

                results["conditions"][key] = gaps

                target_gap = gaps[trait]
                target_rank = sorted(gaps.values(), reverse=True).index(target_gap) + 1
                logger.info("  %s L%d: target=%s gap=%.2f rank=%d",
                            cond_name, inject_layer, trait[:3], target_gap, target_rank)

    # Analysis
    print(f"\n{'='*70}")
    print("CROSS-LAYER STEERING ANALYSIS")
    print(f"{'='*70}")

    for trait in traits_to_test:
        print(f"\n--- Trait: {trait} ---")
        print(f"{'Layer':>6} {'Matched':>10} {'Transfer':>10} {'Δ(M-T)':>10} {'M Rank':>8} {'T Rank':>8} {'Vec Norm':>10}")

        matched_gaps = []
        transfer_gaps = []
        norms = []

        for L in test_layers:
            m_key = f"{trait}_matched_L{L}"
            t_key = f"{trait}_transfer_L{L}"

            m_gaps = results["conditions"].get(m_key, {})
            t_gaps = results["conditions"].get(t_key, {})

            m_val = m_gaps.get(trait, 0)
            t_val = t_gaps.get(trait, 0)
            diff = m_val - t_val

            m_rank = sorted(m_gaps.values(), reverse=True).index(m_val) + 1 if m_gaps else 0
            t_rank = sorted(t_gaps.values(), reverse=True).index(t_val) + 1 if t_gaps else 0

            vec_norm = np.linalg.norm(all_layer_vectors[trait][L + 1])

            matched_gaps.append(m_val)
            transfer_gaps.append(t_val)
            norms.append(vec_norm)

            marker = " ← mid" if L == mid_layer else ""
            print(f"{L:>6} {m_val:>10.2f} {t_val:>10.2f} {diff:>+10.2f} {m_rank:>8} {t_rank:>8} {vec_norm:>10.1f}{marker}")

        matched_gaps = np.array(matched_gaps)
        transfer_gaps = np.array(transfer_gaps)
        norms = np.array(norms)

        print(f"\nSummary:")
        print(f"  Matched: mean={matched_gaps.mean():.2f}, std={matched_gaps.std():.2f}, "
              f"best=L{test_layers[np.argmax(matched_gaps)]} ({matched_gaps.max():.2f})")
        print(f"  Transfer: mean={transfer_gaps.mean():.2f}, std={transfer_gaps.std():.2f}, "
              f"best=L{test_layers[np.argmax(transfer_gaps)]} ({transfer_gaps.max():.2f})")

        # Correlation between matched and transfer
        if len(matched_gaps) > 2:
            corr = np.corrcoef(matched_gaps, transfer_gaps)[0, 1]
            print(f"  Correlation(matched, transfer): r = {corr:.3f}")

            # Correlation with norm
            norm_corr_m = np.corrcoef(norms, matched_gaps)[0, 1]
            norm_corr_t = np.corrcoef(norms, transfer_gaps)[0, 1]
            print(f"  Correlation(norm, matched): r = {norm_corr_m:.3f}")
            print(f"  Correlation(norm, transfer): r = {norm_corr_t:.3f}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cross_layer_steering_{safe_model}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
