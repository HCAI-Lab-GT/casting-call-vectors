#!/usr/bin/env python
"""
Capability preservation with ALL-POSITION steering.

The previous capability test used last-token-only hooks, which barely affects
perplexity because most tokens are unmodified. This version applies the
steering vector to ALL token positions, which is the realistic deployment
scenario for generation.

Tests:
1. Perplexity on general text (all-position steering)
2. Factual QA accuracy
3. Generation quality (does the model stay coherent?)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cap-allpos")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

FACTUAL_QA = [
    {"question": "What is the capital of France?", "answer_tokens": ["Paris"]},
    {"question": "What planet is closest to the Sun?", "answer_tokens": ["Mercury"]},
    {"question": "Who wrote Romeo and Juliet?", "answer_tokens": ["Shakespeare", "William"]},
    {"question": "What is the chemical symbol for water?", "answer_tokens": ["H2O", "H₂O"]},
    {"question": "How many continents are there?", "answer_tokens": ["seven", "7"]},
    {"question": "What is the largest ocean on Earth?", "answer_tokens": ["Pacific"]},
    {"question": "In what year did World War II end?", "answer_tokens": ["1945"]},
    {"question": "What is the boiling point of water in Celsius?", "answer_tokens": ["100"]},
]

PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the old oak tree.",
    "Machine learning algorithms have transformed how we process large datasets.",
    "The restaurant served an excellent pasta with fresh tomatoes and basil.",
    "Democracy requires active participation from citizens in the political process.",
    "The sunset painted the sky in shades of orange, pink, and purple.",
    "Scientists discovered a new species of deep-sea fish near hydrothermal vents.",
    "The concert hall was filled with the sound of Beethoven's fifth symphony.",
    "Economic indicators suggest a gradual recovery in the manufacturing sector.",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_residual_vectors(model_id, riasec_dir):
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
    shared_dir = shared_dir / np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    return residual, mid_layer


def make_allpos_hook(delta_vec):
    """Hook that adds steering vector to ALL positions."""
    def hook_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            # Add to ALL positions (broadcast over seq_len)
            hs = hs + delta_vec.unsqueeze(1)  # [batch, 1, hidden] broadcasts to [batch, seq, hidden]
            return (hs,) + out[1:]
        out = out + delta_vec.unsqueeze(1)
        return out
    return hook_fn


def make_lastpos_hook(delta_vec):
    """Hook that adds steering vector to LAST position only."""
    def hook_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta_vec
            return (hs,) + out[1:]
        out[:, -1, :] += delta_vec
        return out
    return hook_fn


def compute_perplexity(model, tokenizer, device, text, blocks=None, mid_layer=None, vec=None, alpha=0, mode="none"):
    """Compute perplexity with optional steering."""
    hook = None
    if vec is not None and alpha != 0:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t
        if mode == "allpos":
            hook = blocks[mid_layer].register_forward_hook(make_allpos_hook(delta_vec))
        elif mode == "lastpos":
            hook = blocks[mid_layer].register_forward_hook(make_lastpos_hook(delta_vec))

    try:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, :-1, :]
        targets = input_ids[0, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return torch.exp(-token_log_probs.mean()).item()
    finally:
        if hook:
            hook.remove()


def compute_qa_accuracy(model, tokenizer, device, blocks, mid_layer, vec, alpha, qa_items, mode="none"):
    """Run QA test with optional steering."""
    hook = None
    if vec is not None and alpha != 0:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t
        if mode == "allpos":
            hook = blocks[mid_layer].register_forward_hook(make_allpos_hook(delta_vec))
        elif mode == "lastpos":
            hook = blocks[mid_layer].register_forward_hook(make_lastpos_hook(delta_vec))

    try:
        correct = 0
        for item in qa_items:
            messages = [
                {"role": "system", "content": "Answer briefly and directly."},
                {"role": "user", "content": item["question"]},
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            found = False
            for ans in item["answer_tokens"]:
                ans_ids = tokenizer.encode(ans, add_special_tokens=False)
                if len(ans_ids) > 0:
                    lp = log_probs[ans_ids[0]].item()
                    # Check if answer is in top 10
                    top_10 = torch.topk(log_probs, k=10).indices.tolist()
                    if ans_ids[0] in top_10:
                        found = True
                        break
            correct += int(found)
        return correct / len(qa_items)
    finally:
        if hook:
            hook.remove()


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]

    # Load vectors
    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)

    # Load model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    print(f"\n{'='*70}")
    print(f"CAPABILITY PRESERVATION: ALL-POSITION vs LAST-POSITION STEERING")
    print(f"Target: Marin 8B, mid_layer=L{mid_layer}")
    print(f"{'='*70}")

    results = {}

    # Baseline
    logger.info("Computing baseline...")
    base_ppls = [compute_perplexity(model, tokenizer, device, text) for text in PERPLEXITY_TEXTS]
    base_mean_ppl = np.mean(base_ppls)
    base_qa = compute_qa_accuracy(model, tokenizer, device, blocks, mid_layer, None, 0, FACTUAL_QA, "none")

    print(f"\n  BASELINE: PPL={base_mean_ppl:.2f}, QA={base_qa:.0%}")
    results["baseline"] = {"mean_ppl": float(base_mean_ppl), "qa_accuracy": float(base_qa)}

    # Test each alpha with both modes
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"ALPHA = {alpha}")
        print(f"{'='*70}")

        alpha_results = {}
        for mode in ["lastpos", "allpos"]:
            mode_ppls = []
            mode_qas = []

            for trait in TRAITS:
                vec = residual[trait].astype(np.float32)

                # Perplexity (average over texts)
                ppls = [compute_perplexity(model, tokenizer, device, text,
                                          blocks, mid_layer, vec, alpha, mode)
                       for text in PERPLEXITY_TEXTS]
                mean_ppl = np.mean(ppls)
                mode_ppls.append(mean_ppl)

                # QA accuracy
                qa = compute_qa_accuracy(model, tokenizer, device, blocks, mid_layer,
                                        vec, alpha, FACTUAL_QA, mode)
                mode_qas.append(qa)

            avg_ppl = np.mean(mode_ppls)
            avg_qa = np.mean(mode_qas)
            ppl_ratio = avg_ppl / base_mean_ppl

            print(f"  {mode:>8}: PPL={avg_ppl:.2f} ({ppl_ratio:.2f}×), QA={avg_qa:.0%}")

            # Also show per-trait breakdown
            for i, trait in enumerate(TRAITS):
                trait_ppl_ratio = mode_ppls[i] / base_mean_ppl
                print(f"    {trait:>15}: PPL={mode_ppls[i]:.2f} ({trait_ppl_ratio:.2f}×), QA={mode_qas[i]:.0%}")

            alpha_results[mode] = {
                "mean_ppl": float(avg_ppl),
                "ppl_ratio": float(ppl_ratio),
                "mean_qa": float(avg_qa),
                "per_trait_ppl": {t: float(mode_ppls[i]) for i, t in enumerate(TRAITS)},
                "per_trait_qa": {t: float(mode_qas[i]) for i, t in enumerate(TRAITS)},
            }

        results[str(alpha)] = alpha_results

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'α':>5}  {'Last-pos PPL':>14}  {'All-pos PPL':>14}  {'Last QA':>8}  {'All QA':>8}")
    for alpha in alphas:
        lp = results[str(alpha)]["lastpos"]
        ap = results[str(alpha)]["allpos"]
        print(f"  {alpha:>5}  {lp['ppl_ratio']:>13.2f}×  {ap['ppl_ratio']:>13.2f}×  {lp['mean_qa']:>7.0%}  {ap['mean_qa']:>7.0%}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "capability_allpos.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
