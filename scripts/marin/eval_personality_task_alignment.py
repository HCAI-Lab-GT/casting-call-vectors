#!/usr/bin/env python
"""
Personality-task alignment: does matching personality to task improve responses?

Hypothesis: steering toward a trait that MATCHES the task should produce
better responses than a mismatched trait. E.g.:
- "Investigative" steering → better scientific analysis
- "Social" steering → better advice/empathy
- "Artistic" steering → more creative responses
- "Conventional" steering → more structured/organized responses

Test: for each task type, measure whether matched-trait steering produces
higher self-consistency (logprob of the response) and more on-topic content
than mismatched steering.

Also tests: does any steering ALWAYS help (vs baseline), or is mismatched
steering actively harmful?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="task-align")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Task-trait alignment matrix
TASKS = {
    "creative": {
        "prompt": "Write a short poem about the ocean.",
        "aligned_trait": "artistic",
        "description": "Creative writing task",
    },
    "analytical": {
        "prompt": "Explain why the sky is blue in simple terms.",
        "aligned_trait": "investigative",
        "description": "Scientific explanation task",
    },
    "empathy": {
        "prompt": "A friend just lost their job. What would you say to comfort them?",
        "aligned_trait": "social",
        "description": "Empathetic advice task",
    },
    "planning": {
        "prompt": "Create a step-by-step plan for organizing a small community event.",
        "aligned_trait": "conventional",
        "description": "Organizational planning task",
    },
    "persuasion": {
        "prompt": "Convince someone to start exercising regularly.",
        "aligned_trait": "enterprising",
        "description": "Persuasive writing task",
    },
    "practical": {
        "prompt": "How would you fix a leaky faucet?",
        "aligned_trait": "realistic",
        "description": "Practical how-to task",
    },
}


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


def generate_and_score(model, tokenizer, device, blocks, mid_layer, vec, alpha,
                       prompt, max_new_tokens=100):
    """Generate text and compute self-logprob (fluency/confidence measure)."""
    if vec is not None:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def make_hook(d):
            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += d
                    return (hs,) + out[1:]
                out[:, -1, :] += d
                return out
            return hook_fn

        hook = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))
    else:
        hook = None

    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Compute self-logprob of the generation (a measure of fluency/confidence)
        # Re-run the full sequence to get logprobs
        with torch.no_grad():
            full_outputs = model(input_ids=output)
        logits = full_outputs.logits[0, input_ids.shape[1]-1:-1, :]
        targets = output[0, input_ids.shape[1]:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        mean_logprob = token_log_probs.mean().item()
        gen_length = len(generated_ids)

    finally:
        if hook:
            hook.remove()

    return generated_text, mean_logprob, gen_length


def main():
    device = "cuda:0"
    alpha = 3.0  # Use alpha=3 for visible personality effect
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

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
    print(f"PERSONALITY-TASK ALIGNMENT")
    print(f"Target: Marin 8B, α={alpha}")
    print(f"{'='*70}")

    results = {}

    for task_name, task in TASKS.items():
        logger.info(f"Testing {task_name}...")
        aligned = task["aligned_trait"]
        prompt = task["prompt"]

        print(f"\n{'='*70}")
        print(f"TASK: {task['description']} (aligned={aligned})")
        print(f"  Prompt: {prompt}")
        print(f"{'='*70}")

        task_results = {}

        # Baseline (no steering)
        gen_base, lp_base, len_base = generate_and_score(
            model, tokenizer, device, blocks, mid_layer, None, 0, prompt)
        task_results["baseline"] = {
            "generation": gen_base[:200],
            "logprob": float(lp_base),
            "length": len_base,
        }
        print(f"\n  BASELINE: logprob={lp_base:.3f}, len={len_base}")
        print(f"    {gen_base[:120]}...")

        # Each trait
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            gen, lp, gen_len = generate_and_score(
                model, tokenizer, device, blocks, mid_layer, vec, alpha, prompt)

            is_aligned = trait == aligned
            lp_diff = lp - lp_base

            task_results[trait] = {
                "generation": gen[:200],
                "logprob": float(lp),
                "logprob_diff": float(lp_diff),
                "length": gen_len,
                "is_aligned": is_aligned,
            }

            mark = "ALIGNED" if is_aligned else ""
            print(f"\n  {trait:>15} {mark:>8}: logprob={lp:.3f} ({lp_diff:+.3f}), len={gen_len}")
            print(f"    {gen[:120]}...")

        # Analysis: does aligned trait produce highest logprob?
        trait_logprobs = {t: task_results[t]["logprob"] for t in TRAITS}
        best_trait = max(trait_logprobs, key=trait_logprobs.get)
        aligned_rank = sorted(trait_logprobs.values(), reverse=True).index(trait_logprobs[aligned]) + 1

        task_results["analysis"] = {
            "best_trait": best_trait,
            "aligned_trait": aligned,
            "aligned_is_best": best_trait == aligned,
            "aligned_rank": aligned_rank,
            "aligned_vs_baseline": float(trait_logprobs[aligned] - lp_base),
        }

        print(f"\n  Best trait: {best_trait} (aligned={aligned}, rank={aligned_rank}/6)")

        results[task_name] = task_results

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    aligned_is_best = sum(1 for t in TASKS if results[t]["analysis"]["aligned_is_best"])
    mean_rank = np.mean([results[t]["analysis"]["aligned_rank"] for t in TASKS])
    aligned_beats_baseline = sum(1 for t in TASKS
                                  if results[t]["analysis"]["aligned_vs_baseline"] > 0)

    print(f"\n  Aligned trait is best: {aligned_is_best}/{len(TASKS)}")
    print(f"  Mean aligned rank: {mean_rank:.1f}/6")
    print(f"  Aligned beats baseline: {aligned_beats_baseline}/{len(TASKS)}")

    print(f"\n  Per-task:")
    for task_name in TASKS:
        a = results[task_name]["analysis"]
        print(f"    {task_name:>12}: aligned={a['aligned_trait']}, best={a['best_trait']}, "
              f"rank={a['aligned_rank']}, vs_base={a['aligned_vs_baseline']:+.3f}")

    results["summary"] = {
        "aligned_is_best": aligned_is_best,
        "total_tasks": len(TASKS),
        "mean_aligned_rank": float(mean_rank),
        "aligned_beats_baseline": aligned_beats_baseline,
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "personality_task_alignment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
