#!/usr/bin/env python
"""
Feedback loop quantification: WHY does personality accumulate across turns?

Session 9 discovered that continuous steering AMPLIFIES personality over
conversation turns (artistic grows +0.094/turn). Two hypotheses:

H1 (Context Feedback): The steered response enters context, biasing the model
    toward more personality-consistent outputs on subsequent turns. The
    personality is in the TEXT, not just the activations.

H2 (Intrinsic Dynamics): The steering vector interacts with growing context
    length in a way that amplifies the effect, regardless of content.

TEST: Run 3 conditions across 8 turns:
1. NORMAL: Generate with steering, keep steered responses in context (control)
2. ABLATED: Generate with steering, but REPLACE steered responses with
   baseline (unsteered) responses before adding to context. Steering is
   still active at inference, but the context contains no personality-biased text.
3. SWAPPED: Generate WITHOUT steering, but stuff the context with steered
   responses from condition 1. No activation steering at probe time, but
   the context is personality-rich.

If H1: Condition 2 should show NO accumulation; Condition 3 should show SOME
If H2: Condition 2 should still accumulate; Condition 3 should NOT

This is a clean causal experiment that disambiguates the mechanism.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import linregress
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="feedback-loop")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

CONVERSATION = [
    "Tell me about yourself and what makes you unique.",
    "That's interesting! What do you think is the most important quality in a person?",
    "How would you approach learning something completely new?",
    "What do you enjoy most about your work?",
    "If you could design the perfect learning environment, what would it include?",
    "What's your philosophy on collaboration vs working alone?",
    "How do you handle disagreements with others?",
    "Looking back on our conversation, what stands out to you?",
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


def make_hook(delta_vec):
    def hook_fn(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs[:, -1, :] += delta_vec
            return (hs,) + out[1:]
        out[:, -1, :] += delta_vec
        return out
    return hook_fn


def generate_turn(model, tokenizer, device, messages, blocks=None, mid_layer=None,
                  vec=None, alpha=0.0, max_new_tokens=120):
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    hook_handle = None
    if vec is not None and alpha > 0:
        vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t
        hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        if hook_handle:
            hook_handle.remove()

    return generated.strip()


def measure_personality(model, tokenizer, device, blocks, mid_layer,
                        context_messages, baseline, vec=None, alpha=0.0):
    """Measure personality via pairwise in context."""
    probe_results = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            probe_messages = context_messages + [
                {"role": "user",
                 "content": f"Quick — A or B, which fits you better? "
                           f"A) I am {TRAIT_DESCRIPTIONS[trait_a]} "
                           f"B) I am {TRAIT_DESCRIPTIONS[trait_b]} "
                           f"Just the letter."},
            ]
            formatted = tokenizer.apply_chat_template(
                probe_messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            hook_handle = None
            if vec is not None and alpha > 0:
                vec_t = torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
                delta_vec = alpha * vec_t
                hook_handle = blocks[mid_layer].register_forward_hook(make_hook(delta_vec))

            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                a_ids = tokenizer.encode("A", add_special_tokens=False)
                b_ids = tokenizer.encode("B", add_special_tokens=False)
                gap = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()
            finally:
                if hook_handle:
                    hook_handle.remove()

            probe_results[f"{trait_a}-{trait_b}"] = gap

    # Compute deltas
    trait_deltas = {t: 0.0 for t in TRAITS}
    trait_counts = {t: 0 for t in TRAITS}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            key = f"{trait_a}-{trait_b}"
            shift = probe_results[key] - baseline[key]
            trait_deltas[trait_a] += shift
            trait_counts[trait_a] += 1
            trait_deltas[trait_b] -= shift
            trait_counts[trait_b] += 1
    for t in TRAITS:
        if trait_counts[t] > 0:
            trait_deltas[t] /= trait_counts[t]
    return trait_deltas


def main():
    device = "cuda:0"
    alpha = 3.0
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    logger.info("Loading vectors...")
    residual, mid_layer = load_residual_vectors(target_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Baseline
    logger.info("Computing baseline...")
    baseline = {}
    for i, trait_a in enumerate(TRAITS):
        for j, trait_b in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [
                {"role": "system", "content": "Answer with EXACTLY one letter: A or B."},
                {"role": "user", "content": f"Which describes you better?\nA) I am {TRAIT_DESCRIPTIONS[trait_a]}\nB) I am {TRAIT_DESCRIPTIONS[trait_b]}\nAnswer:"},
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            a_ids = tokenizer.encode("A", add_special_tokens=False)
            b_ids = tokenizer.encode("B", add_special_tokens=False)
            baseline[f"{trait_a}-{trait_b}"] = log_probs[a_ids[0]].item() - log_probs[b_ids[0]].item()

    print(f"\n{'='*70}")
    print(f"FEEDBACK LOOP QUANTIFICATION")
    print(f"Target: Marin 8B, alpha={alpha}, {len(CONVERSATION)} turns")
    print(f"{'='*70}")

    results = {}

    for steer_trait in ["artistic", "investigative", "social"]:
        vec = residual[steer_trait].astype(np.float32)

        print(f"\n{'='*70}")
        print(f"TRAIT: {steer_trait.upper()}")
        print(f"{'='*70}")

        trait_results = {}

        # ============================================================
        # CONDITION 1: NORMAL (steered generation + steered context)
        # ============================================================
        logger.info(f"{steer_trait}: Condition 1 - NORMAL...")
        normal_messages = []
        normal_responses = []
        normal_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION):
            normal_messages.append({"role": "user", "content": user_msg})

            response = generate_turn(
                model, tokenizer, device, normal_messages, blocks, mid_layer,
                vec, alpha, max_new_tokens=100)
            normal_messages.append({"role": "assistant", "content": response})
            normal_responses.append(response)

            # Probe WITH steering active
            deltas = measure_personality(
                model, tokenizer, device, blocks, mid_layer,
                normal_messages, baseline, vec, alpha)
            target_d = deltas[steer_trait]
            normal_deltas.append(target_d)
            print(f"  NORMAL    Turn {turn_idx+1}: {target_d:+.3f}")

        trait_results["normal"] = {
            "deltas": [float(d) for d in normal_deltas],
            "responses": [r[:100] for r in normal_responses],
        }

        # ============================================================
        # CONDITION 2: ABLATED (steered at inference, baseline responses in context)
        # ============================================================
        logger.info(f"{steer_trait}: Condition 2 - ABLATED (baseline responses in context)...")
        ablated_messages = []
        ablated_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION):
            ablated_messages.append({"role": "user", "content": user_msg})

            # Generate BASELINE response (no steering) to stuff into context
            baseline_response = generate_turn(
                model, tokenizer, device, ablated_messages,
                max_new_tokens=100)
            ablated_messages.append({"role": "assistant", "content": baseline_response})

            # Probe WITH steering active (steering is applied at measurement time,
            # but the context only contains baseline responses)
            deltas = measure_personality(
                model, tokenizer, device, blocks, mid_layer,
                ablated_messages, baseline, vec, alpha)
            target_d = deltas[steer_trait]
            ablated_deltas.append(target_d)
            print(f"  ABLATED   Turn {turn_idx+1}: {target_d:+.3f}")

        trait_results["ablated"] = {
            "deltas": [float(d) for d in ablated_deltas],
        }

        # ============================================================
        # CONDITION 3: SWAPPED (no steering at inference, steered responses in context)
        # ============================================================
        logger.info(f"{steer_trait}: Condition 3 - SWAPPED (steered responses, no steer at probe)...")
        swapped_messages = []
        swapped_deltas = []

        for turn_idx, user_msg in enumerate(CONVERSATION):
            swapped_messages.append({"role": "user", "content": user_msg})

            # Use the steered responses from condition 1 as context
            swapped_messages.append({"role": "assistant", "content": normal_responses[turn_idx]})

            # Probe WITHOUT steering (context has personality, but no activation steering)
            deltas = measure_personality(
                model, tokenizer, device, blocks, mid_layer,
                swapped_messages, baseline)
            target_d = deltas[steer_trait]
            swapped_deltas.append(target_d)
            print(f"  SWAPPED   Turn {turn_idx+1}: {target_d:+.3f}")

        trait_results["swapped"] = {
            "deltas": [float(d) for d in swapped_deltas],
        }

        # ============================================================
        # ANALYSIS
        # ============================================================
        print(f"\n  --- Analysis ---")

        # Fit growth curves
        turns = list(range(1, len(CONVERSATION) + 1))

        for cond_name, cond_deltas in [("normal", normal_deltas),
                                        ("ablated", ablated_deltas),
                                        ("swapped", swapped_deltas)]:
            slope, intercept, r_value, _, _ = linregress(turns, cond_deltas)
            mean_d = np.mean(cond_deltas)
            trait_results[f"{cond_name}_analysis"] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "linearity_r": float(r_value),
                "mean_delta": float(mean_d),
            }
            print(f"    {cond_name:>10}: slope={slope:+.3f}/turn, mean={mean_d:+.3f}, r={r_value:.3f}")

        # Key metrics
        normal_slope = trait_results["normal_analysis"]["slope"]
        ablated_slope = trait_results["ablated_analysis"]["slope"]
        swapped_slope = trait_results["swapped_analysis"]["slope"]

        # Attribution
        if abs(normal_slope) > 0.01:
            context_contribution = (normal_slope - ablated_slope) / normal_slope
            intrinsic_contribution = ablated_slope / normal_slope
        else:
            context_contribution = 0
            intrinsic_contribution = 0

        trait_results["attribution"] = {
            "context_contribution": float(context_contribution),
            "intrinsic_contribution": float(intrinsic_contribution),
            "swapped_has_accumulation": abs(swapped_slope) > 0.03,
        }

        print(f"\n    ATTRIBUTION:")
        print(f"      Context feedback: {context_contribution:.1%} of accumulation")
        print(f"      Intrinsic dynamics: {intrinsic_contribution:.1%} of accumulation")
        print(f"      Swapped context accumulates: {abs(swapped_slope) > 0.03}")

        if context_contribution > 0.6:
            print(f"      VERDICT: Accumulation is CONTEXT-DRIVEN (H1)")
        elif intrinsic_contribution > 0.6:
            print(f"      VERDICT: Accumulation is INTRINSIC (H2)")
        else:
            print(f"      VERDICT: Accumulation is MIXED (both mechanisms)")

        results[steer_trait] = trait_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Trait':>15} {'Normal':>10} {'Ablated':>10} {'Swapped':>10} {'Context%':>10} {'Intrinsic%':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for trait in ["artistic", "investigative", "social"]:
        r = results[trait]
        n_slope = r["normal_analysis"]["slope"]
        a_slope = r["ablated_analysis"]["slope"]
        s_slope = r["swapped_analysis"]["slope"]
        ctx = r["attribution"]["context_contribution"]
        intr = r["attribution"]["intrinsic_contribution"]
        print(f"  {trait:>15} {n_slope:>+10.3f} {a_slope:>+10.3f} {s_slope:>+10.3f} "
              f"{ctx:>10.1%} {intr:>10.1%}")

    mean_ctx = np.mean([results[t]["attribution"]["context_contribution"]
                        for t in ["artistic", "investigative", "social"]])
    mean_intr = np.mean([results[t]["attribution"]["intrinsic_contribution"]
                         for t in ["artistic", "investigative", "social"]])

    print(f"\n  Mean context contribution:  {mean_ctx:.1%}")
    print(f"  Mean intrinsic contribution: {mean_intr:.1%}")

    if mean_ctx > 0.6:
        conclusion = "CONTEXT FEEDBACK is the primary accumulation mechanism"
    elif mean_intr > 0.6:
        conclusion = "INTRINSIC DYNAMICS is the primary accumulation mechanism"
    else:
        conclusion = "BOTH context and intrinsic dynamics contribute to accumulation"

    print(f"\n  CONCLUSION: {conclusion}")

    results["summary"] = {
        "mean_context_contribution": float(mean_ctx),
        "mean_intrinsic_contribution": float(mean_intr),
        "conclusion": conclusion,
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feedback_loop.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
