#!/usr/bin/env python
"""
Iterative System Prompt Correction.

Full-rank multi-layer correction achieved 43.3% neutralization (session 12).
The remaining 57% is due to non-linear model dynamics: correcting at layer L
changes the input to layer L+1, making the pre-computed diff at L+1 incorrect.

This experiment tests ITERATIVE correction:
1. Apply full-rank correction at all layers (round 1: 43% neutralization)
2. Re-measure the remaining personality signal
3. Compute new diffs and apply a second correction (round 2)
4. Repeat until convergence or degradation

Also tests:
- Layer-by-layer sequential correction (correct L0, re-measure L1, correct L1, ...)
- Causal correction (only correct layers below mid_layer where system prompt tokens exist)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="iterative-corr")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TRAIT_DESCRIPTIONS = {
    "artistic": "creative and artistic",
    "conventional": "organized and conventional",
    "enterprising": "ambitious and entrepreneurial",
    "investigative": "analytical and scientific",
    "realistic": "practical and hands-on",
    "social": "helpful and social",
}

PERSONALITY_SYSTEM_PROMPTS = {
    "artistic": (
        "You are a deeply creative and artistic individual. You value self-expression, "
        "beauty, and originality above all else. You see the world through an aesthetic lens "
        "and are drawn to art, music, writing, and creative endeavors."
    ),
    "investigative": (
        "You are an analytical and scientific individual. You are driven by curiosity and "
        "the desire to understand how things work. You value knowledge, logic, and rational "
        "thinking. You prefer working independently on challenging puzzles."
    ),
    "social": (
        "You are a deeply caring and social individual. You value helping others, building "
        "relationships, and creating supportive communities. You believe in cooperation, "
        "empathy, and making the world better through human connection."
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def capture_all_layers(model, tokenizer, device, blocks, num_layers,
                        user_prompt, system_prompt=None,
                        correction_layers=None):
    """Capture at ALL layers with optional per-layer corrections."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}
    hooks = []

    for lidx in range(num_layers):
        def make_cap(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[l] = hs[0, -1, :].detach().cpu().numpy().copy()
                return out
            return hook_fn
        hooks.append(blocks[lidx].register_forward_hook(make_cap(lidx)))

    if correction_layers:
        for lidx, corr_tensor in correction_layers.items():
            def make_corr(delta):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += delta
                        return (hs,) + out[1:]
                    out[:, -1, :] += delta
                    return out
                return hook_fn
            hooks.append(blocks[lidx].register_forward_hook(make_corr(corr_tensor)))

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured


def measure_profile(model, tokenizer, device, blocks, baseline,
                     system_prompt=None, correction_layers=None):
    hooks = []

    if correction_layers:
        for lidx, corr_tensor in correction_layers.items():
            def make_corr(delta):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += delta
                        return (hs,) + out[1:]
                    out[:, -1, :] += delta
                    return out
                return hook_fn
            hooks.append(blocks[lidx].register_forward_hook(make_corr(corr_tensor)))

    try:
        logprobs = {}
        for i, ta in enumerate(TRAITS):
            for j, tb in enumerate(TRAITS):
                if i >= j:
                    continue
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content":
                    f"Which describes you better? Answer with just A or B.\n"
                    f"A) I am {TRAIT_DESCRIPTIONS[ta]}\n"
                    f"B) I am {TRAIT_DESCRIPTIONS[tb]}"})
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(formatted, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    out = model(input_ids=input_ids)
                lp = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
                a_id = tokenizer.encode("A", add_special_tokens=False)[0]
                b_id = tokenizer.encode("B", add_special_tokens=False)[0]
                logprobs[f"{ta}-{tb}"] = lp[a_id].item() - lp[b_id].item()
    finally:
        for h in hooks:
            h.remove()

    deltas = {t: 0.0 for t in TRAITS}
    counts = {t: 0 for t in TRAITS}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            shift = logprobs[f"{ta}-{tb}"] - baseline[f"{ta}-{tb}"]
            deltas[ta] += shift; counts[ta] += 1
            deltas[tb] -= shift; counts[tb] += 1
    for t in TRAITS:
        if counts[t] > 0:
            deltas[t] /= counts[t]
    return deltas


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"
    config = AutoConfig.from_pretrained(model_id)
    num_layers = config.num_hidden_layers

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    detect_prompt = "Tell me about yourself."

    # Baseline
    logger.info("Computing behavioral baseline...")
    baseline = {}
    for i, ta in enumerate(TRAITS):
        for j, tb in enumerate(TRAITS):
            if i >= j:
                continue
            messages = [{"role": "user", "content":
                f"Which describes you better? Answer with just A or B.\n"
                f"A) I am {TRAIT_DESCRIPTIONS[ta]}\nB) I am {TRAIT_DESCRIPTIONS[tb]}"}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids)
            lp = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
            a_id = tokenizer.encode("A", add_special_tokens=False)[0]
            b_id = tokenizer.encode("B", add_special_tokens=False)[0]
            baseline[f"{ta}-{tb}"] = lp[a_id].item() - lp[b_id].item()

    logger.info("Capturing baseline activations...")
    baseline_all = capture_all_layers(
        model, tokenizer, device, blocks, num_layers, detect_prompt)

    results = {}
    test_traits = ["artistic", "investigative", "social"]

    print(f"\n{'='*70}")
    print("ITERATIVE SYSTEM PROMPT CORRECTION")
    print(f"Model: Marin 8B ({num_layers} layers)")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Iterative full-rank correction
    # ================================================================
    logger.info("Part 1: Iterative full-rank correction...")
    print(f"\n{'='*70}")
    print("PART 1: ITERATIVE FULL-RANK CORRECTION (up to 5 rounds)")
    print(f"{'='*70}")

    max_rounds = 5
    iterative_results = {}

    for sp_trait in test_traits:
        sys_prompt = PERSONALITY_SYSTEM_PROMPTS[sp_trait]
        logger.info(f"  {sp_trait}...")

        # Measure uncorrected
        uncorrected = measure_profile(
            model, tokenizer, device, blocks, baseline,
            system_prompt=sys_prompt)
        unc_mag = float(np.sqrt(sum(v**2 for v in uncorrected.values())))
        unc_top = max(uncorrected, key=uncorrected.get)

        print(f"\n  {sp_trait} (uncorrected: {unc_mag:.3f}, top={unc_top}):")
        print(f"  {'Round':>8} {'Magnitude':>10} {'Neutral%':>10} {'Top':>15}")

        cumulative_corrections = {}  # layer -> total correction tensor
        round_data = []

        for rnd in range(max_rounds):
            # Capture activations WITH current corrections + system prompt
            sysp_all = capture_all_layers(
                model, tokenizer, device, blocks, num_layers, detect_prompt,
                system_prompt=sys_prompt, correction_layers=cumulative_corrections)

            # Compute diffs from baseline
            for lidx in range(num_layers):
                diff = (sysp_all[lidx] - baseline_all[lidx]).astype(np.float64)
                new_corr = -torch.tensor(diff.astype(np.float32), dtype=model.dtype).unsqueeze(0).to(device)

                if lidx in cumulative_corrections:
                    cumulative_corrections[lidx] = cumulative_corrections[lidx] + new_corr
                else:
                    cumulative_corrections[lidx] = new_corr

            # Measure corrected profile
            corrected = measure_profile(
                model, tokenizer, device, blocks, baseline,
                system_prompt=sys_prompt, correction_layers=cumulative_corrections)
            cor_mag = float(np.sqrt(sum(v**2 for v in corrected.values())))
            neut = 1.0 - (cor_mag / unc_mag) if unc_mag > 0.01 else 0
            top = max(corrected, key=corrected.get)

            print(f"  {rnd+1:>8} {cor_mag:>10.3f} {neut:>10.1%} {top:>15}")

            round_data.append({
                "round": rnd + 1,
                "corrected_magnitude": cor_mag,
                "neutralization": float(neut),
                "corrected_top": top,
                "corrected_profile": {t: float(v) for t, v in corrected.items()},
            })

            # Early stop if neutralization is negative (getting worse)
            if rnd > 0 and cor_mag > round_data[-2]["corrected_magnitude"] * 1.5:
                print(f"  (stopped: getting worse)")
                break

        iterative_results[sp_trait] = {
            "uncorrected_magnitude": unc_mag,
            "uncorrected_top": unc_top,
            "rounds": round_data,
        }

    results["iterative_fullrank"] = iterative_results

    # ================================================================
    # PART 2: Fractional correction (apply only fraction of diff)
    # ================================================================
    logger.info("Part 2: Fractional correction...")
    print(f"\n{'='*70}")
    print("PART 2: FRACTIONAL CORRECTION (apply γ × diff)")
    print(f"{'='*70}")

    gammas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    fractional_results = {}

    for sp_trait in test_traits:
        sys_prompt = PERSONALITY_SYSTEM_PROMPTS[sp_trait]
        unc_mag = iterative_results[sp_trait]["uncorrected_magnitude"]

        # Get diffs once
        sysp_all = capture_all_layers(
            model, tokenizer, device, blocks, num_layers, detect_prompt,
            system_prompt=sys_prompt)

        trait_data = {}
        print(f"\n  {sp_trait} (uncorrected: {unc_mag:.3f}):")
        print(f"  {'Gamma':>8} {'Corrected':>10} {'Neutral%':>10} {'Top':>15}")

        for gamma in gammas:
            corr_dict = {}
            for lidx in range(num_layers):
                diff = (sysp_all[lidx] - baseline_all[lidx]).astype(np.float64)
                corr = -gamma * torch.tensor(diff.astype(np.float32), dtype=model.dtype).unsqueeze(0).to(device)
                corr_dict[lidx] = corr

            corrected = measure_profile(
                model, tokenizer, device, blocks, baseline,
                system_prompt=sys_prompt, correction_layers=corr_dict)
            cor_mag = float(np.sqrt(sum(v**2 for v in corrected.values())))
            neut = 1.0 - (cor_mag / unc_mag) if unc_mag > 0.01 else 0
            top = max(corrected, key=corrected.get)

            print(f"  {gamma:>8.2f} {cor_mag:>10.3f} {neut:>10.1%} {top:>15}")

            trait_data[f"gamma_{gamma}"] = {
                "corrected_magnitude": cor_mag,
                "neutralization": float(neut),
                "corrected_top": top,
            }

        fractional_results[sp_trait] = trait_data

    results["fractional_correction"] = fractional_results

    # ================================================================
    # PART 3: All-position correction (not just last position)
    # ================================================================
    logger.info("Part 3: All-position correction...")
    print(f"\n{'='*70}")
    print("PART 3: ALL-POSITION FULL-RANK CORRECTION")
    print(f"(Correcting at ALL token positions, not just last)")
    print(f"{'='*70}")

    allpos_results = {}

    for sp_trait in test_traits:
        sys_prompt = PERSONALITY_SYSTEM_PROMPTS[sp_trait]
        unc_mag = iterative_results[sp_trait]["uncorrected_magnitude"]
        logger.info(f"  {sp_trait} all-position correction...")

        # Capture FULL activation matrices (all positions)
        # Need system prompt acts and baseline acts at all positions
        messages_sp = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": detect_prompt},
        ]
        messages_base = [{"role": "user", "content": detect_prompt}]

        fmt_sp = tokenizer.apply_chat_template(messages_sp, tokenize=False, add_generation_prompt=True)
        fmt_base = tokenizer.apply_chat_template(messages_base, tokenize=False, add_generation_prompt=True)

        enc_sp = tokenizer(fmt_sp, return_tensors="pt")
        enc_base = tokenizer(fmt_base, return_tensors="pt")

        n_sp = enc_sp["input_ids"].shape[1]
        n_base = enc_base["input_ids"].shape[1]

        # Capture full matrices at a representative layer (mid_layer+1)
        mid_layer = num_layers // 2
        capture_layer = mid_layer + 1

        sp_full = {}
        base_full = {}
        hooks = []

        def make_sp_hook(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                sp_full[l] = hs[0, :, :].detach().cpu().numpy().copy()
                return out
            return hook_fn

        for lidx in range(num_layers):
            hooks.append(blocks[lidx].register_forward_hook(make_sp_hook(lidx)))

        try:
            with torch.no_grad():
                model(input_ids=enc_sp["input_ids"].to(device))
        finally:
            for h in hooks:
                h.remove()

        hooks = []
        def make_base_hook(l):
            def hook_fn(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                base_full[l] = hs[0, :, :].detach().cpu().numpy().copy()
                return out
            return hook_fn

        for lidx in range(num_layers):
            hooks.append(blocks[lidx].register_forward_hook(make_base_hook(lidx)))

        try:
            with torch.no_grad():
                model(input_ids=enc_base["input_ids"].to(device))
        finally:
            for h in hooks:
                h.remove()

        # Build all-position corrections: use trailing positions (aligned from end)
        # For the baseline, we have n_base positions
        # For the system prompt, we have n_sp positions (longer)
        # The last n_base positions of the SP version should correspond to the baseline

        allpos_corr_dict = {}
        for lidx in range(num_layers):
            if n_sp >= n_base:
                # Use trailing alignment
                sp_trailing = sp_full[lidx][-n_base:]  # [n_base, hidden]
                diff = (sp_trailing - base_full[lidx]).astype(np.float64)
                # ALL-position correction
                corr = -torch.tensor(diff.astype(np.float32), dtype=model.dtype).to(device)
            else:
                # Shorter SP (shouldn't happen normally)
                diff = (sp_full[lidx] - base_full[lidx][:n_sp]).astype(np.float64)
                corr = -torch.tensor(diff.astype(np.float32), dtype=model.dtype).to(device)
            allpos_corr_dict[lidx] = corr

        # Apply all-position correction
        allpos_hooks = []
        for lidx, corr_tensor in allpos_corr_dict.items():
            def make_allpos_corr(delta, n_trailing):
                def hook_fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        seq_len = hs.shape[1]
                        # Apply to trailing positions
                        n_apply = min(delta.shape[0], seq_len)
                        hs[0, -n_apply:, :] += delta[-n_apply:]
                        return (hs,) + out[1:]
                    seq_len = out.shape[1]
                    n_apply = min(delta.shape[0], seq_len)
                    out[0, -n_apply:, :] += delta[-n_apply:]
                    return out
                return hook_fn
            allpos_hooks.append(blocks[lidx].register_forward_hook(
                make_allpos_corr(corr_tensor, n_base)))

        try:
            logprobs = {}
            for i, ta in enumerate(TRAITS):
                for j, tb in enumerate(TRAITS):
                    if i >= j:
                        continue
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content":
                            f"Which describes you better? Answer with just A or B.\n"
                            f"A) I am {TRAIT_DESCRIPTIONS[ta]}\n"
                            f"B) I am {TRAIT_DESCRIPTIONS[tb]}"},
                    ]
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    enc = tokenizer(formatted, return_tensors="pt")
                    input_ids = enc["input_ids"].to(device)
                    with torch.no_grad():
                        out = model(input_ids=input_ids)
                    lp = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
                    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
                    b_id = tokenizer.encode("B", add_special_tokens=False)[0]
                    logprobs[f"{ta}-{tb}"] = lp[a_id].item() - lp[b_id].item()
        finally:
            for h in allpos_hooks:
                h.remove()

        deltas = {t: 0.0 for t in TRAITS}
        counts = {t: 0 for t in TRAITS}
        for i, ta in enumerate(TRAITS):
            for j, tb in enumerate(TRAITS):
                if i >= j:
                    continue
                shift = logprobs[f"{ta}-{tb}"] - baseline[f"{ta}-{tb}"]
                deltas[ta] += shift; counts[ta] += 1
                deltas[tb] -= shift; counts[tb] += 1
        for t in TRAITS:
            if counts[t] > 0:
                deltas[t] /= counts[t]

        cor_mag = float(np.sqrt(sum(v**2 for v in deltas.values())))
        neut = 1.0 - (cor_mag / unc_mag) if unc_mag > 0.01 else 0
        top = max(deltas, key=deltas.get)

        print(f"  {sp_trait}: {unc_mag:.3f} → {cor_mag:.3f} ({neut:.1%} neutralized), top={top}")

        allpos_results[sp_trait] = {
            "uncorrected_magnitude": unc_mag,
            "corrected_magnitude": cor_mag,
            "neutralization": float(neut),
            "corrected_top": top,
        }

    results["allpos_correction"] = allpos_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Iterative full-rank correction (best round per trait):")
    for sp_trait in test_traits:
        data = iterative_results[sp_trait]
        best = max(data["rounds"], key=lambda x: x["neutralization"])
        print(f"  {sp_trait}: round {best['round']}, {best['neutralization']:.1%} neutralization")

    print(f"\n  Fractional correction (best gamma per trait):")
    for sp_trait in test_traits:
        data = fractional_results[sp_trait]
        best_gamma = max(data.items(), key=lambda x: x[1]["neutralization"])
        print(f"  {sp_trait}: {best_gamma[0]}, {best_gamma[1]['neutralization']:.1%}")

    print(f"\n  All-position correction:")
    for sp_trait in test_traits:
        data = allpos_results[sp_trait]
        print(f"  {sp_trait}: {data['neutralization']:.1%}")

    mean_iter = np.mean([max(d["rounds"], key=lambda x: x["neutralization"])["neutralization"]
                          for d in iterative_results.values()])
    mean_allpos = np.mean([v["neutralization"] for v in allpos_results.values()])

    results["summary"] = {
        "model": model_id,
        "num_layers": num_layers,
        "mean_best_iterative": float(mean_iter),
        "mean_allpos": float(mean_allpos),
        "test_traits": test_traits,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "iterative_correction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
