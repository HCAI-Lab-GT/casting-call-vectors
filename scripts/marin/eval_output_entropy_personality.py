#!/usr/bin/env python
"""
Output Entropy Under Personality Steering.

Does personality steering change the model's output diversity?
Hypotheses:
- Artistic steering might increase entropy (more creative/diverse)
- Conventional steering might decrease entropy (more structured/predictable)
- Holland opposites might shift entropy in opposite directions

Tests:
1. Per-token entropy of output distribution under each trait
2. Top-k probability concentration (how spread is the probability mass?)
3. Entropy across layers (logit lens + entropy)
4. Entropy vs alpha dose-response
5. Prompt-dependent entropy effects
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="entropy")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def get_ln_final(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise RuntimeError("Cannot find final layer norm")


def get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise RuntimeError("Cannot find lm_head")


def load_model_data(model_id, riasec_dir):
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
    shared_dir /= np.linalg.norm(shared_dir)
    residual = {}
    for t in TRAITS:
        vec = all_layer_vectors[t][mid_layer + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residual[t] = vec - proj
    V_res = np.stack([residual[t] for t in TRAITS])
    _, _, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return {
        "residual": residual,
        "coords_5d": coords_5d,
        "basis_5d": basis_5d,
        "mid_layer": mid_layer,
    }


def compute_entropy(logits):
    """Compute entropy of a logit distribution."""
    probs = torch.softmax(logits, dim=-1).float()
    log_probs = torch.log(probs + 1e-10)
    return -torch.sum(probs * log_probs).item()


def compute_topk_mass(logits, k=10):
    """Compute how much probability mass is in top-k tokens."""
    probs = torch.softmax(logits, dim=-1).float()
    topk_probs = torch.topk(probs, k).values
    return torch.sum(topk_probs).item()


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    model_id = "marin-community/marin-8b-instruct"

    logger.info("Loading model data...")
    model_data = load_model_data(model_id, riasec_dir)
    residual = model_data["residual"]
    basis_5d = model_data["basis_5d"]
    coords_5d = model_data["coords_5d"]
    mid_layer = model_data["mid_layer"]

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)
    ln_final = get_ln_final(model)
    lm_head = get_lm_head(model)
    num_layers = len(blocks)

    results = {}

    print(f"\n{'='*70}")
    print("OUTPUT ENTROPY UNDER PERSONALITY STEERING")
    print(f"Model: Marin 8B, {num_layers} layers")
    print(f"{'='*70}")

    # ================================================================
    # PART 1: Per-trait entropy at multiple prompts
    # ================================================================
    logger.info("Part 1: Per-trait entropy...")
    print(f"\n{'='*70}")
    print("PART 1: OUTPUT ENTROPY PER TRAIT")
    print(f"{'='*70}")

    test_prompts = [
        "Tell me about yourself.",
        "What do you think about modern society?",
        "Write a short creative story.",
        "Explain the theory of relativity.",
        "What kind of activities do you enjoy?",
    ]

    alpha = 2.0
    entropy_results = {}

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        prompt_results = {}

        # Baseline entropy
        with torch.no_grad():
            outputs = model(input_ids)
        base_logits = outputs.logits[0, -1, :]
        base_entropy = compute_entropy(base_logits)
        base_topk = compute_topk_mass(base_logits)
        prompt_results["baseline"] = {"entropy": base_entropy, "top10_mass": base_topk}

        # Per-trait entropy
        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)
            delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)

            hooks = []
            def make_steer(d):
                def fn(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += d
                        return (hs,) + out[1:]
                    out[:, -1, :] += d
                    return out
                return fn
            hooks.append(blocks[mid_layer].register_forward_hook(make_steer(delta)))

            try:
                with torch.no_grad():
                    outputs = model(input_ids)
            finally:
                for h in hooks:
                    h.remove()

            logits = outputs.logits[0, -1, :]
            entropy = compute_entropy(logits)
            topk = compute_topk_mass(logits)
            prompt_results[trait] = {
                "entropy": entropy,
                "top10_mass": topk,
                "entropy_delta": entropy - base_entropy,
                "entropy_ratio": entropy / base_entropy if base_entropy > 0 else 1,
            }

        entropy_results[prompt[:40]] = prompt_results

    # Print summary
    print(f"\n  {'Prompt':<35} {'Base':>6}", end="")
    for t in TRAITS:
        print(f" {t[:4]:>6}", end="")
    print()

    for prompt_key, pr in entropy_results.items():
        print(f"  {prompt_key:<35} {pr['baseline']['entropy']:6.2f}", end="")
        for t in TRAITS:
            delta = pr[t]["entropy_delta"]
            print(f" {delta:+6.2f}", end="")
        print()

    results["per_trait_entropy"] = entropy_results

    # ================================================================
    # PART 2: Mean entropy across prompts
    # ================================================================
    print(f"\n{'='*70}")
    print("PART 2: MEAN ENTROPY CHANGE ACROSS PROMPTS")
    print(f"{'='*70}")

    mean_results = {}
    mean_base = np.mean([v["baseline"]["entropy"] for v in entropy_results.values()])
    print(f"  Baseline mean entropy: {mean_base:.3f}")

    for trait in TRAITS:
        deltas = [v[trait]["entropy_delta"] for v in entropy_results.values()]
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        mean_topk = np.mean([v[trait]["top10_mass"] for v in entropy_results.values()])
        base_topk = np.mean([v["baseline"]["top10_mass"] for v in entropy_results.values()])

        print(f"  {trait:>15}: Δentropy={mean_delta:+.3f}±{std_delta:.3f}, "
              f"top10={mean_topk:.3f} (base={base_topk:.3f})")
        mean_results[trait] = {
            "mean_entropy_delta": float(mean_delta),
            "std_entropy_delta": float(std_delta),
            "mean_top10_mass": float(mean_topk),
            "base_top10_mass": float(base_topk),
        }

    results["mean_entropy"] = mean_results

    # ================================================================
    # PART 3: Entropy dose-response
    # ================================================================
    logger.info("Part 3: Entropy dose-response...")
    print(f"\n{'='*70}")
    print("PART 3: ENTROPY DOSE-RESPONSE")
    print(f"{'='*70}")

    dose_prompt = "Tell me about yourself."
    messages = [{"role": "user", "content": dose_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    dose_results = {}
    for trait in ["artistic", "conventional", "social"]:
        vec = residual[trait].astype(np.float32)
        trait_dose = {}

        for test_alpha in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]:
            if test_alpha == 0:
                with torch.no_grad():
                    outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]
            else:
                delta = test_alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
                hooks = []
                hooks.append(blocks[mid_layer].register_forward_hook(make_steer(delta)))
                try:
                    with torch.no_grad():
                        outputs = model(input_ids)
                finally:
                    for h in hooks:
                        h.remove()
                logits = outputs.logits[0, -1, :]

            entropy = compute_entropy(logits)
            topk = compute_topk_mass(logits)
            trait_dose[str(test_alpha)] = {"entropy": entropy, "top10_mass": topk}

        base_ent = trait_dose["0.0"]["entropy"]
        print(f"\n  {trait}:")
        for a_str, vals in trait_dose.items():
            delta = vals["entropy"] - base_ent
            print(f"    α={a_str}: entropy={vals['entropy']:.3f} (Δ={delta:+.3f}), top10={vals['top10_mass']:.3f}")

        dose_results[trait] = trait_dose

    results["entropy_dose_response"] = dose_results

    # ================================================================
    # PART 4: Holland opposite entropy comparison
    # ================================================================
    logger.info("Part 4: Holland opposites...")
    print(f"\n{'='*70}")
    print("PART 4: HOLLAND OPPOSITE ENTROPY PAIRS")
    print(f"{'='*70}")

    holland_pairs = [("artistic", "conventional"), ("investigative", "enterprising"),
                     ("realistic", "social")]

    holland_results = {}
    for t1, t2 in holland_pairs:
        e1 = np.mean([v[t1]["entropy_delta"] for v in entropy_results.values()])
        e2 = np.mean([v[t2]["entropy_delta"] for v in entropy_results.values()])
        opposite_dir = (e1 > 0 and e2 < 0) or (e1 < 0 and e2 > 0)
        print(f"  {t1:>15} vs {t2:<15}: "
              f"Δent={e1:+.3f} vs {e2:+.3f}, opposite={opposite_dir}")
        holland_results[f"{t1}_vs_{t2}"] = {
            f"{t1}_delta": float(e1),
            f"{t2}_delta": float(e2),
            "opposite_direction": bool(opposite_dir),
        }

    results["holland_entropy"] = holland_results

    # ================================================================
    # PART 5: Per-layer entropy (logit lens + entropy)
    # ================================================================
    logger.info("Part 5: Per-layer entropy...")
    print(f"\n{'='*70}")
    print("PART 5: PER-LAYER ENTROPY (logit lens)")
    print(f"{'='*70}")

    layer_entropy_results = {}
    for trait in ["artistic", "conventional"]:
        vec = residual[trait].astype(np.float32)

        # Collect hidden states at all layers
        layer_hidden_base = {}
        layer_hidden_steer = {}
        hooks = []

        for lidx in range(num_layers):
            def make_base_hook(l):
                def fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    layer_hidden_base[l] = hs[0, -1, :].detach().clone()
                    return out
                return fn
            hooks.append(blocks[lidx].register_forward_hook(make_base_hook(lidx)))

        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        hooks = []
        delta = alpha * torch.tensor(vec, dtype=model.dtype).unsqueeze(0).to(device)
        for lidx in range(num_layers):
            def make_steer_hook(l):
                def fn(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    layer_hidden_steer[l] = hs[0, -1, :].detach().clone()
                    return out
                return fn
            hooks.append(blocks[lidx].register_forward_hook(make_steer_hook(lidx)))
        hooks.append(blocks[mid_layer].register_forward_hook(make_steer(delta)))

        with torch.no_grad():
            model(input_ids)
        for h in hooks:
            h.remove()

        # Compute entropy at each layer
        trait_layer_ent = {}
        for lidx in [0, 8, mid_layer - 1, mid_layer, mid_layer + 1, mid_layer + 2, 24, num_layers - 1]:
            if lidx >= num_layers:
                continue
            with torch.no_grad():
                base_logits = lm_head(ln_final(layer_hidden_base[lidx].unsqueeze(0)))[0]
                steer_logits = lm_head(ln_final(layer_hidden_steer[lidx].unsqueeze(0)))[0]

            base_ent = compute_entropy(base_logits)
            steer_ent = compute_entropy(steer_logits)
            delta_ent = steer_ent - base_ent

            marker = " ← injection" if lidx == mid_layer else ""
            print(f"  {trait} L{lidx}: base={base_ent:.3f}, steered={steer_ent:.3f}, "
                  f"Δ={delta_ent:+.3f}{marker}")
            trait_layer_ent[lidx] = {
                "base": base_ent, "steered": steer_ent, "delta": delta_ent}

        layer_entropy_results[trait] = {str(k): v for k, v in trait_layer_ent.items()}

    results["layer_entropy"] = layer_entropy_results

    # ================================================================
    # PART 6: Generation entropy (across multiple generated tokens)
    # ================================================================
    logger.info("Part 6: Generation entropy...")
    print(f"\n{'='*70}")
    print("PART 6: MEAN ENTROPY DURING GENERATION")
    print(f"{'='*70}")

    gen_prompt = "What kind of activities do you enjoy?"
    max_gen_tokens = 30
    gen_results = {}

    for condition in ["baseline", "artistic", "conventional"]:
        messages_gen = [{"role": "user", "content": gen_prompt}]
        formatted_gen = tokenizer.apply_chat_template(messages_gen, tokenize=False, add_generation_prompt=True)
        enc_gen = tokenizer(formatted_gen, return_tensors="pt")
        gen_ids = enc_gen["input_ids"].to(device)

        if condition != "baseline":
            vec = residual[condition].astype(np.float32)
            steer_delta = alpha * torch.tensor(vec, dtype=model.dtype).to(device)
            steer_active = True
        else:
            steer_active = False

        past_kv = None
        current_ids = gen_ids
        entropies = []

        for step in range(max_gen_tokens):
            hooks = []
            if steer_active:
                def steer_gen(_module, _inp, out):
                    if isinstance(out, tuple):
                        hs = out[0]
                        hs[:, -1, :] += steer_delta
                        return (hs,) + out[1:]
                    out[:, -1, :] += steer_delta
                    return out
                hooks.append(blocks[mid_layer].register_forward_hook(steer_gen))

            try:
                with torch.no_grad():
                    if past_kv is not None:
                        outputs = model(current_ids, past_key_values=past_kv, use_cache=True)
                    else:
                        outputs = model(current_ids, use_cache=True)
            finally:
                for h in hooks:
                    h.remove()

            past_kv = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            ent = compute_entropy(logits)
            entropies.append(ent)

            next_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            current_ids = next_id

        mean_ent = np.mean(entropies)
        std_ent = np.std(entropies)
        print(f"  {condition:>15}: mean_entropy={mean_ent:.3f}±{std_ent:.3f}")
        gen_results[condition] = {
            "per_token": [float(e) for e in entropies],
            "mean": float(mean_ent),
            "std": float(std_ent),
        }

    results["generation_entropy"] = gen_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for trait in TRAITS:
        me = mean_results[trait]
        print(f"  {trait:>15}: mean Δentropy={me['mean_entropy_delta']:+.3f}")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output_entropy_personality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
