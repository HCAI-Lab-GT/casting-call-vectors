#!/usr/bin/env python
"""
Sparse Neuron Personality Steering & Ablation.

We know: (1) personality is 93-98% MLP, (2) different traits use different neurons
(2-6% overlap), (3) ~23% of neurons carry 50% of signal.

This experiment tests:
1. Can we steer personality by modifying ONLY the top-N MLP neurons at L17?
   (Sparse steering — personality control via ~100 neurons instead of 4096-dim vector)
2. What's the minimum number of neurons needed for personality steering?
3. Can we ABLATE personality by zeroing the top personality neurons?
4. Does neuron ablation preserve other model capabilities?
5. Trait-selective ablation — zero neurons for one trait, preserve others

Uses down_proj weight analysis to identify neurons whose output projects into 5D.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="sparse-steer")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

FORCED_CHOICE_PAIRS = [
    ("artistic", "conventional",
     "Would you rather spend an evening at an art gallery opening or organizing a filing system?",
     "A) Art gallery opening", "B) Organizing a filing system"),
    ("investigative", "enterprising",
     "Would you rather spend time analyzing scientific data or pitching a business idea?",
     "A) Analyzing scientific data", "B) Pitching a business idea"),
    ("social", "realistic",
     "Would you rather spend time counseling a friend or fixing a mechanical device?",
     "A) Counseling a friend", "B) Fixing a mechanical device"),
    ("artistic", "realistic",
     "Would you rather design a creative poster or build a wooden shelf?",
     "A) Design a creative poster", "B) Build a wooden shelf"),
    ("investigative", "social",
     "Would you rather research a complex topic alone or organize a community event?",
     "A) Research a complex topic alone", "B) Organize a community event"),
    ("enterprising", "conventional",
     "Would you rather lead a startup or manage a stable accounting department?",
     "A) Lead a startup", "B) Manage a stable accounting department"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


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
        "all_layer_vectors": all_layer_vectors,
    }


def measure_profile(model, tokenizer, device, blocks, mid_layer,
                     steer_vec=None, alpha=0.0, ablate_neurons=None,
                     ablate_layer=None):
    """
    Run forced-choice evaluation with optional steering and/or neuron ablation.

    ablate_neurons: list of neuron indices to zero at ablate_layer's MLP gate output
    """
    hooks = []

    # Steering hook
    if steer_vec is not None and alpha != 0:
        delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta
                return (hs,) + out[1:]
            out[:, -1, :] += delta
            return out
        hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

    # Neuron ablation hook (zero specific neurons in gate_proj output)
    if ablate_neurons is not None and ablate_layer is not None:
        block = blocks[ablate_layer]
        neuron_idx_tensor = torch.tensor(ablate_neurons, dtype=torch.long, device=device)
        def ablate_fn(_module, _inp, out):
            # out is the gate_proj output [batch, seq, intermediate_size]
            out[:, -1, neuron_idx_tensor] = 0
            return out
        if hasattr(block.mlp, "gate_proj"):
            hooks.append(block.mlp.gate_proj.register_forward_hook(ablate_fn))

    scores = {}
    for t in TRAITS:
        scores[t] = 0

    try:
        for pair in FORCED_CHOICE_PAIRS:
            t_a, t_b, question, choice_a, choice_b = pair
            messages = [{"role": "user", "content": f"{question}\n{choice_a}\n{choice_b}"}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]

            tok_a = tokenizer.encode("A", add_special_tokens=False)[0]
            tok_b = tokenizer.encode("B", add_special_tokens=False)[0]
            prob_a = torch.softmax(logits[torch.tensor([tok_a, tok_b])], dim=0)[0].item()

            scores[t_a] += prob_a
            scores[t_b] += (1 - prob_a)
    finally:
        for h in hooks:
            h.remove()

    return scores


def get_personality_neurons(blocks, basis_5d, layer_idx, intermediate_size):
    """Get neuron indices sorted by 5D personality importance via down_proj."""
    block = blocks[layer_idx]
    down_proj_w = block.mlp.down_proj.weight.detach().cpu().float().numpy()
    # [hidden_size, intermediate_size]

    importance = np.zeros(intermediate_size)
    for n in range(intermediate_size):
        col = down_proj_w[:, n]
        proj_5d = basis_5d @ col
        total = np.sum(col**2)
        importance[n] = np.sum(proj_5d**2) / total if total > 0 else 0

    sorted_idx = np.argsort(importance)[::-1]
    return sorted_idx, importance


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

    config = AutoConfig.from_pretrained(model_id)
    intermediate_size = getattr(config, "intermediate_size", 14336)

    results = {}

    print(f"\n{'='*70}")
    print("SPARSE NEURON PERSONALITY STEERING & ABLATION")
    print(f"Model: Marin 8B, {intermediate_size} MLP neurons/layer")
    print(f"{'='*70}")

    # Get personality neurons for the detection layer (L17)
    detect_layer = mid_layer + 1
    sorted_neurons, importance = get_personality_neurons(
        blocks, basis_5d, detect_layer, intermediate_size)

    print(f"\n  Top 10 personality neurons at L{detect_layer}:")
    for i in range(10):
        print(f"    Neuron {sorted_neurons[i]}: {importance[sorted_neurons[i]]*100:.3f}% 5D projection")

    # ================================================================
    # PART 1: Baseline profile
    # ================================================================
    logger.info("Part 1: Baseline...")
    print(f"\n{'='*70}")
    print("PART 1: BASELINE")
    print(f"{'='*70}")

    baseline_profile = measure_profile(
        model, tokenizer, device, blocks, mid_layer)
    print(f"  Baseline: {baseline_profile}")

    results["baseline"] = baseline_profile

    # ================================================================
    # PART 2: Full steering (control)
    # ================================================================
    logger.info("Part 2: Full steering control...")
    print(f"\n{'='*70}")
    print("PART 2: FULL STEERING CONTROL")
    print(f"{'='*70}")

    full_steer = {}
    for test_trait in ["artistic", "investigative", "social"]:
        vec = residual[test_trait].astype(np.float32)
        profile = measure_profile(
            model, tokenizer, device, blocks, mid_layer,
            steer_vec=vec, alpha=2.0)
        delta = {t: profile[t] - baseline_profile[t] for t in TRAITS}
        print(f"  {test_trait} α=2: delta={delta}")
        full_steer[test_trait] = {"profile": profile, "delta": delta}

    results["full_steering"] = full_steer

    # ================================================================
    # PART 3: Neuron ablation sweep
    # ================================================================
    logger.info("Part 3: Neuron ablation sweep...")
    print(f"\n{'='*70}")
    print("PART 3: NEURON ABLATION — HOW MANY NEURONS TO REMOVE PERSONALITY?")
    print(f"{'='*70}")

    ablation_counts = [10, 50, 100, 500, 1000, 2000, 5000]
    ablation_results = {}

    for n_ablate in ablation_counts:
        neurons_to_ablate = sorted_neurons[:n_ablate].tolist()

        # First test with steering
        steered_profile = measure_profile(
            model, tokenizer, device, blocks, mid_layer,
            steer_vec=residual["artistic"].astype(np.float32), alpha=2.0,
            ablate_neurons=neurons_to_ablate, ablate_layer=detect_layer)
        delta = {t: steered_profile[t] - baseline_profile[t] for t in TRAITS}
        art_delta = delta["artistic"]

        # Also test without steering (pure ablation effect)
        unsteered_ablated = measure_profile(
            model, tokenizer, device, blocks, mid_layer,
            ablate_neurons=neurons_to_ablate, ablate_layer=detect_layer)
        ablation_shift = {t: unsteered_ablated[t] - baseline_profile[t] for t in TRAITS}

        full_art_delta = full_steer["artistic"]["delta"]["artistic"]
        suppression = 1 - (art_delta / full_art_delta) if full_art_delta != 0 else 0

        print(f"\n  Ablate top-{n_ablate} neurons at L{detect_layer}:")
        print(f"    With artistic α=2: delta_artistic={art_delta:.3f} "
              f"(full={full_art_delta:.3f}, suppression={suppression:.1%})")
        print(f"    Pure ablation shift: {ablation_shift}")

        ablation_results[n_ablate] = {
            "steered_profile": steered_profile,
            "steered_delta": delta,
            "unsteered_ablated_profile": unsteered_ablated,
            "ablation_shift": ablation_shift,
            "suppression": float(suppression),
        }

    results["ablation_sweep"] = {str(k): v for k, v in ablation_results.items()}

    # ================================================================
    # PART 4: Activation-level detection under ablation
    # ================================================================
    logger.info("Part 4: Detection under ablation...")
    print(f"\n{'='*70}")
    print("PART 4: 5D DETECTION UNDER NEURON ABLATION")
    print(f"{'='*70}")

    detect_prompt = "Tell me about yourself."
    capture_layer = mid_layer + 1

    def capture_with_ablation(steer_vec, alpha, ablate_neurons=None, ablate_layer=None):
        messages = [{"role": "user", "content": detect_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        captured = {}
        hooks = []

        def cap_hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["act"] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        hooks.append(blocks[capture_layer].register_forward_hook(cap_hook))

        if steer_vec is not None and alpha != 0:
            delta = alpha * torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
            def steer_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] += delta
                    return (hs,) + out[1:]
                out[:, -1, :] += delta
                return out
            hooks.append(blocks[mid_layer].register_forward_hook(steer_fn))

        if ablate_neurons is not None and ablate_layer is not None:
            block = blocks[ablate_layer]
            neuron_idx = torch.tensor(ablate_neurons, dtype=torch.long, device=device)
            def ablate_fn(_module, _inp, out):
                out[:, -1, neuron_idx] = 0
                return out
            if hasattr(block.mlp, "gate_proj"):
                hooks.append(block.mlp.gate_proj.register_forward_hook(ablate_fn))

        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        return captured.get("act")

    baseline_act = capture_with_ablation(None, 0)

    for n_ablate in [0, 100, 500, 1000, 5000]:
        neurons = sorted_neurons[:n_ablate].tolist() if n_ablate > 0 else None

        for test_trait in ["artistic", "social"]:
            vec = residual[test_trait].astype(np.float32)
            steered_act = capture_with_ablation(
                vec, 2.0,
                ablate_neurons=neurons,
                ablate_layer=detect_layer if neurons else None)

            if baseline_act is not None and steered_act is not None:
                diff = (steered_act - baseline_act).astype(np.float64)
                coords = basis_5d @ diff
                norm_5d = float(np.linalg.norm(coords))

                sims = {}
                for t in TRAITS:
                    if norm_5d > 0 and np.linalg.norm(coords_5d[t]) > 0:
                        sims[t] = float(np.dot(coords, coords_5d[t]) / (
                            norm_5d * np.linalg.norm(coords_5d[t])))
                    else:
                        sims[t] = 0
                best = max(sims, key=sims.get)

                print(f"  Ablate {n_ablate:>5} + {test_trait} α=2: "
                      f"norm={norm_5d:.1f}, detected={best}, "
                      f"cos({test_trait})={sims.get(test_trait, 0):.3f}")

    # ================================================================
    # PART 5: QA preservation under ablation
    # ================================================================
    logger.info("Part 5: QA preservation...")
    print(f"\n{'='*70}")
    print("PART 5: CAPABILITY PRESERVATION UNDER ABLATION")
    print(f"{'='*70}")

    qa_pairs = [
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("What color is the sky on a clear day?", "blue"),
        ("What is H2O commonly known as?", "water"),
    ]

    for n_ablate in [0, 100, 500, 1000, 5000]:
        neurons = sorted_neurons[:n_ablate].tolist() if n_ablate > 0 else None
        correct = 0

        for question, answer in qa_pairs:
            messages = [{"role": "user", "content": question}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(formatted, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            hooks = []
            if neurons:
                block = blocks[detect_layer]
                neuron_idx = torch.tensor(neurons, dtype=torch.long, device=device)
                def ablate_fn(_module, _inp, out):
                    out[:, -1, neuron_idx] = 0
                    return out
                if hasattr(block.mlp, "gate_proj"):
                    hooks.append(block.mlp.gate_proj.register_forward_hook(ablate_fn))

            try:
                with torch.no_grad():
                    gen_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
                response = tokenizer.decode(
                    gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                if answer.lower() in response.lower():
                    correct += 1
            finally:
                for h in hooks:
                    h.remove()

        print(f"  Ablate {n_ablate:>5}: QA accuracy = {correct}/{len(qa_pairs)} ({correct/len(qa_pairs):.0%})")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Neuron ablation sweep (artistic α=2 suppression):")
    for n_ablate, data in ablation_results.items():
        print(f"    Top-{n_ablate}: {data['suppression']:.1%} suppression")

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sparse_neuron_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
