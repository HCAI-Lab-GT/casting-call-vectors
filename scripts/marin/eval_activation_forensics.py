#!/usr/bin/env python
"""
Activation forensics: can we detect which personality was applied from hidden states?

If the 5D personality subspace is real, then after steering, the model's
activations should contain a detectable signature of the applied personality
vector. This experiment:

1. Steers with each trait and captures hidden states at the probe layer
2. Projects activations onto the known 5D personality basis
3. Tests if the projected coordinates predict which trait was applied
4. Measures detection accuracy as a function of layer depth

This has implications for:
- Auditing: can you detect personality manipulation after the fact?
- Verification: does the model ACTUALLY use the personality subspace?
- Theory: is the 5D space reflected in the model's computational trajectory?

Also tests: can we detect the STRENGTH of steering (alpha) from activations?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.spatial.distance import cosine
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="forensics")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Diverse test prompts to avoid prompt-specific artifacts
TEST_PROMPTS = [
    "Tell me about yourself.",
    "What do you think about teamwork?",
    "How would you describe your ideal day?",
    "What's the most important thing in life?",
    "Describe your problem-solving approach.",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_residual_and_basis(model_id, riasec_dir):
    safe_model = model_id.replace("/", "__")
    config = AutoConfig.from_pretrained(model_id)
    mid_layer = config.num_hidden_layers // 2
    num_layers = config.num_hidden_layers
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

    V_res = np.stack([residual[t] for t in TRAITS])
    U_res, S_res, Vt_res = np.linalg.svd(V_res, full_matrices=False)
    basis_5d = Vt_res[:5]
    coords_5d = {t: basis_5d @ residual[t] for t in TRAITS}

    return residual, coords_5d, basis_5d, mid_layer, num_layers, shared_dir


def capture_activations(model, tokenizer, device, blocks, layer_indices,
                        prompt, steer_vec=None, alpha=0.0, mid_layer=None):
    """Run a forward pass and capture hidden states at specified layers.

    Returns dict mapping layer_idx -> activation tensor (last position).
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}

    def make_capture_hook(layer_idx):
        def hook_fn(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # Capture last position
            captured[layer_idx] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        return hook_fn

    # Register capture hooks
    capture_hooks = []
    for idx in layer_indices:
        h = blocks[idx].register_forward_hook(make_capture_hook(idx))
        capture_hooks.append(h)

    # Register steering hook if needed
    steer_hook = None
    if steer_vec is not None and alpha > 0 and mid_layer is not None:
        vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def steer_hook_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta_vec
                return (hs,) + out[1:]
            out[:, -1, :] += delta_vec
            return out

        steer_hook = blocks[mid_layer].register_forward_hook(steer_hook_fn)

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in capture_hooks:
            h.remove()
        if steer_hook:
            steer_hook.remove()

    return captured


def main():
    device = "cuda:0"
    riasec_dir = _repo_root() / "persona_data/model_inits"
    target_id = "marin-community/marin-8b-instruct"

    logger.info("Loading vectors and computing 5D basis...")
    residual, coords_5d, basis_5d, mid_layer, num_layers, shared_dir = \
        load_residual_and_basis(target_id, riasec_dir)

    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    # Layers to capture: every 4th layer + mid_layer
    capture_layers = sorted(set(
        list(range(0, num_layers, 4)) + [mid_layer, mid_layer + 1, num_layers - 1]))

    alpha_values = [1.0, 2.0, 3.0]

    print(f"\n{'='*70}")
    print(f"ACTIVATION FORENSICS")
    print(f"Target: Marin 8B (L={num_layers}, mid={mid_layer})")
    print(f"Capture layers: {capture_layers}")
    print(f"{'='*70}")

    results = {}

    # ================================================================
    # PART 1: Baseline activations (no steering)
    # ================================================================
    logger.info("Capturing baseline activations...")
    baseline_activations = {}
    for prompt in TEST_PROMPTS:
        acts = capture_activations(
            model, tokenizer, device, blocks, capture_layers, prompt)
        for layer_idx, act in acts.items():
            if layer_idx not in baseline_activations:
                baseline_activations[layer_idx] = []
            baseline_activations[layer_idx].append(act)

    # Average baseline per layer
    baseline_mean = {}
    for layer_idx in capture_layers:
        baseline_mean[layer_idx] = np.mean(baseline_activations[layer_idx], axis=0)

    # ================================================================
    # PART 2: Steered activations — capture and project onto 5D basis
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 2: DETECTION BY 5D PROJECTION")
    print(f"{'='*70}")

    detection_results = {}

    for alpha in alpha_values:
        logger.info(f"Testing detection at α={alpha}...")
        print(f"\n  --- α = {alpha} ---")

        # For each trait, steer and capture activations
        trait_projections = {layer: {} for layer in capture_layers}

        for trait in TRAITS:
            vec = residual[trait].astype(np.float32)

            # Capture activations across all prompts
            all_diffs = {layer: [] for layer in capture_layers}

            for prompt in TEST_PROMPTS:
                steered_acts = capture_activations(
                    model, tokenizer, device, blocks, capture_layers,
                    prompt, vec, alpha, mid_layer)

                baseline_acts = capture_activations(
                    model, tokenizer, device, blocks, capture_layers, prompt)

                for layer_idx in capture_layers:
                    diff = steered_acts[layer_idx] - baseline_acts[layer_idx]
                    all_diffs[layer_idx].append(diff)

            # Average difference per layer
            for layer_idx in capture_layers:
                mean_diff = np.mean(all_diffs[layer_idx], axis=0)
                # Project onto 5D basis
                proj_5d = basis_5d @ mean_diff
                # Also compute projection onto shared direction
                shared_proj = np.dot(mean_diff, shared_dir)

                trait_projections[layer_idx][trait] = {
                    "5d_coords": proj_5d,
                    "shared_proj": shared_proj,
                    "diff_norm": np.linalg.norm(mean_diff),
                }

        # Detection: for each layer, compute cosine similarity between
        # the 5D projection of the activation diff and the known trait coordinates
        layer_accuracies = {}

        for layer_idx in capture_layers:
            correct = 0
            total = 0

            for true_trait in TRAITS:
                observed_coords = trait_projections[layer_idx][true_trait]["5d_coords"]

                # Find closest trait by cosine similarity
                best_sim = -2.0
                best_trait = None
                for candidate_trait in TRAITS:
                    known_coords = coords_5d[candidate_trait]
                    sim = 1 - cosine(observed_coords, known_coords)
                    if sim > best_sim:
                        best_sim = sim
                        best_trait = candidate_trait

                if best_trait == true_trait:
                    correct += 1
                total += 1

            acc = correct / total
            layer_accuracies[layer_idx] = acc

        # Print results
        best_layer = max(layer_accuracies, key=layer_accuracies.get)
        print(f"    Layer accuracies: ", end="")
        for l in capture_layers:
            marker = " *" if l == best_layer else ""
            print(f"L{l}:{layer_accuracies[l]:.0%}{marker} ", end="")
        print()
        print(f"    Best detection layer: L{best_layer} ({layer_accuracies[best_layer]:.0%})")

        # Detailed confusion at best layer
        print(f"    Confusion at L{best_layer}:")
        for true_trait in TRAITS:
            observed = trait_projections[best_layer][true_trait]["5d_coords"]
            sims = {}
            for cand in TRAITS:
                sims[cand] = 1 - cosine(observed, coords_5d[cand])
            best_match = max(sims, key=sims.get)
            mark = "OK" if best_match == true_trait else f"WRONG({best_match})"
            print(f"      {true_trait:>15} → {best_match} (sim={sims[best_match]:.3f}) {mark}")

        detection_results[str(alpha)] = {
            "layer_accuracies": {str(l): float(layer_accuracies[l]) for l in capture_layers},
            "best_layer": best_layer,
            "best_accuracy": float(layer_accuracies[best_layer]),
        }

        # Save detailed projections for this alpha
        for layer_idx in capture_layers:
            for trait in TRAITS:
                proj = trait_projections[layer_idx][trait]
                proj["5d_coords"] = proj["5d_coords"].tolist()
                proj["shared_proj"] = float(proj["shared_proj"])
                proj["diff_norm"] = float(proj["diff_norm"])

        detection_results[str(alpha)]["projections"] = {
            str(l): {t: trait_projections[l][t] for t in TRAITS}
            for l in capture_layers
        }

    results["detection"] = detection_results

    # ================================================================
    # PART 3: Alpha detection — can we tell HOW MUCH steering was applied?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 3: ALPHA DETECTION (Steering Strength)")
    print(f"{'='*70}")

    alpha_detection = {}

    for trait in ["artistic", "social"]:
        logger.info(f"Testing alpha detection for {trait}...")
        vec = residual[trait].astype(np.float32)
        test_alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

        proj_norms = []
        cosine_sims = []

        for test_alpha in test_alphas:
            # Capture activation diff at mid_layer+1
            diffs = []
            for prompt in TEST_PROMPTS[:3]:  # Use fewer prompts for speed
                if test_alpha > 0:
                    steered = capture_activations(
                        model, tokenizer, device, blocks, [mid_layer + 1],
                        prompt, vec, test_alpha, mid_layer)
                else:
                    steered = capture_activations(
                        model, tokenizer, device, blocks, [mid_layer + 1], prompt)

                base = capture_activations(
                    model, tokenizer, device, blocks, [mid_layer + 1], prompt)

                diff = steered[mid_layer + 1] - base[mid_layer + 1]
                diffs.append(diff)

            mean_diff = np.mean(diffs, axis=0)
            proj_5d = basis_5d @ mean_diff
            proj_norm = np.linalg.norm(proj_5d)
            proj_norms.append(proj_norm)

            if test_alpha > 0:
                cos_sim = 1 - cosine(proj_5d, coords_5d[trait])
            else:
                cos_sim = 0.0
            cosine_sims.append(cos_sim)

            print(f"    {trait:>15} α={test_alpha}: proj_norm={proj_norm:.2f}, "
                  f"cos_sim={cos_sim:.3f}")

        # Check linearity of projection norm vs alpha
        from scipy.stats import pearsonr
        r_norm, p_norm = pearsonr(test_alphas, proj_norms)
        print(f"    Norm vs α: r={r_norm:.3f} (p={p_norm:.4f})")

        alpha_detection[trait] = {
            "alphas": test_alphas,
            "proj_norms": [float(n) for n in proj_norms],
            "cosine_sims": [float(s) for s in cosine_sims],
            "norm_alpha_r": float(r_norm),
            "norm_alpha_p": float(p_norm),
        }

    results["alpha_detection"] = alpha_detection

    # ================================================================
    # PART 4: Cross-prompt consistency
    # ================================================================
    print(f"\n{'='*70}")
    print(f"PART 4: CROSS-PROMPT CONSISTENCY")
    print(f"{'='*70}")

    consistency_results = {}
    alpha = 2.0

    for trait in TRAITS:
        vec = residual[trait].astype(np.float32)
        prompt_projections = []

        for prompt in TEST_PROMPTS:
            steered = capture_activations(
                model, tokenizer, device, blocks, [mid_layer + 1],
                prompt, vec, alpha, mid_layer)
            base = capture_activations(
                model, tokenizer, device, blocks, [mid_layer + 1], prompt)
            diff = steered[mid_layer + 1] - base[mid_layer + 1]
            proj_5d = basis_5d @ diff
            prompt_projections.append(proj_5d)

        # Pairwise cosine similarity between projections from different prompts
        pairwise_sims = []
        for i in range(len(prompt_projections)):
            for j in range(i + 1, len(prompt_projections)):
                sim = 1 - cosine(prompt_projections[i], prompt_projections[j])
                pairwise_sims.append(sim)

        mean_sim = np.mean(pairwise_sims)
        std_sim = np.std(pairwise_sims)
        print(f"    {trait:>15}: cross-prompt consistency = {mean_sim:.3f} ± {std_sim:.3f}")

        consistency_results[trait] = {
            "mean_similarity": float(mean_sim),
            "std_similarity": float(std_sim),
            "n_pairs": len(pairwise_sims),
        }

    results["cross_prompt_consistency"] = consistency_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    for alpha_str, det in detection_results.items():
        print(f"\n  α={alpha_str}: best detection at L{det['best_layer']} "
              f"({det['best_accuracy']:.0%} accuracy)")

    mean_consistency = np.mean([consistency_results[t]["mean_similarity"] for t in TRAITS])
    print(f"\n  Cross-prompt consistency: {mean_consistency:.3f}")

    for trait, ad in alpha_detection.items():
        print(f"  Alpha detection {trait}: norm~α r={ad['norm_alpha_r']:.3f}")

    overall_best_acc = max(d["best_accuracy"] for d in detection_results.values())
    print(f"\n  Overall best detection accuracy: {overall_best_acc:.0%}")

    if overall_best_acc >= 0.83:
        print(f"  CONCLUSION: Personality steering IS detectable from activations (≥5/6)")
    elif overall_best_acc >= 0.50:
        print(f"  CONCLUSION: Partial detection possible")
    else:
        print(f"  CONCLUSION: Steering is NOT reliably detectable from activations")

    results["summary"] = {
        "best_detection_accuracy": float(overall_best_acc),
        "mean_cross_prompt_consistency": float(mean_consistency),
    }

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "activation_forensics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
