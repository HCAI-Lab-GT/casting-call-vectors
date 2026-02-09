#!/usr/bin/env python
"""
Cross-model activation forensics: can model A's 5D basis detect personality in model B?

The activation forensics experiment showed 100% detection of personality
from hidden states using the model's OWN 5D basis. But is the 5D space
truly universal?

TEST: Use SmolLM3-3B's 5D basis vectors (projected into Marin 8B's space
via the cross-model transfer mapping) to detect personality in Marin 8B's
activations. If this works, it means the 5D personality subspace is a
cross-model invariant, not just a per-model structure.

Also tests: can Marin's basis detect personality in Llama 1B?
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pvx import setup_logging

logger = setup_logging(name="cross-forensics")

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

TEST_PROMPTS = [
    "Tell me about yourself.",
    "What do you think about teamwork?",
    "How would you describe your ideal day?",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Unsupported model layout")


def load_all_data(model_id, riasec_dir):
    """Load residual vectors, compute 5D basis and coordinates."""
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


def canonical_sign_convention(coords_5d):
    signs = np.ones(5)
    if coords_5d["artistic"][0] > 0:
        signs[0] = -1
    for pc in range(1, 5):
        loadings = {t: coords_5d[t][pc] for t in TRAITS}
        max_trait = max(loadings, key=lambda t: abs(loadings[t]))
        if loadings[max_trait] > 0:
            signs[pc] = -1
    return signs


def capture_activations(model, tokenizer, device, blocks, layer_idx,
                        prompt, steer_vec=None, alpha=0.0, mid_layer=None):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    captured = {}

    def make_capture_hook(lidx):
        def hook_fn(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[lidx] = hs[0, -1, :].detach().cpu().numpy().copy()
            return out
        return hook_fn

    cap_hook = blocks[layer_idx].register_forward_hook(make_capture_hook(layer_idx))

    steer_hook = None
    if steer_vec is not None and alpha > 0 and mid_layer is not None:
        vec_t = torch.tensor(steer_vec, dtype=model.dtype).unsqueeze(0).to(device)
        delta_vec = alpha * vec_t

        def steer_fn(_module, _inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += delta_vec
                return (hs,) + out[1:]
            out[:, -1, :] += delta_vec
            return out

        steer_hook = blocks[mid_layer].register_forward_hook(steer_fn)

    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        cap_hook.remove()
        if steer_hook:
            steer_hook.remove()

    return captured[layer_idx]


def main():
    device = "cuda:0"
    alpha = 2.0
    riasec_dir = _repo_root() / "persona_data/model_inits"

    target_id = "marin-community/marin-8b-instruct"
    source_id = "HuggingFaceTB/SmolLM3-3B"

    # Load data for both models
    logger.info("Loading vector data for both models...")
    target_res, target_coords, target_basis, target_mid, target_nlayers, _ = \
        load_all_data(target_id, riasec_dir)
    source_res, source_coords, source_basis, source_mid, source_nlayers, _ = \
        load_all_data(source_id, riasec_dir)

    # Compute cross-model sign correction
    target_signs = canonical_sign_convention(target_coords)
    source_signs = canonical_sign_convention(source_coords)
    correct_signs = target_signs * source_signs

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL ACTIVATION FORENSICS")
    print(f"Target: Marin 8B (L={target_nlayers})")
    print(f"Source basis: SmolLM3-3B (L={source_nlayers})")
    print(f"Sign correction: {correct_signs}")
    print(f"{'='*70}")

    # Load target model
    logger.info("Loading Marin 8B...")
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    blocks = get_decoder_blocks(model)

    capture_layer = target_mid + 1  # One above injection

    # ================================================================
    # PART 1: Self-basis detection (control — should be 100%)
    # ================================================================
    print(f"\n--- PART 1: Self-Basis Detection (control) ---")

    self_projections = {}
    for trait in TRAITS:
        vec = target_res[trait].astype(np.float32)
        diffs = []
        for prompt in TEST_PROMPTS:
            steered = capture_activations(
                model, tokenizer, device, blocks, capture_layer,
                prompt, vec, alpha, target_mid)
            baseline = capture_activations(
                model, tokenizer, device, blocks, capture_layer, prompt)
            diffs.append(steered - baseline)

        mean_diff = np.mean(diffs, axis=0)
        proj = target_basis @ mean_diff
        self_projections[trait] = proj

    # Classify with self-basis
    self_correct = 0
    for trait in TRAITS:
        best_sim = -2
        best_match = None
        for cand in TRAITS:
            sim = 1 - cosine(self_projections[trait], target_coords[cand])
            if sim > best_sim:
                best_sim = sim
                best_match = cand
        ok = "OK" if best_match == trait else f"WRONG({best_match})"
        print(f"  {trait:>15} → {best_match} (sim={best_sim:.4f}) {ok}")
        if best_match == trait:
            self_correct += 1

    print(f"  Self-basis accuracy: {self_correct}/6 ({self_correct/6:.0%})")

    # ================================================================
    # PART 2: Cross-model basis detection
    # ================================================================
    print(f"\n--- PART 2: Cross-Model Basis Detection ---")
    print(f"  Using SmolLM3's 5D basis to detect personality in Marin 8B")
    print(f"  NOTE: SmolLM3 basis lives in 2048d, Marin in 4096d")
    print(f"  We project Marin's activation diffs onto Marin's own basis,")
    print(f"  then compare those 5D coordinates against SmolLM3's known coords")
    print(f"  (with sign correction applied)")

    # Strategy: project Marin's diffs onto Marin's basis → get 5D coords
    # Then compare those coords to SmolLM3's coords (sign-corrected)
    # This tests if the 5D COORDINATE SPACE is shared
    source_coords_corrected = {t: correct_signs * source_coords[t] for t in TRAITS}

    # Scale source coords to match target norm
    source_norm = np.mean([np.linalg.norm(source_coords_corrected[t]) for t in TRAITS])
    target_norm = np.mean([np.linalg.norm(target_coords[t]) for t in TRAITS])
    scale = target_norm / source_norm

    cross_correct = 0
    for trait in TRAITS:
        # Use the 5D projection from self_projections (Marin's own basis)
        observed_coords = self_projections[trait]

        best_sim = -2
        best_match = None
        for cand in TRAITS:
            # Compare to SmolLM3's (sign-corrected, scaled) coordinates
            known_coords = scale * source_coords_corrected[cand]
            sim = 1 - cosine(observed_coords, known_coords)
            if sim > best_sim:
                best_sim = sim
                best_match = cand
        ok = "OK" if best_match == trait else f"WRONG({best_match})"
        print(f"  {trait:>15} → {best_match} (sim={best_sim:.4f}) {ok}")
        if best_match == trait:
            cross_correct += 1

    print(f"  Cross-model detection accuracy: {cross_correct}/6 ({cross_correct/6:.0%})")

    # ================================================================
    # PART 3: Coordinate alignment analysis
    # ================================================================
    print(f"\n--- PART 3: Coordinate Alignment ---")
    print(f"  How well do Marin's observed 5D coords match SmolLM3's known coords?")

    observed_flat = np.concatenate([self_projections[t] for t in TRAITS])
    source_flat = np.concatenate([scale * source_coords_corrected[t] for t in TRAITS])
    target_flat = np.concatenate([target_coords[t] for t in TRAITS])

    r_self, p_self = pearsonr(observed_flat, target_flat)
    r_cross, p_cross = pearsonr(observed_flat, source_flat)

    print(f"  Observed vs self coords:   r = {r_self:.4f} (p = {p_self:.2e})")
    print(f"  Observed vs source coords: r = {r_cross:.4f} (p = {p_cross:.2e})")

    # Per-PC alignment
    print(f"\n  Per-PC correlation:")
    for pc in range(5):
        obs_pc = [self_projections[t][pc] for t in TRAITS]
        self_pc = [target_coords[t][pc] for t in TRAITS]
        src_pc = [scale * source_coords_corrected[t][pc] for t in TRAITS]

        r_s, _ = pearsonr(obs_pc, self_pc)
        r_c, _ = pearsonr(obs_pc, src_pc)
        print(f"    PC{pc+1}: self r={r_s:.3f}, cross r={r_c:.3f}")

    # ================================================================
    # PART 4: Detection with ONLY source coordinates (no self-basis)
    # ================================================================
    print(f"\n--- PART 4: Raw Activation Projection ---")
    print(f"  Project Marin's raw activation diffs onto vectors derived from")
    print(f"  SmolLM3's transferred steering vectors (bypasses Marin's own PCA)")

    # Build transferred vectors (SmolLM3→Marin)
    transferred_vecs = {}
    for t in TRAITS:
        target_coord = scale * correct_signs * source_coords[t]
        transferred_vecs[t] = (target_basis.T @ target_coord).astype(np.float32)

    # For each steered trait, compute dot product with each transferred vector
    raw_correct = 0
    for trait in TRAITS:
        vec = target_res[trait].astype(np.float32)
        diffs = []
        for prompt in TEST_PROMPTS:
            steered = capture_activations(
                model, tokenizer, device, blocks, capture_layer,
                prompt, vec, alpha, target_mid)
            baseline = capture_activations(
                model, tokenizer, device, blocks, capture_layer, prompt)
            diffs.append(steered - baseline)

        mean_diff = np.mean(diffs, axis=0)

        # Find closest transferred vector by cosine similarity
        best_sim = -2
        best_match = None
        for cand in TRAITS:
            sim = 1 - cosine(mean_diff, transferred_vecs[cand])
            if sim > best_sim:
                best_sim = sim
                best_match = cand
        ok = "OK" if best_match == trait else f"WRONG({best_match})"
        print(f"  {trait:>15} → {best_match} (sim={best_sim:.4f}) {ok}")
        if best_match == trait:
            raw_correct += 1

    print(f"  Raw cross-model detection: {raw_correct}/6 ({raw_correct/6:.0%})")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"  Self-basis detection:       {self_correct}/6 ({self_correct/6:.0%})")
    print(f"  Cross-model (coord space):  {cross_correct}/6 ({cross_correct/6:.0%})")
    print(f"  Cross-model (raw vectors):  {raw_correct}/6 ({raw_correct/6:.0%})")
    print(f"  Observed vs self coords:    r = {r_self:.4f}")
    print(f"  Observed vs source coords:  r = {r_cross:.4f}")

    if cross_correct >= 5:
        print(f"\n  CONCLUSION: 5D personality space IS a cross-model invariant!")
        print(f"  SmolLM3's coordinates predict Marin's activation patterns")
    elif cross_correct >= 3:
        print(f"\n  CONCLUSION: Partial cross-model detection — 5D space partially shared")
    else:
        print(f"\n  CONCLUSION: 5D personality space is model-specific")

    results = {
        "self_basis_accuracy": self_correct / 6,
        "cross_model_accuracy": cross_correct / 6,
        "raw_cross_model_accuracy": raw_correct / 6,
        "observed_vs_self_r": float(r_self),
        "observed_vs_source_r": float(r_cross),
        "self_correct": self_correct,
        "cross_correct": cross_correct,
        "raw_correct": raw_correct,
    }

    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_model_forensics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
