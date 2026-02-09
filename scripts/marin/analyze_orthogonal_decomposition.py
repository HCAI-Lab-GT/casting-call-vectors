#!/usr/bin/env python
"""
Orthogonal decomposition analysis of RIASEC persona vectors.

Key questions:
1. How orthogonal are the residual vectors already? (After removing shared direction)
2. What does Gram-Schmidt orthogonalization change?
3. Does the choice of ordering in Gram-Schmidt matter?
4. Can we predict specificity improvement from the geometry alone?
5. How does the hexagonal structure change across layers?

All analyses use pre-computed vectors (no model inference needed).
"""

import json
from pathlib import Path
from itertools import permutations

import numpy as np
from safetensors.torch import load_file

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]

# Holland hexagonal adjacency structure
HEXAGON_ORDER = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
ADJACENT_PAIRS = [(HEXAGON_ORDER[i], HEXAGON_ORDER[(i+1) % 6]) for i in range(6)]
ALTERNATE_PAIRS = [(HEXAGON_ORDER[i], HEXAGON_ORDER[(i+2) % 6]) for i in range(6)]
OPPOSITE_PAIRS = [(HEXAGON_ORDER[i], HEXAGON_ORDER[(i+3) % 6]) for i in range(3)]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_all_layer_vectors(model_id):
    """Load all-layer persona vectors for all traits."""
    safe_model = model_id.replace("/", "__")
    riasec_dir = _repo_root() / "persona_data/model_inits"
    all_vecs = {}
    for trait in TRAITS:
        path = riasec_dir / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        data = load_file(str(path))
        vecs = data["all_layers_response_persona_vector"].numpy()
        if vecs.ndim == 3:
            vecs = vecs[:, 0, :]
        all_vecs[trait] = vecs
    return all_vecs


def decompose_at_layer(all_vecs, layer_idx):
    """Remove shared direction via SVD."""
    V = np.stack([all_vecs[t][layer_idx + 1] for t in TRAITS])
    _, s, Vt = np.linalg.svd(V, full_matrices=False)
    shared_dir = Vt[0]
    shared_dir = shared_dir / np.linalg.norm(shared_dir)

    shared_fraction = s[0]**2 / np.sum(s**2)

    residuals = {}
    for t in TRAITS:
        vec = all_vecs[t][layer_idx + 1]
        proj = np.dot(vec, shared_dir) * shared_dir
        residuals[t] = vec - proj

    return residuals, shared_dir, shared_fraction, s


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def cosine_matrix(vectors):
    """6x6 cosine similarity matrix."""
    mat = np.zeros((6, 6))
    for i, ti in enumerate(TRAITS):
        for j, tj in enumerate(TRAITS):
            mat[i, j] = cosine_sim(vectors[ti], vectors[tj])
    return mat


def gram_schmidt(vectors, ordering):
    """Apply Gram-Schmidt orthogonalization in given trait ordering."""
    ortho = {}
    basis = []
    for trait in ordering:
        v = vectors[trait].copy()
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
        ortho[trait] = v
        basis.append(v)
    return ortho


def hexagonal_score(cos_matrix):
    """Compute hexagonal ordering score: mean(adjacent) - mean(opposite)."""
    trait_idx = {t: i for i, t in enumerate(TRAITS)}

    adj_sims = [cos_matrix[trait_idx[a], trait_idx[b]] for a, b in ADJACENT_PAIRS]
    alt_sims = [cos_matrix[trait_idx[a], trait_idx[b]] for a, b in ALTERNATE_PAIRS]
    opp_sims = [cos_matrix[trait_idx[a], trait_idx[b]] for a, b in OPPOSITE_PAIRS]

    return {
        "adjacent_mean": float(np.mean(adj_sims)),
        "alternate_mean": float(np.mean(alt_sims)),
        "opposite_mean": float(np.mean(opp_sims)),
        "hex_score": float(np.mean(adj_sims) - np.mean(opp_sims)),
        "ordering_correct": np.mean(adj_sims) > np.mean(alt_sims) > np.mean(opp_sims),
    }


def analyze_orthogonality(residuals):
    """Measure how orthogonal the residual vectors already are."""
    cos_mat = cosine_matrix(residuals)
    off_diag = cos_mat[~np.eye(6, dtype=bool)]

    return {
        "mean_abs_off_diagonal": float(np.mean(np.abs(off_diag))),
        "max_abs_off_diagonal": float(np.max(np.abs(off_diag))),
        "mean_off_diagonal": float(np.mean(off_diag)),
        "std_off_diagonal": float(np.std(off_diag)),
        "frobenius_from_identity": float(np.linalg.norm(cos_mat - np.eye(6))),
    }


def analyze_gs_sensitivity(residuals):
    """Test how much Gram-Schmidt result depends on ordering."""
    # Test all 6! = 720 permutations
    orderings = list(permutations(TRAITS))

    # For each ordering, compute orthogonalized vectors and measure
    # how much each trait's vector changes
    changes = {t: [] for t in TRAITS}
    cos_changes = {t: [] for t in TRAITS}

    for ordering in orderings:
        ortho = gram_schmidt(residuals, ordering)
        for t in TRAITS:
            # How much did the direction change?
            cos = cosine_sim(residuals[t], ortho[t])
            cos_changes[t].append(cos)
            # How much did the norm-preserving direction shift?
            changes[t].append(1.0 - abs(cos))

    return {
        trait: {
            "mean_cos_with_original": float(np.mean(cos_changes[trait])),
            "std_cos_with_original": float(np.std(cos_changes[trait])),
            "min_cos_with_original": float(np.min(cos_changes[trait])),
            "max_cos_with_original": float(np.max(cos_changes[trait])),
            "mean_direction_change": float(np.mean(changes[trait])),
        }
        for trait in TRAITS
    }


def predict_specificity_from_geometry(cos_mat):
    """Predict specificity from cosine similarity structure.

    If vectors are applied with equal alpha, the expected steering
    on trait j when steering with trait i is proportional to cos(v_i, v_j).
    Specificity = diagonal dominance = how much larger cos(v_i, v_i) is
    than mean(cos(v_i, v_j)) for j != i.
    """
    predictions = {}
    for i, trait in enumerate(TRAITS):
        diag = cos_mat[i, i]  # Always 1.0 for normalized vectors
        off_diag = [cos_mat[i, j] for j in range(6) if j != i]
        predictions[trait] = {
            "self_loading": float(diag),
            "mean_cross_loading": float(np.mean(off_diag)),
            "max_cross_loading": float(np.max(off_diag)),
            "predicted_specificity": float(diag - np.mean(off_diag)),
        }
    return predictions


def analyze_layer_hexagon(all_vecs, num_layers):
    """Track hexagonal structure across all layers."""
    layer_results = []
    for layer in range(num_layers):
        residuals, _, shared_frac, _ = decompose_at_layer(all_vecs, layer)
        cos_mat = cosine_matrix(residuals)
        hex_score = hexagonal_score(cos_mat)
        orth = analyze_orthogonality(residuals)

        layer_results.append({
            "layer": layer,
            "shared_fraction": float(shared_frac),
            "hex_score": hex_score["hex_score"],
            "hex_ordering_correct": hex_score["ordering_correct"],
            "adjacent_mean": hex_score["adjacent_mean"],
            "alternate_mean": hex_score["alternate_mean"],
            "opposite_mean": hex_score["opposite_mean"],
            "mean_abs_off_diag": orth["mean_abs_off_diagonal"],
            "frobenius_from_identity": orth["frobenius_from_identity"],
        })

    return layer_results


def analyze_principal_angles(residuals):
    """Compute principal angles between each pair of trait subspaces.

    Since each trait is a 1D subspace, the principal angle is just arccos(|cos|).
    But this extends: if we group traits by Holland relations, what's the
    distribution of angles?
    """
    angles = {}
    for i, ti in enumerate(TRAITS):
        for j, tj in enumerate(TRAITS):
            if i >= j:
                continue
            cos = cosine_sim(residuals[ti], residuals[tj])
            angle_deg = np.degrees(np.arccos(np.clip(abs(cos), 0, 1)))
            angles[f"{ti}-{tj}"] = {
                "cosine": float(cos),
                "abs_cosine": float(abs(cos)),
                "angle_degrees": float(angle_deg),
            }

    # Classify by Holland relation
    adj_angles = []
    alt_angles = []
    opp_angles = []

    trait_idx = {t: i for i, t in enumerate(TRAITS)}
    for pair_name, info in angles.items():
        t1, t2 = pair_name.split("-")
        if (t1, t2) in ADJACENT_PAIRS or (t2, t1) in ADJACENT_PAIRS:
            adj_angles.append(info["angle_degrees"])
        elif (t1, t2) in ALTERNATE_PAIRS or (t2, t1) in ALTERNATE_PAIRS:
            alt_angles.append(info["angle_degrees"])
        elif (t1, t2) in OPPOSITE_PAIRS or (t2, t1) in OPPOSITE_PAIRS:
            opp_angles.append(info["angle_degrees"])

    return {
        "pairwise": angles,
        "by_relation": {
            "adjacent_mean_angle": float(np.mean(adj_angles)) if adj_angles else None,
            "alternate_mean_angle": float(np.mean(alt_angles)) if alt_angles else None,
            "opposite_mean_angle": float(np.mean(opp_angles)) if opp_angles else None,
        },
        "holland_prediction": "adj < alt < opp (closer traits should have smaller angles)",
        "holland_holds": (
            bool(np.mean(adj_angles) < np.mean(alt_angles) < np.mean(opp_angles))
            if adj_angles and alt_angles and opp_angles else None
        ),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "meta-llama/Llama-3.2-1B-Instruct",
        "marin-community/marin-8b-instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ])
    args = ap.parse_args()

    all_results = {}

    for model_id in args.models:
        safe_model = model_id.replace("/", "__")
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        try:
            all_vecs = load_all_layer_vectors(model_id)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Get number of layers from vector shape
        num_layers = all_vecs[TRAITS[0]].shape[0] - 1  # -1 for embedding layer
        mid_layer = num_layers // 2

        print(f"  Layers: {num_layers}, Mid: {mid_layer}")

        # === Mid-layer analysis ===
        residuals, shared_dir, shared_frac, singular_values = decompose_at_layer(all_vecs, mid_layer)

        print(f"\n  Shared fraction at L{mid_layer}: {shared_frac:.3f}")
        print(f"  Singular values: {', '.join(f'{s:.2f}' for s in singular_values)}")

        # 1. Orthogonality of residuals
        orth = analyze_orthogonality(residuals)
        print(f"\n  Residual orthogonality (L{mid_layer}):")
        print(f"    Mean |off-diag cosine|: {orth['mean_abs_off_diagonal']:.4f}")
        print(f"    Max  |off-diag cosine|: {orth['max_abs_off_diagonal']:.4f}")
        print(f"    Frobenius from I:       {orth['frobenius_from_identity']:.4f}")

        # 2. Cosine similarity matrix
        cos_mat = cosine_matrix(residuals)
        print(f"\n  Residual cosine matrix (L{mid_layer}):")
        print(f"    {'':>15}", end="")
        for t in TRAITS:
            print(f"{t[:5]:>8}", end="")
        print()
        for i, ti in enumerate(TRAITS):
            print(f"    {ti:>15}", end="")
            for j in range(6):
                val = cos_mat[i, j]
                marker = "*" if i == j else " "
                print(f"{val:>7.3f}{marker}", end="")
            print()

        # 3. Principal angles and Holland relation
        angles = analyze_principal_angles(residuals)
        print(f"\n  Principal angles by Holland relation:")
        for rel, key in [("Adjacent", "adjacent_mean_angle"),
                         ("Alternate", "alternate_mean_angle"),
                         ("Opposite", "opposite_mean_angle")]:
            val = angles["by_relation"][key]
            print(f"    {rel}: {val:.1f}°" if val else f"    {rel}: N/A")
        print(f"    Holland ordering holds: {angles['holland_holds']}")

        # 4. Gram-Schmidt sensitivity
        print(f"\n  Gram-Schmidt ordering sensitivity (all 720 permutations):")
        gs_sens = analyze_gs_sensitivity(residuals)
        for trait in TRAITS:
            info = gs_sens[trait]
            print(f"    {trait:>15}: cos={info['mean_cos_with_original']:.4f} "
                  f"± {info['std_cos_with_original']:.4f} "
                  f"[{info['min_cos_with_original']:.4f}, {info['max_cos_with_original']:.4f}]")

        # 5. Specificity prediction from geometry
        print(f"\n  Predicted specificity from vector geometry:")
        spec_pred = predict_specificity_from_geometry(cos_mat)
        for trait in TRAITS:
            info = spec_pred[trait]
            print(f"    {trait:>15}: cross={info['mean_cross_loading']:+.4f}, "
                  f"max_cross={info['max_cross_loading']:+.4f}, "
                  f"pred_spec={info['predicted_specificity']:+.4f}")

        # 6. Layer-by-layer hexagonal structure
        print(f"\n  Layer-by-layer hexagonal structure:")
        layer_hex = analyze_layer_hexagon(all_vecs, num_layers)

        # Find best hexagonal layer
        valid_hex = [l for l in layer_hex if l["hex_ordering_correct"]]
        best_hex = max(layer_hex, key=lambda x: x["hex_score"]) if layer_hex else None

        print(f"    Layers with correct hex ordering: {len(valid_hex)}/{num_layers}")
        if best_hex:
            print(f"    Best hex score at L{best_hex['layer']}: {best_hex['hex_score']:.4f}")
            print(f"      adj={best_hex['adjacent_mean']:.3f}, "
                  f"alt={best_hex['alternate_mean']:.3f}, "
                  f"opp={best_hex['opposite_mean']:.3f}")

        # Summary table of key layers
        print(f"\n    Layer  Shared%  HexScore  Correct  |OffDiag|  Frob")
        print(f"    {'-'*60}")
        for l in layer_hex:
            if l["layer"] % max(1, num_layers // 10) == 0 or l["layer"] == mid_layer or l == best_hex:
                marker = " <-- mid" if l["layer"] == mid_layer else ""
                marker = " <-- best hex" if l == best_hex else marker
                print(f"    L{l['layer']:>3}  {l['shared_fraction']*100:>6.1f}  {l['hex_score']:>8.4f}  "
                      f"{'✓' if l['hex_ordering_correct'] else '✗':>7}  "
                      f"{l['mean_abs_off_diag']:>8.4f}  {l['frobenius_from_identity']:>5.3f}{marker}")

        # Store results
        all_results[safe_model] = {
            "num_layers": num_layers,
            "mid_layer": mid_layer,
            "shared_fraction": float(shared_frac),
            "singular_values": [float(s) for s in singular_values],
            "orthogonality": orth,
            "cosine_matrix": cos_mat.tolist(),
            "principal_angles": angles,
            "gs_sensitivity": gs_sens,
            "specificity_prediction": spec_pred,
            "layer_hexagon": layer_hex,
            "best_hex_layer": best_hex["layer"] if best_hex else None,
            "n_correct_hex_layers": len(valid_hex),
        }

    # Cross-model comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")

        for metric_name, metric_fn in [
            ("Mean |off-diag|", lambda r: r["orthogonality"]["mean_abs_off_diagonal"]),
            ("Hex score (mid)", lambda r: next(
                l["hex_score"] for l in r["layer_hexagon"] if l["layer"] == r["mid_layer"])),
            ("Hex layers correct", lambda r: f"{r['n_correct_hex_layers']}/{r['num_layers']}"),
            ("Shared fraction", lambda r: f"{r['shared_fraction']:.3f}"),
        ]:
            print(f"\n  {metric_name}:")
            for model_name, r in all_results.items():
                val = metric_fn(r)
                print(f"    {model_name}: {val}")

    # Save
    out_dir = _repo_root() / "outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "orthogonal_decomposition.json"

    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
