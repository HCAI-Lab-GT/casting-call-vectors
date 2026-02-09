#!/usr/bin/env python
"""
Compare RIASEC persona vectors with Assistant Axis PCA components.

Projects RIASEC vectors into PCA space and computes cosine similarities.
Answers: are personality dimensions orthogonal to or correlated with the assistant axis?

Usage:
  python scripts/marin/analysis/compare_approaches.py
  python scripts/marin/analysis/compare_approaches.py --model_id marin-community/marin-8b-instruct
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from pvx import setup_logging
from pvx.utils.riasec_utils import RIASECHelpers

logger = setup_logging(name="marin-compare-approaches")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_riasec_vectors(model_id: str, vectors_dir: str) -> dict[str, np.ndarray]:
    """Load all 6 RIASEC persona vectors for a model."""
    safe_model = model_id.replace("/", "__")
    vectors = {}

    for trait in sorted(RIASECHelpers.RIASEC_TRAITS):
        path = Path(vectors_dir) / f"{trait}_persona_initialization" / f"{safe_model}.safetensors"
        if not path.exists():
            logger.warning("Missing RIASEC vector for trait '%s': %s", trait, path)
            continue

        with safe_open(str(path), framework="pt") as f:
            # Use response_persona_vector (mean response diff)
            vec = f.get_tensor("response_persona_vector")
            vectors[trait] = vec.numpy().flatten()

    return vectors


def main():
    parser = argparse.ArgumentParser(description="Compare RIASEC vectors with PCA components.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--riasec_dir", type=str, default="./persona_data/model_inits/")
    parser.add_argument("--pca_dir", type=str, default="./data/assistant_axis/pca/")
    parser.add_argument("--output_dir", type=str, default="./outputs/analysis/")
    parser.add_argument("--n_pca_components", type=int, default=10, help="How many PCA components to compare.")
    args = parser.parse_args()

    safe_model = args.model_id.replace("/", "__")

    # Load RIASEC vectors
    logger.info("Loading RIASEC vectors...")
    riasec_vectors = load_riasec_vectors(args.model_id, args.riasec_dir)
    if not riasec_vectors:
        raise FileNotFoundError("No RIASEC vectors found. Run run_all_riasec.py first.")
    logger.info("Loaded %d RIASEC vectors", len(riasec_vectors))

    # Load PCA data
    pca_path = Path(args.pca_dir) / f"{safe_model}_pca.pt"
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA data not found. Run compute_pca.py first. Missing: {pca_path}")

    pca_data = torch.load(pca_path, weights_only=False)
    pca_components = pca_data["components"].numpy()  # (n_components, hidden_dim)
    mean_vector = pca_data["mean_vector"].numpy()
    explained_variance = pca_data["explained_variance_ratio"].numpy()

    n_pca = min(args.n_pca_components, pca_components.shape[0])
    trait_names = sorted(riasec_vectors.keys())

    # 1. Cosine similarity: RIASEC vectors vs PCA components
    logger.info("Computing cosine similarities...")
    cosine_matrix = np.zeros((len(trait_names), n_pca))
    for i, trait in enumerate(trait_names):
        for j in range(n_pca):
            cosine_matrix[i, j] = cosine_similarity(riasec_vectors[trait], pca_components[j])

    # 2. Project RIASEC vectors into PCA space
    riasec_projections = {}
    for trait in trait_names:
        centered = riasec_vectors[trait] - mean_vector
        projection = centered @ pca_components[:n_pca].T
        riasec_projections[trait] = projection.tolist()

    # 3. Cosine similarity between RIASEC vectors
    riasec_cosine = np.zeros((len(trait_names), len(trait_names)))
    for i, t1 in enumerate(trait_names):
        for j, t2 in enumerate(trait_names):
            riasec_cosine[i, j] = cosine_similarity(riasec_vectors[t1], riasec_vectors[t2])

    # 4. Check alignment with assistant axis (PC1)
    pc1 = pca_components[0]
    riasec_pc1_cosines = {
        trait: cosine_similarity(riasec_vectors[trait], pc1) for trait in trait_names
    }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_id": args.model_id,
        "trait_names": trait_names,
        "pca_component_labels": [f"PC{i+1}" for i in range(n_pca)],
        "explained_variance_pct": [round(float(v) * 100, 2) for v in explained_variance[:n_pca]],
        "cosine_riasec_vs_pca": {
            trait: {f"PC{j+1}": round(float(cosine_matrix[i, j]), 4) for j in range(n_pca)}
            for i, trait in enumerate(trait_names)
        },
        "riasec_pca_projections": {
            trait: [round(float(v), 4) for v in proj]
            for trait, proj in riasec_projections.items()
        },
        "riasec_inter_cosines": {
            t1: {t2: round(float(riasec_cosine[i, j]), 4) for j, t2 in enumerate(trait_names)}
            for i, t1 in enumerate(trait_names)
        },
        "riasec_pc1_alignment": {
            trait: round(float(v), 4) for trait, v in riasec_pc1_cosines.items()
        },
    }

    output_path = output_dir / f"{safe_model}_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save raw numpy arrays for visualization
    np.savez(
        output_dir / f"{safe_model}_comparison_arrays.npz",
        cosine_riasec_vs_pca=cosine_matrix,
        riasec_inter_cosines=riasec_cosine,
        trait_names=np.array(trait_names),
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Comparison Summary: {args.model_id}")
    print(f"{'='*60}")

    print("\nRIASEC vector alignment with Assistant Axis (PC1):")
    for trait in trait_names:
        cos = riasec_pc1_cosines[trait]
        print(f"  {trait:15s}: cosine = {cos:+.4f}")

    print("\nRIASEC inter-trait cosine similarities:")
    header = "              " + "".join(f"{t[:6]:>8s}" for t in trait_names)
    print(header)
    for i, t1 in enumerate(trait_names):
        row = f"  {t1[:12]:12s}" + "".join(f"{riasec_cosine[i,j]:8.3f}" for j in range(len(trait_names)))
        print(row)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
