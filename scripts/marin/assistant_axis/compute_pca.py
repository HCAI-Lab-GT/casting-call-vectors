#!/usr/bin/env python
"""
Compute PCA on role vectors to find the Assistant Axis.

Based on methodology from arXiv 2601.10387 (Christina Lu et al):
  - PC1 = "Assistant Axis" (direction of maximum variance among role vectors)
  - Default assistant should have highest PC1 projection
  - Also computes simple axis: mean(default) - mean(all_roles) for comparison

Usage:
  python scripts/marin/assistant_axis/compute_pca.py
  python scripts/marin/assistant_axis/compute_pca.py --model_id marin-community/marin-8b-instruct
  python scripts/marin/assistant_axis/compute_pca.py --n_components 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from pvx import setup_logging

logger = setup_logging(name="marin-compute-pca")

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Compute PCA on role vectors.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--vectors_dir", type=str, default="./data/assistant_axis/role_vectors/")
    parser.add_argument("--output_dir", type=str, default="./data/assistant_axis/pca/")
    parser.add_argument("--n_components", type=int, default=20, help="Number of PCA components.")
    args = parser.parse_args()

    safe_model = args.model_id.replace("/", "__")
    vectors_path = Path(args.vectors_dir) / f"{safe_model}.pt"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Run extract_activations.py first. Missing: {vectors_path}")

    # Load role vectors
    data = torch.load(vectors_path, weights_only=False)
    vectors = data["vectors"].numpy()  # (n_roles, hidden_dim)
    role_names = data["role_names"]
    role_categories = data["role_categories"]
    layer = data["layer"]

    n_roles, hidden_dim = vectors.shape
    logger.info("Loaded %d role vectors of dim %d", n_roles, hidden_dim)

    # Find default assistant index
    default_idx = None
    for i, name in enumerate(role_names):
        if name == "default_assistant":
            default_idx = i
            break

    if default_idx is None:
        logger.warning("No 'default_assistant' role found. PCA will proceed but validation skipped.")

    # Compute mean and center
    mean_vector = vectors.mean(axis=0)
    centered = vectors - mean_vector

    # PCA
    n_components = min(args.n_components, n_roles, hidden_dim)
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(centered)  # (n_roles, n_components)

    # Results
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    logger.info("PCA Results:")
    logger.info("  PC1 explained variance: %.2f%%", explained_variance[0] * 100)
    for k in [5, 10, 15, 20]:
        if k <= len(cumulative_variance):
            logger.info("  Top %d components explain: %.2f%%", k, cumulative_variance[k-1] * 100)

    # Validate: default assistant should have highest PC1 projection
    if default_idx is not None:
        pc1_projections = projections[:, 0]
        default_pc1 = pc1_projections[default_idx]
        max_pc1_idx = np.argmax(pc1_projections)
        max_pc1_name = role_names[max_pc1_idx]

        # PC1 direction might be flipped; check if default is at an extreme
        default_rank_high = np.sum(pc1_projections >= default_pc1)
        default_rank_low = np.sum(pc1_projections <= default_pc1)
        is_extreme = (default_rank_high <= 3) or (default_rank_low <= 3)

        if max_pc1_name == "default_assistant":
            logger.info("  PASS: default_assistant has highest PC1 projection (%.4f)", default_pc1)
        elif is_extreme:
            logger.info(
                "  NOTE: default_assistant at PC1 extreme (projection=%.4f, rank_high=%d, rank_low=%d). "
                "PC1 might be flipped; highest is %s (%.4f)",
                default_pc1, default_rank_high, default_rank_low,
                max_pc1_name, pc1_projections[max_pc1_idx],
            )
        else:
            logger.warning(
                "  WARN: default_assistant NOT at PC1 extreme (projection=%.4f, rank=%d/%d). "
                "Highest: %s (%.4f)",
                default_pc1, default_rank_high, n_roles, max_pc1_name, pc1_projections[max_pc1_idx],
            )

    # Compute simple axis: mean(default) - mean(all)
    simple_axis = None
    cosine_simple_vs_pc1 = None
    if default_idx is not None:
        default_vector = vectors[default_idx]
        simple_axis = default_vector - mean_vector
        simple_axis_norm = simple_axis / (np.linalg.norm(simple_axis) + 1e-10)
        pc1_direction = pca.components_[0]
        pc1_norm = pc1_direction / (np.linalg.norm(pc1_direction) + 1e-10)
        cosine_simple_vs_pc1 = float(np.dot(simple_axis_norm, pc1_norm))
        logger.info("  Cosine(simple_axis, PC1): %.4f", cosine_simple_vs_pc1)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PCA components and axis as torch tensors
    pca_data = {
        "components": torch.from_numpy(pca.components_.copy()),  # (n_components, hidden_dim)
        "explained_variance_ratio": torch.from_numpy(explained_variance.copy()),
        "cumulative_variance": torch.from_numpy(cumulative_variance.copy()),
        "mean_vector": torch.from_numpy(mean_vector.copy()),
        "projections": torch.from_numpy(projections.copy()),  # (n_roles, n_components)
        "role_names": role_names,
        "role_categories": role_categories,
        "model_id": args.model_id,
        "layer": layer,
        "n_components": n_components,
    }

    if simple_axis is not None:
        pca_data["simple_axis"] = torch.from_numpy(simple_axis.copy())
        pca_data["cosine_simple_vs_pc1"] = cosine_simple_vs_pc1

    torch.save(pca_data, output_dir / f"{safe_model}_pca.pt")

    # Save human-readable summary
    summary = {
        "model_id": args.model_id,
        "layer": layer,
        "n_roles": n_roles,
        "hidden_dim": hidden_dim,
        "n_components": n_components,
        "explained_variance_pct": [round(float(v) * 100, 2) for v in explained_variance],
        "cumulative_variance_pct": [round(float(v) * 100, 2) for v in cumulative_variance],
        "role_pc1_projections": {
            name: round(float(projections[i, 0]), 4) for i, name in enumerate(role_names)
        },
        "cosine_simple_vs_pc1": cosine_simple_vs_pc1,
    }
    with open(output_dir / f"{safe_model}_pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPCA Summary ({args.model_id}):")
    print(f"  PC1 explains {explained_variance[0]*100:.1f}% of variance")
    top4 = cumulative_variance[min(3, len(cumulative_variance)-1)]
    print(f"  Top 4 components explain {top4*100:.1f}%")
    if cosine_simple_vs_pc1 is not None:
        print(f"  Cosine(simple_axis, PC1) = {cosine_simple_vs_pc1:.4f}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
