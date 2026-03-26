# in trait_coverage_inits, there are json files corresponding to traits (personas).
# each trait subfolder has a varying structure. one of the files it contains or in a file in a folder it contains has a file called ..._count40.json. 
# these are the important files. each file has "prompt_persona_vector" key, which is a list of vectors for each layer.
# choose the 16th vector (corresponding to the 16th layer).
# do a PCA on these vectors to find the principal components (maybe like the first 20 principal components?) and see which traits are closest to each component.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import os
import glob

os.makedirs("figures/trait-figures", exist_ok=True)

# ============================================================
# Load trait vectors from trait_coverage_inits
# ============================================================
trait_dir = "trait_coverage_inits"
vectors = {}

for count40_path in glob.glob(os.path.join(trait_dir, "**", "*count40.json"), recursive=True):
    data = json.load(open(count40_path))
    trait_name = data.get("concept", os.path.basename(count40_path))
    vec = np.array(data["prompt_persona_vector"]).squeeze().astype(np.float32)
    if vec.shape != (4096,):
        print(f"Warning: {trait_name} has shape {vec.shape}, skipping")
        continue
    vectors[trait_name] = vec

print(f"Loaded {len(vectors)} trait vectors")

names = sorted(vectors.keys())
matrix = np.array([vectors[n] for n in names])

# Mean-center
matrix_centered = matrix - matrix.mean(axis=0)

# ============================================================
# PCA (20 components)
# ============================================================
n_components = 20
pca = PCA(n_components=n_components)
projected = pca.fit_transform(matrix_centered)

print(f"\nExplained variance per component:")
cumulative = 0
for i, ev in enumerate(pca.explained_variance_ratio_):
    cumulative += ev
    print(f"  PC{i+1:2d}: {ev*100:5.2f}%  (cumulative: {cumulative*100:5.2f}%)")

# ============================================================
# For each PC, find top traits (highest and lowest projections)
# ============================================================
n_top = 10
print(f"\nTop {n_top} traits per principal component (+ and - directions):")
for i in range(n_components):
    scores = projected[:, i]
    top_pos = np.argsort(-scores)[:n_top]
    top_neg = np.argsort(scores)[:n_top]
    
    print(f"\n--- PC{i+1} ({pca.explained_variance_ratio_[i]*100:.2f}%) ---")
    print(f"  + direction: {', '.join(f'{names[j]} ({scores[j]:.2f})' for j in top_pos)}")
    print(f"  - direction: {', '.join(f'{names[j]} ({scores[j]:.2f})' for j in top_neg)}")

# ============================================================
# Extract PC directions as a list of vectors
# ============================================================
pc_vectors = [pca.components_[i] for i in range(n_components)]  # list of 20 arrays, each (4096,)

# ============================================================
# Map each trait to the PC it correlates with most (by absolute projection)
# ============================================================
trait_to_best_pc = {}
for i, name in enumerate(names):
    abs_scores = np.abs(projected[i])
    best_pc = int(np.argmax(abs_scores)) + 1  # 1-indexed
    trait_to_best_pc[name] = {
        "best_pc": best_pc,
        "score": float(projected[i, best_pc - 1]),
        "abs_score": float(abs_scores[best_pc - 1]),
    }

# Print summary grouped by PC
from collections import defaultdict
pc_to_traits = defaultdict(list)
for trait, info in trait_to_best_pc.items():
    pc_to_traits[info["best_pc"]].append((trait, info["score"]))

print(f"\n{'='*60}")
print(f"Trait -> Best PC mapping ({len(trait_to_best_pc)} traits)")
print(f"{'='*60}")
for pc_num in sorted(pc_to_traits.keys()):
    traits = sorted(pc_to_traits[pc_num], key=lambda x: -abs(x[1]))
    trait_str = ", ".join(f"{t} ({s:+.2f})" for t, s in traits)
    print(f"  PC{pc_num:2d} ({len(traits):3d} traits): {trait_str}")

# ============================================================
# Plot: explained variance
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_ * 100)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.set_title("PCA of Trait Vectors — Explained Variance")
ax.set_xticks(range(1, n_components + 1))
plt.tight_layout()
plt.savefig("figures/trait-figures/trait_pca_variance.png", dpi=150)
plt.show()

# ============================================================
# Plot: top traits per component (horizontal bar chart)
# ============================================================
fig, axes = plt.subplots(4, 5, figsize=(28, 20))
for i, ax in enumerate(axes.flat):
    if i >= n_components:
        ax.axis('off')
        continue
    scores = projected[:, i]
    top_idx = np.argsort(np.abs(scores))[-n_top:][::-1]
    top_names = [names[j] for j in top_idx]
    top_scores = [scores[j] for j in top_idx]
    colors = ['#d73027' if s < 0 else '#4575b4' for s in top_scores]
    
    ax.barh(range(len(top_names)), top_scores, color=colors)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=7)
    ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)", fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='k', linewidth=0.5)

plt.suptitle("Top Traits per Principal Component", fontsize=14)
plt.tight_layout()
plt.savefig("figures/trait-figures/trait_pca_loadings.png", dpi=150)
plt.show()




# ============================================================
# Plot role vectors along user-chosen trait axes
# ============================================================
from safetensors.numpy import load_file
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

# --- Choose sets of 3 traits as axes ---
input_axes_list = [
    ["critical", "evil", "serene"],
    ["nurturing", "theatrical", "erudite"],
    ["hostile", "playful", "analytical"],
    ["diplomatic", "dramatic", "technical"],
    ["cynical", "whimsical", "methodical"],
    ["assertive", "poetic", "practical"],
    ["empathetic", "enigmatic", "stoic"],
]

# --- Load role vectors from model_layer_inits ---
role_dir = "model_layer_inits"
layer_number = 16
num_questions_used = 50
role_vectors = {}

for folder in os.listdir(role_dir):
    full_path = os.path.join(role_dir, folder)
    if not os.path.isdir(full_path):
        continue
    files = os.listdir(full_path)
    safetensor_files = [f for f in files if f.endswith(".safetensors") and f"count{num_questions_used}" in f]
    if not safetensor_files:
        continue
    data = load_file(os.path.join(full_path, safetensor_files[0]))
    key = list(data.keys())[0]
    role_name = folder.replace("_persona_initialization", "")
    role_vectors[role_name] = data[key].astype(np.float32)[layer_number - 1].squeeze()

print(f"\nLoaded {len(role_vectors)} role vectors for trait-axis projection")

# --- Build role name list and matrix (shared across all axis combos) ---
role_names = sorted(role_vectors.keys())
role_matrix = np.array([role_vectors[n] for n in role_names])

for input_axes in input_axes_list:
    print(f"\n--- Plotting trait axes: {input_axes} ---")

    # --- Build trait axis directions (normalized) ---
    trait_axes = []
    for trait in input_axes:
        if trait not in vectors:
            print(f"Warning: trait '{trait}' not found in trait vectors, skipping")
            continue
        axis = vectors[trait].copy()
        axis = axis / np.linalg.norm(axis)
        trait_axes.append((trait, axis))

    if len(trait_axes) != 3:
        print(f"Need exactly 3 valid trait axes, got {len(trait_axes)} — skipping this combo")
        continue

    # --- Project each role vector onto the 3 trait axes ---
    projections = np.column_stack([role_matrix @ ax for _, ax in trait_axes])  # (N, 3)

    # --- 3D scatter plot ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by magnitude of projection (distance from origin)
    magnitudes = np.linalg.norm(projections, axis=1)
    sc = ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2],
                    c=magnitudes, cmap='viridis',
                    s=60, edgecolors='k', linewidths=0.3, alpha=0.85)

    # Label points with offsets pushed away from neighbors
    tree = cKDTree(projections)
    k = min(6, len(role_names) - 1)
    dd, ii = tree.query(projections, k=k + 1)
    scale = np.ptp(projections, axis=0) * 0.025
    for i, name in enumerate(role_names):
        neighbors = projections[ii[i, 1:]]
        direction = projections[i] - neighbors.mean(axis=0)
        norm_d = np.linalg.norm(direction)
        if norm_d > 0:
            direction = direction / norm_d
        else:
            direction = np.array([1, 0, 0])
        offset = direction * scale
        ax.text(projections[i, 0] + offset[0], projections[i, 1] + offset[1],
                projections[i, 2] + offset[2], name,
                fontsize=4.5, ha='center', va='center', alpha=0.85)

    ax.set_xlabel(f"← less {trait_axes[0][0]}  |  more {trait_axes[0][0]} →")
    ax.set_ylabel(f"← less {trait_axes[1][0]}  |  more {trait_axes[1][0]} →")
    ax.set_zlabel(f"← less {trait_axes[2][0]}  |  more {trait_axes[2][0]} →")
    ax.set_title(f"Role Vectors in Trait Space: {', '.join(input_axes)}")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Magnitude")

    plt.tight_layout()
    plt.savefig(f"figures/trait-figures/roles_in_trait_space_{'_'.join(input_axes)}.png", dpi=150)
    plt.show()

print("\nDone.")


