"""
full_analysis_sans_mythical.py

Same pipeline as `full_analysis.py`, but first auto-detects the "mythical"
cluster of role vectors that sits outside the main cluster, removes those
roles, and then re-runs the PCA / t-SNE / NMF analyses on the cleaned set.

Adds a vector-distance analysis section (pairwise cosine & Euclidean
distances, distance to assistant, distance to centroid) on the cleaned set
and, for comparison, on the full set.

Outlier cluster detection:
    1. Project role vectors to a low-dimensional PCA space.
    2. Fit KMeans(k=2) on the projections to split into two clusters.
    3. Treat the smaller cluster as the outlier ("mythical") cluster.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.colors import TwoSlopeNorm
from safetensors.numpy import load_file
import os
from adjustText import adjust_text
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

FIG_DIR = "analysis/geometry/figures_sans_mythical"
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load vectors
# ------------------------------------------------------------------
base_path = "persona_data/model_inits"
vector_list = os.listdir(base_path)
layer_number = 16
num_questions_used = 50

vectors = {}
print("Number of files:", len(vector_list))

for vector in vector_list:
    full_path = os.path.join(base_path, vector)
    if not os.path.isdir(full_path):
        continue
    files = os.listdir(full_path)
    safetensor_files = [
        f for f in files
        if f.endswith(".safetensors") and f"count{num_questions_used}" in f
    ]
    if not safetensor_files:
        continue
    file = safetensor_files[0]
    data = load_file(os.path.join(full_path, file))
    key = list(data.keys())[0]
    vector_name = vector.replace("_persona_initialization", "")
    vectors[vector_name] = (
        data[key].astype(np.float32)[layer_number - 1].squeeze()
    )
print("Number of vectors extracted:", len(vectors))


# ------------------------------------------------------------------
# Detect & remove the mythical / outlier cluster.
#
# We cluster in the same 2D t-SNE space that `tsne_layer_16.png` is drawn
# from — that way the exclusion matches what's visually separated in that
# figure. KMeans in the raw / high-dim PCA space tended to split the main
# blob instead of isolating the outlier clump.
# ------------------------------------------------------------------
assistant_vector_full = vectors["assistant"]
role_names_full = [n for n in vectors.keys() if n != "assistant"]
role_matrix_full = np.array([vectors[n] for n in role_names_full])

# Same centering convention used downstream for the t-SNE figure.
all_vectors_raw_full = np.vstack(
    [role_matrix_full, assistant_vector_full.reshape(1, -1)]
)
all_vectors_centered_full = all_vectors_raw_full - all_vectors_raw_full.mean(axis=0)

tsne_cluster = TSNE(
    n_components=2,
    perplexity=min(5, len(role_names_full) + 1 - 1),
    random_state=42,
)
tsne_all = tsne_cluster.fit_transform(all_vectors_centered_full)
# Drop the assistant row (last) before clustering the role vectors.
tsne_roles = tsne_all[:-1]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(tsne_roles)

counts = np.bincount(cluster_labels)
outlier_label = int(np.argmin(counts))
main_label = int(np.argmax(counts))
outlier_mask = cluster_labels == outlier_label

print(
    f"[detect] KMeans(k=2) on t-SNE 2D: "
    f"main cluster={counts[main_label]} roles, "
    f"outlier cluster={counts[outlier_label]} roles."
)

# Save the cluster diagnostic so you can sanity-check against tsne_layer_16.png
fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(
    tsne_roles[cluster_labels == main_label, 0],
    tsne_roles[cluster_labels == main_label, 1],
    c="tab:blue", label=f"kept ({counts[main_label]})",
    s=40, edgecolors="k", linewidths=0.3, alpha=0.85,
)
ax.scatter(
    tsne_roles[cluster_labels == outlier_label, 0],
    tsne_roles[cluster_labels == outlier_label, 1],
    c="tab:red", label=f"excluded ({counts[outlier_label]})",
    s=40, edgecolors="k", linewidths=0.3, alpha=0.85,
)
ax.scatter(
    tsne_all[-1, 0], tsne_all[-1, 1],
    c="black", marker="*", s=200, label="assistant",
)
texts = [
    ax.text(tsne_roles[i, 0], tsne_roles[i, 1],
            os.path.splitext(n)[0], fontsize=5, alpha=0.85)
    for i, n in enumerate(role_names_full)
]
adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5),
    expand=(1.5, 1.5), force_text=(0.8, 0.8), force_points=(0.3, 0.3),
)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title(f"Mythical cluster detection on t-SNE (Layer {layer_number})")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, f"cluster_detection_tsne_layer_{layer_number}.png"),
    dpi=150,
)
plt.close()

excluded = [role_names_full[i] for i in np.where(outlier_mask)[0]]
kept = [role_names_full[i] for i in np.where(~outlier_mask)[0]]

print(f"[detect] Excluded mythical cluster ({len(excluded)} roles):")
for n in sorted(excluded):
    print(f"          - {n}")
print(f"[detect] Kept {len(kept)} role vectors (+ assistant).")

# Save membership so it can be inspected
with open(os.path.join(FIG_DIR, "cluster_membership.txt"), "w") as f:
    f.write("# Excluded (mythical cluster)\n")
    for n in sorted(excluded):
        f.write(f"{n}\n")
    f.write("\n# Kept (main cluster)\n")
    for n in sorted(kept):
        f.write(f"{n}\n")

# Build filtered vectors dict
vectors_full = dict(vectors)  # keep a reference before mutation
vectors = {n: vectors[n] for n in (["assistant"] + kept)}

# ------------------------------------------------------------------
# Line plot of kept vectors
# ------------------------------------------------------------------
plt.figure(figsize=(10, 10))
for name, vector in vectors.items():
    plt.plot(range(len(vector)), vector, label=os.path.splitext(name)[0])
plt.legend(fontsize=5, ncol=3)
plt.title(f"Role Vectors — mythical cluster removed (Layer {layer_number})")
plt.savefig(os.path.join(FIG_DIR, f"layer_{layer_number}_lines.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------
# Heatmap of kept vectors
# ------------------------------------------------------------------
sorted_names = sorted(vectors.keys())
matrix = np.array([vectors[n] for n in sorted_names])
fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_names) * 0.12)))
im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=5)
ax.set_xlabel("Dimension")
ax.set_ylabel("Role")
ax.set_title(
    f"Activation Vectors — mythical cluster removed (Layer {layer_number})"
)
fig.colorbar(im, ax=ax, shrink=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"layer_{layer_number}_heatmap.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------
# PCA with Assistant-Axis as dim 1
# ------------------------------------------------------------------
assistant_vector = vectors.pop("assistant")
names = list(vectors.keys())
role_matrix = np.array([vectors[n] for n in names])

assistant_axis = assistant_vector - role_matrix.mean(axis=0)
assistant_axis = assistant_axis / np.linalg.norm(assistant_axis)

all_vectors_raw = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
all_names = [os.path.splitext(n)[0] for n in names] + ["assistant"]
all_vectors_centered = all_vectors_raw - all_vectors_raw.mean(axis=0)

dim1 = all_vectors_centered @ assistant_axis
residual = all_vectors_centered - np.outer(dim1, assistant_axis)
pca_resid = PCA(n_components=2)
dims23 = pca_resid.fit_transform(residual)
projected = np.column_stack([dim1, dims23])

proj_centered = dim1 - dim1.mean()
proj_spread = np.max(np.abs(proj_centered))
proj_normalized = proj_centered / proj_spread if proj_spread > 0 else proj_centered

vmin = min(proj_normalized.min(), -1e-8)
vmax = max(proj_normalized.max(), 1e-8)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                c=proj_normalized, cmap='coolwarm_r', norm=norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

_tree = cKDTree(projected)
_k = min(6, len(all_names) - 1)
_dd, _ii = _tree.query(projected, k=_k + 1)
_scale = np.ptp(projected, axis=0) * 0.025
for i, name in enumerate(all_names):
    neighbors = projected[_ii[i, 1:]]
    direction = projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    direction = direction / norm_d if norm_d > 0 else np.array([1, 0, 0])
    offset = direction * _scale
    ax.text(projected[i, 0] + offset[0], projected[i, 1] + offset[1],
            projected[i, 2] + offset[2], name,
            fontsize=4.5, ha='center', va='center', alpha=0.85)

t = np.linspace(-1, 1, 2) * max(np.abs(projected[:, 0])) * 1.2
ax.plot(t, [0, 0], [0, 0], 'k--', linewidth=1.5, label="Assistant Axis")
ax.set_xlabel("Assistant Axis")
ax.set_ylabel(f"Residual PC1 ({pca_resid.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_zlabel(f"Residual PC2 ({pca_resid.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(
    f"Role Vectors in Persona Space — mythical removed (Layer {layer_number})"
)
ax.legend(loc='upper left')
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, f"pca3d_assistantaxis_layer_{layer_number}.png"),
    dpi=150,
)
plt.close()

# ------------------------------------------------------------------
# Standard PCA (3D)
# ------------------------------------------------------------------
pca = PCA(n_components=3)
pca_projected = pca.fit_transform(all_vectors_centered)

pca_axis_proj = pca.components_ @ assistant_axis
pca_axis_proj = pca_axis_proj / np.linalg.norm(pca_axis_proj)
pca_proj = pca_projected @ pca_axis_proj
pca_proj_centered = pca_proj - pca_proj.mean()
pca_proj_spread = np.max(np.abs(pca_proj_centered))
pca_proj_normalized = (
    pca_proj_centered / pca_proj_spread if pca_proj_spread > 0 else pca_proj_centered
)
pca_vmin = min(pca_proj_normalized.min(), -1e-8)
pca_vmax = max(pca_proj_normalized.max(), 1e-8)
pca_norm = TwoSlopeNorm(vmin=pca_vmin, vcenter=0, vmax=pca_vmax)

fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2],
                c=pca_proj_normalized, cmap='coolwarm_r', norm=pca_norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

_tree = cKDTree(pca_projected)
_dd, _ii = _tree.query(pca_projected, k=_k + 1)
_scale = np.ptp(pca_projected, axis=0) * 0.025
for i, name in enumerate(all_names):
    neighbors = pca_projected[_ii[i, 1:]]
    direction = pca_projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    direction = direction / norm_d if norm_d > 0 else np.array([1, 0, 0])
    offset = direction * _scale
    ax.text(pca_projected[i, 0] + offset[0], pca_projected[i, 1] + offset[1],
            pca_projected[i, 2] + offset[2], name,
            fontsize=4.5, ha='center', va='center', alpha=0.85)

t = np.linspace(-1, 1, 2) * max(np.abs(pca_projected[:, 0])) * 1.2
ax.plot(t * pca_axis_proj[0], t * pca_axis_proj[1], t * pca_axis_proj[2],
        'k--', linewidth=1.5, label="Assistant Axis")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title(
    f"Standard PCA — mythical removed (Layer {layer_number})"
)
ax.legend(loc='upper left')
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"pca3d_layer_{layer_number}.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------
# t-SNE 2D and 3D
# ------------------------------------------------------------------
tsne_names = all_names
tsne = TSNE(
    n_components=2,
    perplexity=min(5, len(tsne_names) - 1),
    random_state=42,
)
tsne_projected = tsne.fit_transform(all_vectors_centered)

tsne_proj = all_vectors_raw @ assistant_axis
tsne_proj_centered = tsne_proj - tsne_proj.mean()
tsne_proj_spread = np.max(np.abs(tsne_proj_centered))
tsne_proj_normalized = (
    tsne_proj_centered / tsne_proj_spread if tsne_proj_spread > 0 else tsne_proj_centered
)
tsne_vmin = min(tsne_proj_normalized.min(), -1e-8)
tsne_vmax = max(tsne_proj_normalized.max(), 1e-8)
tsne_norm = TwoSlopeNorm(vmin=tsne_vmin, vcenter=0, vmax=tsne_vmax)

fig, ax = plt.subplots(figsize=(16, 11))
sc = ax.scatter(tsne_projected[:, 0], tsne_projected[:, 1],
                c=tsne_proj_normalized, cmap='coolwarm_r', norm=tsne_norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)
texts = [
    ax.text(tsne_projected[i, 0], tsne_projected[i, 1], name,
            fontsize=5.5, alpha=0.85)
    for i, name in enumerate(tsne_names)
]
adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5),
    expand=(1.5, 1.5), force_text=(0.8, 0.8), force_points=(0.3, 0.3),
)
fig.colorbar(sc, ax=ax, label="Projection onto Assistant Axis", shrink=0.8)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title(f"t-SNE — mythical removed (Layer {layer_number})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"tsne_layer_{layer_number}.png"), dpi=150)
plt.close()

tsne3d = TSNE(
    n_components=3,
    perplexity=min(5, len(tsne_names) - 1),
    random_state=42,
)
tsne3d_projected = tsne3d.fit_transform(all_vectors_centered)

fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(tsne3d_projected[:, 0], tsne3d_projected[:, 1],
                tsne3d_projected[:, 2],
                c=tsne_proj_normalized, cmap='coolwarm_r', norm=tsne_norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

_tree_t3 = cKDTree(tsne3d_projected)
_dd_t3, _ii_t3 = _tree_t3.query(tsne3d_projected, k=_k + 1)
_scale_t3 = np.ptp(tsne3d_projected, axis=0) * 0.025
for i, name in enumerate(tsne_names):
    neighbors = tsne3d_projected[_ii_t3[i, 1:]]
    direction = tsne3d_projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    direction = direction / norm_d if norm_d > 0 else np.array([1, 0, 0])
    offset = direction * _scale_t3
    ax.text(tsne3d_projected[i, 0] + offset[0],
            tsne3d_projected[i, 1] + offset[1],
            tsne3d_projected[i, 2] + offset[2], name,
            fontsize=4.5, ha='center', va='center', alpha=0.85)

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_zlabel("t-SNE 3")
ax.set_title(f"3D t-SNE — mythical removed (Layer {layer_number})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"tsne3d_layer_{layer_number}.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------
# NMF on the cleaned set
# ------------------------------------------------------------------
n_components = 6
nmf_input = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
nmf_input_shifted = nmf_input - nmf_input.min()
nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
W = nmf.fit_transform(nmf_input_shifted)
H = nmf.components_
nmf_names = all_names

fig, axes = plt.subplots(n_components, 1, figsize=(14, 2 * n_components), sharex=True)
for i in range(n_components):
    axes[i].plot(H[i], linewidth=0.5)
    axes[i].set_ylabel(f"Basis {i+1}")
    axes[i].set_xlim(0, H.shape[1])
axes[-1].set_xlabel("Dimension")
fig.suptitle(
    f"NMF Basis Vectors — mythical removed (Layer {layer_number}, k={n_components})",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"nmf_bases_layer_{layer_number}.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, max(6, len(nmf_names) * 0.4)))
im = ax.imshow(W, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(nmf_names)))
ax.set_yticklabels(nmf_names, fontsize=8)
ax.set_xticks(range(n_components))
ax.set_xticklabels([f"Basis {i+1}" for i in range(n_components)])
ax.set_xlabel("NMF Component"); ax.set_ylabel("Role")
ax.set_title(
    f"Role Coefficients on NMF Bases — mythical removed "
    f"(Layer {layer_number}, k={n_components})"
)
fig.colorbar(im, ax=ax, label="Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"nmf_roles_layer_{layer_number}.png"), dpi=150)
plt.close()

# ==================================================================
# Vector-distance analysis
# ==================================================================
def _compute_distances(vec_dict, label):
    """Compute standard distance metrics for a set of (role + assistant) vectors."""
    if "assistant" not in vec_dict:
        raise ValueError("assistant vector required")
    order = ["assistant"] + sorted(n for n in vec_dict if n != "assistant")
    M = np.array([vec_dict[n] for n in order])

    # Normalize for cosine
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    cos_sim = Mn @ Mn.T
    cos_dist = 1.0 - cos_sim
    eucl_dist = squareform(pdist(M, metric="euclidean"))

    # Distance to assistant
    assistant_idx = order.index("assistant")
    d_to_assistant_cos = cos_dist[assistant_idx]
    d_to_assistant_eucl = eucl_dist[assistant_idx]

    # Distance to centroid of roles (excluding assistant)
    role_mat = np.array(
        [vec_dict[n] for n in order if n != "assistant"]
    )
    centroid = role_mat.mean(axis=0)
    d_to_centroid = np.linalg.norm(M - centroid, axis=1)

    # Save a numeric summary
    summary_path = os.path.join(FIG_DIR, f"distance_summary_{label}.tsv")
    with open(summary_path, "w") as f:
        f.write(
            "role\tcos_dist_to_assistant\teucl_dist_to_assistant\t"
            "dist_to_centroid\n"
        )
        for i, n in enumerate(order):
            f.write(
                f"{n}\t{d_to_assistant_cos[i]:.6f}\t"
                f"{d_to_assistant_eucl[i]:.6f}\t{d_to_centroid[i]:.6f}\n"
            )
    print(f"[distances:{label}] wrote {summary_path}")

    # Heatmaps
    for mat, metric in [(cos_dist, "cosine"), (eucl_dist, "euclidean")]:
        fig, ax = plt.subplots(
            figsize=(max(8, len(order) * 0.18), max(7, len(order) * 0.18))
        )
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=4.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=4.5)
        ax.set_title(
            f"Pairwise {metric} distance ({label}, layer {layer_number})"
        )
        fig.colorbar(im, ax=ax, shrink=0.6, label=f"{metric} distance")
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"dist_{metric}_{label}_layer_{layer_number}.png"),
            dpi=150,
        )
        plt.close()

    # Bar chart: distance-to-assistant ordered
    sort_idx = np.argsort(d_to_assistant_cos)
    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.2), 6))
    ax.bar(
        range(len(order)),
        d_to_assistant_cos[sort_idx],
        color="steelblue",
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([order[i] for i in sort_idx], rotation=90, fontsize=5)
    ax.set_ylabel("Cosine distance to assistant")
    ax.set_title(
        f"Cosine distance to assistant ({label}, layer {layer_number})"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"dist_to_assistant_{label}_layer_{layer_number}.png"),
        dpi=150,
    )
    plt.close()

    return {
        "order": order,
        "cos_dist": cos_dist,
        "eucl_dist": eucl_dist,
        "d_to_assistant_cos": d_to_assistant_cos,
        "d_to_assistant_eucl": d_to_assistant_eucl,
        "d_to_centroid": d_to_centroid,
    }


cleaned_vecs = {"assistant": assistant_vector}
cleaned_vecs.update({n: vectors_full[n] for n in kept})

full_vecs = {n: vectors_full[n] for n in (["assistant"] + role_names_full)}

stats_clean = _compute_distances(cleaned_vecs, label="sans_mythical")
stats_full = _compute_distances(full_vecs, label="full")

# Aggregate comparison: mean pairwise distance
def _mean_offdiag(M):
    n = M.shape[0]
    return (M.sum() - np.trace(M)) / (n * (n - 1))

print(
    f"[distances] mean pairwise cosine distance: "
    f"full={_mean_offdiag(stats_full['cos_dist']):.4f}, "
    f"sans_mythical={_mean_offdiag(stats_clean['cos_dist']):.4f}"
)
print(
    f"[distances] mean pairwise euclidean distance: "
    f"full={_mean_offdiag(stats_full['eucl_dist']):.4f}, "
    f"sans_mythical={_mean_offdiag(stats_clean['eucl_dist']):.4f}"
)
print(
    f"[distances] mean cosine distance to assistant: "
    f"full={stats_full['d_to_assistant_cos'][1:].mean():.4f}, "
    f"sans_mythical={stats_clean['d_to_assistant_cos'][1:].mean():.4f}"
)

print(f"\nAll figures written to: {FIG_DIR}")
