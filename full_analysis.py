import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from matplotlib.colors import TwoSlopeNorm
from safetensors.numpy import load_file
import os
import sys
from adjustText import adjust_text
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from role_list import role_mystical_ratings, role_assistant_ratings

os.makedirs("figures/role-figures", exist_ok=True)

# Load vectors
vector_list = os.listdir("model_layer_inits")
layer_number = 16
num_questions_used = 50

base_path = "model_layer_inits"
vectors = {}
print("Number of files:", len(vector_list))

for vector in vector_list:
    full_path = os.path.join(base_path, vector)
    if not os.path.isdir(full_path):
        continue
    files = os.listdir(full_path)
    safetensor_files = [f for f in files if f.endswith(".safetensors") and f"count{num_questions_used}" in f]
    if not safetensor_files:
        # print(f"Skipping {vector}: no safetensors files found")
        continue
    file = safetensor_files[0]
    data = load_file(os.path.join(full_path, file))
    key = list(data.keys())[0]
    vector_name = vector.replace("_persona_initialization", "")
    vectors[vector_name] = data[key].astype(np.float32)[layer_number - 1].squeeze()
print("Number of vectors extracted:", len(vectors))


# Plot vectors
plt.figure(figsize=(10, 10))
for name, vector in vectors.items():
    plt.plot([i for i in range(len(vector))], vector, label=os.path.splitext(name)[0])
plt.legend()
plt.savefig("figures/role-figures/layer_" + str(layer_number) + ".png")
plt.show()



# Plot vectors as heatmap
sorted_names = sorted(vectors.keys())
matrix = np.array([vectors[n] for n in sorted_names])
fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_names) * 0.12)))
im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=5)
ax.set_xlabel("Dimension")
ax.set_ylabel("Role")
ax.set_title(f"Activation Vectors (Layer {layer_number})")
fig.colorbar(im, ax=ax, shrink=0.6)
plt.tight_layout()
plt.savefig("figures/role-figures/layer_" + str(layer_number) + ".png", dpi=150)
plt.show()




# pca
# Separate assistant from role vectors
assistant_vector = vectors.pop("assistant")
names = list(vectors.keys())
role_matrix = np.array([vectors[n] for n in names])

# Compute Assistant Axis: mean difference between assistant and the role activations
assistant_axis = assistant_vector - role_matrix.mean(axis=0)
assistant_axis = assistant_axis / np.linalg.norm(assistant_axis)

# Mean-subtract all vectors before PCA
all_vectors_raw = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
all_names = [os.path.splitext(n)[0] for n in names] + ["assistant"]
all_vectors_centered = all_vectors_raw - all_vectors_raw.mean(axis=0)

# Custom 3D basis: Assistant Axis as dim 1, PCA on residual for dims 2 & 3
# Dim 1: projection onto assistant axis
dim1 = all_vectors_centered @ assistant_axis  # (N,)

# Remove assistant axis component, then PCA on the residual for 2 more dims
residual = all_vectors_centered - np.outer(dim1, assistant_axis)
pca_resid = PCA(n_components=2)
dims23 = pca_resid.fit_transform(residual)

projected = np.column_stack([dim1, dims23])

# Color by projection onto Assistant Axis (dim 1)
proj_centered = dim1 - dim1.mean()
proj_spread = np.max(np.abs(proj_centered))
proj_normalized = proj_centered / proj_spread if proj_spread > 0 else proj_centered

cmap = plt.cm.coolwarm_r
vmin = min(proj_normalized.min(), -1e-8)
vmax = max(proj_normalized.max(), 1e-8)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# 3D scatter plot
fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                c=proj_normalized, cmap='coolwarm_r', norm=norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

# Label all points with offsets pushed away from nearest neighbors
_tree = cKDTree(projected)
_k = min(6, len(all_names) - 1)
_dd, _ii = _tree.query(projected, k=_k + 1)  # includes self
_scale = np.ptp(projected, axis=0) * 0.025
for i, name in enumerate(all_names):
    neighbors = projected[_ii[i, 1:]]  # exclude self
    direction = projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    if norm_d > 0:
        direction = direction / norm_d
    else:
        direction = np.array([1, 0, 0])
    offset = direction * _scale
    ax.text(projected[i, 0] + offset[0], projected[i, 1] + offset[1],
            projected[i, 2] + offset[2], name,
            fontsize=4.5, ha='center', va='center', alpha=0.85)

# Assistant Axis line along dim 1 (the x-axis)
t = np.linspace(-1, 1, 2) * max(np.abs(projected[:, 0])) * 1.2
ax.plot(t, [0, 0], [0, 0], 'k--', linewidth=1.5, label="Assistant Axis")

ax.set_xlabel("Assistant Axis")
ax.set_ylabel(f"Residual PC1 ({pca_resid.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_zlabel(f"Residual PC2 ({pca_resid.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"Role Vectors in Persona Space (Layer {layer_number})")
ax.legend(loc='upper left')

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")

plt.tight_layout()
plt.savefig("figures/role-figures/pca3d_assistantaxis_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()


# Standard PCA (3D)
pca = PCA(n_components=3)
pca_projected = pca.fit_transform(all_vectors_centered)

# Assistant Axis direction in PCA space
pca_axis_proj = pca.components_ @ assistant_axis
pca_axis_proj = pca_axis_proj / np.linalg.norm(pca_axis_proj)

# Color by projection onto Assistant Axis in PCA space
pca_proj = pca_projected @ pca_axis_proj
pca_proj_centered = pca_proj - pca_proj.mean()
pca_proj_spread = np.max(np.abs(pca_proj_centered))
pca_proj_normalized = pca_proj_centered / pca_proj_spread if pca_proj_spread > 0 else pca_proj_centered

pca_vmin = min(pca_proj_normalized.min(), -1e-8)
pca_vmax = max(pca_proj_normalized.max(), 1e-8)
pca_norm = TwoSlopeNorm(vmin=pca_vmin, vcenter=0, vmax=pca_vmax)

fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2],
                c=pca_proj_normalized, cmap='coolwarm_r', norm=pca_norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

# Label all points with offsets pushed away from nearest neighbors
_tree = cKDTree(pca_projected)
_k = min(6, len(all_names) - 1)
_dd, _ii = _tree.query(pca_projected, k=_k + 1)
_scale = np.ptp(pca_projected, axis=0) * 0.025
for i, name in enumerate(all_names):
    neighbors = pca_projected[_ii[i, 1:]]
    direction = pca_projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    if norm_d > 0:
        direction = direction / norm_d
    else:
        direction = np.array([1, 0, 0])
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
ax.set_title(f"Standard PCA — Role Vectors (Layer {layer_number})")
ax.legend(loc='upper left')

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")

plt.tight_layout()
plt.savefig("figures/role-figures/pca3d_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()


# Standard PCA colored by mystical rating
colors_mystical_pca = np.array([role_mystical_ratings.get(n, 0.0) for n in all_names])

fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2],
                c=colors_mystical_pca, cmap='viridis', vmin=0, vmax=10,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

_tree_mp = cKDTree(pca_projected)
_k_mp = min(6, len(all_names) - 1)
_dd_mp, _ii_mp = _tree_mp.query(pca_projected, k=_k_mp + 1)
_scale_mp = np.ptp(pca_projected, axis=0) * 0.025
for i, name in enumerate(all_names):
    neighbors = pca_projected[_ii_mp[i, 1:]]
    direction = pca_projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    if norm_d > 0:
        direction = direction / norm_d
    else:
        direction = np.array([1, 0, 0])
    offset = direction * _scale_mp
    ax.text(pca_projected[i, 0] + offset[0], pca_projected[i, 1] + offset[1],
            pca_projected[i, 2] + offset[2], name,
            fontsize=4.5, ha='center', va='center', alpha=0.85)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title(f"Standard PCA — colored by mystical rating (Layer {layer_number})")

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Mystical Rating (0–10)")

plt.tight_layout()
plt.savefig(f"figures/role-figures/pca3d_mystical_layer_{layer_number}.png", dpi=150)
plt.show()


# Standard PCA colored by assistant rating
colors_assistant_pca = np.array([role_assistant_ratings.get(n, 0.0) for n in all_names])

fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2],
                c=colors_assistant_pca, cmap='viridis', vmin=0, vmax=10,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

_tree_ap = cKDTree(pca_projected)
_k_ap = min(6, len(all_names) - 1)
_dd_ap, _ii_ap = _tree_ap.query(pca_projected, k=_k_ap + 1)
_scale_ap = np.ptp(pca_projected, axis=0) * 0.025
for i, name in enumerate(all_names):
    neighbors = pca_projected[_ii_ap[i, 1:]]
    direction = pca_projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    if norm_d > 0:
        direction = direction / norm_d
    else:
        direction = np.array([1, 0, 0])
    offset = direction * _scale_ap
    ax.text(pca_projected[i, 0] + offset[0], pca_projected[i, 1] + offset[1],
            pca_projected[i, 2] + offset[2], name,
            fontsize=4.5, ha='center', va='center', alpha=0.85)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title(f"Standard PCA — colored by assistant rating (Layer {layer_number})")

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Assistant Rating (0–10)")

plt.tight_layout()
plt.savefig(f"figures/role-figures/pca3d_assistant_layer_{layer_number}.png", dpi=150)
plt.show()


# tSNE

tsne_names = [os.path.splitext(n)[0] for n in names] + ["assistant"]

tsne = TSNE(n_components=2, perplexity=min(5, len(tsne_names) - 1), random_state=42)
tsne_projected = tsne.fit_transform(all_vectors_centered)

# Color by projection onto Assistant Axis (same centering as PCA plot)
tsne_proj = all_vectors_raw @ assistant_axis
tsne_proj_centered = tsne_proj - tsne_proj.mean()
tsne_proj_spread = np.max(np.abs(tsne_proj_centered))
tsne_proj_normalized = tsne_proj_centered / tsne_proj_spread if tsne_proj_spread > 0 else tsne_proj_centered

tsne_vmin = min(tsne_proj_normalized.min(), -1e-8)
tsne_vmax = max(tsne_proj_normalized.max(), 1e-8)
tsne_norm = TwoSlopeNorm(vmin=tsne_vmin, vcenter=0, vmax=tsne_vmax)

fig, ax = plt.subplots(figsize=(16, 11))
sc = ax.scatter(tsne_projected[:, 0], tsne_projected[:, 1],
                c=tsne_proj_normalized, cmap='coolwarm_r', norm=tsne_norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

texts = []
for i, name in enumerate(tsne_names):
    texts.append(ax.text(tsne_projected[i, 0], tsne_projected[i, 1], name,
                         fontsize=5.5, alpha=0.85))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5),
            expand=(1.5, 1.5), force_text=(0.8, 0.8), force_points=(0.3, 0.3))

fig.colorbar(sc, ax=ax, label="Projection onto Assistant Axis", shrink=0.8)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title(f"t-SNE of Role Vectors (Layer {layer_number})")
plt.tight_layout()
plt.savefig("figures/role-figures/tsne_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()


# tSNE 3D
tsne3d = TSNE(n_components=3, perplexity=min(5, len(tsne_names) - 1), random_state=42)
tsne3d_projected = tsne3d.fit_transform(all_vectors_centered)

fig = plt.figure(figsize=(16, 11))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(tsne3d_projected[:, 0], tsne3d_projected[:, 1], tsne3d_projected[:, 2],
                c=tsne_proj_normalized, cmap='coolwarm_r', norm=tsne_norm,
                s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

_tree_t3 = cKDTree(tsne3d_projected)
_k_t3 = min(6, len(tsne_names) - 1)
_dd_t3, _ii_t3 = _tree_t3.query(tsne3d_projected, k=_k_t3 + 1)
_scale_t3 = np.ptp(tsne3d_projected, axis=0) * 0.025
for i, name in enumerate(tsne_names):
    neighbors = tsne3d_projected[_ii_t3[i, 1:]]
    direction = tsne3d_projected[i] - neighbors.mean(axis=0)
    norm_d = np.linalg.norm(direction)
    if norm_d > 0:
        direction = direction / norm_d
    else:
        direction = np.array([1, 0, 0])
    offset = direction * _scale_t3
    ax.text(tsne3d_projected[i, 0] + offset[0], tsne3d_projected[i, 1] + offset[1],
            tsne3d_projected[i, 2] + offset[2], name,
            fontsize=4.5, ha='center', va='center', alpha=0.85)

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")

ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.set_title(f"3D t-SNE of Role Vectors (Layer {layer_number})")
plt.tight_layout()
plt.savefig("figures/role-figures/tsne3d_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()


# ============================================================
# Rating-colored plots (mystical & assistant ratings)
# ============================================================
rating_dicts = {
    "mystical": role_mystical_ratings,
    "assistant": role_assistant_ratings,
}

for rating_name, rating_dict in rating_dicts.items():
    # Build color array matching all_names order
    colors_rating = np.array([rating_dict.get(n, 0.0) for n in all_names])

    # --- PCA (assistant axis) colored by rating ---
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                    c=colors_rating, cmap='viridis', vmin=0, vmax=10,
                    s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

    _tree_r = cKDTree(projected)
    _k_r = min(6, len(all_names) - 1)
    _dd_r, _ii_r = _tree_r.query(projected, k=_k_r + 1)
    _scale_r = np.ptp(projected, axis=0) * 0.025
    for i, name in enumerate(all_names):
        neighbors = projected[_ii_r[i, 1:]]
        direction = projected[i] - neighbors.mean(axis=0)
        norm_d = np.linalg.norm(direction)
        if norm_d > 0:
            direction = direction / norm_d
        else:
            direction = np.array([1, 0, 0])
        offset = direction * _scale_r
        ax.text(projected[i, 0] + offset[0], projected[i, 1] + offset[1],
                projected[i, 2] + offset[2], name,
                fontsize=4.5, ha='center', va='center', alpha=0.85)

    t_r = np.linspace(-1, 1, 2) * max(np.abs(projected[:, 0])) * 1.2
    ax.plot(t_r, [0, 0], [0, 0], 'k--', linewidth=1.5, label="Assistant Axis")

    ax.set_xlabel("Assistant Axis")
    ax.set_ylabel(f"Residual PC1 ({pca_resid.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_zlabel(f"Residual PC2 ({pca_resid.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Role Vectors — colored by {rating_name} rating (Layer {layer_number})")
    ax.legend(loc='upper left')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f"{rating_name.capitalize()} Rating (0–10)")

    plt.tight_layout()
    plt.savefig(f"figures/role-figures/role_colored_pca_{rating_name}_layer_{layer_number}.png", dpi=150)
    plt.show()

    # --- t-SNE colored by rating ---
    fig, ax = plt.subplots(figsize=(16, 11))
    sc = ax.scatter(tsne_projected[:, 0], tsne_projected[:, 1],
                    c=colors_rating, cmap='viridis', vmin=0, vmax=10,
                    s=60, edgecolors='k', linewidths=0.3, zorder=5, alpha=0.85)

    texts = []
    for i, name in enumerate(all_names):
        texts.append(ax.text(tsne_projected[i, 0], tsne_projected[i, 1], name,
                             fontsize=5.5, alpha=0.85))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, alpha=0.5),
                expand=(1.5, 1.5), force_text=(0.8, 0.8), force_points=(0.3, 0.3))

    fig.colorbar(sc, ax=ax, label=f"{rating_name.capitalize()} Rating (0–10)", shrink=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of Role Vectors — colored by {rating_name} rating (Layer {layer_number})")
    plt.tight_layout()
    plt.savefig(f"figures/role-figures/role_colored_tsne_{rating_name}_layer_{layer_number}.png", dpi=150)
    plt.show()


# non-negative matrix factorization
n_components = 6

# NMF requires non-negative input; shift vectors so min is 0
nmf_input = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
nmf_input_shifted = nmf_input - nmf_input.min()

nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
W = nmf.fit_transform(nmf_input_shifted)  # (n_roles+1, n_components) - coefficients
H = nmf.components_                        # (n_components, 4096) - basis vectors

nmf_names = [os.path.splitext(n)[0] for n in names] + ["assistant"]

# Figure 1: Basis vectors heatmap
fig, axes = plt.subplots(n_components, 1, figsize=(14, 2 * n_components), sharex=True)
for i in range(n_components):
    axes[i].plot(H[i], linewidth=0.5)
    axes[i].set_ylabel(f"Basis {i+1}")
    axes[i].set_xlim(0, H.shape[1])
axes[-1].set_xlabel("Dimension")
fig.suptitle(f"NMF Basis Vectors (Layer {layer_number}, k={n_components})", fontsize=14)
plt.tight_layout()
plt.savefig("figures/role-figures/nmf_bases_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()

# Figure 2: Roles' coefficients on the bases (heatmap)
fig, ax = plt.subplots(figsize=(10, max(6, len(nmf_names) * 0.4)))
im = ax.imshow(W, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(nmf_names)))
ax.set_yticklabels(nmf_names, fontsize=8)
ax.set_xticks(range(n_components))
ax.set_xticklabels([f"Basis {i+1}" for i in range(n_components)])
ax.set_xlabel("NMF Component")
ax.set_ylabel("Role")
ax.set_title(f"Role Coefficients on NMF Bases (Layer {layer_number}, k={n_components})")
fig.colorbar(im, ax=ax, label="Coefficient")
plt.tight_layout()
plt.savefig("figures/role-figures/nmf_roles_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()


