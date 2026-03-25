import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from matplotlib.colors import TwoSlopeNorm
from safetensors.numpy import load_file
import os

os.makedirs("figures/role-figures", exist_ok=True)

# Load vectors
vector_list = os.listdir("personality-vectors/persona_data/model_layer_inits")
layer_number = 16
num_questions_used = 50

base_path = "personality-vectors/persona_data/model_layer_inits"
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
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                c=proj_normalized, cmap='coolwarm_r', norm=norm,
                s=80, edgecolors='k', linewidths=0.5, zorder=5)

for i, name in enumerate(all_names):
    ax.text(projected[i, 0], projected[i, 1], projected[i, 2], f"  {name}",
            fontsize=7, ha='left', va='center')

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

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2],
                c=pca_proj_normalized, cmap='coolwarm_r', norm=pca_norm,
                s=80, edgecolors='k', linewidths=0.5, zorder=5)

for i, name in enumerate(all_names):
    ax.text(pca_projected[i, 0], pca_projected[i, 1], pca_projected[i, 2], f"  {name}",
            fontsize=7, ha='left', va='center')

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

plt.figure(figsize=(12, 8))
sc = plt.scatter(tsne_projected[:, 0], tsne_projected[:, 1],
                 c=tsne_proj_normalized, cmap='coolwarm_r', norm=tsne_norm,
                 s=80, edgecolors='k', linewidths=0.5, zorder=5)

for i, name in enumerate(tsne_names):
    plt.annotate(name, (tsne_projected[i, 0], tsne_projected[i, 1]),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)

plt.colorbar(sc, label="Projection onto Assistant Axis")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title(f"t-SNE of Role Vectors (Layer {layer_number})")
plt.tight_layout()
plt.savefig("figures/role-figures/tsne_layer_" + str(layer_number) + ".png", dpi=150)
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
