import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import TwoSlopeNorm
import torch
import os

os.makedirs("analysis/geometry/figures", exist_ok=True)

# Load vectors from christina-vectors
vector_dir = "christina-vectors"
layer_number = 16
vectors = {}

for fname in os.listdir(vector_dir):
    if not fname.endswith(".pt"):
        continue
    data = torch.load(os.path.join(vector_dir, fname), map_location="cpu", weights_only=False)
    name = os.path.splitext(fname)[0]
    vec = data["vector"].float().numpy().squeeze()
    vectors[name] = vec
    if vec.shape != (4096,):
        print(f"Warning: {name} has shape {vec.shape}")

print(f"Loaded {len(vectors)} vectors")

# Separate assistant from role vectors
assistant_vector = vectors.pop("assistant")
names = sorted(vectors.keys())
role_matrix = np.array([vectors[n] for n in names])
print(f"{len(names)} role vectors + 1 assistant")

# Assistant Axis: mean assistant activation - mean of all role vectors
role_mean = role_matrix.mean(axis=0)
assistant_axis = assistant_vector - role_mean
assistant_axis = assistant_axis / np.linalg.norm(assistant_axis)

# Mean-subtract all vectors before PCA
all_vectors_raw = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
all_names = names + ["assistant"]
all_vectors_centered = all_vectors_raw - all_vectors_raw.mean(axis=0)

# ============================================================
# PCA (3D)
# ============================================================
pca = PCA(n_components=3)
projected = pca.fit_transform(all_vectors_centered)

# Color by projection onto Assistant Axis (centered)
projections = all_vectors_raw @ assistant_axis
proj_centered = projections - projections.mean()
proj_spread = np.max(np.abs(proj_centered))
proj_normalized = proj_centered / proj_spread if proj_spread > 0 else proj_centered

cmap = plt.cm.coolwarm_r
vmin = min(proj_normalized.min(), -1e-8)
vmax = max(proj_normalized.max(), 1e-8)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# 3D scatter
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                c=proj_normalized, cmap='coolwarm_r', norm=norm,
                s=80, edgecolors='k', linewidths=0.5, zorder=5)

for i, name in enumerate(all_names):
    ax.text(projected[i, 0], projected[i, 1], projected[i, 2], f"  {name}",
            fontsize=5, ha='left', va='center')

# Assistant Axis direction in PCA space (project direction, not data point)
axis_proj = pca.components_ @ assistant_axis
axis_proj = axis_proj / np.linalg.norm(axis_proj)
t = np.linspace(-1, 1, 2) * max(np.abs(projected[:, 0])) * 1.2
ax.plot(t * axis_proj[0], t * axis_proj[1], t * axis_proj[2],
        'k--', linewidth=1.5, label="Assistant Axis")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title(f"Christina Vectors — Persona Space PCA (Layer {layer_number})")
ax.legend(loc='upper left')

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")

plt.tight_layout()
plt.savefig("analysis/geometry/figures/christina_pca3d_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()


# ============================================================
# t-SNE 2D
# ============================================================
tsne = TSNE(n_components=2, perplexity=min(5, len(all_names) - 1), random_state=42)
tsne_projected = tsne.fit_transform(all_vectors_centered)

tsne_vmin = min(proj_normalized.min(), -1e-8)
tsne_vmax = max(proj_normalized.max(), 1e-8)
tsne_norm = TwoSlopeNorm(vmin=tsne_vmin, vcenter=0, vmax=tsne_vmax)

plt.figure(figsize=(14, 10))
sc = plt.scatter(tsne_projected[:, 0], tsne_projected[:, 1],
                 c=proj_normalized, cmap='coolwarm_r', norm=tsne_norm,
                 s=80, edgecolors='k', linewidths=0.5, zorder=5)

for i, name in enumerate(all_names):
    plt.annotate(name, (tsne_projected[i, 0], tsne_projected[i, 1]),
                 textcoords="offset points", xytext=(5, 5), fontsize=5)

plt.colorbar(sc, label="Projection onto Assistant Axis")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title(f"Christina Vectors — t-SNE 2D (Layer {layer_number})")
plt.tight_layout()
plt.savefig("analysis/geometry/figures/christina_tsne_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()



# ============================================================
# t-SNE 3D
# ============================================================
tsne3d = TSNE(n_components=3, perplexity=min(30, len(all_names) - 1), random_state=42)
tsne3d_projected = tsne3d.fit_transform(all_vectors_centered)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(tsne3d_projected[:, 0], tsne3d_projected[:, 1], tsne3d_projected[:, 2],
                c=proj_normalized, cmap='coolwarm_r', norm=tsne_norm,
                s=80, edgecolors='k', linewidths=0.5, zorder=5)

for i, name in enumerate(all_names):
    ax.text(tsne3d_projected[i, 0], tsne3d_projected[i, 1], tsne3d_projected[i, 2],
            f"  {name}", fontsize=4, ha='left', va='center')

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Projection onto Assistant Axis")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.set_title(f"Christina Vectors — t-SNE 3D (Layer {layer_number})")
plt.tight_layout()
plt.savefig("analysis/geometry/figures/christina_tsne3d_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()

print("Done.")