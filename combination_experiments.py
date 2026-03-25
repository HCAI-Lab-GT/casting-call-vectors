import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, FastICA
from safetensors.numpy import load_file
import os

# Load vectors
vector_list = os.listdir("personality-vectors/persona_data/model_layer_inits")
layer_number = 16
num_questions_used = 50

os.makedirs("figures/role-figures", exist_ok=True)

base_path = "personality-vectors/persona_data/model_layer_inits"
vectors = {}

for vector in vector_list:
    full_path = os.path.join(base_path, vector)
    if not os.path.isdir(full_path):
        continue
    files = os.listdir(full_path)
    safetensor_files = [f for f in files if f.endswith(".safetensors") and f"count{num_questions_used}" in f]
    if not safetensor_files:
        continue
    file = safetensor_files[0]
    data = load_file(os.path.join(full_path, file))
    key = list(data.keys())[0]
    vector_name = vector.replace("_persona_initialization", "")
    vectors[vector_name] = data[key].astype(np.float32)[layer_number - 1].squeeze()

assistant_vector = vectors.pop("assistant")
names = sorted(vectors.keys())
role_matrix = np.array([vectors[n] for n in names])
all_vectors = np.vstack([role_matrix, assistant_vector.reshape(1, -1)])
all_names = names + ["assistant"]

print(f"Loaded {len(names)} role vectors + assistant ({role_matrix.shape[1]} dims)")


# ============================================================
# 1. Hierarchical Clustering (cosine distance)
# ============================================================
print("\n--- Hierarchical Clustering ---")
cos_dists = pdist(all_vectors, metric='cosine')

linkage_matrix = linkage(cos_dists, method='ward')

fig, ax = plt.subplots(figsize=(20, 10))
dendrogram(
    linkage_matrix,
    labels=all_names,
    leaf_rotation=90,
    leaf_font_size=6,
    ax=ax,
)
ax.set_ylabel("Distance")
ax.set_title(f"Hierarchical Clustering of Persona Vectors (Layer {layer_number}, cosine, ward)")
plt.tight_layout()
plt.savefig("figures/role-figures/dendrogram_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()

# Cosine similarity heatmap (ordered by dendrogram)
from scipy.cluster.hierarchy import leaves_list
order = leaves_list(linkage_matrix)
cos_sim = 1 - squareform(cos_dists)
cos_sim_ordered = cos_sim[np.ix_(order, order)]
ordered_names = [all_names[i] for i in order]

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cos_sim_ordered, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
ax.set_xticks(range(len(ordered_names)))
ax.set_xticklabels(ordered_names, rotation=90, fontsize=4)
ax.set_yticks(range(len(ordered_names)))
ax.set_yticklabels(ordered_names, fontsize=4)
ax.set_title(f"Cosine Similarity Matrix (Layer {layer_number}, ordered by clustering)")
fig.colorbar(im, ax=ax, shrink=0.6, label="Cosine Similarity")
plt.tight_layout()
plt.savefig("figures/role-figures/cosine_sim_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()


# ============================================================
# 2. Vector Arithmetic
# ============================================================
print("\n--- Vector Arithmetic ---")

def nearest_persona(target_vec, names, role_matrix, exclude=None):
    """Find the closest persona to a target vector by cosine similarity."""
    sims = role_matrix @ target_vec / (
        np.linalg.norm(role_matrix, axis=1) * np.linalg.norm(target_vec)
    )
    ranked = np.argsort(-sims)
    results = []
    for idx in ranked:
        if exclude and names[idx] in exclude:
            continue
        results.append((names[idx], sims[idx]))
        if len(results) >= 5:
            break
    return results

# a) Nearest neighbors for each persona
print("\nTop-3 nearest neighbors (cosine):")
for i, name in enumerate(names):
    neighbors = nearest_persona(role_matrix[i], names, role_matrix, exclude={name})
    neighbor_str = ", ".join(f"{n} ({s:.3f})" for n, s in neighbors[:3])
    print(f"  {name}: {neighbor_str}")

# b) Arithmetic: A - B + C = ?
arithmetic_tests = [
    ("warrior", "stoic", "pacifist"),
    ("scientist", "theorist", "actor"),
    ("hermit", "loner", "comedian"),
    ("teacher", "tutor", "pirate"),
    ("scientist", "mathematician", "poet"),
    ("warrior", "zealot", "scholar"),
    ("detective", "smuggler", "therapist"),
    ("teacher", "guide", "destroyer"),
    ("hermit", "ascetic", "networker"),
    ("prophet", "spirit", "scientist"),
    ("pirate", "merchant", "philosopher"),
    ("activist", "revolutionary", "peacekeeper"),
    ("addict", "smuggler", "guru"),
    ("scientist", "critic", "criminal")
]

print("\nVector arithmetic (A - B + C -> nearest):")
for a, b, c in arithmetic_tests:
    if a in vectors and b in vectors and c in vectors:
        result_vec = vectors[a] - vectors[b] + vectors[c]
        neighbors = nearest_persona(result_vec, names, role_matrix, exclude={a, b, c})
        neighbor_str = ", ".join(f"{n} ({s:.3f})" for n, s in neighbors[:3])
        print(f"  {a} - {b} + {c} = {neighbor_str}")

# c) Interpolation between two distant personas projected into PCA space
pca = PCA(n_components=3)
pca_projected = pca.fit_transform(all_vectors)

pair_a, pair_b = "warrior", "pacifist"¬
if pair_a in vectors and pair_b in vectors:
    interp_steps = 10
    interp_vecs = np.array([
        vectors[pair_a] * (1 - t) + vectors[pair_b] * t
        for t in np.linspace(0, 1, interp_steps)
    ])
    interp_proj = pca.transform(interp_vecs)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_projected[:, 0], pca_projected[:, 1], pca_projected[:, 2],
               c='gray', s=20, alpha=0.4)
    for i, n in enumerate(all_names):
        ax.text(pca_projected[i, 0], pca_projected[i, 1], pca_projected[i, 2],
                f"  {n}", fontsize=4, alpha=0.5)

    ax.plot(interp_proj[:, 0], interp_proj[:, 1], interp_proj[:, 2],
            'r-o', linewidth=2, markersize=5, label=f"{pair_a} -> {pair_b}")
    ax.scatter(*interp_proj[0], c='blue', s=100, zorder=10)
    ax.scatter(*interp_proj[-1], c='red', s=100, zorder=10)
    ax.text(*interp_proj[0], f"  {pair_a}", fontsize=9, color='blue')
    ax.text(*interp_proj[-1], f"  {pair_b}", fontsize=9, color='red')

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax.set_title(f"Interpolation: {pair_a} -> {pair_b} (Layer {layer_number})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/role-figures/interpolation_layer_" + str(layer_number) + ".png", dpi=150)
    plt.show()

    # What persona is nearest at each interpolation step?
    print(f"\nInterpolation {pair_a} -> {pair_b}:")
    for step, t in enumerate(np.linspace(0, 1, interp_steps)):
        nn = nearest_persona(interp_vecs[step], names, role_matrix, exclude=set())
        print(f"  t={t:.1f}: {nn[0][0]} ({nn[0][1]:.3f}), {nn[1][0]} ({nn[1][1]:.3f})")


# ============================================================
# 3. Independent Component Analysis (ICA)
# ============================================================
print("\n--- Independent Component Analysis ---")
n_ica_components = 6

ica = FastICA(n_components=n_ica_components, random_state=42, max_iter=1000)
S = ica.fit_transform(all_vectors)  # (n_samples, n_components) - source signals
A = ica.mixing_                     # (n_features, n_components) - mixing matrix

# Figure 1: Independent components (basis signals)
fig, axes = plt.subplots(n_ica_components, 1, figsize=(14, 2 * n_ica_components), sharex=True)
for i in range(n_ica_components):
    axes[i].plot(A[:, i], linewidth=0.5)
    axes[i].set_ylabel(f"IC {i+1}")
    axes[i].set_xlim(0, A.shape[0])
axes[-1].set_xlabel("Dimension")
fig.suptitle(f"ICA Independent Components (Layer {layer_number}, k={n_ica_components})", fontsize=14)
plt.tight_layout()
plt.savefig("figures/role-figures/ica_components_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()

# Figure 2: Persona loadings on independent components (heatmap)
fig, ax = plt.subplots(figsize=(10, max(8, len(all_names) * 0.15)))
im = ax.imshow(S, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_yticks(range(len(all_names)))
ax.set_yticklabels(all_names, fontsize=5)
ax.set_xticks(range(n_ica_components))
ax.set_xticklabels([f"IC {i+1}" for i in range(n_ica_components)])
ax.set_xlabel("Independent Component")
ax.set_ylabel("Persona")
ax.set_title(f"Persona Loadings on Independent Components (Layer {layer_number}, k={n_ica_components})")
fig.colorbar(im, ax=ax, shrink=0.6, label="Loading")
plt.tight_layout()
plt.savefig("figures/role-figures/ica_loadings_layer_" + str(layer_number) + ".png", dpi=150)
plt.show()

# Figure 3: Top personas per IC
print("\nTop 5 personas per Independent Component:")
for i in range(n_ica_components):
    top_pos = np.argsort(-S[:, i])[:5]
    top_neg = np.argsort(S[:, i])[:5]
    pos_str = ", ".join(f"{all_names[j]} ({S[j, i]:.2f})" for j in top_pos)
    neg_str = ", ".join(f"{all_names[j]} ({S[j, i]:.2f})" for j in top_neg)
    print(f"  IC {i+1} (+): {pos_str}")
    print(f"  IC {i+1} (-): {neg_str}")

print("\nDone.")
