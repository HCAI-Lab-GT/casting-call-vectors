import numpy as np
import matplotlib.pyplot as plt
from safetensors.numpy import load_file
import os

os.makedirs("figures/role-figures", exist_ok=True)

# Load vectors (same as full_analysis.py)
base_path = "model_layer_inits"
layer_number = 16
num_questions_used = 50
vectors = {}

for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder)
    if not os.path.isdir(full_path):
        continue
    files = os.listdir(full_path)
    safetensor_files = [f for f in files if f.endswith(".safetensors") and f"count{num_questions_used}" in f]
    if not safetensor_files:
        continue
    data = load_file(os.path.join(full_path, safetensor_files[0]))
    key = list(data.keys())[0]
    name = folder.replace("_persona_initialization", "")
    vectors[name] = data[key].astype(np.float32)[layer_number - 1].squeeze()

names = sorted(vectors.keys())
matrix = np.array([vectors[n] for n in names])  # (num_roles, 4096)
print(f"Matrix shape: {matrix.shape}")

# ============================================================
# SVD analysis
# ============================================================
U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

# Numerical rank (default tolerance)
rank = np.linalg.matrix_rank(matrix)
print(f"\nNumerical rank: {rank}  (out of {matrix.shape[0]} vectors in {matrix.shape[1]}-d space)")

# Effective rank via Shannon entropy of normalized singular values
# (Vershynin / Roy-Vetterli definition)
p = S / S.sum()
effective_rank_entropy = np.exp(-np.sum(p * np.log(p + 1e-30)))
print(f"Effective rank (entropy): {effective_rank_entropy:.1f}")

# How many singular values needed to capture X% of total variance
total_var = np.sum(S**2)
cumvar = np.cumsum(S**2) / total_var
for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
    k = int(np.searchsorted(cumvar, threshold)) + 1
    print(f"  {threshold*100:.0f}% variance captured by top {k} singular values")

# Condition number
print(f"\nLargest singular value:  {S[0]:.4f}")
print(f"Smallest singular value: {S[-1]:.6f}")
print(f"Condition number: {S[0]/S[-1]:.1f}")

# ============================================================
# Plot 1: Singular value spectrum
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Raw singular values
axes[0].semilogy(range(1, len(S) + 1), S, 'b.-', markersize=3)
axes[0].set_xlabel("Singular value index")
axes[0].set_ylabel("Singular value (log scale)")
axes[0].set_title("Singular Value Spectrum")
axes[0].grid(True, alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(S) + 1), cumvar * 100, 'r.-', markersize=3)
axes[1].axhline(90, color='gray', linestyle='--', alpha=0.5, label='90%')
axes[1].axhline(95, color='gray', linestyle=':', alpha=0.5, label='95%')
axes[1].axhline(99, color='gray', linestyle='-.', alpha=0.5, label='99%')
axes[1].set_xlabel("Number of components")
axes[1].set_ylabel("Cumulative variance explained (%)")
axes[1].set_title("Cumulative Variance Explained")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Normalized singular values (fraction of total)
axes[2].bar(range(1, min(51, len(S) + 1)), (S[:50]**2 / total_var) * 100, color='steelblue')
axes[2].set_xlabel("Singular value index")
axes[2].set_ylabel("Variance explained (%)")
axes[2].set_title("Top 50 Components — Variance Share")
axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Rank Analysis of Role Vector Matrix ({matrix.shape[0]} vectors × {matrix.shape[1]} dims, Layer {layer_number})", fontsize=13)
plt.tight_layout()
plt.savefig("figures/role-figures/rank_analysis.png", dpi=150)
plt.show()

# ============================================================
# Also do it on mean-centered matrix
# ============================================================
matrix_centered = matrix - matrix.mean(axis=0)
U_c, S_c, Vt_c = np.linalg.svd(matrix_centered, full_matrices=False)
rank_c = np.linalg.matrix_rank(matrix_centered)

p_c = S_c / S_c.sum()
effective_rank_c = np.exp(-np.sum(p_c * np.log(p_c + 1e-30)))

total_var_c = np.sum(S_c**2)
cumvar_c = np.cumsum(S_c**2) / total_var_c

print(f"\n{'='*60}")
print(f"Mean-centered matrix:")
print(f"  Numerical rank: {rank_c}")
print(f"  Effective rank (entropy): {effective_rank_c:.1f}")
for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
    k = int(np.searchsorted(cumvar_c, threshold)) + 1
    print(f"  {threshold*100:.0f}% variance captured by top {k} singular values")
