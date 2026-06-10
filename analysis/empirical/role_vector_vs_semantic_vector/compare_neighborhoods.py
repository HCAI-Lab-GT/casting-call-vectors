"""
Representational-similarity comparison.

Question: even if persona[i] doesn't line up with semantic[i] in absolute
direction, does the *similarity structure across roles* match? I.e., if
role A and role B are close in persona space, are they also close in
semantic space?

For each variant we compute:
  - Sim_P:  N x N cosine similarity matrix of persona vectors
  - Sim_S:  N x N cosine similarity matrix of semantic vectors
  - Pearson + Spearman correlation between the two upper triangles  (RSA)
  - kNN overlap@k:  for each role, take its k nearest neighbors in each
                    space and compute Jaccard similarity, averaged over roles
  - Mantel-style permutation test for the Pearson RSA value

Outputs (analysis/empirical/role_vector_vs_semantic_vector/figures/):
  - rsa_summary.json
  - rsa_scatter_{variant}.png      -- scatter of pairwise sims, P vs S
  - similarity_heatmaps_{variant}.png  -- side-by-side heatmaps with the same
                                         ordering (hierarchical on persona)
  - knn_overlap_{variant}.png      -- mean Jaccard@k for k=1..50
  - example_neighbors_{variant}.txt -- top-10 neighbors per example role in
                                       both spaces

Usage:
    python analysis/empirical/role_vector_vs_semantic_vector/compare_neighborhoods.py
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


VECTOR_DIR = "persona_data/model_inits"
SEMANTIC_FILE = "analysis/empirical/role_vector_vs_semantic_vector/semantic_vectors.safetensors"
OUTPUT_DIR = "analysis/empirical/role_vector_vs_semantic_vector/figures"

VARIANT_ROWS = {
    "bare_last": 0,
    "bare_mean": 1,
    "context_last": 2,
    "context_mean": 3,
}

EXAMPLE_ROLES = [
    "accountant", "doctor", "wizard", "alien", "child", "king",
    "philosopher", "soldier", "vampire", "robot",
]


def load_persona_vectors(layer: int, sample_count: int) -> Dict[str, np.ndarray]:
    suffix = "_persona_initialization"
    out: Dict[str, np.ndarray] = {}
    for entry in sorted(os.listdir(VECTOR_DIR)):
        full = os.path.join(VECTOR_DIR, entry)
        if not os.path.isdir(full) or not entry.endswith(suffix):
            continue
        files = [
            f for f in os.listdir(full)
            if f.endswith(".safetensors") and f"count{sample_count}" in f
        ]
        if not files:
            continue
        data = load_file(os.path.join(full, files[0]))
        key = sorted(data.keys())[0]
        vec = data[key].astype(np.float32)[layer - 1].squeeze()
        out[entry[: -len(suffix)]] = vec
    return out


def cosine_self(A: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return A_norm @ A_norm.T


def upper_tri(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def mantel_pvalue(Sp: np.ndarray, Ss: np.ndarray, n_perm: int = 1000,
                  seed: int = 0) -> float:
    """Permutation test on Pearson RSA. Permute one matrix's row/col indices."""
    rng = np.random.default_rng(seed)
    n = Sp.shape[0]
    obs = pearsonr(upper_tri(Sp), upper_tri(Ss))[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Ss_perm = Ss[perm][:, perm]
        r = pearsonr(upper_tri(Sp), upper_tri(Ss_perm))[0]
        if r >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def knn_overlap(Sp: np.ndarray, Ss: np.ndarray, ks: List[int]) -> Dict[int, float]:
    """For each role, mean Jaccard between top-k neighbors in P and S."""
    n = Sp.shape[0]
    # ignore self-similarity by zeroing diagonal
    Sp_ = Sp.copy()
    Ss_ = Ss.copy()
    np.fill_diagonal(Sp_, -np.inf)
    np.fill_diagonal(Ss_, -np.inf)

    # rank ordering once: argsort each row descending
    order_p = np.argsort(-Sp_, axis=1)
    order_s = np.argsort(-Ss_, axis=1)

    results: Dict[int, float] = {}
    for k in ks:
        jaccs = np.empty(n)
        for i in range(n):
            sp_set = set(order_p[i, :k].tolist())
            ss_set = set(order_s[i, :k].tolist())
            inter = len(sp_set & ss_set)
            union = len(sp_set | ss_set)
            jaccs[i] = inter / union if union else 0.0
        results[k] = float(jaccs.mean())
    return results


def random_knn_baseline(n: int, ks: List[int]) -> Dict[int, float]:
    # expected Jaccard for two independent uniform-random size-k sets from
    # a universe of (n-1):  k/(2*(n-1) - k)  approx for small k
    out = {}
    for k in ks:
        N = n - 1
        out[k] = k / (2 * N - k)
    return out


def plot_rsa_scatter(Sp: np.ndarray, Ss: np.ndarray, variant: str, path: str,
                     subsample: int = 20000, seed: int = 0):
    p = upper_tri(Sp)
    s = upper_tri(Ss)
    rng = np.random.default_rng(seed)
    if len(p) > subsample:
        idx = rng.choice(len(p), subsample, replace=False)
        p = p[idx]
        s = s[idx]
    r_p = pearsonr(p, s)[0]
    r_s = spearmanr(p, s)[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(p, s, s=2, alpha=0.15, color="C0")
    lim = [
        min(p.min(), s.min()) - 0.02,
        max(p.max(), s.max()) + 0.02,
    ]
    ax.plot(lim, lim, "k--", lw=0.6, alpha=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("pairwise cosine sim — persona space")
    ax.set_ylabel("pairwise cosine sim — semantic space")
    ax.set_title(
        f"RSA scatter, variant = {variant}\n"
        f"Pearson = {r_p:.3f},  Spearman = {r_s:.3f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hierarchical_leaf_order(D: np.ndarray) -> np.ndarray:
    """1-D spectral-style ordering: project rows of the distance matrix onto
    PC1 and sort. Roles close in D end up adjacent. This is not true
    average-linkage clustering, but produces a clean diagonal block structure
    in heatmaps and avoids scipy.cluster entirely (the venv's scipy.cluster
    fails to import — _vq C extension missing)."""
    proj = PCA(n_components=1).fit_transform(D)[:, 0]
    return np.argsort(proj)


def plot_similarity_heatmaps(names: List[str], Sp: np.ndarray, Ss: np.ndarray,
                             variant: str, path: str):
    # order by hierarchical clustering on Sp (so persona-similar roles group)
    # convert similarity to distance in [0, 2]
    dist = 1.0 - Sp
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # ensure symmetric
    order = hierarchical_leaf_order(dist)

    Sp_o = Sp[order][:, order]
    Ss_o = Ss[order][:, order]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    vmax_p = max(abs(Sp.min()), abs(Sp.max()))
    vmax_s = max(abs(Ss.min()), abs(Ss.max()))
    im0 = axes[0].imshow(Sp_o, cmap="RdBu_r", vmin=-vmax_p, vmax=vmax_p, aspect="auto")
    im1 = axes[1].imshow(Ss_o, cmap="RdBu_r", vmin=-vmax_s, vmax=vmax_s, aspect="auto")
    axes[0].set_title("persona x persona  (rows ordered by hierarchical clustering)")
    axes[1].set_title(f"semantic x semantic  (same order, variant = {variant})")
    for ax in axes:
        ax.set_xlabel("role")
        ax.set_ylabel("role")
    fig.colorbar(im0, ax=axes[0], shrink=0.7, label="cosine")
    fig.colorbar(im1, ax=axes[1], shrink=0.7, label="cosine")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_knn_curve(curve: Dict[int, float], baseline: Dict[int, float],
                   variant: str, path: str):
    ks = sorted(curve.keys())
    vals = [curve[k] for k in ks]
    base = [baseline[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, vals, "-o", color="C0", label="observed Jaccard@k")
    ax.plot(ks, base, "--", color="gray", label="random baseline")
    ax.set_xlabel("k (neighborhood size)")
    ax.set_ylabel("mean Jaccard overlap (persona kNN ∩ semantic kNN)")
    ax.set_title(f"kNN neighborhood overlap, variant = {variant}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_example_neighbors(names: List[str], Sp: np.ndarray, Ss: np.ndarray,
                            variant: str, path: str, k: int = 10):
    name_to_idx = {n: i for i, n in enumerate(names)}
    Sp_ = Sp.copy()
    Ss_ = Ss.copy()
    np.fill_diagonal(Sp_, -np.inf)
    np.fill_diagonal(Ss_, -np.inf)
    with open(path, "w") as f:
        f.write(f"# Top-{k} neighbors per example role, variant = {variant}\n")
        f.write(f"# (cosine in respective space, decreasing)\n\n")
        for r in EXAMPLE_ROLES:
            if r not in name_to_idx:
                f.write(f"## {r}: NOT IN DATASET\n\n")
                continue
            i = name_to_idx[r]
            top_p = np.argsort(-Sp_[i])[:k]
            top_s = np.argsort(-Ss_[i])[:k]
            f.write(f"## {r}\n")
            f.write(f"  persona-space neighbors:  "
                    + ", ".join(f"{names[j]} ({Sp[i, j]:+.3f})" for j in top_p)
                    + "\n")
            f.write(f"  semantic-space neighbors: "
                    + ", ".join(f"{names[j]} ({Ss[i, j]:+.3f})" for j in top_s)
                    + "\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_ROWS.keys()),
                        choices=list(VARIANT_ROWS.keys()))
    parser.add_argument("--n-perm", type=int, default=500,
                        help="Permutations for Mantel test (0 to skip)")
    parser.add_argument("--ks", nargs="+", type=int,
                        default=[1, 3, 5, 10, 20, 50])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    persona_dict = load_persona_vectors(args.layer, args.sample_count)
    semantic_raw = load_file(SEMANTIC_FILE)
    common = sorted(set(persona_dict) & set(semantic_raw))
    print(f"{len(common)} common roles")

    persona_mat = np.stack([persona_dict[r] for r in common]).astype(np.float32)
    Sp = cosine_self(persona_mat)

    summary: Dict[str, Dict] = {}
    for variant in args.variants:
        row = VARIANT_ROWS[variant]
        sem_mat = np.stack([semantic_raw[r][row] for r in common]).astype(np.float32)
        Ss = cosine_self(sem_mat)

        p = upper_tri(Sp)
        s = upper_tri(Ss)
        r_pearson, _ = pearsonr(p, s)
        r_spearman, _ = spearmanr(p, s)

        knn_curve = knn_overlap(Sp, Ss, args.ks)
        baseline = random_knn_baseline(len(common), args.ks)

        if args.n_perm > 0:
            p_value = mantel_pvalue(Sp, Ss, n_perm=args.n_perm)
        else:
            p_value = None

        summary[variant] = {
            "rsa_pearson": float(r_pearson),
            "rsa_spearman": float(r_spearman),
            "mantel_p_value_pearson": p_value,
            "knn_overlap_jaccard": {str(k): v for k, v in knn_curve.items()},
            "knn_overlap_random_baseline": {str(k): v for k, v in baseline.items()},
            "n_roles": int(len(common)),
        }

        print(
            f"[{variant}] RSA Pearson={r_pearson:+.3f}  Spearman={r_spearman:+.3f}  "
            f"Jaccard@10={knn_curve[10]:.3f} (random={baseline[10]:.3f})  "
            f"Mantel p={p_value}"
        )

        plot_rsa_scatter(Sp, Ss, variant,
                         os.path.join(OUTPUT_DIR, f"rsa_scatter_{variant}.png"))
        plot_similarity_heatmaps(common, Sp, Ss, variant,
                                 os.path.join(OUTPUT_DIR, f"similarity_heatmaps_{variant}.png"))
        plot_knn_curve(knn_curve, baseline, variant,
                       os.path.join(OUTPUT_DIR, f"knn_overlap_{variant}.png"))
        write_example_neighbors(common, Sp, Ss, variant,
                                os.path.join(OUTPUT_DIR, f"example_neighbors_{variant}.txt"))

    with open(os.path.join(OUTPUT_DIR, "rsa_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote rsa_summary.json")


if __name__ == "__main__":
    main()
