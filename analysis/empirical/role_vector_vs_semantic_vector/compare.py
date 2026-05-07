"""
Compare persona role vectors with semantic vectors of the role names.

Both sets live in the same 4096-dim residual-stream space (layer 16 of
allenai/Olmo-3-7B-Instruct), so we can directly compute cosine similarity,
overlay them in a shared PCA / t-SNE, and check whether the geometry of the
role-name embeddings tracks the geometry of the persona vectors.

Inputs:
    persona_data/model_inits/{role}_persona_initialization/...layer16_count{N}.safetensors
    analysis/empirical/role_vector_vs_semantic_vector/semantic_vectors.safetensors

Outputs (analysis/empirical/role_vector_vs_semantic_vector/figures/):
    cosine_similarity_hist_{variant}.png        -- distribution of per-role cosine
    cosine_similarity_top_bottom_{variant}.png  -- bar chart of best/worst aligned roles
    cross_similarity_heatmap_{variant}.png      -- full role x role cosine heatmap
    pca_overlay_{variant}.png                   -- shared 2D PCA, role + name pairs
    tsne_overlay_{variant}.png                  -- shared t-SNE, role + name pairs
    procrustes_{variant}.json                   -- procrustes disparity / scale
    summary.csv                                 -- per-role cos sim across variants

Usage:
    python analysis/empirical/role_vector_vs_semantic_vector/compare.py
    python analysis/empirical/role_vector_vs_semantic_vector/compare.py --variants context_last
"""

import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from safetensors.numpy import load_file
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


VECTOR_DIR = "persona_data/model_inits"
SEMANTIC_FILE = "analysis/empirical/role_vector_vs_semantic_vector/semantic_vectors.safetensors"
OUTPUT_DIR = "analysis/empirical/role_vector_vs_semantic_vector/figures"

VARIANT_ROWS = {
    "bare_last": 0,
    "bare_mean": 1,
    "context_last": 2,
    "context_mean": 3,
}


def load_persona_vectors(layer: int, sample_count: int) -> Dict[str, np.ndarray]:
    """Mirror analysis/geometry/full_analysis.py loading."""
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
        # full_analysis.py uses list(data.keys())[0]; in alphabetical order this
        # is "all_layers_response_persona_vector" with shape (L+1, 1, H).
        key = sorted(data.keys())[0]
        vec = data[key].astype(np.float32)[layer - 1].squeeze()
        out[entry[: -len(suffix)]] = vec
    return out


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def plot_cosine_hist(diag: np.ndarray, off: np.ndarray, variant: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(diag.min(), off.min()) - 0.05,
                       max(diag.max(), off.max()) + 0.05, 60)
    ax.hist(off, bins=bins, alpha=0.5, color="gray",
            label=f"role x other-role-name (n={len(off)})", density=True)
    ax.hist(diag, bins=bins, alpha=0.7, color="C0",
            label=f"role x own-role-name (n={len(diag)})", density=True)
    ax.axvline(diag.mean(), color="C0", ls="--",
               label=f"matched mean = {diag.mean():.3f}")
    ax.axvline(off.mean(), color="black", ls=":",
               label=f"unmatched mean = {off.mean():.3f}")
    ax.set_xlabel("cosine similarity (persona vector ↔ semantic vector)")
    ax.set_ylabel("density")
    ax.set_title(f"Cosine similarity, variant = {variant}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_bottom(names: List[str], diag: np.ndarray, variant: str, path: str, k: int = 25):
    order = np.argsort(diag)
    bottom_idx = order[:k]
    top_idx = order[-k:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, k * 0.25)))
    for ax, idx, title in [
        (axes[0], top_idx, f"top {k} roles where name ≈ persona"),
        (axes[1], bottom_idx, f"bottom {k} roles"),
    ]:
        ax.barh(range(len(idx))[::-1], diag[idx], color="C0")
        ax.set_yticks(range(len(idx))[::-1])
        ax.set_yticklabels([names[i] for i in idx], fontsize=8)
        ax.set_xlabel("cosine similarity")
        ax.set_title(title)
        ax.axvline(0, color="k", lw=0.5)
    fig.suptitle(f"variant = {variant}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_heatmap(names: List[str], M: np.ndarray, variant: str, path: str):
    order = np.argsort(-np.diag(M))  # high-similarity rows first
    M_sorted = M[order][:, order]
    fig, ax = plt.subplots(figsize=(12, 10))
    vmax = max(abs(M.min()), abs(M.max()))
    im = ax.imshow(M_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(
        f"Cross-cosine: persona[i] ⋅ semantic[j], variant={variant}\n"
        "rows/cols sorted by diagonal (matched) similarity"
    )
    ax.set_xlabel("semantic vector (role name)")
    ax.set_ylabel("persona vector (role)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="cosine")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca_overlay(
    names: List[str],
    persona: np.ndarray,
    semantic: np.ndarray,
    variant: str,
    path: str,
    label_n: int = 30,
):
    stacked = np.vstack([persona, semantic])
    joint_mean = stacked.mean(axis=0)
    pca = PCA(n_components=2).fit(stacked - joint_mean)
    p_proj = pca.transform(persona - joint_mean)
    s_proj = pca.transform(semantic - joint_mean)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(p_proj[:, 0], p_proj[:, 1], s=24, c="C0",
               alpha=0.7, edgecolors="k", linewidths=0.3, label="persona vector")
    ax.scatter(s_proj[:, 0], s_proj[:, 1], s=24, c="C3", marker="^",
               alpha=0.7, edgecolors="k", linewidths=0.3, label="semantic name vector")

    # connect matched pairs
    for i in range(len(names)):
        ax.plot([p_proj[i, 0], s_proj[i, 0]], [p_proj[i, 1], s_proj[i, 1]],
                color="gray", lw=0.3, alpha=0.4)

    # label only a subset to keep the plot readable
    if label_n and len(names) > label_n:
        sample_idx = np.linspace(0, len(names) - 1, label_n).astype(int)
    else:
        sample_idx = np.arange(len(names))
    texts = []
    for i in sample_idx:
        texts.append(ax.text(p_proj[i, 0], p_proj[i, 1], names[i], fontsize=6, color="C0"))
        texts.append(ax.text(s_proj[i, 0], s_proj[i, 1], names[i], fontsize=6, color="C3"))
    adjust_text(texts, ax=ax, only_move={"text": "xy"}, expand=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.3, alpha=0.4))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"Shared PCA — persona vs semantic, variant = {variant}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne_overlay(
    names: List[str],
    persona: np.ndarray,
    semantic: np.ndarray,
    variant: str,
    path: str,
    label_n: int = 30,
):
    n = len(names)
    stacked = np.vstack([persona, semantic])
    stacked = stacked - stacked.mean(axis=0)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, n // 6)),
                random_state=42, init="pca")
    proj = tsne.fit_transform(stacked)
    p_proj = proj[:n]
    s_proj = proj[n:]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(p_proj[:, 0], p_proj[:, 1], s=24, c="C0",
               alpha=0.7, edgecolors="k", linewidths=0.3, label="persona vector")
    ax.scatter(s_proj[:, 0], s_proj[:, 1], s=24, c="C3", marker="^",
               alpha=0.7, edgecolors="k", linewidths=0.3, label="semantic name vector")
    for i in range(n):
        ax.plot([p_proj[i, 0], s_proj[i, 0]], [p_proj[i, 1], s_proj[i, 1]],
                color="gray", lw=0.3, alpha=0.4)

    if label_n and n > label_n:
        sample_idx = np.linspace(0, n - 1, label_n).astype(int)
    else:
        sample_idx = np.arange(n)
    texts = []
    for i in sample_idx:
        texts.append(ax.text(p_proj[i, 0], p_proj[i, 1], names[i], fontsize=6, color="C0"))
        texts.append(ax.text(s_proj[i, 0], s_proj[i, 1], names[i], fontsize=6, color="C3"))
    adjust_text(texts, ax=ax, only_move={"text": "xy"}, expand=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.3, alpha=0.4))

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Shared t-SNE — persona vs semantic, variant = {variant}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_procrustes(persona: np.ndarray, semantic: np.ndarray) -> Dict[str, float]:
    # procrustes returns standardized inputs and disparity; lower is better,
    # 0 means a perfect orthogonal+scale match.
    _, _, disparity = procrustes(persona, semantic)
    # Also report alignment via Frobenius after Orthogonal-Procrustes-style fit
    # using SVD: argmin_R || A R - B ||_F over orthogonal R.
    A = persona - persona.mean(axis=0)
    B = semantic - semantic.mean(axis=0)
    A /= np.linalg.norm(A) + 1e-12
    B /= np.linalg.norm(B) + 1e-12
    M = A.T @ B
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return {
        "procrustes_disparity": float(disparity),
        "orth_procrustes_alignment": float(S.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_ROWS.keys()),
                        choices=list(VARIANT_ROWS.keys()))
    parser.add_argument("--semantic-file", default=SEMANTIC_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading persona vectors (layer={args.layer}, count={args.sample_count})")
    persona_dict = load_persona_vectors(args.layer, args.sample_count)
    print(f"  -> {len(persona_dict)} persona vectors")

    print(f"Loading semantic vectors from {args.semantic_file}")
    semantic_raw = load_file(args.semantic_file)
    print(f"  -> {len(semantic_raw)} semantic vectors")

    common = sorted(set(persona_dict) & set(semantic_raw))
    print(f"Common roles: {len(common)}")
    if not common:
        raise RuntimeError("No overlap between persona and semantic role sets.")

    persona_mat = np.stack([persona_dict[r] for r in common]).astype(np.float32)

    # Per-role results across variants
    per_role: Dict[str, Dict[str, float]] = {r: {} for r in common}

    for variant in args.variants:
        row = VARIANT_ROWS[variant]
        sem_mat = np.stack([semantic_raw[r][row] for r in common]).astype(np.float32)

        cos = cosine_matrix(persona_mat, sem_mat)
        diag = np.diag(cos)
        off_mask = ~np.eye(len(common), dtype=bool)
        off = cos[off_mask]

        for i, r in enumerate(common):
            per_role[r][f"cos_{variant}"] = float(diag[i])

        proc = run_procrustes(persona_mat, sem_mat)
        with open(os.path.join(args.output_dir, f"procrustes_{variant}.json"), "w") as f:
            json.dump({
                "variant": variant,
                "matched_cos_mean": float(diag.mean()),
                "matched_cos_median": float(np.median(diag)),
                "matched_cos_std": float(diag.std()),
                "unmatched_cos_mean": float(off.mean()),
                "unmatched_cos_std": float(off.std()),
                "frac_matched_above_unmatched_mean": float((diag > off.mean()).mean()),
                **proc,
                "n_roles": int(len(common)),
            }, f, indent=2)

        print(
            f"[{variant}] matched cos = {diag.mean():.3f} ± {diag.std():.3f} | "
            f"unmatched = {off.mean():.3f} ± {off.std():.3f} | "
            f"procrustes disparity = {proc['procrustes_disparity']:.4f}"
        )

        plot_cosine_hist(diag, off, variant,
                         os.path.join(args.output_dir, f"cosine_similarity_hist_{variant}.png"))
        plot_top_bottom(common, diag, variant,
                        os.path.join(args.output_dir, f"cosine_similarity_top_bottom_{variant}.png"))
        plot_cross_heatmap(common, cos, variant,
                           os.path.join(args.output_dir, f"cross_similarity_heatmap_{variant}.png"))
        plot_pca_overlay(common, persona_mat, sem_mat, variant,
                         os.path.join(args.output_dir, f"pca_overlay_{variant}.png"))
        plot_tsne_overlay(common, persona_mat, sem_mat, variant,
                          os.path.join(args.output_dir, f"tsne_overlay_{variant}.png"))

    # CSV summary
    csv_path = os.path.join(args.output_dir, "summary.csv")
    fields = ["role"] + [f"cos_{v}" for v in args.variants]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in common:
            row = {"role": r}
            row.update(per_role[r])
            w.writerow(row)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
