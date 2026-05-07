"""
RSA between role-vector geometry and steered-behavior geometry (n = 275 roles).

Pipeline
--------
1. Load every persona vector in persona_data/pt_vectors/*.pt (4096-dim, layer 16).
2. Aggregate per-role gold-prompt judge scores in
   experiment_data/gold_prompt_experiments/Comparison_GoldStandard_*.csv into a
   per-role behavioral profile of 4 alphas x (steered_score + 5 sub-dims) = 24
   features. Also build per-alpha 6-feature profiles for an alpha breakdown.
3. Build pairwise distance matrices (RDMs):
     - RDM_repr_cos : 1 - cosine(v_i, v_j)        on role vectors
     - RDM_repr_l2  : || v_i - v_j ||_2           on role vectors
     - RDM_beh_corr : 1 - pearson(b_i, b_j)       on behavioral profile
     - RDM_beh_l2   : || b_i - b_j ||_2           on z-scored behavioral profile
4. RSA = Spearman correlation between upper-triangles of RDM_repr and RDM_beh.
   Compute for all 4 (repr_distance, beh_distance) combinations, plus per-alpha.
5. Mantel permutation test (n_perm=10000): shuffle role labels of one RDM,
   recompute, get a null distribution -> two-sided p-value.
6. Partial RSA controlling for ||v|| (does the link survive after partialling
   out the magnitude effect documented in vector_magnitude_effect_on_stability)?

Outputs (data/)
---------------
- behavioral_profiles.csv : one row per role, the 24-feature vector
- rdm_repr_cos.npy        : 275x275 representational RDM (cosine)
- rdm_repr_l2.npy         : 275x275 representational RDM (l2)
- rdm_beh_corr.npy        : 275x275 behavioral RDM (corr-dist on full profile)
- rdm_beh_l2.npy          : 275x275 behavioral RDM (l2 on z-scored profile)
- rdm_beh_corr_alpha{a}.npy : per-alpha behavioral RDMs (4 files)
- role_order.json         : role names in matrix index order
- rsa_main.csv            : (repr_dist, beh_dist) -> RSA, Mantel p-value
- rsa_per_alpha.csv       : RSA at each alpha (uses only that alpha's 6 features)
- rsa_partial.csv         : RSA after partialling out role-vector norm
- mantel_null_distribution.npy : the null Spearman r values (for histograms)
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform

REPO = Path(__file__).resolve().parents[3]
VECTORS_DIR = REPO / "persona_data" / "pt_vectors"
GOLD_DIR = REPO / "experiment_data" / "gold_prompt_experiments"
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]
SUBDIMS = [
    "cmp_emotional_register",
    "cmp_vocab_choice",
    "cmp_social_dynamic",
    "cmp_motivation",
    "cmp_worldview_alignment",
]
BEH_FEATURES_PER_ALPHA = ["steered_score"] + SUBDIMS  # 6 features per alpha
N_PERM = 10_000
RNG_SEED = 0


# ------------------------------ data loading ----------------------------------

def load_role_vectors() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (role_names, vector_matrix [n_roles, dim], norms [n_roles])."""
    names: list[str] = []
    mats: list[np.ndarray] = []
    for fp in sorted(glob.glob(str(VECTORS_DIR / "*.pt"))):
        role = os.path.splitext(os.path.basename(fp))[0]
        v = torch.load(fp, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            v = next(t for t in v.values() if torch.is_tensor(t))
        v = v.float().squeeze().numpy()
        names.append(role)
        mats.append(v)
    mat = np.stack(mats, axis=0)
    norms = np.linalg.norm(mat, axis=1)
    return names, mat, norms


def load_behavioral_profiles(role_filter: list[str]) -> pd.DataFrame:
    """One row per role with 4 * 6 = 24 columns: <feature>_alpha_<a>."""
    rows = []
    for fp in sorted(glob.glob(str(GOLD_DIR / "Comparison_GoldStandard_*.csv"))):
        role = os.path.basename(fp).replace("Comparison_GoldStandard_", "").replace(".csv", "")
        if role not in role_filter:
            continue
        df = pd.read_csv(fp)
        if not set(ALPHAS).issubset(set(df["alpha"].unique())):
            continue
        per_alpha = df.groupby("alpha")[BEH_FEATURES_PER_ALPHA].mean()
        flat: dict = {"role": role}
        for a in ALPHAS:
            for feat in BEH_FEATURES_PER_ALPHA:
                flat[f"{feat}__alpha_{a}"] = float(per_alpha.loc[a, feat])
        rows.append(flat)
    return pd.DataFrame(rows)


# ------------------------------ RDMs / RSA ------------------------------------

def cosine_rdm(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def l2_rdm(X: np.ndarray) -> np.ndarray:
    return squareform(pdist(X, metric="euclidean"))


def corr_rdm(X: np.ndarray) -> np.ndarray:
    """1 - Pearson correlation between rows. Works for short feature vectors."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def rdm_to_vec(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(rdm, k=1)
    return rdm[iu]


def rsa_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    return float(stats.spearmanr(rdm_to_vec(rdm_a), rdm_to_vec(rdm_b)).statistic)


def mantel_test(
    rdm_a: np.ndarray, rdm_b: np.ndarray, n_perm: int = N_PERM, seed: int = RNG_SEED
) -> tuple[float, float, np.ndarray]:
    """Two-sided Mantel test using Spearman. Returns (observed, p, null_dist)."""
    rng = np.random.default_rng(seed)
    observed = rsa_spearman(rdm_a, rdm_b)
    n = rdm_a.shape[0]
    null = np.empty(n_perm, dtype=float)
    a_vec = rdm_to_vec(rdm_a)
    for i in range(n_perm):
        perm = rng.permutation(n)
        b_perm = rdm_b[perm][:, perm]
        null[i] = float(stats.spearmanr(a_vec, rdm_to_vec(b_perm)).statistic)
    p = float(np.mean(np.abs(null) >= np.abs(observed)))
    # avoid p == 0 reporting; floor at 1/n_perm
    p = max(p, 1.0 / n_perm)
    return observed, p, null


def partial_rsa(
    rdm_a: np.ndarray, rdm_b: np.ndarray, rdm_c: np.ndarray
) -> float:
    """Spearman partial correlation of (vec(a), vec(b)) controlling for vec(c).

    Uses rank residuals: rank-transform each vector, regress out c, correlate.
    """
    a = stats.rankdata(rdm_to_vec(rdm_a))
    b = stats.rankdata(rdm_to_vec(rdm_b))
    c = stats.rankdata(rdm_to_vec(rdm_c))

    def residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        x_ = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(x_, y, rcond=None)
        return y - x_ @ beta

    return float(stats.pearsonr(residual(a, c), residual(b, c)).statistic)


# ------------------------------ main ------------------------------------------

def main() -> None:
    print(f"Loading vectors from {VECTORS_DIR}")
    all_names, all_vecs, all_norms = load_role_vectors()
    print(f"  vectors: {len(all_names)}, dim={all_vecs.shape[1]}")

    print(f"Loading behavioral data from {GOLD_DIR}")
    beh = load_behavioral_profiles(role_filter=all_names)
    print(f"  roles with full alpha coverage: {len(beh)}")

    # Align the two sources on the intersection, preserving role order from beh.
    keep = beh["role"].tolist()
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    idx = np.array([name_to_idx[r] for r in keep], dtype=int)
    vec_mat = all_vecs[idx]            # (n, 4096)
    norms = all_norms[idx]             # (n,)
    role_names = keep                  # length n
    n = len(role_names)

    feat_cols = [
        f"{feat}__alpha_{a}" for a in ALPHAS for feat in BEH_FEATURES_PER_ALPHA
    ]
    beh_mat_raw = beh[feat_cols].to_numpy(dtype=float)  # (n, 24)
    # Z-score per feature so L2 isn't dominated by the highest-variance dim
    beh_mu = beh_mat_raw.mean(axis=0, keepdims=True)
    beh_sd = beh_mat_raw.std(axis=0, keepdims=True) + 1e-12
    beh_mat_z = (beh_mat_raw - beh_mu) / beh_sd

    # ---- Build RDMs ----
    print("Building RDMs ...")
    rdm_repr_cos = cosine_rdm(vec_mat)
    rdm_repr_l2 = l2_rdm(vec_mat)
    rdm_beh_corr = corr_rdm(beh_mat_raw)   # corr-dist on raw 24-d profile
    rdm_beh_l2 = l2_rdm(beh_mat_z)         # L2 on z-scored profile

    # Norm-difference RDM (control for partialling)
    rdm_norm = squareform(pdist(norms.reshape(-1, 1), metric="euclidean"))

    # ---- Save matrices ----
    np.save(DATA_DIR / "rdm_repr_cos.npy", rdm_repr_cos)
    np.save(DATA_DIR / "rdm_repr_l2.npy", rdm_repr_l2)
    np.save(DATA_DIR / "rdm_beh_corr.npy", rdm_beh_corr)
    np.save(DATA_DIR / "rdm_beh_l2.npy", rdm_beh_l2)
    np.save(DATA_DIR / "rdm_norm.npy", rdm_norm)
    with open(DATA_DIR / "role_order.json", "w") as f:
        json.dump(role_names, f, indent=2)
    beh.to_csv(DATA_DIR / "behavioral_profiles.csv", index=False)
    pd.DataFrame({"role": role_names, "norm": norms}).to_csv(
        DATA_DIR / "role_norms.csv", index=False
    )

    # ---- Main RSA grid (4 combos), with Mantel p-values ----
    print(f"Running Mantel tests (n_perm={N_PERM}) ...")
    main_rows = []
    null_to_save: dict[str, np.ndarray] = {}
    combos = [
        ("repr_cos", rdm_repr_cos, "beh_corr", rdm_beh_corr),
        ("repr_cos", rdm_repr_cos, "beh_l2", rdm_beh_l2),
        ("repr_l2", rdm_repr_l2, "beh_corr", rdm_beh_corr),
        ("repr_l2", rdm_repr_l2, "beh_l2", rdm_beh_l2),
    ]
    for repr_name, repr_rdm, beh_name, beh_rdm in combos:
        obs, p, null = mantel_test(repr_rdm, beh_rdm)
        main_rows.append(
            {
                "repr_distance": repr_name,
                "beh_distance": beh_name,
                "rsa_spearman": obs,
                "mantel_p": p,
                "n_roles": n,
                "n_perm": N_PERM,
            }
        )
        null_to_save[f"{repr_name}__{beh_name}"] = null
        print(f"  {repr_name:9s} <-> {beh_name:9s}: r={obs:+.4f}  p={p:.2e}")
    pd.DataFrame(main_rows).to_csv(DATA_DIR / "rsa_main.csv", index=False)
    np.savez(DATA_DIR / "mantel_null_distribution.npz", **null_to_save)

    # ---- Per-alpha RSA: behavioral profile = 6 features at single alpha ----
    print("Per-alpha RSA breakdown ...")
    per_alpha_rows = []
    for a in ALPHAS:
        cols = [f"{feat}__alpha_{a}" for feat in BEH_FEATURES_PER_ALPHA]
        b_a_raw = beh[cols].to_numpy(dtype=float)
        b_a_z = (b_a_raw - b_a_raw.mean(axis=0)) / (b_a_raw.std(axis=0) + 1e-12)
        rdm_b_corr = corr_rdm(b_a_raw)
        rdm_b_l2 = l2_rdm(b_a_z)
        np.save(DATA_DIR / f"rdm_beh_corr_alpha{a}.npy", rdm_b_corr)

        for repr_name, repr_rdm in [("repr_cos", rdm_repr_cos)]:
            for beh_name, beh_rdm in [("beh_corr", rdm_b_corr), ("beh_l2", rdm_b_l2)]:
                obs, p, _ = mantel_test(repr_rdm, beh_rdm, n_perm=2000)
                per_alpha_rows.append(
                    {
                        "alpha": a,
                        "repr_distance": repr_name,
                        "beh_distance": beh_name,
                        "rsa_spearman": obs,
                        "mantel_p": p,
                    }
                )
                print(
                    f"  alpha={a}  {repr_name} <-> {beh_name}: r={obs:+.4f}  p={p:.2e}"
                )
    pd.DataFrame(per_alpha_rows).to_csv(DATA_DIR / "rsa_per_alpha.csv", index=False)

    # ---- Partial RSA controlling for ||v|| ----
    print("Partial RSA controlling for vector norm ...")
    part_rows = []
    for repr_name, repr_rdm, beh_name, beh_rdm in combos:
        full = rsa_spearman(repr_rdm, beh_rdm)
        part = partial_rsa(repr_rdm, beh_rdm, rdm_norm)
        part_rows.append(
            {
                "repr_distance": repr_name,
                "beh_distance": beh_name,
                "rsa_full": full,
                "rsa_partial_minus_norm": part,
                "delta": part - full,
            }
        )
        print(
            f"  {repr_name} <-> {beh_name}: full r={full:+.4f}  "
            f"partial r={part:+.4f}  delta={part-full:+.4f}"
        )
    pd.DataFrame(part_rows).to_csv(DATA_DIR / "rsa_partial.csv", index=False)

    print("\nDone. Wrote:")
    for fp in sorted(DATA_DIR.iterdir()):
        print(f"  {fp.relative_to(REPO)}")


if __name__ == "__main__":
    main()
