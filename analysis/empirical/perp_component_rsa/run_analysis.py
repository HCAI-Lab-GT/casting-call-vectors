"""
Perp-component RSA: does the role-specific residual drive behavioral specificity?

Each proposed role vector is decomposed as v_i = v_∥ + v_⊥ where:
  v_∥ = (v_i · d_aa) * d_aa   — captured by the assistant axis
  v_⊥ = v_i − v_∥              — role-specific residual the AA discards

We build pairwise cosine-distance RDMs from both components and ask:
  RSA(perp_rdm,  beh_steered)  vs  RSA(perp_rdm,  beh_aa)
  RSA(parallel_rdm, beh_aa)   vs  RSA(parallel_rdm, beh_steered)

Expected result if our method is geometrically motivated:
  • perp_rdm  better predicts steered behavior (our method uses v_⊥)
  • parallel_rdm better predicts AA behavior (AA only captures v_∥)
  • Together this shows the two methods are sensitive to different geometric subspaces

Outputs (data/):
  perp_rdm.npy          275×275 cosine-distance RDM of role-specific residuals
  parallel_rdm.npy      275×275 cosine-distance RDM of assistant-axis projections
  rsa_perp_components.csv  RSA results with Mantel p-values
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform

REPO = Path(__file__).resolve().parents[3]
PT_VECTORS_DIR = REPO / "persona_data" / "pt_vectors"
AA_VECTORS_DIR = REPO / "persona_data" / "assistant-axis" / "olmo-3-7b-instruct" / "vectors"
RSA_DATA_DIR = REPO / "analysis" / "empirical" / "rsa_geometry_behavior" / "data"
BEH_DIV_DIR = REPO / "analysis" / "empirical" / "behavioral_diversity" / "data"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 5_000
RNG_SEED = 42


# ── vector loading ─────────────────────────────────────────────────────────────

def _extract_tensor(payload) -> torch.Tensor | None:
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, dict):
        for k in ("vector", "persona_vector", "delta", "axis"):
            if k in payload and torch.is_tensor(payload[k]):
                return payload[k]
        for v in payload.values():
            if torch.is_tensor(v):
                return v
    return None


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().float()
    if t.ndim == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 2:
        idx = 15 if t.shape[0] > 15 else t.shape[0] - 1
        t = t[idx]
    return t.numpy()


def load_vectors(directory: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fp in sorted(directory.glob("*.pt")):
        role = fp.stem
        try:
            payload = torch.load(fp, map_location="cpu", weights_only=False)
            t = _extract_tensor(payload)
            if t is None:
                continue
            out[role] = _to_numpy(t)
        except Exception as e:
            print(f"  skip {fp.name}: {e}")
    return out


# ── geometry ───────────────────────────────────────────────────────────────────

def compute_d_aa(aa_vectors: dict[str, np.ndarray]) -> np.ndarray:
    mats = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in aa_vectors.values()])
    mean_dir = mats.mean(axis=0)
    return mean_dir / (np.linalg.norm(mean_dir) + 1e-12)


def decompose_vectors(vectors: dict[str, np.ndarray],
                      d_aa: np.ndarray,
                      role_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (perp_matrix, parallel_matrix), each shape (n_roles, dim)."""
    perps, pars = [], []
    for role in role_order:
        v = vectors[role]
        dot = float(np.dot(v, d_aa))   # d_aa is unit, so this is the scalar projection
        v_par = dot * d_aa
        v_perp = v - v_par
        perps.append(v_perp)
        pars.append(v_par)
    return np.stack(perps), np.stack(pars)


def cos_rdm(mat: np.ndarray) -> np.ndarray:
    """Pairwise cosine-distance RDM. Rows with near-zero norm get distance 1 to all others."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = norms > 1e-12
    mat_n = np.where(safe, mat / (norms + 1e-12), 0.0)
    sim = np.clip(mat_n @ mat_n.T, -1.0, 1.0)
    return 1.0 - sim


# ── Mantel test ────────────────────────────────────────────────────────────────

def rdm_vec(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices_from(rdm, k=1)]


def mantel_test(rdm_a: np.ndarray, rdm_b: np.ndarray,
                n_perm: int = N_PERM, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = float(stats.spearmanr(rdm_vec(rdm_a), rdm_vec(rdm_b)).statistic)
    n = rdm_a.shape[0]
    a_vec = rdm_vec(rdm_a)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = float(stats.spearmanr(a_vec, rdm_vec(rdm_b[perm][:, perm])).statistic)
    p = max(float(np.mean(np.abs(null) >= np.abs(obs))), 1.0 / n_perm)
    return obs, p


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load role order
    with open(RSA_DATA_DIR / "role_order.json") as f:
        role_order = json.load(f)
    print(f"Role order: {len(role_order)} roles")

    # Load vectors
    print(f"Loading proposed vectors from {PT_VECTORS_DIR} ...")
    proposed = load_vectors(PT_VECTORS_DIR)
    print(f"  loaded {len(proposed)}")

    print(f"Loading AA vectors from {AA_VECTORS_DIR} ...")
    aa_vecs = load_vectors(AA_VECTORS_DIR)
    print(f"  loaded {len(aa_vecs)}")

    common = [r for r in role_order if r in proposed and r in aa_vecs]
    print(f"  common roles in role_order: {len(common)}")

    # Compute d_aa and decompose
    d_aa = compute_d_aa({r: aa_vecs[r] for r in common})
    perp_mat, par_mat = decompose_vectors(proposed, d_aa, common)
    print(f"  perp_mat shape: {perp_mat.shape}  par_mat shape: {par_mat.shape}")

    # Build component RDMs
    print("Building component RDMs ...")
    perp_rdm = cos_rdm(perp_mat)
    par_rdm = cos_rdm(par_mat)
    np.save(DATA_DIR / "perp_rdm.npy", perp_rdm)
    np.save(DATA_DIR / "parallel_rdm.npy", par_rdm)
    print(f"  perp_rdm: min={perp_rdm[perp_rdm > 0].min():.3f}  max={perp_rdm.max():.3f}")

    # Load behavioral and repr RDMs (aligned to role_order.json)
    full_repr_rdm = np.load(RSA_DATA_DIR / "rdm_repr_cos.npy")
    full_beh_s_rdm = np.load(RSA_DATA_DIR / "rdm_beh_corr.npy")  # steered, corr-dist
    full_beh_aa_rdm = np.load(BEH_DIV_DIR / "rdm_beh_aa_corr.npy")

    # Slice to common roles (role_order is the index for full RDMs)
    full_role_to_idx = {r: i for i, r in enumerate(role_order)}
    common_idx = np.array([full_role_to_idx[r] for r in common], dtype=int)

    repr_rdm = full_repr_rdm[np.ix_(common_idx, common_idx)]
    beh_s_rdm = full_beh_s_rdm[np.ix_(common_idx, common_idx)]
    beh_aa_rdm = full_beh_aa_rdm[np.ix_(common_idx, common_idx)]
    print(f"  Sliced all RDMs to {len(common)}×{len(common)}")

    # ── RSA comparisons ────────────────────────────────────────────────────────
    comparisons = [
        ("perp_cos",     perp_rdm, "beh_steered_corr", beh_s_rdm),
        ("perp_cos",     perp_rdm, "beh_aa_corr",      beh_aa_rdm),
        ("perp_cos",     perp_rdm, "repr_cos",          repr_rdm),
        ("parallel_cos", par_rdm,  "beh_steered_corr", beh_s_rdm),
        ("parallel_cos", par_rdm,  "beh_aa_corr",      beh_aa_rdm),
        ("parallel_cos", par_rdm,  "repr_cos",          repr_rdm),
        ("repr_cos",     repr_rdm, "beh_steered_corr", beh_s_rdm),  # reference
        ("repr_cos",     repr_rdm, "beh_aa_corr",      beh_aa_rdm),  # reference
    ]

    print(f"\nRunning Mantel tests (n_perm={N_PERM}) ...")
    rows = []
    for rdm_a_name, rdm_a, rdm_b_name, rdm_b in comparisons:
        obs, p = mantel_test(rdm_a, rdm_b)
        rows.append(dict(
            rdm_a=rdm_a_name, rdm_b=rdm_b_name,
            comparison=f"{rdm_a_name}_vs_{rdm_b_name}",
            rsa_spearman=obs, mantel_p=p, n_roles=len(common),
        ))
        print(f"  {rdm_a_name:14s} vs {rdm_b_name:20s}: r={obs:+.4f}  p={p:.2e}")

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "rsa_perp_components.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'rsa_perp_components.csv'}")

    # Summary
    perp_s = out[out["comparison"] == "perp_cos_vs_beh_steered_corr"]["rsa_spearman"].iloc[0]
    perp_a = out[out["comparison"] == "perp_cos_vs_beh_aa_corr"]["rsa_spearman"].iloc[0]
    par_s  = out[out["comparison"] == "parallel_cos_vs_beh_steered_corr"]["rsa_spearman"].iloc[0]
    par_a  = out[out["comparison"] == "parallel_cos_vs_beh_aa_corr"]["rsa_spearman"].iloc[0]

    print(f"\nKey result:")
    print(f"  RSA(perp,    beh_steered) = {perp_s:+.4f}  "
          f"RSA(perp,    beh_aa) = {perp_a:+.4f}  "
          f"→ perp favours {'steered' if perp_s > perp_a else 'AA'}")
    print(f"  RSA(parallel, beh_steered) = {par_s:+.4f}  "
          f"RSA(parallel, beh_aa) = {par_a:+.4f}  "
          f"→ parallel favours {'steered' if par_s > par_a else 'AA'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
