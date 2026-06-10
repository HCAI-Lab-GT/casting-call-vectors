"""
Intrinsic dimensionality of the role-vector cloud.

Loads 275 role vectors (4096-dim, layer 16) and asks: how many "true" persona
directions are there in the cloud? If the answer is, say, 8, then most cosine
differences between two random roles are noise on the remaining 4088 axes,
which would explain why RSA(repr_cos, beh_corr) ~ 0.14 instead of ~ 0.5.

Metrics
-------
- Singular value spectrum (raw and centered).
- Cumulative explained variance: smallest d s.t. >=50/80/90/95/99% explained.
- Effective rank (Roy & Vetterli, 2007):
      eff_rank = exp(- sum p_i log p_i),  where p_i = sigma_i^2 / sum_j sigma_j^2
- Participation ratio:
      PR = (sum sigma_i^2)^2 / sum sigma_i^4
- Anisotropy / mean cosine baseline:
      mean cosine of all pairs (high anisotropy => global bias direction).
- Random-Gaussian baseline of the same shape as a null reference (so we can
  separate "real low-rank structure" from "rank constrained by sample count").

Behavioral side too
-------------------
We compute the same diagnostics for the 24-dim behavioral profile (output of
analysis/empirical/rsa_geometry_behavior/data/behavioral_profiles.csv) so we
can compare representational vs behavioral effective rank in one plot.

Outputs (data/)
---------------
- spectrum_repr_raw.csv       sigma_i (raw, untouched)
- spectrum_repr_centered.csv  sigma_i (after subtracting role-mean direction)
- spectrum_beh.csv            sigma_i for the 275x24 behavioral matrix
- spectrum_random.csv         sigma_i for an iid-Gaussian 275x4096 null
- summary.csv                 effective_rank, PR, mean cos, etc, per matrix
- explained_variance_curves.csv  cumulative variance vs d, for plotting
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
VECTORS_DIR = REPO / "persona_data" / "pt_vectors"
RSA_DATA_DIR = REPO / "analysis" / "empirical" / "rsa_geometry_behavior" / "data"
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EV_THRESHOLDS = [0.50, 0.80, 0.90, 0.95, 0.99]


def load_role_matrix(role_filter: list[str]) -> tuple[list[str], np.ndarray]:
    name_to_path = {
        os.path.splitext(os.path.basename(fp))[0]: fp
        for fp in sorted(glob.glob(str(VECTORS_DIR / "*.pt")))
    }
    names: list[str] = []
    mats: list[np.ndarray] = []
    for r in role_filter:
        if r not in name_to_path:
            continue
        v = torch.load(name_to_path[r], map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            v = next(t for t in v.values() if torch.is_tensor(t))
        v = v.float().squeeze().numpy()
        names.append(r)
        mats.append(v)
    return names, np.stack(mats, axis=0)


def spectrum_diagnostics(name: str, X: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Return (spectrum dataframe, summary dict) for matrix X (n_samples, dim)."""
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    var = s ** 2
    var_total = var.sum()
    p = var / (var_total + 1e-12)
    cumvar = np.cumsum(p)

    # Effective rank (Shannon entropy of normalized squared singular values)
    nz = p[p > 0]
    eff_rank = float(np.exp(-(nz * np.log(nz)).sum()))
    # Participation ratio
    pr = float((var.sum() ** 2) / (np.sum(var ** 2) + 1e-12))

    # Mean pairwise cosine (anisotropy proxy) on row-centered X
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos_full = Xn @ Xn.T
    iu = np.triu_indices_from(cos_full, k=1)
    mean_cos = float(cos_full[iu].mean())
    median_cos = float(np.median(cos_full[iu]))

    summary = {
        "matrix": name,
        "n_samples": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "max_rank": int(min(X.shape)),
        "effective_rank": eff_rank,
        "participation_ratio": pr,
        "mean_pairwise_cosine": mean_cos,
        "median_pairwise_cosine": median_cos,
        "frobenius_norm": float(np.linalg.norm(X)),
    }
    for thr in EV_THRESHOLDS:
        d = int(np.searchsorted(cumvar, thr) + 1)
        summary[f"d_explains_{int(thr*100)}pct"] = d

    spec_rows = []
    for i, (sv, pi, ci) in enumerate(zip(s, p, cumvar)):
        spec_rows.append(
            {
                "rank_index": i + 1,
                "singular_value": float(sv),
                "variance_share": float(pi),
                "cumulative_variance": float(ci),
            }
        )
    return pd.DataFrame(spec_rows), summary


def main() -> None:
    role_order = json.loads((RSA_DATA_DIR / "role_order.json").read_text())
    print(f"Loading role vectors (target n={len(role_order)}) ...")
    names, X_repr = load_role_matrix(role_order)
    print(f"  loaded {X_repr.shape[0]} x {X_repr.shape[1]}")

    # Centered (subtract mean direction) -- removes the assistant-axis-like
    # rank-1 component so we measure dispersion *around* the cloud center.
    X_centered = X_repr - X_repr.mean(axis=0, keepdims=True)

    # Behavioral matrix (n x 24), z-scored per column
    beh_df = pd.read_csv(RSA_DATA_DIR / "behavioral_profiles.csv")
    beh_df = beh_df.set_index("role").loc[names]
    X_beh = beh_df.to_numpy(dtype=float)
    X_beh = (X_beh - X_beh.mean(axis=0, keepdims=True)) / (X_beh.std(axis=0, keepdims=True) + 1e-12)

    # iid Gaussian null with same shape and global std as X_repr -- reference for
    # "what does an unstructured cloud's spectrum look like at this aspect ratio?"
    rng = np.random.default_rng(0)
    X_null = rng.normal(0, X_repr.std(), size=X_repr.shape)

    print("Computing spectra ...")
    spec_raw, sum_raw = spectrum_diagnostics("repr_raw", X_repr)
    spec_cen, sum_cen = spectrum_diagnostics("repr_centered", X_centered)
    spec_beh, sum_beh = spectrum_diagnostics("behavioral", X_beh)
    spec_null, sum_null = spectrum_diagnostics("repr_iid_gaussian_null", X_null)

    spec_raw.to_csv(DATA_DIR / "spectrum_repr_raw.csv", index=False)
    spec_cen.to_csv(DATA_DIR / "spectrum_repr_centered.csv", index=False)
    spec_beh.to_csv(DATA_DIR / "spectrum_beh.csv", index=False)
    spec_null.to_csv(DATA_DIR / "spectrum_random.csv", index=False)

    summary = pd.DataFrame([sum_raw, sum_cen, sum_beh, sum_null])
    summary.to_csv(DATA_DIR / "summary.csv", index=False)

    # Long-format explained-variance curves for easy plotting
    long_rows = []
    for label, spec in [
        ("repr_raw", spec_raw),
        ("repr_centered", spec_cen),
        ("behavioral", spec_beh),
        ("repr_random_null", spec_null),
    ]:
        for _, row in spec.iterrows():
            long_rows.append(
                {
                    "matrix": label,
                    "rank_index": int(row["rank_index"]),
                    "cumulative_variance": float(row["cumulative_variance"]),
                    "variance_share": float(row["variance_share"]),
                }
            )
    pd.DataFrame(long_rows).to_csv(DATA_DIR / "explained_variance_curves.csv", index=False)

    print("\n=== Effective rank summary ===")
    cols_to_show = [
        "matrix", "n_samples", "dim", "max_rank",
        "effective_rank", "participation_ratio",
        "mean_pairwise_cosine",
        "d_explains_50pct", "d_explains_90pct", "d_explains_99pct",
    ]
    print(summary[cols_to_show].to_string(index=False))

    print(f"\nWrote outputs to {DATA_DIR}")


if __name__ == "__main__":
    main()
