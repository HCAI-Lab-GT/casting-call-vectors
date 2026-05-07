"""
Estimate the noise floor / reliability ceiling of the behavioral RDM used in
analysis/empirical/rsa_geometry_behavior/.

Question
--------
We measured RSA(repr_cos, beh_corr) = +0.14 and RSA(repr_cos, beh_l2) = +0.23
across 275 roles. Is that small because the geometry-behavior link is weak,
or because judge noise caps the achievable RSA at, say, 0.30?

Method (split-half reliability)
-------------------------------
For each random seed:
  1. For each role, randomly split its 50 gold-prompt rows into halves A and B.
  2. Build a behavioral profile (24-d: 4 alphas x 6 features) on each half ->
     two 275 x 24 matrices, M_A and M_B.
  3. Build behavioral RDMs (corr-dist, L2) on M_A and M_B.
  4. Compute Spearman r between the upper-triangles of RDM_A and RDM_B.
     This is the "self-RSA" of the behavioral measurement: an empirical upper
     bound on how well any *external* RDM (representational, taxonomic, ...)
     can possibly correlate with it.
  5. Spearman-Brown correction extrapolates from "half-data" reliability to
     "full-data" reliability:  rho_full = 2 * rho_half / (1 + rho_half)

We then re-express the observed geometry-behavior RSA as a fraction of this
ceiling, giving a noise-corrected RSA that is comparable to other studies.

Outputs (data/)
---------------
- noise_floor_split_half.csv : per-seed reliability for both beh distances
- noise_floor_summary.csv    : mean +/- 95% CI of split-half + Spearman-Brown
- noise_corrected_rsa.csv    : observed RSA / ceiling, per (repr, beh) combo
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

REPO = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO / "experiment_data" / "gold_prompt_experiments"
RSA_DATA_DIR = REPO / "analysis" / "empirical" / "rsa_geometry_behavior" / "data"
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
FEATURES = ["steered_score"] + SUBDIMS
N_SEEDS = 200


def load_role_csvs(role_filter: list[str]) -> dict[str, pd.DataFrame]:
    """Load only the alpha + score columns we need. The full CSVs include the
    baseline / steered text, which is ~7 MB per file -> 2 GB total."""
    use_cols = ["alpha"] + FEATURES
    out: dict[str, pd.DataFrame] = {}
    for fp in sorted(glob.glob(str(GOLD_DIR / "Comparison_GoldStandard_*.csv"))):
        role = os.path.basename(fp).replace("Comparison_GoldStandard_", "").replace(".csv", "")
        if role not in role_filter:
            continue
        df = pd.read_csv(fp, usecols=use_cols)
        if not set(ALPHAS).issubset(set(df["alpha"].unique())):
            continue
        out[role] = df.reset_index(drop=True)
    return out


def profile_from_subset(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """24-feature behavioral profile from a subset of rows in a role's csv."""
    sub = df.iloc[mask]
    means = sub.groupby("alpha")[FEATURES].mean()
    flat: list[float] = []
    for a in ALPHAS:
        for feat in FEATURES:
            flat.append(float(means.loc[a, feat]))
    return np.asarray(flat, dtype=float)


def cosine_rdm_from_centered(X: np.ndarray) -> np.ndarray:
    """1 - Pearson correlation distance on rows of X."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return 1.0 - sim


def l2_rdm_z(X: np.ndarray) -> np.ndarray:
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    return squareform(pdist(Xz, metric="euclidean"))


def upper(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(rdm, k=1)
    return rdm[iu]


def split_half_one_seed(
    role_dfs: dict[str, pd.DataFrame], rng: np.random.Generator
) -> dict[str, float]:
    roles = list(role_dfs.keys())
    M_a = np.zeros((len(roles), len(ALPHAS) * len(FEATURES)), dtype=float)
    M_b = np.zeros_like(M_a)
    for i, r in enumerate(roles):
        df = role_dfs[r]
        # Stratify split by alpha so each half is balanced across alphas.
        idx_a, idx_b = [], []
        for a in ALPHAS:
            rows = np.flatnonzero(df["alpha"].to_numpy() == a)
            rng.shuffle(rows)
            mid = len(rows) // 2
            idx_a.extend(rows[:mid].tolist())
            idx_b.extend(rows[mid : 2 * mid].tolist())
        M_a[i] = profile_from_subset(df, np.asarray(idx_a, dtype=int))
        M_b[i] = profile_from_subset(df, np.asarray(idx_b, dtype=int))

    rdm_a_corr = cosine_rdm_from_centered(M_a)
    rdm_b_corr = cosine_rdm_from_centered(M_b)
    rdm_a_l2 = l2_rdm_z(M_a)
    rdm_b_l2 = l2_rdm_z(M_b)

    return {
        "split_half_corr": float(stats.spearmanr(upper(rdm_a_corr), upper(rdm_b_corr)).statistic),
        "split_half_l2": float(stats.spearmanr(upper(rdm_a_l2), upper(rdm_b_l2)).statistic),
    }


def spearman_brown(r: float) -> float:
    """Project half-data reliability to full-data reliability."""
    if not np.isfinite(r):
        return r
    return 2 * r / (1 + r) if (1 + r) != 0 else r


def main() -> None:
    rsa_main = pd.read_csv(RSA_DATA_DIR / "rsa_main.csv")
    role_order = json.loads((RSA_DATA_DIR / "role_order.json").read_text())
    print(f"Loading per-role gold csvs (target n={len(role_order)}) ...")
    role_dfs = load_role_csvs(role_filter=role_order)
    print(f"  loaded {len(role_dfs)} roles with full alpha coverage")

    rng = np.random.default_rng(0)
    rows = []
    for seed in range(N_SEEDS):
        sub_rng = np.random.default_rng(rng.integers(2**31 - 1))
        out = split_half_one_seed(role_dfs, sub_rng)
        rows.append({"seed": seed, **out})
        if (seed + 1) % 25 == 0:
            print(f"  seed {seed+1}/{N_SEEDS}: split_half_corr="
                  f"{out['split_half_corr']:.4f}  split_half_l2={out['split_half_l2']:.4f}")
    df = pd.DataFrame(rows)
    df["full_corr_spearmanbrown"] = df["split_half_corr"].map(spearman_brown)
    df["full_l2_spearmanbrown"] = df["split_half_l2"].map(spearman_brown)
    df.to_csv(DATA_DIR / "noise_floor_split_half.csv", index=False)

    summary_rows = []
    for col in [
        "split_half_corr",
        "split_half_l2",
        "full_corr_spearmanbrown",
        "full_l2_spearmanbrown",
    ]:
        vals = df[col].to_numpy()
        summary_rows.append(
            {
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "ci95_low": float(np.percentile(vals, 2.5)),
                "ci95_high": float(np.percentile(vals, 97.5)),
                "n_seeds": len(vals),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / "noise_floor_summary.csv", index=False)

    # --- Noise-corrected RSA: observed_RSA / Spearman-Brown ceiling ---
    ceiling_corr = df["full_corr_spearmanbrown"].mean()
    ceiling_l2 = df["full_l2_spearmanbrown"].mean()
    print(f"\nCeiling (Spearman-Brown corrected): "
          f"corr={ceiling_corr:.4f}  l2={ceiling_l2:.4f}")

    noise_rows = []
    for _, r in rsa_main.iterrows():
        ceil = ceiling_corr if r["beh_distance"] == "beh_corr" else ceiling_l2
        noise_rows.append(
            {
                "repr_distance": r["repr_distance"],
                "beh_distance": r["beh_distance"],
                "rsa_observed": float(r["rsa_spearman"]),
                "ceiling": float(ceil),
                "rsa_noise_corrected": float(r["rsa_spearman"] / ceil),
                "mantel_p": float(r["mantel_p"]),
            }
        )
    nc = pd.DataFrame(noise_rows)
    nc.to_csv(DATA_DIR / "noise_corrected_rsa.csv", index=False)

    print("\n=== Noise-corrected RSA ===")
    print(nc.to_string(index=False))


if __name__ == "__main__":
    main()
