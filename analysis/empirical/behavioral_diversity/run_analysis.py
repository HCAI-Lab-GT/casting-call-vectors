"""
Behavioral diversity and cross-method RSA analysis.

Idea 2 — Behavioral diversity collapse:
  Build parallel steered and assistant-axis (AA) behavioral profiles from the
  gold-prompt CSVs. At each alpha, compare:
    - Effective dimensionality of the 275×6 behavioral matrix (per method)
    - Inter-role variance (mean pairwise L2 distance between role profiles)
    - Score variance across roles (how spread out are roles in score space?)
  Prediction: the AA method drives all roles toward the same point as alpha
  increases (effective rank → 1), while our method preserves inter-role diversity.

Idea 4 — RSA between steered and AA behavioral RDMs:
  Build a 275×24 AA behavioral profile matrix and compute its behavioral RDM.
  Then ask:
    - RSA(repr_cos, beh_aa)     vs RSA(repr_cos, beh_steered) [already known]
    - RSA(beh_steered, beh_aa)  [how similar are the two methods' structures?]
    - Per-alpha: which method's behavior is better predicted by geometry?
  All tested with Mantel permutation tests.

Outputs (data/):
  aa_behavioral_profiles.csv      275×24 AA behavioral profile matrix
  diversity_by_alpha.csv          effective_rank, mean_pairwise_l2, score_var per method per alpha
  rdm_beh_aa_corr.npy             275×275 AA behavioral RDM (corr-dist)
  rdm_beh_aa_l2.npy               275×275 AA behavioral RDM (l2 on z-scored)
  rsa_cross_method.csv            RSA results: (beh_steered vs beh_aa), (repr_cos vs beh_aa)
  rsa_per_alpha_method_comparison.csv  per-alpha RSA for both methods vs repr_cos
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

REPO = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO / "experiment_data" / "gold_prompt_experiments"
RSA_DATA_DIR = REPO / "analysis" / "empirical" / "rsa_geometry_behavior" / "data"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]

STEERED_FEATURES = [
    "steered_score",
    "cmp_emotional_register",
    "cmp_vocab_choice",
    "cmp_social_dynamic",
    "cmp_motivation",
    "cmp_worldview_alignment",
]
AA_FEATURES = [
    "assistant_axis_score",
    "assistant_axis_cmp_emotional_register",
    "assistant_axis_cmp_vocab_choice",
    "assistant_axis_cmp_social_dynamic",
    "assistant_axis_cmp_motivation",
    "assistant_axis_cmp_worldview_alignment",
]
N_FEATURES = len(STEERED_FEATURES)   # 6

N_PERM = 5_000
RNG_SEED = 42


# ── data loading ───────────────────────────────────────────────────────────────

def _parse_score(val) -> float:
    if pd.isna(val):
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        m = re.search(r"\b(\d+)\s*\n", str(val))
        if m:
            return float(m.group(1))
        m = re.search(r"\d+", str(val))
        if m:
            return float(m.group())
        return float("nan")


def load_profiles(role_filter: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (steered_df, aa_df), each with one row per role and columns
    {feature}__alpha_{a} for 4 alphas × 6 features = 24 columns.

    Only roles that have:
      - Full alpha coverage (all 4 alphas present)
      - Non-NaN assistant_axis_cmp_* values (filters out old CSVs)
    are included in aa_df. steered_df is built for all roles with full alpha
    coverage regardless.
    """
    steered_rows, aa_rows = [], []

    all_cols = (
        {"alpha", "sample_count"}
        | set(STEERED_FEATURES)
        | set(AA_FEATURES)
    )

    for fp in sorted(GOLD_DIR.glob("Comparison_GoldStandard_*.csv")):
        role = fp.stem.replace("Comparison_GoldStandard_", "")
        if role not in role_filter:
            continue

        header = pd.read_csv(fp, nrows=0).columns.tolist()
        use_cols = [c for c in header if c in all_cols]
        df = pd.read_csv(fp, usecols=use_cols)
        df = df[df["sample_count"] == 50].copy()

        for col in STEERED_FEATURES + AA_FEATURES:
            if col in df.columns:
                df[col] = df[col].apply(_parse_score)

        covered_alphas = set(df["alpha"].unique())
        if not set(ALPHAS).issubset(covered_alphas):
            continue

        per_alpha = df.groupby("alpha")

        # Steered profile
        s_row: dict = {"role": role}
        for a in ALPHAS:
            grp = per_alpha.get_group(a) if a in per_alpha.groups else None
            for feat in STEERED_FEATURES:
                col_name = f"{feat}__alpha_{a}"
                if grp is not None and feat in grp.columns:
                    s_row[col_name] = float(grp[feat].mean())
                else:
                    s_row[col_name] = float("nan")
        steered_rows.append(s_row)

        # AA profile — only if AA cmp columns have data
        aa_cmp_col = "assistant_axis_cmp_emotional_register"
        if aa_cmp_col in df.columns:
            grp_25 = per_alpha.get_group(2.5) if 2.5 in per_alpha.groups else None
            if grp_25 is not None and grp_25[aa_cmp_col].notna().mean() > 0.5:
                a_row: dict = {"role": role}
                valid = True
                for a in ALPHAS:
                    grp = per_alpha.get_group(a) if a in per_alpha.groups else None
                    for feat in AA_FEATURES:
                        col_name = f"{feat}__alpha_{a}"
                        if grp is not None and feat in grp.columns:
                            v = float(grp[feat].mean())
                        else:
                            v = float("nan")
                            valid = False
                        a_row[col_name] = v
                if valid:
                    aa_rows.append(a_row)

    steered_df = pd.DataFrame(steered_rows)
    aa_df = pd.DataFrame(aa_rows)
    return steered_df, aa_df


# ── diversity metrics ──────────────────────────────────────────────────────────

def effective_rank(mat: np.ndarray) -> float:
    """Shannon-entropy effective rank: exp(-Σ p_i log p_i), p_i = σ_i² / Σσ²."""
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(mat: np.ndarray) -> float:
    """(Σσ²)² / Σσ⁴."""
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def mean_pairwise_l2(mat: np.ndarray) -> float:
    """Mean L2 distance between all pairs of rows."""
    dists = pdist(mat, metric="euclidean")
    return float(dists.mean())


def compute_diversity(df: pd.DataFrame, features: list[str],
                      method_name: str) -> pd.DataFrame:
    rows = []
    for a in ALPHAS:
        cols = [f"{feat}__alpha_{a}" for feat in features]
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        mat = df[available].dropna().to_numpy(dtype=float)
        if mat.shape[0] < 5:
            continue
        # Z-score per feature before computing L2 diversity
        mat_z = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)
        rows.append(dict(
            method=method_name,
            alpha=a,
            n_roles=mat.shape[0],
            effective_rank=effective_rank(mat_z),
            participation_ratio=participation_ratio(mat_z),
            mean_pairwise_l2=mean_pairwise_l2(mat_z),
            score_var=float(mat[:, 0].var()),  # variance of the overall score across roles
            score_std=float(mat[:, 0].std()),
        ))
    return pd.DataFrame(rows)


# ── RDM helpers ────────────────────────────────────────────────────────────────

def corr_rdm(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return 1.0 - sim


def l2_rdm(X: np.ndarray) -> np.ndarray:
    return squareform(pdist(X, metric="euclidean"))


def rdm_vec(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices_from(rdm, k=1)]


def rsa_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    return float(stats.spearmanr(rdm_vec(rdm_a), rdm_vec(rdm_b)).statistic)


def mantel_test(rdm_a: np.ndarray, rdm_b: np.ndarray,
                n_perm: int = N_PERM, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = rsa_spearman(rdm_a, rdm_b)
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
    # Load existing role order from the rsa_geometry_behavior analysis
    role_order_path = RSA_DATA_DIR / "role_order.json"
    if role_order_path.exists():
        with open(role_order_path) as f:
            existing_roles = json.load(f)
        print(f"Loaded {len(existing_roles)} roles from rsa_geometry_behavior role_order.json")
    else:
        print("Warning: role_order.json not found, will use all available roles")
        existing_roles = []

    role_filter = existing_roles if existing_roles else None

    print(f"Loading behavioral profiles from {GOLD_DIR} ...")
    steered_df, aa_df = load_profiles(
        role_filter=role_filter if role_filter else [
            fp.stem.replace("Comparison_GoldStandard_", "")
            for fp in GOLD_DIR.glob("Comparison_GoldStandard_*.csv")
        ]
    )
    print(f"  steered profiles: {len(steered_df)} roles")
    print(f"  AA profiles: {len(aa_df)} roles")

    aa_df.to_csv(DATA_DIR / "aa_behavioral_profiles.csv", index=False)
    print(f"Saved: {DATA_DIR / 'aa_behavioral_profiles.csv'}")

    # ── Diversity metrics ──────────────────────────────────────────────────────
    print("\nComputing diversity metrics ...")
    s_div = compute_diversity(steered_df, STEERED_FEATURES, "steered")
    a_div = compute_diversity(aa_df, AA_FEATURES, "assistant_axis")
    diversity = pd.concat([s_div, a_div], ignore_index=True)
    diversity.to_csv(DATA_DIR / "diversity_by_alpha.csv", index=False)
    print(f"Saved: {DATA_DIR / 'diversity_by_alpha.csv'}")

    print("\nDiversity summary:")
    for _, row in diversity.iterrows():
        print(f"  {row['method']:16s}  α={row['alpha']}  "
              f"eff_rank={row['effective_rank']:.2f}  "
              f"mean_pw_l2={row['mean_pairwise_l2']:.3f}  "
              f"score_std={row['score_std']:.2f}  "
              f"n={row['n_roles']}")

    # ── RSA analysis ───────────────────────────────────────────────────────────
    # We need a common role set for RDM comparisons.
    # Use intersection of AA roles and existing role order.
    common_roles = [r for r in (existing_roles or steered_df["role"].tolist())
                    if r in aa_df["role"].values and r in steered_df["role"].values]
    print(f"\nCommon roles for RSA: {len(common_roles)}")

    if len(common_roles) < 10:
        print("Too few common roles for RSA. Skipping.")
        return

    # Align dataframes to common_roles order
    s_idx = steered_df.set_index("role").loc[common_roles]
    a_idx = aa_df.set_index("role").loc[common_roles]

    steered_feat_cols = [
        f"{feat}__alpha_{a}" for a in ALPHAS for feat in STEERED_FEATURES
    ]
    aa_feat_cols = [
        f"{feat}__alpha_{a}" for a in ALPHAS for feat in AA_FEATURES
    ]

    s_mat = s_idx[steered_feat_cols].to_numpy(dtype=float)
    a_mat = a_idx[aa_feat_cols].to_numpy(dtype=float)

    # Drop roles that have any NaN in either profile (edge case from missing data)
    valid_mask = ~(np.isnan(s_mat).any(axis=1) | np.isnan(a_mat).any(axis=1))
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        print(f"  Dropping {n_dropped} roles with NaN values in profiles")
        s_mat = s_mat[valid_mask]
        a_mat = a_mat[valid_mask]
        common_roles = [r for r, v in zip(common_roles, valid_mask) if v]
    print(f"  Final RSA role count: {len(common_roles)}")

    # Z-score per feature
    s_mat_z = (s_mat - s_mat.mean(axis=0)) / (s_mat.std(axis=0) + 1e-12)
    a_mat_z = (a_mat - a_mat.mean(axis=0)) / (a_mat.std(axis=0) + 1e-12)

    # Build RDMs
    print("Building behavioral RDMs ...")
    rdm_beh_s_corr = corr_rdm(s_mat)
    rdm_beh_s_l2 = l2_rdm(s_mat_z)
    rdm_beh_aa_corr = corr_rdm(a_mat)
    rdm_beh_aa_l2 = l2_rdm(a_mat_z)

    np.save(DATA_DIR / "rdm_beh_aa_corr.npy", rdm_beh_aa_corr)
    np.save(DATA_DIR / "rdm_beh_aa_l2.npy", rdm_beh_aa_l2)

    # Load repr_cos RDM from rsa_geometry_behavior (aligned to role_order.json)
    repr_rdm_path = RSA_DATA_DIR / "rdm_repr_cos.npy"
    if not repr_rdm_path.exists():
        print(f"Warning: {repr_rdm_path} not found. Skipping repr RSA.")
        repr_rdm = None
    else:
        full_repr_rdm = np.load(repr_rdm_path)
        full_roles = existing_roles  # role_order.json order
        role_to_idx = {r: i for i, r in enumerate(full_roles)}
        # common_roles may have shrunk after NaN drop above
        common_idx = np.array([role_to_idx[r] for r in common_roles
                                if r in role_to_idx], dtype=int)
        common_roles = [r for r in common_roles if r in role_to_idx]
        repr_rdm = full_repr_rdm[np.ix_(common_idx, common_idx)]
        print(f"  repr_cos RDM sliced to {repr_rdm.shape[0]}×{repr_rdm.shape[1]}")

    # ── Cross-method comparisons ───────────────────────────────────────────────
    print(f"\nRunning Mantel tests (n_perm={N_PERM}) ...")
    rsa_rows = []

    # beh_steered vs beh_aa (both corr-dist and L2)
    for s_name, s_rdm in [("beh_steered_corr", rdm_beh_s_corr),
                           ("beh_steered_l2", rdm_beh_s_l2)]:
        for a_name, a_rdm in [("beh_aa_corr", rdm_beh_aa_corr),
                               ("beh_aa_l2", rdm_beh_aa_l2)]:
            obs, p = mantel_test(s_rdm, a_rdm)
            rsa_rows.append(dict(comparison=f"{s_name}_vs_{a_name}",
                                 rdm_a=s_name, rdm_b=a_name,
                                 rsa_spearman=obs, mantel_p=p, n_roles=len(common_roles)))
            print(f"  {s_name} vs {a_name}: r={obs:+.4f}  p={p:.2e}")

    # repr_cos vs beh_aa (and vs beh_steered for comparison)
    if repr_rdm is not None:
        for beh_name, beh_rdm in [("beh_steered_corr", rdm_beh_s_corr),
                                   ("beh_steered_l2", rdm_beh_s_l2),
                                   ("beh_aa_corr", rdm_beh_aa_corr),
                                   ("beh_aa_l2", rdm_beh_aa_l2)]:
            obs, p = mantel_test(repr_rdm, beh_rdm)
            rsa_rows.append(dict(comparison=f"repr_cos_vs_{beh_name}",
                                 rdm_a="repr_cos", rdm_b=beh_name,
                                 rsa_spearman=obs, mantel_p=p, n_roles=len(common_roles)))
            print(f"  repr_cos vs {beh_name}: r={obs:+.4f}  p={p:.2e}")

    pd.DataFrame(rsa_rows).to_csv(DATA_DIR / "rsa_cross_method.csv", index=False)
    print(f"Saved: {DATA_DIR / 'rsa_cross_method.csv'}")

    # ── Per-alpha RSA breakdown ────────────────────────────────────────────────
    print("\nPer-alpha RSA (repr_cos vs each method's behavioral RDM) ...")
    per_alpha_rows = []
    for a in ALPHAS:
        for method_name, feat_list, df_idx in [
            ("steered", STEERED_FEATURES, s_idx),
            ("assistant_axis", AA_FEATURES, a_idx),
        ]:
            cols = [f"{feat}__alpha_{a}" for feat in feat_list]
            avail = [c for c in cols if c in df_idx.columns]
            if not avail:
                continue
            mat_a = df_idx[avail].to_numpy(dtype=float)
            mat_a_z = (mat_a - mat_a.mean(axis=0)) / (mat_a.std(axis=0) + 1e-12)
            rdm_corr = corr_rdm(mat_a)
            rdm_l2 = l2_rdm(mat_a_z)

            for beh_name, beh_rdm in [("corr", rdm_corr), ("l2", rdm_l2)]:
                if repr_rdm is not None:
                    obs, p = mantel_test(repr_rdm, beh_rdm, n_perm=2000)
                    per_alpha_rows.append(dict(
                        alpha=a, method=method_name, beh_distance=beh_name,
                        rsa_spearman=obs, mantel_p=p, n_roles=len(common_roles),
                    ))
                    print(f"  α={a}  {method_name:16s}  beh_{beh_name}: r={obs:+.4f}  p={p:.2e}")

    pd.DataFrame(per_alpha_rows).to_csv(DATA_DIR / "rsa_per_alpha_method_comparison.csv", index=False)
    print(f"Saved: {DATA_DIR / 'rsa_per_alpha_method_comparison.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
