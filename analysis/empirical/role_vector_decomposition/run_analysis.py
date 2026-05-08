"""
Decompose each role's persona vector into assistant-axis component and role-specific residual.

  v_i = v_parallel + v_perp
  v_parallel = (v_i · d_aa) * d_aa   — what the assistant axis captures (same direction for all roles)
  v_perp     = v_i - v_parallel        — role-specific information the assistant axis discards

d_aa is the global assistant axis direction: the normalized mean of all per-role
assistant-axis vectors. Each per-role AA vector points toward "assistant" behavior from
that role; their mean gives the dominant shared direction.

Key questions:
  1. How much of v_i is role-specific? (‖v_⊥‖ / ‖v_i‖ = perp_frac)
  2. Does the score advantage (steered − assistant_axis) correlate with the misalignment
     angle between v_i and d_aa? With ‖v_⊥‖? With perp_frac?
  3. Is the role-specific residual the geometric explanation for our method's advantage?

Outputs (data/):
  decomposition.csv      per-role geometry + score_advantage at each alpha
  correlations.csv       Pearson + Spearman of geometric features vs score_advantage
  d_aa_consistency.csv   per-role cos(v_aa_i, d_aa) sanity check
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
PT_VECTORS_DIR = REPO / "persona_data" / "pt_vectors"
AA_VECTORS_DIR = REPO / "persona_data" / "assistant-axis" / "olmo-3-7b-instruct" / "vectors"
GOLD_DIR = REPO / "experiment_data" / "gold_prompt_experiments"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]
AA_SUBDIMS = [
    "assistant_axis_cmp_emotional_register",
    "assistant_axis_cmp_vocab_choice",
    "assistant_axis_cmp_social_dynamic",
    "assistant_axis_cmp_motivation",
    "assistant_axis_cmp_worldview_alignment",
]


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
                print(f"  skip {fp.name}: no tensor found")
                continue
            out[role] = _to_numpy(t)
        except Exception as e:
            print(f"  skip {fp.name}: {e}")
    return out


# ── score loading ──────────────────────────────────────────────────────────────

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


def load_score_advantage(roles: list[str]) -> pd.DataFrame:
    """Per-role, per-alpha mean(steered_score - assistant_axis_score)."""
    records: list[dict] = []
    for fp in sorted(GOLD_DIR.glob("Comparison_GoldStandard_*.csv")):
        role = fp.stem.replace("Comparison_GoldStandard_", "")
        if role not in roles:
            continue
        df = pd.read_csv(fp, usecols=lambda c: c in
            {"alpha", "sample_count", "steered_score", "assistant_axis_score"})
        df = df[df["sample_count"] == 50].copy()
        for col in ("steered_score", "assistant_axis_score"):
            df[col] = df[col].apply(_parse_score)
        df["advantage"] = df["steered_score"] - df["assistant_axis_score"]
        per_alpha = df.groupby("alpha")[["steered_score", "assistant_axis_score", "advantage"]].mean()
        row: dict = {"role": role}
        for a in ALPHAS:
            if a in per_alpha.index:
                row[f"steered_score_alpha_{a}"] = per_alpha.loc[a, "steered_score"]
                row[f"assistant_axis_score_alpha_{a}"] = per_alpha.loc[a, "assistant_axis_score"]
                row[f"advantage_alpha_{a}"] = per_alpha.loc[a, "advantage"]
        records.append(row)
    return pd.DataFrame(records)


# ── decomposition ──────────────────────────────────────────────────────────────

def compute_d_aa(aa_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """
    Global assistant axis direction = normalized mean of all (normalized) AA vectors.
    Each per-role AA vector points toward 'assistant' from that role; their mean
    direction is the dominant shared component.
    """
    mats = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in aa_vectors.values()])
    mean_dir = mats.mean(axis=0)
    return mean_dir / (np.linalg.norm(mean_dir) + 1e-12)


def decompose(v: np.ndarray, d_aa: np.ndarray) -> dict[str, float]:
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return dict(
            role_norm=0.0, parallel_norm=0.0, perp_norm=0.0,
            cos_with_daa=float("nan"), angle_deg=float("nan"), perp_frac=float("nan"),
        )
    v_hat = v / norm_v
    dot = float(np.dot(v_hat, d_aa))
    dot_clamped = float(np.clip(dot, -1.0, 1.0))
    v_parallel = dot_clamped * norm_v * d_aa          # v_hat·d_aa * ‖v‖ * d_aa
    v_perp = v - v_parallel
    parallel_norm = float(np.linalg.norm(v_parallel))
    perp_norm = float(np.linalg.norm(v_perp))
    angle_deg = float(np.degrees(np.arccos(np.abs(dot_clamped))))
    perp_frac = perp_norm / (norm_v + 1e-12)
    return dict(
        role_norm=float(norm_v),
        parallel_norm=parallel_norm,
        perp_norm=perp_norm,
        cos_with_daa=float(dot_clamped),
        angle_deg=angle_deg,
        perp_frac=perp_frac,
    )


def d_aa_consistency(aa_vectors: dict[str, np.ndarray], d_aa: np.ndarray) -> pd.DataFrame:
    rows = []
    for role, v in aa_vectors.items():
        v_hat = v / (np.linalg.norm(v) + 1e-12)
        cos = float(np.clip(np.dot(v_hat, d_aa), -1.0, 1.0))
        rows.append({"role": role, "cos_with_daa": cos, "aa_norm": float(np.linalg.norm(v))})
    return pd.DataFrame(rows)


# ── correlations ───────────────────────────────────────────────────────────────

GEO_FEATURES = ["angle_deg", "perp_frac", "perp_norm", "role_norm"]


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ALPHAS:
        adv_col = f"advantage_alpha_{a}"
        if adv_col not in df.columns:
            continue
        sub = df.dropna(subset=[adv_col])
        y = sub[adv_col].to_numpy(dtype=float)
        for feat in GEO_FEATURES:
            if feat not in sub.columns:
                continue
            x = sub[feat].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue
            pear_r, pear_p = stats.pearsonr(x[mask], y[mask])
            spear_r, spear_p = stats.spearmanr(x[mask], y[mask])
            rows.append(dict(
                alpha=a, feature=feat,
                pearson_r=float(pear_r), pearson_p=float(pear_p),
                spearman_r=float(spear_r), spearman_p=float(spear_p),
                n=int(mask.sum()),
            ))
    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading proposed role vectors from {PT_VECTORS_DIR} ...")
    proposed = load_vectors(PT_VECTORS_DIR)
    print(f"  loaded {len(proposed)} proposed vectors")

    print(f"Loading assistant-axis vectors from {AA_VECTORS_DIR} ...")
    aa_vecs = load_vectors(AA_VECTORS_DIR)
    print(f"  loaded {len(aa_vecs)} assistant-axis vectors")

    common_roles = sorted(set(proposed) & set(aa_vecs))
    print(f"  common roles: {len(common_roles)}")

    # Global assistant axis direction
    d_aa = compute_d_aa({r: aa_vecs[r] for r in common_roles})
    print(f"  d_aa computed (dim={d_aa.shape[0]})")

    # Sanity check: how consistent are individual AA vectors with d_aa?
    consistency = d_aa_consistency({r: aa_vecs[r] for r in common_roles}, d_aa)
    consistency.to_csv(DATA_DIR / "d_aa_consistency.csv", index=False)
    cos_vals = consistency["cos_with_daa"]
    print(f"  AA vector cos(v_aa_i, d_aa): mean={cos_vals.mean():.3f}  "
          f"std={cos_vals.std():.3f}  min={cos_vals.min():.3f}  max={cos_vals.max():.3f}")

    # Decompose each proposed vector
    decomp_rows = []
    for role in common_roles:
        v = proposed[role]
        stats_dict = decompose(v, d_aa)
        stats_dict["role"] = role
        decomp_rows.append(stats_dict)
    decomp_df = pd.DataFrame(decomp_rows)

    print(f"\nDecomposition summary (n={len(decomp_df)}):")
    print(f"  perp_frac: mean={decomp_df['perp_frac'].mean():.3f}  "
          f"std={decomp_df['perp_frac'].std():.3f}")
    print(f"  angle_deg: mean={decomp_df['angle_deg'].mean():.1f}°  "
          f"std={decomp_df['angle_deg'].std():.1f}°")

    # Score advantage per alpha
    print(f"\nLoading judge scores from {GOLD_DIR} ...")
    score_df = load_score_advantage(common_roles)
    print(f"  roles with judge data: {len(score_df)}")

    # Merge
    merged = decomp_df.merge(score_df, on="role", how="inner")
    print(f"  roles after merge: {len(merged)}")

    # Summary of advantages
    for a in ALPHAS:
        col = f"advantage_alpha_{a}"
        if col in merged.columns:
            print(f"  alpha={a}: mean advantage = {merged[col].mean():.2f}  "
                  f"std = {merged[col].std():.2f}")

    merged.to_csv(DATA_DIR / "decomposition.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'decomposition.csv'}")

    # Correlations
    corr_df = compute_correlations(merged)
    corr_df.to_csv(DATA_DIR / "correlations.csv", index=False)
    print(f"Saved: {DATA_DIR / 'correlations.csv'}")

    print("\nCorrelation table (geometric feature vs score advantage):")
    for _, row in corr_df.iterrows():
        print(f"  alpha={row['alpha']}  {row['feature']:12s}  "
              f"Pearson r={row['pearson_r']:+.3f} (p={row['pearson_p']:.3e})  "
              f"Spearman r={row['spearman_r']:+.3f} (p={row['spearman_p']:.3e})  "
              f"n={row['n']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
