"""
Sub-dimension advantage analysis: steered vs assistant-axis per judge dimension.

For each role × alpha × sub-dimension we compute:
  advantage = mean(steered_cmp_X) − mean(aa_cmp_X)

Then ask:
  1. Which sub-dimensions consistently favour our method?
  2. Does our method preserve more sub-dimension variance across roles?
     (AA homogenisation should collapse sub-dim variance too)
  3. Do roles with larger perp_frac (more role-specific geometry) show
     larger sub-dimension advantage?

Outputs (data/):
  subdim_long.csv          per-role × alpha × subdim: steered, aa, advantage
  subdim_summary.csv       mean advantage per subdim per alpha (across roles)
  subdim_variance.csv      std of each subdim across roles per method per alpha
  subdim_geo_corr.csv      Pearson + Spearman: perp_frac vs per-subdim advantage
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO / "experiment_data" / "gold_prompt_experiments"
DECOMP_CSV = (
    REPO / "analysis" / "empirical" / "role_vector_decomposition" / "data" / "decomposition.csv"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]

# (short_name, steered_col, aa_col)
SUBDIMS = [
    ("emotional_register",  "cmp_emotional_register",  "assistant_axis_cmp_emotional_register"),
    ("vocab_choice",        "cmp_vocab_choice",         "assistant_axis_cmp_vocab_choice"),
    ("social_dynamic",      "cmp_social_dynamic",       "assistant_axis_cmp_social_dynamic"),
    ("motivation",          "cmp_motivation",           "assistant_axis_cmp_motivation"),
    ("worldview_alignment", "cmp_worldview_alignment",  "assistant_axis_cmp_worldview_alignment"),
]
SUBDIM_NAMES = [s[0] for s in SUBDIMS]


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


def load_subdim_profiles() -> pd.DataFrame:
    """
    Returns long-format DataFrame with columns:
      role, alpha, subdim, steered_mean, aa_mean, advantage
    """
    all_cols = (
        {"alpha", "sample_count"}
        | {s for _, s, _ in SUBDIMS}
        | {a for _, _, a in SUBDIMS}
    )
    records = []
    for fp in sorted(GOLD_DIR.glob("Comparison_GoldStandard_*.csv")):
        role = fp.stem.replace("Comparison_GoldStandard_", "")
        header = pd.read_csv(fp, nrows=0).columns.tolist()
        use_cols = [c for c in header if c in all_cols]
        if not any(a for _, _, a in SUBDIMS if a in use_cols):
            continue  # old CSV without AA cmp columns

        df = pd.read_csv(fp, usecols=use_cols)
        df = df[df["sample_count"] == 50].copy()

        for _, s_col, a_col in SUBDIMS:
            for col in (s_col, a_col):
                if col in df.columns:
                    df[col] = df[col].apply(_parse_score)

        covered = set(df["alpha"].unique())
        if not set(ALPHAS).issubset(covered):
            continue

        per_alpha = df.groupby("alpha")
        for a in ALPHAS:
            grp = per_alpha.get_group(a)
            for name, s_col, a_col in SUBDIMS:
                s_mean = float(grp[s_col].mean()) if s_col in grp.columns else float("nan")
                a_mean = float(grp[a_col].mean()) if a_col in grp.columns else float("nan")
                if not (np.isfinite(s_mean) and np.isfinite(a_mean)):
                    continue
                records.append(dict(
                    role=role, alpha=a, subdim=name,
                    steered_mean=s_mean, aa_mean=a_mean,
                    advantage=s_mean - a_mean,
                ))

    return pd.DataFrame(records)


def compute_variance(long_df: pd.DataFrame) -> pd.DataFrame:
    """Std of each subdim score across roles, per method per alpha."""
    rows = []
    for a in ALPHAS:
        sub = long_df[long_df["alpha"] == a]
        for subdim in SUBDIM_NAMES:
            s = sub[sub["subdim"] == subdim]
            if len(s) < 5:
                continue
            rows.append(dict(
                alpha=a, subdim=subdim,
                steered_std=float(s["steered_mean"].std()),
                aa_std=float(s["aa_mean"].std()),
                steered_mean=float(s["steered_mean"].mean()),
                aa_mean=float(s["aa_mean"].mean()),
                advantage_mean=float(s["advantage"].mean()),
                n_roles=len(s),
            ))
    return pd.DataFrame(rows)


def compute_geo_correlations(long_df: pd.DataFrame, decomp: pd.DataFrame) -> pd.DataFrame:
    """Pearson + Spearman: perp_frac vs sub-dimension advantage."""
    rows = []
    merged = long_df.merge(decomp[["role", "perp_frac", "angle_deg"]], on="role", how="inner")
    for a in ALPHAS:
        for subdim in SUBDIM_NAMES:
            sub = merged[(merged["alpha"] == a) & (merged["subdim"] == subdim)].dropna(
                subset=["perp_frac", "advantage"]
            )
            if len(sub) < 10:
                continue
            x = sub["perp_frac"].to_numpy(dtype=float)
            y = sub["advantage"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue
            pr, pp = stats.pearsonr(x[mask], y[mask])
            sr, sp = stats.spearmanr(x[mask], y[mask])
            rows.append(dict(
                alpha=a, subdim=subdim,
                pearson_r=float(pr), pearson_p=float(pp),
                spearman_r=float(sr), spearman_p=float(sp),
                n=int(mask.sum()),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Loading sub-dimension profiles from {GOLD_DIR} ...")
    long_df = load_subdim_profiles()
    print(f"  loaded {len(long_df)} role×alpha×subdim records")
    print(f"  roles: {long_df['role'].nunique()}  alphas: {sorted(long_df['alpha'].unique())}")

    long_df.to_csv(DATA_DIR / "subdim_long.csv", index=False)
    print(f"Saved: {DATA_DIR / 'subdim_long.csv'}")

    summary = compute_variance(long_df)
    summary.to_csv(DATA_DIR / "subdim_summary.csv", index=False)
    print(f"Saved: {DATA_DIR / 'subdim_summary.csv'}")

    print("\nSub-dimension summary (mean advantage = steered − AA):")
    for a in ALPHAS:
        sub = summary[summary["alpha"] == a]
        print(f"  α={a}:")
        for _, row in sub.iterrows():
            print(f"    {row['subdim']:25s}  advantage={row['advantage_mean']:+.2f}  "
                  f"steered_std={row['steered_std']:.2f}  aa_std={row['aa_std']:.2f}")

    if DECOMP_CSV.exists():
        decomp = pd.read_csv(DECOMP_CSV)
        geo_corr = compute_geo_correlations(long_df, decomp)
        geo_corr.to_csv(DATA_DIR / "subdim_geo_corr.csv", index=False)
        print(f"\nSaved: {DATA_DIR / 'subdim_geo_corr.csv'}")
        print("\nGeometric correlations (perp_frac vs sub-dim advantage):")
        for _, row in geo_corr.iterrows():
            print(f"  α={row['alpha']}  {row['subdim']:25s}  "
                  f"Pearson r={row['pearson_r']:+.3f} (p={row['pearson_p']:.3e})  "
                  f"Spearman ρ={row['spearman_r']:+.3f}  n={row['n']}")
    else:
        print(f"\nWarning: {DECOMP_CSV} not found, skipping geo correlations")

    print("\nDone.")


if __name__ == "__main__":
    main()
