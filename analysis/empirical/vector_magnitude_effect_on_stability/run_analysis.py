"""
Test the hypothesis: role-vector internal magnitude (L2 norm) predicts the
steering stability of that vector at high alpha.

Pipeline
--------
1. Load every persona vector in persona_data/pt_vectors/*.pt and record its
   L2 norm (and a few related geometry stats).
2. Aggregate per-role gold-standard judge scores in
   experiment_data/gold_prompt_experiments/Comparison_GoldStandard_*.csv
   across alphas {1.0, 1.5, 2.0, 2.5}.
3. Compute per-role stability metrics:
     - score_at_alpha_2_5            : mean steered_score at alpha=2.5
     - score_at_alpha_1_0            : mean steered_score at alpha=1.0
     - delta_high_minus_low          : score(2.5) - score(1.0)
     - monotonic                     : did the curve increase monotonically?
     - peak_alpha                    : alpha at which steered_score peaks
     - late_alpha_drop               : max(score) - score(2.5)  (positive => unstable)
     - score_slope                   : OLS slope of mean-score vs alpha
     - subdim_consistency            : cross-alpha std of mean cmp_* scores
4. Correlate (Pearson + Spearman) magnitude vs each stability metric.
5. Save:
     - data/vector_magnitude_stability.csv
     - data/correlations.csv
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

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


def load_vector_geometry() -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(str(VECTORS_DIR / "*.pt"))):
        role = os.path.splitext(os.path.basename(fp))[0]
        v = torch.load(fp, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            # be defensive: pick the first tensor
            v = next(t for t in v.values() if torch.is_tensor(t))
        v = v.float().squeeze()
        rows.append(
            {
                "role": role,
                "norm": float(v.norm().item()),
                "dim": int(v.numel()),
                "max_abs": float(v.abs().max().item()),
                "mean_abs": float(v.abs().mean().item()),
            }
        )
    df = pd.DataFrame(rows)
    # Centered direction: subtract mean direction across all roles to remove a
    # shared "assistant-axis"-like component. Norm of the centered vector is a
    # cleaner measure of "how strongly does this role differ from the mean".
    mat = []
    for fp in sorted(glob.glob(str(VECTORS_DIR / "*.pt"))):
        v = torch.load(fp, map_location="cpu", weights_only=False)
        if isinstance(v, dict):
            v = next(t for t in v.values() if torch.is_tensor(t))
        mat.append(v.float().squeeze().numpy())
    mat = np.stack(mat, axis=0)
    centered = mat - mat.mean(axis=0, keepdims=True)
    df["norm_centered"] = np.linalg.norm(centered, axis=1)
    return df


def load_gold_scores() -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(str(GOLD_DIR / "Comparison_GoldStandard_*.csv"))):
        role = os.path.basename(fp).replace("Comparison_GoldStandard_", "").replace(".csv", "")
        df = pd.read_csv(fp)
        if not set(ALPHAS).issubset(set(df["alpha"].unique())):
            continue
        # Mean score per alpha
        per_alpha = df.groupby("alpha")["steered_score"].mean().to_dict()
        sub_means = (
            df.groupby("alpha")[SUBDIMS].mean().rename_axis("alpha").reset_index()
        )
        # Score curve features
        ys = np.array([per_alpha[a] for a in ALPHAS], dtype=float)
        xs = np.array(ALPHAS, dtype=float)
        slope, intercept, r_value, p_value, _ = stats.linregress(xs, ys)
        peak_alpha = float(ALPHAS[int(np.argmax(ys))])
        # Sub-dimension stability: how variable are the sub-scores across alpha
        subdim_consistency = float(
            sub_means[SUBDIMS].std(axis=0).mean()
        )  # mean across dims of std-across-alpha
        # Sub-dim score at alpha 2.5
        sub_at_25 = sub_means[sub_means["alpha"] == 2.5][SUBDIMS].iloc[0].to_dict()
        rows.append(
            {
                "role": role,
                "score_at_alpha_1_0": per_alpha[1.0],
                "score_at_alpha_1_5": per_alpha[1.5],
                "score_at_alpha_2_0": per_alpha[2.0],
                "score_at_alpha_2_5": per_alpha[2.5],
                "delta_high_minus_low": per_alpha[2.5] - per_alpha[1.0],
                "score_slope": slope,
                "score_curve_r2": r_value ** 2,
                "peak_alpha": peak_alpha,
                "late_alpha_drop": float(ys.max() - per_alpha[2.5]),
                "monotonic": bool(np.all(np.diff(ys) >= 0)),
                "subdim_consistency": subdim_consistency,
                **{f"sub25_{k}": v for k, v in sub_at_25.items()},
                "n_questions_total": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def correlate(merged: pd.DataFrame, mag_col: str) -> pd.DataFrame:
    targets = [
        "score_at_alpha_1_0",
        "score_at_alpha_1_5",
        "score_at_alpha_2_0",
        "score_at_alpha_2_5",
        "delta_high_minus_low",
        "score_slope",
        "late_alpha_drop",
        "subdim_consistency",
    ] + [f"sub25_{s}" for s in SUBDIMS]
    out = []
    for col in targets:
        x = merged[mag_col].values
        y = merged[col].values
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        out.append(
            {
                "magnitude_metric": mag_col,
                "stability_metric": col,
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_r": r_s,
                "spearman_p": p_s,
                "n": len(merged),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    print(f"Loading vectors from {VECTORS_DIR}")
    geom = load_vector_geometry()
    print(f"  vectors: {len(geom)}, dim={geom['dim'].iloc[0]}, "
          f"norm range [{geom['norm'].min():.3f}, {geom['norm'].max():.3f}]")

    print(f"Loading gold-prompt scores from {GOLD_DIR}")
    scores = load_gold_scores()
    print(f"  roles with full alpha coverage: {len(scores)}")

    merged = geom.merge(scores, on="role", how="inner")
    print(f"  merged roles: {len(merged)}")

    out_csv = DATA_DIR / "vector_magnitude_stability.csv"
    merged.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    cor_raw = correlate(merged, "norm")
    cor_centered = correlate(merged, "norm_centered")
    cor = pd.concat([cor_raw, cor_centered], ignore_index=True)
    cor_csv = DATA_DIR / "correlations.csv"
    cor.to_csv(cor_csv, index=False)
    print(f"  wrote {cor_csv}")

    # Also dump a tidy long table of (role, alpha, score, norm) for plotting.
    long_rows = []
    for _, row in merged.iterrows():
        for a in ALPHAS:
            col = f"score_at_alpha_{str(a).replace('.', '_')}"
            long_rows.append(
                {
                    "role": row["role"],
                    "alpha": a,
                    "steered_score_mean": row[col],
                    "norm": row["norm"],
                    "norm_centered": row["norm_centered"],
                }
            )
    pd.DataFrame(long_rows).to_csv(DATA_DIR / "alpha_curves_long.csv", index=False)
    print(f"  wrote {DATA_DIR / 'alpha_curves_long.csv'}")

    print("\n=== Summary correlations (Pearson) ===")
    pivot = cor[cor["magnitude_metric"] == "norm"][
        ["stability_metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
    ]
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
