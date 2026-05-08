"""
Score trajectory comparison: steered vs assistant-axis across alphas.

For each role we summarise the score-vs-alpha curve for BOTH methods:
  - AUC (area under the curve, trapezoidal, normalised by alpha range)
  - Monotonicity: is the curve non-decreasing at every step?
  - Peak alpha: at which alpha does the score peak?
  - Score slope: OLS slope across the four alpha points

Outputs (data/):
  trajectory_comparison.csv   per-role metrics for steered and AA
  alpha_curves_both.csv       tidy (role, alpha, steered_score, aa_score) for plotting
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO / "experiment_data" / "gold_prompt_experiments"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = np.array([1.0, 1.5, 2.0, 2.5])
ALPHA_RANGE = float(ALPHAS[-1] - ALPHAS[0])   # 1.5


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


def _auc(scores: np.ndarray) -> float:
    """Trapezoidal AUC normalised by the alpha range."""
    mask = np.isfinite(scores)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapz(scores[mask], ALPHAS[mask]) / ALPHA_RANGE)


def _slope(scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    if mask.sum() < 2:
        return float("nan")
    m, _ = np.polyfit(ALPHAS[mask], scores[mask], 1)
    return float(m)


def _peak_alpha(scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    if not mask.any():
        return float("nan")
    return float(ALPHAS[mask][np.argmax(scores[mask])])


def _monotonic(scores: np.ndarray) -> bool | None:
    mask = np.isfinite(scores)
    if mask.sum() < 2:
        return None
    return bool(np.all(np.diff(scores[mask]) >= 0))


def load_trajectories() -> tuple[pd.DataFrame, pd.DataFrame]:
    traj_rows = []
    curve_rows = []

    for fp in sorted(GOLD_DIR.glob("Comparison_GoldStandard_*.csv")):
        role = fp.stem.replace("Comparison_GoldStandard_", "")
        header = pd.read_csv(fp, nrows=0).columns.tolist()
        need = {"alpha", "sample_count", "steered_score", "assistant_axis_score"}
        if not need.issubset(set(header)):
            continue

        df = pd.read_csv(fp, usecols=list(need))
        df = df[df["sample_count"] == 50].copy()
        for col in ("steered_score", "assistant_axis_score"):
            df[col] = df[col].apply(_parse_score)

        covered = set(df["alpha"].unique())
        if not set(ALPHAS.tolist()).issubset(covered):
            continue

        per_alpha = df.groupby("alpha")[["steered_score", "assistant_axis_score"]].mean()

        s_scores = np.array([per_alpha.loc[a, "steered_score"] for a in ALPHAS])
        a_scores = np.array([per_alpha.loc[a, "assistant_axis_score"] for a in ALPHAS])

        traj_rows.append(dict(
            role=role,
            steered_auc=_auc(s_scores),
            aa_auc=_auc(a_scores),
            auc_advantage=_auc(s_scores) - _auc(a_scores),
            steered_monotonic=_monotonic(s_scores),
            aa_monotonic=_monotonic(a_scores),
            steered_peak_alpha=_peak_alpha(s_scores),
            aa_peak_alpha=_peak_alpha(a_scores),
            steered_slope=_slope(s_scores),
            aa_slope=_slope(a_scores),
            slope_advantage=_slope(s_scores) - _slope(a_scores),
            steered_score_1_0=float(s_scores[0]),
            steered_score_2_5=float(s_scores[-1]),
            aa_score_1_0=float(a_scores[0]),
            aa_score_2_5=float(a_scores[-1]),
            steered_gain=float(s_scores[-1] - s_scores[0]),
            aa_gain=float(a_scores[-1] - a_scores[0]),
        ))

        for a, s, aa in zip(ALPHAS, s_scores, a_scores):
            curve_rows.append(dict(role=role, alpha=float(a),
                                   steered_score=float(s), aa_score=float(aa)))

    return pd.DataFrame(traj_rows), pd.DataFrame(curve_rows)


def main() -> None:
    print(f"Loading score trajectories from {GOLD_DIR} ...")
    traj_df, curve_df = load_trajectories()
    print(f"  roles loaded: {len(traj_df)}")

    traj_df.to_csv(DATA_DIR / "trajectory_comparison.csv", index=False)
    curve_df.to_csv(DATA_DIR / "alpha_curves_both.csv", index=False)
    print(f"Saved: {DATA_DIR / 'trajectory_comparison.csv'}")
    print(f"Saved: {DATA_DIR / 'alpha_curves_both.csv'}")

    print("\nTrajectory summary:")
    print(f"  Steered  — mean AUC: {traj_df['steered_auc'].mean():.2f}  "
          f"monotonic: {traj_df['steered_monotonic'].mean():.1%}  "
          f"mean slope: {traj_df['steered_slope'].mean():.2f}")
    print(f"  AA       — mean AUC: {traj_df['aa_auc'].mean():.2f}  "
          f"monotonic: {traj_df['aa_monotonic'].mean():.1%}  "
          f"mean slope: {traj_df['aa_slope'].mean():.2f}")

    print(f"\n  AUC advantage (steered − AA): mean={traj_df['auc_advantage'].mean():.2f}  "
          f"std={traj_df['auc_advantage'].std():.2f}")

    print("\n  Peak alpha distribution:")
    for method, col in [("Steered", "steered_peak_alpha"), ("AA", "aa_peak_alpha")]:
        counts = traj_df[col].value_counts().sort_index()
        print(f"    {method}: " + "  ".join(f"α={a}: {c}" for a, c in counts.items()))

    print("\nDone.")


if __name__ == "__main__":
    main()
