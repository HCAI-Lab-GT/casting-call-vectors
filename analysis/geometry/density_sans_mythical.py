"""
density_sans_mythical.py

Plot the density (KDE + histogram) of judge scores for
`assistant_axis_score` vs `steered_score` (and baseline as a reference),
after dropping the mythical cluster identified by
`full_analysis_sans_mythical.py`.

Inputs
------
  - analysis/geometry/figures_sans_mythical/cluster_membership.txt
        The membership file written by full_analysis_sans_mythical.py.
        Roles under "# Excluded (mythical cluster)" are dropped.
  - experiment_data/gold_prompt_experiments/Comparison_GoldStandard_*.csv
        One CSV per role; we use the per-row judge scores.

Outputs (in analysis/geometry/figures_sans_mythical/density/)
-------
  - density_scores_sans_mythical.png  overlaid KDE + histograms
  - density_scores_full.png           same but over all roles (comparison)
  - density_scores_summary.tsv        per-method mean/median/std/count
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp

MEMBERSHIP_PATH = "analysis/geometry/figures_sans_mythical/cluster_membership.txt"
GOLD_DIR = "experiment_data/gold_prompt_experiments"
OUT_DIR = "analysis/geometry/figures_sans_mythical/density"
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = [
    ("assistant_axis_score", "assistant_axis",  "tab:blue"),
    ("steered_score",        "steered",         "tab:red"),
    ("baseline_score",       "baseline",        "tab:gray"),
]


def _load_excluded(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run full_analysis_sans_mythical.py first."
        )
    excluded, in_excl = set(), False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("# Excluded"):
                in_excl = True
                continue
            if line.startswith("# Kept"):
                in_excl = False
                continue
            if in_excl and line:
                # strip any trailing description/suffix
                excluded.add(os.path.splitext(line.split()[0])[0])
    return excluded


def _role_from_path(p):
    m = re.match(r"Comparison_GoldStandard_(.+)\.csv$", os.path.basename(p))
    return m.group(1) if m else None


def _load_scores(csv_paths, excluded_roles):
    """Return {method_col: np.ndarray of floats} aggregated across roles."""
    rows = []
    dropped_rows = 0
    for p in csv_paths:
        role = _role_from_path(p)
        if role is None:
            continue
        if role in excluded_roles:
            dropped_rows += 1
            continue
        try:
            df = pd.read_csv(p, usecols=[c for c, _, _ in METHODS])
        except (ValueError, pd.errors.ParserError):
            # Missing columns or malformed row — skip
            continue
        df["_role"] = role
        rows.append(df)
    if not rows:
        return {col: np.array([]) for col, _, _ in METHODS}, 0
    full = pd.concat(rows, ignore_index=True)
    scores = {
        col: pd.to_numeric(full[col], errors="coerce").dropna().to_numpy()
        for col, _, _ in METHODS
    }
    return scores, dropped_rows


def _plot_density(scores, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    all_vals = np.concatenate([v for v in scores.values() if len(v)])
    if len(all_vals) == 0:
        print(f"[density] no data to plot for {title}")
        plt.close(fig)
        return
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    xs = np.linspace(x_min - pad, x_max + pad, 400)

    for col, label, color in METHODS:
        vals = scores[col]
        if len(vals) < 2:
            continue
        ax.hist(
            vals, bins=30, density=True, color=color, alpha=0.18,
            range=(xs[0], xs[-1]),
        )
        try:
            kde = gaussian_kde(vals)
            ax.plot(
                xs, kde(xs), color=color, lw=2.0,
                label=f"{label} (n={len(vals)}, μ={vals.mean():.2f}, "
                      f"med={np.median(vals):.2f})",
            )
        except np.linalg.LinAlgError:
            # Constant distribution — KDE fails; fall back to a stick
            ax.axvline(vals[0], color=color, lw=2.0,
                       label=f"{label} (constant={vals[0]:.2f})")
        ax.axvline(vals.mean(), color=color, lw=1.0, ls="--", alpha=0.6)

    ax.set_xlabel("judge score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[density] wrote {out_path}")


def _summary_rows(scores, label):
    rows = []
    for col, name, _ in METHODS:
        v = scores[col]
        rows.append({
            "set": label, "method": name, "n": len(v),
            "mean": float(v.mean()) if len(v) else float("nan"),
            "median": float(np.median(v)) if len(v) else float("nan"),
            "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "p05": float(np.percentile(v, 5)) if len(v) else float("nan"),
            "p95": float(np.percentile(v, 95)) if len(v) else float("nan"),
        })
    return rows


def main():
    excluded = _load_excluded(MEMBERSHIP_PATH)
    print(f"[density] mythical cluster size: {len(excluded)}")

    csvs = sorted(glob.glob(os.path.join(GOLD_DIR, "Comparison_GoldStandard_*.csv")))
    print(f"[density] found {len(csvs)} gold-prompt csvs in {GOLD_DIR}")

    scores_sans, dropped = _load_scores(csvs, excluded_roles=excluded)
    scores_full, _ = _load_scores(csvs, excluded_roles=set())
    print(f"[density] dropped {dropped} role CSVs as mythical")

    _plot_density(
        scores_sans,
        title="Judge-score density (mythical cluster removed)",
        out_path=os.path.join(OUT_DIR, "density_scores_sans_mythical.png"),
    )
    _plot_density(
        scores_full,
        title="Judge-score density (all roles)",
        out_path=os.path.join(OUT_DIR, "density_scores_full.png"),
    )

    # KS test: assistant_axis vs steered on the cleaned set
    aa, st = scores_sans["assistant_axis_score"], scores_sans["steered_score"]
    if len(aa) and len(st):
        ks = ks_2samp(aa, st)
        print(
            f"[density] sans-mythical KS(assistant_axis vs steered): "
            f"stat={ks.statistic:.4f}, p={ks.pvalue:.3e}"
        )

    # Summary table
    rows = _summary_rows(scores_sans, "sans_mythical") + _summary_rows(scores_full, "full")
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_DIR, "density_scores_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"[density] wrote {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
