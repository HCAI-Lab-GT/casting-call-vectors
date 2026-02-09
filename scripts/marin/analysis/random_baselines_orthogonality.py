from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

import random_baselines_common as c


def orthogonality_observed(traits_unit: np.ndarray, pcs: np.ndarray) -> dict:
    pcs_unit = c.unit_rows(pcs)
    cos = traits_unit @ pcs_unit.T
    abs_cos = np.abs(cos)
    per_trait_r2 = (cos * cos).sum(axis=1)
    return {
        "cos_matrix": cos,
        "max_abs_cos": float(abs_cos.max()),
        "mean_abs_cos": float(abs_cos.mean()),
        "per_trait_subspace_r2": {t: float(v) for t, v in zip(c.TRAITS, per_trait_r2, strict=True)},
        "mean_subspace_r2": float(per_trait_r2.mean()),
    }


def analytic_p_max_abs_cos(x: float, d: int, n_comparisons: int) -> float:
    a = 0.5
    b = (d - 1) / 2
    p_single = float(beta_dist.sf(x * x, a=a, b=b))
    return float(-np.expm1(n_comparisons * np.log1p(-p_single)))


def _mean_pair_dots(x: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    dots = (x[:, pairs[:, 0], :] * x[:, pairs[:, 1], :]).sum(axis=2)
    return dots.mean(axis=1)


def run_monte_carlo(pcs: np.ndarray) -> tuple[dict, int]:
    rng = np.random.default_rng(c.SEED)
    pcs32 = c.unit_rows(pcs).astype(np.float32)

    max_abs = np.empty(c.N_TRIALS, dtype=np.float32)
    mean_abs = np.empty(c.N_TRIALS, dtype=np.float32)
    mean_r2 = np.empty(c.N_TRIALS, dtype=np.float32)
    n_hex_monotonic = 0

    done = 0
    while done < c.N_TRIALS:
        b = min(c.BATCH_SIZE, c.N_TRIALS - done)
        x = rng.standard_normal((b, len(c.TRAITS), c.DIMENSIONS), dtype=np.float32)
        x /= np.sqrt((x * x).sum(axis=2, keepdims=True))

        cos = (x.reshape(b * len(c.TRAITS), c.DIMENSIONS) @ pcs32.T).reshape(b, len(c.TRAITS), c.N_PCS)
        abs_cos = np.abs(cos).reshape(b, -1)
        max_abs[done : done + b] = abs_cos.max(axis=1)
        mean_abs[done : done + b] = abs_cos.mean(axis=1)
        mean_r2[done : done + b] = (cos * cos).sum(axis=2).mean(axis=1)

        adj_m = _mean_pair_dots(x, c.HEX_ADJ)
        alt_m = _mean_pair_dots(x, c.HEX_ALT)
        opp_m = _mean_pair_dots(x, c.HEX_OPP)
        n_hex_monotonic += int(((adj_m > alt_m) & (alt_m > opp_m)).sum())

        done += b

    def q(arr: np.ndarray, ps: list[float]) -> dict:
        vals = np.quantile(arr.astype(np.float64), ps).tolist()
        out = {}
        for p, v in zip(ps, vals, strict=True):
            out["p999" if p == 0.999 else f"p{int(p * 100)}"] = float(v)
        return out

    max_abs_q = q(max_abs, [0.5, 0.95, 0.99, 0.999])
    max_abs_q["max"] = float(max_abs.max())
    mean_abs_q = q(mean_abs, [0.5, 0.95, 0.99])
    mean_r2_q = q(mean_r2, [0.5, 0.95, 0.99])

    return {
        "max_abs_cos": max_abs,
        "mean_abs_cos": mean_abs,
        "mean_subspace_r2": mean_r2,
        "summary": {
            "n_trials": c.N_TRIALS,
            "max_abs_cos": max_abs_q,
            "mean_abs_cos": mean_abs_q,
            "mean_subspace_r2": mean_r2_q,
        },
    }, n_hex_monotonic

