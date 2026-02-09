from __future__ import annotations

import itertools

import numpy as np

import random_baselines_common as c


def hexagon_observed(traits_unit: np.ndarray) -> dict:
    sim = traits_unit @ traits_unit.T
    adj = float(sim[c.HEX_ADJ[:, 0], c.HEX_ADJ[:, 1]].mean())
    alt = float(sim[c.HEX_ALT[:, 0], c.HEX_ALT[:, 1]].mean())
    opp = float(sim[c.HEX_OPP[:, 0], c.HEX_OPP[:, 1]].mean())
    return {
        "sim_matrix": sim,
        "adjacent_mean": adj,
        "alternate_mean": alt,
        "opposite_mean": opp,
        "is_monotonic": bool(adj > alt > opp),
    }


def hexagon_label_permutation(sim: np.ndarray) -> tuple[int, float]:
    n_monotonic = 0
    for perm in itertools.permutations(range(len(c.TRAITS))):
        p = np.asarray(perm, dtype=np.int64)
        adj = sim[p[c.HEX_ADJ[:, 0]], p[c.HEX_ADJ[:, 1]]].mean()
        alt = sim[p[c.HEX_ALT[:, 0]], p[c.HEX_ALT[:, 1]]].mean()
        opp = sim[p[c.HEX_OPP[:, 0]], p[c.HEX_OPP[:, 1]]].mean()
        if adj > alt > opp:
            n_monotonic += 1
    return n_monotonic, float(n_monotonic / 720)

