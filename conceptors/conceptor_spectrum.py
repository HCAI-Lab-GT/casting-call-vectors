"""What a per-token conceptor keeps, direction by direction, against what a cap touches (deck data).

For each persona: the eigen-directions of its per-token cloud at hidden module + 1 (the cloud conceptor_tok is fit
on), the keep share lam / (lam + a) the conceptor applies along each, how much of the persona direction vhat lies in
each, and how much of the default tokens' gap to the persona center does. A cap edits only along vhat; the conceptor
reweights every direction, so the question is where the gap and the persona direction sit on its keep curve.
Directions are binned on a log scale of their rank for plotting.

  python conceptors/conceptor_spectrum.py --run runs/smollm3_full --personas pirate ghost bard consultant \
      --module 17 --out spectrum.json [--default_run runs/smollm3_full]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import steering


def default_tokens(run: Path, hidden: int, default: str = "assistant") -> np.ndarray:
    z = np.load(run / "acts.npz")
    k = json.loads((run / "manifest.json").read_text())["token_hidden_idx"].index(hidden)
    role, jobs, offs, acts = z["p1_role"], z["p1_tok_job"], z["p1_tok_offsets"], z["p1_tok_acts"]
    return np.concatenate([acts[offs[i]:offs[i + 1], k, :] for i, j in enumerate(jobs) if role[j] == default]).astype(np.float64)


def spectrum(f: dict, H: np.ndarray, n_bins: int = 90) -> dict:
    C, mu, vhat = f["tok_C"], f["tok_mu"], f["vhat"]
    keep, U = np.linalg.eigh(C)                     # C = U diag(keep) U^T; keep = lam / (lam + a)
    order = np.argsort(-keep)
    keep, U = keep[order], U[:, order]
    v_w = (U.T @ vhat) ** 2                         # share of the persona direction in each eigen-direction
    G = mu - H
    g_w = ((G @ U) ** 2).sum(0) / (G ** 2).sum()    # share of the default tokens' gap energy in each
    d = len(keep)
    edges = np.unique(np.round(np.logspace(0, np.log10(d + 1), n_bins + 1)).astype(int))  # 1-based ranks, end exclusive
    edges[-1] = d + 1
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = slice(lo - 1, hi - 1)
        bins.append(dict(lo=int(lo), hi=int(hi - 1), keep=round(float(keep[s].mean()), 4),
                         v=round(float(v_w[s].sum()), 5), gap=round(float(g_w[s].sum()), 5)))
    return dict(d=d, n_tokens=int(f["n_tokens"]), aperture=float(f["tok_aperture"]),
                keep_along_vhat=round(float(vhat @ C @ vhat), 4),
                gap_energy_kept=round(float((g_w * keep ** 2).sum()), 4),
                gap_share_in_directions_kept_below_half=round(float(g_w[keep < .5].sum()), 4),
                vhat_share_in_top_10=round(float(v_w[:10].sum()), 4),
                n_kept_above_090=int((keep > .9).sum()), n_kept_below_010=int((keep < .1).sum()), bins=bins)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--personas", nargs="+", required=True)
    ap.add_argument("--module", type=int, default=None)
    ap.add_argument("--default_run", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge", action="store_true", help="add to an existing --out file instead of replacing it")
    a = ap.parse_args(argv)
    out = json.loads(a.out.read_text()) if a.merge and a.out.exists() else {}
    H = None
    for p in a.personas:
        f = steering.fit(a.run, persona=p, module=a.module, default_run=a.default_run)
        if "tok_C" not in f:
            raise SystemExit(f"{p}: no per-token rows at hidden {f['module'] + 1} in {a.run}")
        if H is None:
            H = default_tokens(a.default_run or a.run, f["module"] + 1)
        out[p] = spectrum(f, H)
        s = out[p]
        print(f"{p}: {s['n_tokens']} tokens, keeps {s['keep_along_vhat']} along vhat, gap energy kept {s['gap_energy_kept']}, "
              f"gap share where keep < 0.5: {s['gap_share_in_directions_kept_below_half']}, "
              f"{s['n_kept_above_090']} directions kept > 0.9, {s['n_kept_below_010']} < 0.1", flush=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
