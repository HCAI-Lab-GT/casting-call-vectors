"""How much of the default-to-persona gap a fitted conceptor closes, measured where the steering hook acts.

At beta 1 the conceptor hook maps h -> mu + C (h - mu), so a default activation's gap to the persona center becomes
C (mu - h). Along the persona direction vhat the closed share is 1 - vhat.C(mu - h) / vhat.(mu - h); the mean-shift
control (C = 0) closes all of it. Measured on the default role's per-token activations at hidden module + 1, the cloud
the hook sees during generation (needs a run collected with --save_tokens at that index).

  python conceptors/gap_closure.py --run runs/smollm3_full --module 17
"""
import argparse
import json
from pathlib import Path

import numpy as np

import steering


def gap_report(H: np.ndarray, vhat: np.ndarray, mu: np.ndarray, C: np.ndarray) -> dict:
    G = mu - H                                       # each default token's gap to the persona center
    g, rem = G @ vhat, (G @ C) @ vhat                # C is symmetric: vhat.C g_i
    lam, U = np.linalg.eigh(C)                       # eigenvalues = per-direction keep shares
    e = (G @ U) ** 2
    return dict(keep_along_vhat=float(vhat @ C @ vhat), closed_along_vhat_median=float(np.median(1 - rem / g)),
                gap_energy_kept=float(np.mean((e * lam ** 2).sum(1) / e.sum(1))),
                gap_energy_in_kept_dirs=float(np.mean(e[:, lam > 0.9].sum(1) / e.sum(1))), n_kept_dirs=int((lam > 0.9).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--module", type=int, default=None)
    ap.add_argument("--persona", default="pirate")
    ap.add_argument("--default", dest="default_role", default="assistant")
    a = ap.parse_args(argv)
    f = steering.fit(a.run, persona=a.persona, default=a.default_role, module=a.module)
    hidden = f["module"] + 1
    z = np.load(a.run / "acts.npz")
    k = json.loads((a.run / "manifest.json").read_text())["token_hidden_idx"].index(hidden)
    role, jobs, offs, acts = z["p1_role"], z["p1_tok_job"], z["p1_tok_offsets"], z["p1_tok_acts"]
    H = np.concatenate([acts[offs[i]:offs[i + 1], k, :] for i, j in enumerate(jobs)
                        if role[j] == a.default_role]).astype(np.float64)
    out = dict(run=str(a.run), hidden=hidden, n_default_tokens=len(H),
               answer_average=gap_report(H, f["vhat"], f["mu"], f["C"]))
    if "tok_C" in f:
        out["per_token"] = gap_report(H, f["vhat"], f["tok_mu"], f["tok_C"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
