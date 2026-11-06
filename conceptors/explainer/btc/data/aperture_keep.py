"""Per-aperture keep and gap closure along the persona axis for the per-token conceptor (SmolLM3 pirate, module 17).

Uses the parent session's own fit code (steering.fit / tok_conceptor / gap_closure.gap_report), so k = 1 must
reproduce the logged 0.890 keep and 11.0% closure.
usage: python aperture_keep.py RUN_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/glenn/dev/casting-call-vectors/conceptors")
import numpy as np  # noqa: E402

import gap_closure  # noqa: E402
import steering  # noqa: E402


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    f = steering.fit(run, persona="pirate", default="assistant", module=17)
    hidden = f["module"] + 1
    z = np.load(run / "acts.npz")
    k_idx = json.loads((run / "manifest.json").read_text())["token_hidden_idx"].index(hidden)
    role, jobs, offs, acts = z["p1_role"], z["p1_tok_job"], z["p1_tok_offsets"], z["p1_tok_acts"]
    H = np.concatenate([acts[offs[i]:offs[i + 1], k_idx, :] for i, j in enumerate(jobs)
                        if role[j] == "assistant"]).astype(np.float64)
    lam, U = f["_tok_eig"]
    a0 = f["tok_aperture"]
    vhat = f["vhat"]
    w = (U.T @ vhat) ** 2                      # how vhat spreads over the token eigenbasis
    top = np.argsort(lam)[::-1]
    res = dict(n_default_tokens=int(len(H)), n_persona_tokens=f["n_tokens"], median_eig=a0,
               vhat_variance_over_median=float(vhat @ (U * lam) @ U.T @ vhat / a0),
               top_eigs_over_median=[float(x / a0) for x in lam[top[:8]]], rows={})
    for k in (1, 3, 10, 30, 100):
        C = steering.tok_conceptor(f, k)
        r = gap_closure.gap_report(H, vhat, f["tok_mu"], C)
        gain = lam / (lam + k * a0)
        r["pull_along_vhat"] = 1 - r["keep_along_vhat"]
        r["dirs_kept_over_half"] = int((gain > 0.5).sum())
        r["vhat_weight_in_dirs_kept_over_half"] = float(w[gain > 0.5].sum())
        res["rows"][str(k)] = r
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
