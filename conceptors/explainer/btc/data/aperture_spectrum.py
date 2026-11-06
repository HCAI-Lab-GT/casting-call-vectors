"""Export the SmolLM3 pirate per-token spectrum for the explainer's aperture plate.

For each eigendirection i of the persona token cloud (hook layer 17, hidden 18): lam_i / median eigenvalue,
w_i = (u_i . vhat)^2 (where the persona axis lives), e_i = mean over assistant tokens of (u_i . gap)^2 / total
(where the assistant-to-persona gap lives). Sorted by eigenvalue, largest first.
usage: python aperture_spectrum.py RUN_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/glenn/dev/casting-call-vectors/conceptors")
import numpy as np  # noqa: E402

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
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    a0 = f["tok_aperture"]
    w = (U.T @ f["vhat"]) ** 2
    G = f["tok_mu"] - H
    e = ((G @ U) ** 2).mean(0)
    r = lambda x, n=5: [float(f"{v:.{n}g}") for v in x]  # noqa: E731
    res = dict(source="runs/smollm3_full, pirate vs assistant, SmolLM3-3B hook layer 17 (hidden 18), per-token",
               n_persona_tokens=f["n_tokens"], n_assistant_tokens=int(len(H)), dim=int(len(lam)),
               lam_over_median=r(lam / a0), vhat_weight=r(w / w.sum()), gap_energy=r(e / e.sum()))
    out.write_text(json.dumps(res, separators=(",", ":")))
    print(len(lam), "dirs; vhat weight sum", float(w.sum()), "; top-10 vhat weight", float((w / w.sum())[:10].sum()))


if __name__ == "__main__":
    main()
