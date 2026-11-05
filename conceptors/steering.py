"""Fit steering interventions from a collected persona_acts.py run and turn them into Steer hook specs.

All fits use the steering module m (default n_layers // 2), i.e. hidden index m + 1 (output of decoder layer m),
on answer-averaged activations; persona rows are restricted to judge-"fully" rows when judge.jsonl exists.

Methods and their strength parameter:
  add           h + c v, v = mean(persona) - mean(default); c = 1 moves the default mean onto the persona mean
  cap           h + v^ max(0, tau - v^.h); tau = this percentile of the persona rows' projections (a floor)
  conceptor     (1 - b) h + b (mu + C (h - mu)), C = R (R + a I)^-1 on the persona cloud (rank <= n, median aperture)
  lowrank       same with a Ledoit-Wolf-shrunk rank-k conceptor (analyze_acts.lowrank_conceptor)
  meanshift     (1 - b) h + b mu: the conceptor's center and gain with C removed (matrix-removed control)
  none          no intervention

Read-out: a persona score from an unsteered teacher-forced pass, projected on the persona axis at the read layer
(0.75 n, not the steered layer) and scaled so default mean = 0, persona mean = 1.
"""

import json
from pathlib import Path

import numpy as np
import torch

from analyze_acts import column_map, layer_plan, load_judge, lowrank_conceptor
from persona_acts import conceptor_from

DEFAULT_GRID = {
    "none": [0.0],
    "add": [0.5, 1.0, 2.0, 4.0],
    "cap": [10, 25, 50, 75],
    "conceptor": [0.1, 0.3, 0.5, 0.8],
    "lowrank": [0.1, 0.3, 0.5, 0.8],
    "meanshift": [0.1, 0.3, 0.5, 0.8],
}


def fit(run_dir: Path, persona: str = "pirate", default: str = "assistant", module: int | None = None,
        lowrank_k: int = 16, use_judge: bool = True) -> dict:
    run_dir = Path(run_dir)
    z = np.load(run_dir / "acts.npz")
    man = json.loads((run_dir / "manifest.json").read_text())
    n_layers = man["n_layers"]
    module = n_layers // 2 if module is None else module
    read_hidden = layer_plan(n_layers)[1]
    col = column_map(man.get("saved_hidden_idx", list(range(z["p1_resp_avg"].shape[1]))))
    role = z["p1_role"]
    P, A = role == persona, role == default
    n_judged = None
    if use_judge and (run_dir / "judge.jsonl").exists():
        labels, _ = load_judge(run_dir / "judge.jsonl", len(role))
        P = P & (labels == "fully")
        n_judged = int(P.sum())
    if P.sum() < 2 or A.sum() < 2:
        raise ValueError(f"need >= 2 persona and default rows, got {int(P.sum())} and {int(A.sum())}")

    X = z["p1_resp_avg"][:, col(module + 1), :].astype(np.float64)
    Xp, Xa = X[P], X[A]
    v = Xp.mean(0) - Xa.mean(0)
    vhat = v / np.linalg.norm(v)
    mu = Xp.mean(0)
    C, keep, aperture = conceptor_from(Xp - mu)
    lr = lowrank_conceptor(Xp, lowrank_k)

    Xr = z["p1_resp_avg"][:, col(read_hidden), :].astype(np.float64)
    vr = Xr[P].mean(0) - Xr[A].mean(0)
    ur = vr / np.linalg.norm(vr)
    a0, p0 = float((Xr[A] @ ur).mean()), float((Xr[P] @ ur).mean())

    return dict(
        module=module, read_hidden=read_hidden, n_layers=n_layers, d=int(X.shape[1]), persona=persona, default=default,
        n_persona=int(P.sum()), n_default=int(A.sum()), n_persona_judged_fully=n_judged, source=str(run_dir),
        source_model=man["model"], source_snapshot=man.get("model_snapshot"),
        v=v, vhat=vhat, proj_persona=Xp @ vhat, mu=mu, C=C, conceptor_rank=int((keep > 1e-6).sum()),
        conceptor_aperture=float(aperture), lr_U=lr["U"], lr_keep=lr["keep"], lr_delta=lr["delta"],
        read_u=ur, read_a0=a0, read_p0=p0,
    )


def specs(fits: dict, method: str, strength: float, device: str = "cpu") -> list[dict]:
    """Steer specs for one method at one strength, with tensors already on the target device."""
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    m = fits["module"]
    if method == "none":
        return []
    if method == "add":
        return [dict(mode="add", module=m, coef=float(strength), v=t(fits["v"]))]
    if method == "cap":
        tau = float(np.percentile(fits["proj_persona"], strength))
        return [dict(mode="cap", module=m, vhat=t(fits["vhat"]), tau=tau)]
    if method == "conceptor":
        return [dict(mode="conceptor", module=m, beta=float(strength), mu=t(fits["mu"]), C=t(fits["C"]))]
    if method == "lowrank":
        return [dict(mode="lowrank", module=m, beta=float(strength), mu=t(fits["mu"]), U=t(fits["lr_U"]),
                     keep=t(fits["lr_keep"]))]
    if method == "meanshift":
        return [dict(mode="meanshift", module=m, beta=float(strength), mu=t(fits["mu"]))]
    raise ValueError(f"unknown method {method}")


def readout(fits: dict, resp_at_read: np.ndarray) -> np.ndarray:
    """Persona score of answer-averaged activations at the read layer (default mean 0, persona mean 1)."""
    s = resp_at_read.astype(np.float64) @ fits["read_u"]
    return (s - fits["read_a0"]) / (fits["read_p0"] - fits["read_a0"])


def summary(fits: dict) -> dict:
    """JSON-safe provenance of a fit (no arrays)."""
    keep = ("module", "read_hidden", "n_layers", "d", "persona", "default", "n_persona", "n_default",
            "n_persona_judged_fully", "source", "source_model", "source_snapshot", "conceptor_rank",
            "conceptor_aperture", "lr_delta")
    out = {k: fits[k] for k in keep}
    out["v_norm"] = float(np.linalg.norm(fits["v"]))
    out["lowrank_k"] = int(len(fits["lr_keep"]))
    return out
