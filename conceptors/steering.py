"""Fit steering interventions from a collected persona_acts.py run and turn them into Steer hook specs.

All fits use the steering module m (default n_layers // 2), i.e. hidden index m + 1 (output of decoder layer m),
on answer-averaged activations; persona rows are restricted to judge-"fully" rows when judge.jsonl exists.

Methods and their strength parameter:
  add           h + c v, v = mean(persona) - mean(default); c = 1 moves the default mean onto the persona mean
  cap           h + v^ max(0, tau - v^.h); tau = this percentile of the persona rows' projections (a floor)
  conceptor     (1 - b) h + b (mu + C (h - mu)), C = R (R + a I)^-1 on the persona cloud (rank <= n, median aperture)
  lowrank       same with a Ledoit-Wolf-shrunk rank-k conceptor (analyze_acts.lowrank_conceptor); directions outside the
                top k keep the shrinkage floor null_keep instead of 0
  meanshift     (1 - b) h + b mu: the conceptor's center and gain with C removed (matrix-removed control)
  conceptor_tok conceptor and center fitted on per-token persona activations (needs --save_tokens at hidden m + 1)
  meanshift_tok its matrix-removed control
  add_ctok      add at c, then conceptor_tok at beta 1 on the same module (hooks run in registration order): the vector
                sets the persona and the per-token conceptor pulls everything else back to the persona cloud; strength c
  add_cap       add at c, then a cap on the same module with its floor at the CAP_HOLD_PCT percentile: the vector sets the
                persona and the cap holds it along the persona axis only; strength c (the cap counterpart of add_ctok)
  cap_aa        the Assistant Axis paper's capping (Lu et al. 2026, Section 5), mirrored to hold a persona: a floor on
                each layer's own persona axis at every module in [0.70 n, 0.85 n), the late-middle band the paper caps
                (Qwen3-32B layers 46-53 of 64, Llama-3.3-70B 56-71 of 80), each floor at this percentile of the persona's
                projections at that layer; the paper's own cap is the 25th percentile
  ctok_cap      conceptor_tok at beta 1, then a cap on the same module (floor at this percentile): does the conceptor add
                anything over the cap? Paired with cap at the same percentile; strength = the cap percentile
  ctok_tight    conceptor_tok with the aperture TIGHT_K times smaller (a = TIGHT_K x median eigenvalue): the aperture
                knob the median rule fixes. The offline screen (conceptor_variants.py) predicts it slides toward the
                mean-shift control rather than toward the cap
  diffpull      h + b C_d (mu - h): pull toward the per-token persona mean only inside C_d, an uncentered conceptor
                (aperture DIFF_K x median) on per-question differences (each persona answer minus the default role's
                mean answer to the same question). C_d keeps the directions that tell the persona from the default
                role, a many-direction version of the cap's axis; content directions they share pass untouched
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

DEFAULT_GRID = {  # widened after the first SmolLM3 sweep (5 Nov 2026): the conceptor family acts only above beta 0.5
    "none": [0.0],
    "add": [0.5, 1.0, 1.5, 2.0, 3.0],
    "cap": [25, 50, 75, 90, 99],
    "conceptor": [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    "lowrank": [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    "meanshift": [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    "conceptor_tok": [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    "meanshift_tok": [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
    "add_ctok": [0.5, 1.0, 1.5, 2.0, 3.0],
    "add_cap": [0.5, 1.0, 1.5, 2.0, 3.0],
    "cap_aa": [10, 25, 50, 75, 90],
    "ctok_cap": [25, 50, 75, 90, 99],
    "ctok_tight": [0.5, 0.7, 0.9, 1.0],
    "diffpull": [0.3, 0.5, 0.7, 1.0],
}
CAP_HOLD_PCT = 25  # add_cap's floor: the Assistant Axis paper caps at the 25th percentile of the role's own range
TOKEN_METHODS = ("conceptor_tok", "meanshift_tok", "add_ctok", "ctok_cap", "ctok_tight", "diffpull")
TIGHT_K = 10.0  # ctok_tight's aperture: a = TIGHT_K x the median eigenvalue (conceptor_tok uses 1)
DIFF_K = 10.0   # diffpull's aperture on the difference cloud; the offline screen's best restore-for-distortion point


def conceptor_matrix(R: np.ndarray, k: float = 1.0) -> np.ndarray:
    """C = R (R + a I)^-1 with a = k x the median positive eigenvalue of R."""
    lam, U = np.linalg.eigh(R)
    lam = np.clip(lam, 0, None)
    a = k * float(np.median(lam[lam > lam.max() * 1e-10]))
    return (U * (lam / (lam + a))) @ U.T


CAP_AA_BAND = (0.70, 0.85)


def cap_modules(n_layers: int, band: tuple = CAP_AA_BAND) -> list[int]:
    """Modules the Assistant-Axis-style cap acts on: [band[0] n, band[1] n) of the network, at least one.
    The default band matches the paper's choices (about 70-88% of depth)."""
    lo = round(band[0] * n_layers)
    return list(range(lo, max(lo + 1, round(band[1] * n_layers))))


def fit(run_dir: Path, persona: str = "pirate", default: str = "assistant", module: int | None = None,
        lowrank_k: int = 16, use_judge: bool = True, default_run: Path | None = None) -> dict:
    """default_run: take the default role's answer averages from another run of the same model (e.g. a run that only
    collected new personas), instead of from run_dir."""
    run_dir = Path(run_dir)
    z = dict(np.load(run_dir / "acts.npz"))  # load once: NpzFile decompresses on every access
    man = json.loads((run_dir / "manifest.json").read_text())
    n_layers = man["n_layers"]
    module = n_layers // 2 if module is None else module
    read_hidden = layer_plan(n_layers)[1]
    col = column_map(man.get("saved_hidden_idx", list(range(z["p1_resp_avg"].shape[1]))))
    role = z["p1_role"]
    P = role == persona
    if default_run is None:
        A, Zd, cold = role == default, z, col
    else:
        Zd = dict(np.load(Path(default_run) / "acts.npz"))
        mand = json.loads((Path(default_run) / "manifest.json").read_text())
        if mand["n_layers"] != n_layers or Zd["p1_resp_avg"].shape[-1] != z["p1_resp_avg"].shape[-1]:
            raise ValueError(f"{default_run} is a different model shape than {run_dir}")
        cold = column_map(mand.get("saved_hidden_idx", list(range(Zd["p1_resp_avg"].shape[1]))))
        A = Zd["p1_role"] == default
    defaults = lambda hidden: Zd["p1_resp_avg"][A][:, cold(hidden), :].astype(np.float64)  # noqa: E731
    n_judged = None
    if use_judge and (run_dir / "judge.jsonl").exists():
        labels, _ = load_judge(run_dir / "judge.jsonl", len(role))
        P = P & (labels == "fully")
        n_judged = int(P.sum())
    if P.sum() < 2 or A.sum() < 2:
        raise ValueError(f"need >= 2 persona and default rows, got {int(P.sum())} and {int(A.sum())}")

    X = z["p1_resp_avg"][:, col(module + 1), :].astype(np.float64)
    Xp, Xa = X[P], defaults(module + 1)
    v = Xp.mean(0) - Xa.mean(0)
    vhat = v / np.linalg.norm(v)
    mu = Xp.mean(0)
    C, keep, aperture = conceptor_from(Xp - mu)
    lr = lowrank_conceptor(Xp, lowrank_k)
    # diffpull: per-question differences from the default role's mean answer (its overall mean where a question is
    # missing, e.g. a default run with other questions)
    q, qd = z["p1_q_idx"], Zd["p1_q_idx"][A]
    a_by_q = {qq: Xa[qd == qq].mean(0) for qq in np.unique(qd)}
    D = np.stack([X[i] - a_by_q.get(q[i], Xa.mean(0)) for i in np.flatnonzero(P)])
    diff_C = conceptor_matrix(D.T @ D / len(D), DIFF_K)

    Xr = z["p1_resp_avg"][:, col(read_hidden), :].astype(np.float64)
    Xra = defaults(read_hidden)
    vr = Xr[P].mean(0) - Xra.mean(0)
    ur = vr / np.linalg.norm(vr)
    a0, p0 = float((Xra @ ur).mean()), float((Xr[P] @ ur).mean())
    # per-layer persona axes for the Assistant-Axis-style cap
    mods = cap_modules(n_layers)
    ml_vhat, ml_proj = [], []
    for m_ in mods:
        Xl = z["p1_resp_avg"][P][:, col(m_ + 1), :].astype(np.float64)
        vl = Xl.mean(0) - defaults(m_ + 1).mean(0)
        ul = vl / np.linalg.norm(vl)
        ml_vhat.append(ul); ml_proj.append(Xl @ ul)

    out = dict(
        module=module, read_hidden=read_hidden, n_layers=n_layers, d=int(X.shape[1]), persona=persona, default=default,
        n_persona=int(P.sum()), n_default=int(A.sum()), n_persona_judged_fully=n_judged, source=str(run_dir),
        source_model=man["model"], source_snapshot=man.get("model_snapshot"),
        v=v, vhat=vhat, proj_persona=Xp @ vhat, mu=mu, C=C, conceptor_rank=int((keep > 1e-6).sum()),
        conceptor_aperture=float(aperture), lr_U=lr["U"], lr_keep=lr["keep"], lr_delta=lr["delta"],
        lr_null_keep=lr["null_keep"], read_u=ur, read_a0=a0, read_p0=p0, n_tokens=0,
        steer_a0=float((Xa @ vhat).mean()), steer_p0=float((Xp @ vhat).mean()),
        ml_modules=np.array(mods), ml_vhat=np.stack(ml_vhat), ml_proj=np.stack(ml_proj),
        default_source=str(default_run or run_dir), diff_C=diff_C, n_diff=len(D), diff_k=DIFF_K, tight_k=TIGHT_K,
    )
    out.update(token_fit(z, man, P, module + 1))
    if "tok_C" in out:  # typical off-axis spread of real persona tokens, the unit for trajectory read-outs
        out["tok_off_ref"] = out.pop("_tok_off_ref_fn")(vhat)
    return out


def token_fit(z: dict, man: dict, persona_rows: np.ndarray, hidden: int) -> dict:
    """Conceptor and mean on per-token persona activations at `hidden`, when the run saved them there.

    The hooks act on every token, so a token-level cloud is the matching reference: answer averages vary far less
    than tokens do, and pulling each token toward an answer-average center erases token identity.
    """
    tok_idx = man.get("token_hidden_idx", [])
    if "p1_tok_acts" not in z or hidden not in tok_idx:
        return {}
    k = tok_idx.index(hidden)
    acts, offs, jobs = z["p1_tok_acts"], z["p1_tok_offsets"], z["p1_tok_job"]
    rows = [acts[offs[i]:offs[i + 1], k, :] for i, j in enumerate(jobs) if persona_rows[j]]
    if not rows:
        return {}
    T = np.concatenate(rows).astype(np.float64)
    mu_t = T.mean(0)
    Tc = T - mu_t
    lam, U = np.linalg.eigh(Tc.T @ Tc / len(T))
    lam = np.clip(lam, 0, None)
    a = float(np.median(lam[lam > lam.max() * 1e-10]))
    C_t = (U * (lam / (lam + a))) @ U.T
    C_tight = (U * (lam / (lam + TIGHT_K * a))) @ U.T

    def off_ref(vhat):
        Y = Tc - np.outer(Tc @ vhat, vhat)
        return float(np.sqrt((Y ** 2).sum(1).mean()))
    return dict(tok_mu=mu_t, tok_C=C_t, tok_C_tight=C_tight, n_tokens=int(len(T)), tok_rows=len(rows), tok_aperture=a,
                tok_rank=int((lam > lam.max() * 1e-10).sum()), _tok_off_ref_fn=off_ref)


def available(fits: dict, method: str) -> bool:
    return method not in TOKEN_METHODS or "tok_C" in fits


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
                     keep=t(fits["lr_keep"]), null_keep=float(fits.get("lr_null_keep", 0.0)))]
    if method == "meanshift":
        return [dict(mode="meanshift", module=m, beta=float(strength), mu=t(fits["mu"]))]
    if method in TOKEN_METHODS and "tok_C" not in fits:
        raise ValueError(f"{method} needs per-token activations at hidden {m + 1} in the fit run")
    if method == "conceptor_tok":
        return [dict(mode="conceptor", module=m, beta=float(strength), mu=t(fits["tok_mu"]), C=t(fits["tok_C"]))]
    if method == "meanshift_tok":
        return [dict(mode="meanshift", module=m, beta=float(strength), mu=t(fits["tok_mu"]))]
    if method == "ctok_tight":
        return [dict(mode="conceptor", module=m, beta=float(strength), mu=t(fits["tok_mu"]), C=t(fits["tok_C_tight"]))]
    if method == "ctok_cap":
        return specs(fits, "conceptor_tok", 1.0, device) + specs(fits, "cap", strength, device)
    if method == "diffpull":
        return [dict(mode="pull", module=m, beta=float(strength), mu=t(fits["tok_mu"]), C=t(fits["diff_C"]))]
    if method == "add_ctok":
        return specs(fits, "add", strength, device) + specs(fits, "conceptor_tok", 1.0, device)
    if method == "add_cap":
        return specs(fits, "add", strength, device) + specs(fits, "cap", CAP_HOLD_PCT, device)
    if method == "cap_aa":
        return [dict(mode="cap", module=int(m_), vhat=t(u), tau=float(np.percentile(pr, strength)))
                for m_, u, pr in zip(fits["ml_modules"], fits["ml_vhat"], fits["ml_proj"])]
    raise ValueError(f"unknown method {method}")


def readout(fits: dict, resp_at_read: np.ndarray) -> np.ndarray:
    """Persona score of answer-averaged activations at the read layer (default mean 0, persona mean 1)."""
    s = resp_at_read.astype(np.float64) @ fits["read_u"]
    return (s - fits["read_a0"]) / (fits["read_p0"] - fits["read_a0"])


def summary(fits: dict) -> dict:
    """JSON-safe provenance of a fit (no arrays)."""
    keep = ("module", "read_hidden", "n_layers", "d", "persona", "default", "n_persona", "n_default",
            "n_persona_judged_fully", "source", "source_model", "source_snapshot", "conceptor_rank",
            "conceptor_aperture", "lr_delta", "lr_null_keep", "n_tokens", "tok_rows", "tok_rank", "tok_aperture",
            "steer_a0", "steer_p0", "tok_off_ref", "default_source", "n_diff", "diff_k", "tight_k")
    out = {k: fits[k] for k in keep if k in fits}
    if "ml_modules" in fits:
        out["cap_aa_modules"] = [int(m_) for m_ in fits["ml_modules"]]
    out["cap_hold_pct"] = CAP_HOLD_PCT
    out["v_norm"] = float(np.linalg.norm(fits["v"]))
    out["lowrank_k"] = int(len(fits["lr_keep"]))
    return out
