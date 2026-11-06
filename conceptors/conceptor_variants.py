"""Offline screen of conceptor designs for holding a persona, measured where the steering hook acts.

Every operator f is scored on held-out per-token activations at hidden module + 1 (the cloud the hook sees during
generation), fit on the other answers (every 5th answer of each role is held out):

  restore      how far f moves drifted tokens back along the persona axis: mean position after minus before, over 1
               minus before (1 = all the way to the persona answer mean, 0 = no help). "full" uses real assistant
               tokens, "half" uses midpoints of paired persona and assistant tokens.
  distort_p    rms size of the edit on tokens already in character, over the rms spread of those tokens around their
               mean: what an in-character conversation pays every token.
  spread_a     off-axis spread of drifted tokens after f, over the persona tokens' own off-axis spread: near 1 looks
               like the persona cloud, near 0 is collapse (the mean-shift control's mush), far above 1 is off-manifold.

Positions use the GPU harness's units (steering.fit: answer-average persona axis, assistant mean 0, persona mean 1).
Reference rows (none, add, cap, meanshift_tok, the current conceptor_tok) let the screen be checked against the 20-turn
holds before it ranks anything new.

Operators (C = R (R + a I)^-1, a = k x median eigenvalue of R):
  ctok k        centered per-token conceptor at aperture k (k = 1 is the harness's conceptor_tok): mu + C (h - mu)
  unc_p k b     uncentered persona-only conceptor (Postmus & Abreu 2024), Replace: b C h
  unc_bi k b    uncentered conceptor on both poles pooled (Triantafyllopoulos et al. 2026), Replace: b C h
  andnot k      centered C_persona AND NOT C_assistant (Jaeger's Boolean ops):
                C = (C_p^-1 + (I - C_a)^-1 - I)^-1 = (I + a S_p^-1 + S_a / a)^-1, applied as mu_p + C (h - mu_p)
  diffpull k b  uncentered conceptor C_d on per-question differences (each persona answer minus the assistant's mean
                answer to the same question): the directions that tell the persona from the assistant, a
                many-direction cap axis. Pulls only inside them: h + b C_d (mu - h)
  gated X       X applied only to tokens whose persona-axis projection is below the cap's floor (25th percentile)

  python conceptors/conceptor_variants.py --run runs/smollm3_full --module 17 --personas pirate,ghost,bard,consultant
"""
import argparse
import json
from pathlib import Path

import numpy as np
from analyze_acts import column_map, load_judge

GATE_PCT = 25


def split(jobs_of_role: list[int]) -> tuple[list[int], list[int]]:
    """Every 5th answer of a role is held out."""
    return [j for i, j in enumerate(jobs_of_role) if i % 5], [j for i, j in enumerate(jobs_of_role) if i % 5 == 0]


def eig(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lam, U = np.linalg.eigh(R)
    lam = np.clip(lam, 0, None)
    return lam, U, float(np.median(lam[lam > lam.max() * 1e-10]))


def conceptor(R: np.ndarray, k: float) -> np.ndarray:
    lam, U, med = eig(R)
    a = k * med
    return (U * (lam / (lam + a))) @ U.T


def and_not(Sp: np.ndarray, Sa: np.ndarray, k: float) -> np.ndarray:
    """C_p AND NOT C_a for centered conceptors at a shared aperture a: (I + a S_p^-1 + S_a / a)^-1.
    S_p gets a tiny ridge so its inverse exists; it changes nothing above 1e-6 of its top eigenvalue."""
    lam_p, Up, med = eig(Sp)
    a = k * med
    inv_p = (Up / (lam_p + lam_p.max() * 1e-6)) @ Up.T
    d = Sp.shape[0]
    return np.linalg.inv(np.eye(d) + a * inv_p + Sa / a)


def operators(Tp: np.ndarray, Ta: np.ndarray, fit: dict) -> dict:
    """name -> f(H). Tp, Ta: training tokens (persona, assistant). fit: v, vhat, tau (gate floor)."""
    mu = Tp.mean(0)
    Sp = np.cov(Tp, rowvar=False, bias=True)
    Sa = np.cov(Ta, rowvar=False, bias=True)
    Rp = Tp.T @ Tp / len(Tp)
    n = min(len(Tp), len(Ta))
    Rb = (Tp[:n].T @ Tp[:n] + Ta[:n].T @ Ta[:n]) / (2 * n)
    v, vhat, tau, caps = fit["v"], fit["vhat"], fit["tau"], fit["caps"]

    def centered(C):
        return lambda H: mu + (H - mu) @ C

    def gated(f):
        def g(H):
            out = H.copy()
            low = H @ vhat < tau
            out[low] = f(H[low])
            return out
        return g

    ops = {"none": lambda H: H}
    for c in (0.5, 1.0):
        ops[f"add c{c}"] = lambda H, c=c: H + c * v
    for pct, t in caps.items():
        ops[f"cap p{pct}"] = lambda H, t=t: H + np.clip(t - H @ vhat, 0, None)[:, None] * vhat
    for b in (0.5, 1.0):
        ops[f"meanshift_tok b{b}"] = lambda H, b=b: (1 - b) * H + b * mu
    for k in (0.01, 0.1, 1.0, 10.0):
        ops[f"ctok k{k}"] = centered(conceptor(Sp, k))
    for k in (0.1, 1.0, 10.0):
        Cp, Cb = conceptor(Rp, k), conceptor(Rb, k)
        for b in (1.0, 1.5, 2.0):
            ops[f"unc_p k{k} b{b}"] = lambda H, C=Cp, b=b: b * H @ C
            ops[f"unc_bi k{k} b{b}"] = lambda H, C=Cb, b=b: b * H @ C
    for k in (0.1, 1.0, 10.0):
        ops[f"andnot k{k}"] = centered(and_not(Sp, Sa, k))
    for k in (0.1, 1.0, 10.0):
        Cd = conceptor(fit["Rd"], k)
        for b in (0.5, 1.0):
            ops[f"diffpull k{k} b{b}"] = lambda H, C=Cd, b=b: H + b * (mu - H) @ C
    for name in ("ctok k1.0", "andnot k1.0", "diffpull k1.0 b1.0", "diffpull k0.1 b1.0"):
        ops[f"gated {name}"] = gated(ops[name])
    return ops


def score(f, Pt: np.ndarray, At: np.ndarray, mu: np.ndarray, fit: dict, rng) -> dict:
    vhat, a0, p0 = fit["vhat"], fit["a0"], fit["p0"]
    pos = lambda H: ((H @ vhat - a0) / (p0 - a0)).mean()
    off = lambda H: np.sqrt((((H - mu) - np.outer((H - mu) @ vhat, vhat)) ** 2).sum(1).mean())
    off_p = off(Pt)
    m = min(len(Pt), len(At))
    half = 0.5 * (Pt[rng.permutation(len(Pt))[:m]] + At[rng.permutation(len(At))[:m]])
    out = {}
    for name, H in (("full", At), ("half", half)):
        b, a = pos(H), pos(f(H))
        out[f"restore_{name}"] = float((a - b) / (1 - b))
    Pf = f(Pt)
    out["pos_p_after"] = float(pos(Pf))
    out["distort_p"] = float(np.sqrt(((Pf - Pt) ** 2).sum(1).mean() / ((Pt - Pt.mean(0)) ** 2).sum(1).mean()))
    out["spread_a"] = float(off(f(At)) / off_p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--module", type=int, required=True)
    ap.add_argument("--personas", default="pirate")
    ap.add_argument("--default", dest="default_role", default="assistant")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    hidden = a.module + 1
    man = json.loads((a.run / "manifest.json").read_text())
    z = np.load(a.run / "acts.npz")
    role, jobs, offs = z["p1_role"], z["p1_tok_job"], z["p1_tok_offsets"]
    tok = z["p1_tok_acts"][:, man["token_hidden_idx"].index(hidden), :].astype(np.float64)
    col = column_map(man.get("saved_hidden_idx", list(range(z["p1_resp_avg"].shape[1]))))
    X = z["p1_resp_avg"][:, col(hidden), :].astype(np.float64)
    labels, _ = load_judge(a.run / "judge.jsonl", len(role)) if (a.run / "judge.jsonl").exists() else (None, None)
    rng = np.random.default_rng(0)

    def tokens(js):
        return np.concatenate([tok[offs[i]:offs[i + 1]] for i, j in enumerate(jobs) if j in set(js)])

    report = dict(run=str(a.run), hidden=hidden, gate_pct=GATE_PCT, personas={})
    a_tr, a_te = split([j for j in jobs if role[j] == a.default_role])
    Ta, At = tokens(a_tr), tokens(a_te)
    A_rows = role == a.default_role
    for persona in a.personas.split(","):
        P_rows = (role == persona) & ((labels == "fully") if labels is not None else True)
        p_tr, p_te = split([j for j in jobs if P_rows[j]])
        Tp, Pt = tokens(p_tr), tokens(p_te)
        v = X[P_rows].mean(0) - X[A_rows].mean(0)
        vhat = v / np.linalg.norm(v)
        proj = X[P_rows] @ vhat
        # per-question differences, leaving out the held-out answers so the test tokens don't shape C_d
        q = z["p1_q_idx"]
        a_mean = {qq: X[A_rows & (q == qq)].mean(0) for qq in np.unique(q[P_rows])}
        held = set(p_te)
        D = np.stack([X[i] - a_mean[q[i]] for i in np.flatnonzero(P_rows) if i not in held])
        fit = dict(v=v, vhat=vhat, a0=float((X[A_rows] @ vhat).mean()), p0=float(proj.mean()), Rd=D.T @ D / len(D),
                   tau=float(np.percentile(proj, GATE_PCT)), caps={p: float(np.percentile(proj, p)) for p in (25, 50, 75)})
        mu = Tp.mean(0)
        rows = {name: score(f, Pt, At, mu, fit, rng) for name, f in operators(Tp, Ta, fit).items()}
        report["personas"][persona] = dict(n_train_tokens=len(Tp), n_test_tokens=len(Pt), n_assistant_test=len(At),
                                           rows=rows)
        print(f"\n{persona}: {len(Tp)} train / {len(Pt)} test persona tokens, {len(At)} assistant test tokens")
        print(f"  {'operator':24s} {'restore_full':>12s} {'restore_half':>12s} {'distort_p':>9s} {'spread_a':>8s} {'pos_p':>6s}")
        for name, r in rows.items():
            print(f"  {name:24s} {r['restore_full']:12.2f} {r['restore_half']:12.2f} {r['distort_p']:9.2f} "
                  f"{r['spread_a']:8.2f} {r['pos_p_after']:6.2f}")
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
