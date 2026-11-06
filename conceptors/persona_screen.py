"""Offline screen of persona-holding operators on real activations: replay_acts.py output plus the hold judges.

Everything is measured at the steering layer (module 17 -> hidden 18 on SmolLM3), per persona, on these populations:
  single      single-turn answers judged fully in character; every 5th answer per role is held out, the rest fit
  assistant   single-turn assistant answers, same split
  hold_in     prompted 20-turn holds, unsteered: turns at least 3 of 4 judges rate fully in character
  hold_out    same holds: turns at least 3 of 4 judges rate not in character (real drift)
  hold_ai / hold_role  probe turns where at least 3 of 4 judges say the answer admits / does not admit being an AI
  noprompt    the same conversations with no system prompt (the model as itself)
Hold conversations are split by id: even ids fit (judge probes, the identity direction, shrinkage), odd ids evaluate.

Operators (each over a strength grid; the cap family's floors are percentiles of persona answer projections):
  add, cap, conceptor_tok, ctok_tight, meanshift_tok, diffpull, ctok_cap   the methods the GPU holds ran
  fisher_cap      gate on the Fisher score w = (S_W + lam I)^-1 d, edit along d (the minimal edit in the data metric)
  region          one-sided ellipse over Fisher and identity scores, minimal data-metric edit back to its boundary
  gated_diffpull  diffpull applied only to tokens below the Fisher floor
  gated_ot        Gaussian optimal transport assistant -> persona, weighted by a soft Fisher gate
  ccs_pull        pull toward the persona mean inside a contrastive second-moment subspace, Fisher-gated

Readouts on edited states (none of them is the axis an operator edits along):
  judge probe     logistic regression on hold tokens, in vs out character (fit conversations): P(in)
  identity probe  logistic regression on probe-turn tokens, admits AI vs not (fit conversations): P(not AI)
plus axis restore, distortion on in-character tokens, off-axis spread, and trigger rates.

  python conceptors/persona_screen.py --replay DATA/replay_smollm3_v2 --judges DATA --out results/persona_screen
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

JUDGES = ("qwen", "gemma", "mistral", "llama")
PERSONAS = ("pirate", "ghost", "bard", "consultant", "genie", "alien")
DIFF_K, TIGHT_K = 10.0, 10.0
RNG = np.random.default_rng(7)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replay", required=True, help="replay_acts.py output folder")
    ap.add_argument("--judges", required=True, help="folder holding <hold run>.judge-<name>/judge.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--personas", default=",".join(PERSONAS))
    ap.add_argument("--hold_prefix", default="steer_hold_v2_")
    ap.add_argument("--max_tokens", type=int, default=60000, help="cap per population (random subsample)")
    ap.add_argument("--shrink", type=float, default=None,
                    help="fixed Fisher shrinkage (x mean eigenvalue of S_W), as a deployed method would use; "
                         "default: chosen per persona on fit-conversation hold labels")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- data
class Replay:
    def __init__(self, folder: Path):
        z = np.load(folder / "acts.npz")
        self.rows = [json.loads(ln) for ln in (folder / "rows.jsonl").read_text().splitlines()]
        self.resp, self.tok, self.off = z["resp_avg"], z["tok_acts"], z["tok_offsets"]
        self.hidden_idx = [int(h) for h in z["hidden_idx"]]
        self.token_hidden = [int(h) for h in z["token_hidden"]]
        for r in self.rows:
            r["run"] = Path(r["source"]).name

    def tokens(self, idx, cap=None):
        """Stacked per-token activations (float32) of rows idx, and the row id of every token."""
        parts = [(self.tok[self.off[i]:self.off[i + 1], 0, :], i) for i in idx]
        if not parts:
            return np.zeros((0, self.tok.shape[-1]), np.float32), np.zeros(0, int)
        H = np.concatenate([p for p, _ in parts]).astype(np.float32)
        rid = np.concatenate([np.full(len(p), i) for p, i in parts])
        if cap and len(H) > cap:
            keep = np.sort(RNG.choice(len(H), cap, replace=False))
            H, rid = H[keep], rid[keep]
        return H, rid

    def means(self, idx, hidden):
        return self.resp[idx, self.hidden_idx.index(hidden), :].astype(np.float32)


def consensus(judge_root: Path, run: str) -> dict:
    """line -> (role verdict, identity verdict) from the four judges: 'in'/'out'/None and 'ai'/'role'/None."""
    votes = defaultdict(list)
    for j in JUDGES:
        for ln in (judge_root / f"{run}.judge-{j}" / "judge.jsonl").read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                votes[r["idx"]].append((r.get("roleplay_label"), r.get("identity_r_label") or r.get("identity_label")))
    out = {}
    for i, v in votes.items():
        rp = [a for a, _ in v]
        idl = [b for _, b in v]
        role = "in" if rp.count("fully") >= 3 else "out" if rp.count("not") >= 3 else None
        ident = "ai" if idl.count("ai") >= 3 else "role" if idl.count("not_ai") >= 3 else None
        out[i] = (role, ident)
    return out


# ----------------------------------------------------------------------------- linear algebra
def cov(X, center=None):
    c = X.mean(0) if center is None else center
    Y = X - c
    return (Y.T @ Y) / len(X)


def eigh(S):
    lam, U = np.linalg.eigh(S)
    return np.clip(lam, 0, None), U


def conceptor(S, k):
    lam, U = eigh(S)
    a = k * float(np.median(lam[lam > lam.max() * 1e-10]))
    return (U * (lam / (lam + a))) @ U.T


def sym_fn(S, f):
    lam, U = eigh(S)
    return (U * f(lam)) @ U.T


def quota(C):
    return float(np.trace(C) / len(C))


# ----------------------------------------------------------------------------- fits
def fit_persona(rp, p, args, judge_root):
    """All statistics and populations for one persona."""
    hid = rp.token_hidden[0]
    by_role = defaultdict(list)
    for i, r in enumerate(rp.rows):
        if r["kind"] == "answer":
            by_role[r["role"]].append(i)
    split = lambda idx: ([i for k, i in enumerate(idx) if k % 5], [i for k, i in enumerate(idx) if k % 5 == 0])
    p_fit, p_ev = split(by_role[p])
    a_fit, a_ev = split(by_role["assistant"])
    hold_run, np_run = f"{args.hold_prefix}{p}_prompt", f"{args.hold_prefix}{p}_steer_only"
    lab = consensus(judge_root, hold_run)
    hold = [i for i, r in enumerate(rp.rows) if r["kind"] == "hold" and r["run"] == hold_run]
    noprompt = [i for i, r in enumerate(rp.rows) if r["kind"] == "hold" and r["run"] == np_run and r["conv"] % 2]
    pop = defaultdict(list)
    for i in hold:
        r = rp.rows[i]
        role, ident = lab.get(r["line"], (None, None))
        part = "fit" if r["conv"] % 2 == 0 else "ev"
        if role:
            pop[f"hold_{role}_{part}"].append(i)
        if r["probe"] and ident:
            pop[f"hold_{ident}_{part}"].append(i)
    cap = args.max_tokens
    T = {}
    T["P_fit"], _ = rp.tokens(p_fit, cap)
    T["A_fit"], _ = rp.tokens(a_fit, cap)
    T["single"], _ = rp.tokens(p_ev, cap)
    T["assistant"], _ = rp.tokens(a_ev, cap)
    for k in ("hold_in_fit", "hold_out_fit", "hold_in_ev", "hold_out_ev", "hold_ai_fit", "hold_role_fit",
              "hold_ai_ev", "hold_role_ev"):
        T[k], _ = rp.tokens(pop[k], cap)
    T["noprompt"], _ = rp.tokens(noprompt, cap)
    counts = {k: len(v) for k, v in pop.items()} | {"single_fit_answers": len(p_fit), "noprompt_rows": len(noprompt)}

    # answer-level statistics, as steering.fit builds them
    Pr, Ar = rp.means(p_fit, hid), rp.means(a_fit, hid)
    v = Pr.mean(0) - Ar.mean(0)
    vhat = v / np.linalg.norm(v)
    a0, p0 = float((Ar @ vhat).mean()), float((Pr @ vhat).mean())
    qa = defaultdict(list)
    for i, x in zip(a_fit, Ar, strict=True):
        qa[rp.rows[i]["q_idx"]].append(x)
    qa = {q: np.mean(xs, 0) for q, xs in qa.items()}
    D = np.stack([x - qa.get(rp.rows[i]["q_idx"], Ar.mean(0)) for i, x in zip(p_fit, Pr, strict=True)])

    # token-level statistics
    P, A = T["P_fit"], T["A_fit"]
    mu, mu_a = P.mean(0), A.mean(0)
    Sp, Sa = cov(P), cov(A)
    Sw = (Sp + Sa) / 2
    d_tok = mu - mu_a
    F = {"p": p, "hid": hid, "v": v, "vhat": vhat, "a0": a0, "p0": p0, "Pr": Pr, "D": D, "mu": mu, "mu_a": mu_a, "Sp": Sp, "Sa": Sa, "Sw": Sw,
             "d_tok": d_tok, "C_t": conceptor(Sp, 1.0), "C_tight": conceptor(Sp, TIGHT_K), "C_d": conceptor(D.T @ D / len(D), DIFF_K),
             "counts": counts}
    F["proj_persona"] = Pr @ vhat
    return F, T


def fisher(F, lam_rel):
    d = F["d_tok"]
    lam = lam_rel * np.trace(F["Sw"]) / len(F["Sw"])
    S = F["Sw"] + lam * np.eye(len(d))
    w = np.linalg.solve(S, d)
    return w, S


def choose_shrink(F, T):
    """Shrinkage for the Fisher direction: the value that best separates fit-conversation hold tokens in vs out."""
    best = None
    for lam_rel in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        w, _ = fisher(F, lam_rel)
        s_in, s_out = T["hold_in_fit"] @ w, T["hold_out_fit"] @ w
        if not len(s_in) or not len(s_out):
            continue
        auc = roc_auc_score(np.r_[np.ones(len(s_in)), np.zeros(len(s_out))], np.r_[s_in, s_out])
        if best is None or auc > best[1]:
            best = (lam_rel, auc)
    return best or (0.1, float("nan"))


def probe(Xpos, Xneg):
    """Standardized logistic regression; returns P(pos | x) as a function."""
    if len(Xpos) < 20 or len(Xneg) < 20:
        return None
    X = np.r_[Xpos, Xneg]
    y = np.r_[np.ones(len(Xpos)), np.zeros(len(Xneg))]
    m, s = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.01, max_iter=400, class_weight="balanced").fit((X - m) / s, y)
    return lambda H: clf.predict_proba((H - m) / s)[:, 1]


# ----------------------------------------------------------------------------- operators
def operators(F, T, lam_rel):
    """name -> list of (setting label, f) with f(H) -> (H', edited mask)."""
    w, S = fisher(F, lam_rel)
    d = F["d_tok"]
    vhat, mu, mu_a = F["vhat"], F["mu"], F["mu_a"]
    I = np.eye(len(mu))
    fis_ans = F["Pr"] @ w  # answer-level Fisher scores of persona answers, for floors
    pct = lambda x, q: float(np.percentile(x, q))
    none = lambda H: np.zeros(len(H), bool)

    def cap_at(q):
        tau = pct(F["proj_persona"], q)

        def f(H):
            g = tau - H @ vhat
            on = g > 0
            return H + np.outer(np.where(on, g, 0), vhat), on
        return f

    def fisher_cap_at(q):
        tau = pct(fis_ans, q)
        wd = float(w @ d)

        def f(H):
            g = tau - H @ w
            on = g > 0
            return H + np.outer(np.where(on, g / wd, 0), d), on
        return f

    def gate_fisher(q):
        tau = pct(fis_ans, q)
        return lambda H: (H @ w) < tau

    out = {}
    out["none"] = [("-", lambda H: (H, none(H)))]
    def add_at(c):
        return lambda H: (H + c * F["v"], ~none(H))

    def meanshift_at(b):
        return lambda H: ((1 - b) * H + b * mu, ~none(H))

    def diffpull_at(b):
        return lambda H: (H + b * (mu - H) @ F["C_d"].T, ~none(H))

    out["add"] = [(f"c={c}", add_at(c)) for c in (0.25, 0.5, 1.0, 1.5)]
    out["cap"] = [(f"p{q}", cap_at(q)) for q in (10, 25, 50, 75, 90, 99)]
    out["fisher_cap"] = [(f"p{q}", fisher_cap_at(q)) for q in (10, 25, 50, 75, 90, 99)]

    def pull_center(C, b, center):
        M = I - C
        return lambda H: (H - b * (H - center) @ M.T, ~none(H))
    out["conceptor_tok"] = [(f"b={b}", pull_center(F["C_t"], b, mu)) for b in (0.3, 0.5, 0.7, 1.0)]
    out["ctok_tight"] = [(f"b={b}", pull_center(F["C_tight"], b, mu)) for b in (0.5, 0.7, 1.0)]
    out["meanshift_tok"] = [(f"b={b}", meanshift_at(b)) for b in (0.3, 0.5, 1.0)]
    Cd = F["C_d"]
    out["diffpull"] = [(f"b={b}", diffpull_at(b)) for b in (0.3, 0.5, 1.0)]

    def gated_diffpull(b, q):
        gate = gate_fisher(q)

        def f(H):
            on = gate(H)
            return H + b * np.where(on[:, None], (mu - H) @ Cd.T, 0), on
        return f
    out["gated_diffpull"] = [(f"b={b},p{q}", gated_diffpull(b, q)) for b in (0.5, 1.0) for q in (25, 50, 90)]

    def ctok_cap(q):
        c = cap_at(q)
        Ct = F["C_t"]

        def f(H):
            H1 = H - (H - mu) @ (I - Ct).T
            H2, _on = c(H1)
            return H2, np.ones(len(H), bool)
        return f
    out["ctok_cap"] = [(f"p{q}", ctok_cap(q)) for q in (25, 50, 99)]

    # Gaussian optimal transport assistant -> persona (shrunk covariances), weighted by a soft Fisher gate
    shrink = lambda M: M + 0.05 * np.trace(M) / len(M) * I
    Sa_h = sym_fn(shrink(F["Sa"]), np.sqrt)
    Sa_hi = sym_fn(shrink(F["Sa"]), lambda x: 1 / np.sqrt(x))
    Amap = Sa_hi @ sym_fn(Sa_h @ shrink(F["Sp"]) @ Sa_h, np.sqrt) @ Sa_hi
    s_scale = float(np.std(fis_ans))

    def gated_ot(q):
        tau = pct(fis_ans, q)

        def f(H):
            g = 1 / (1 + np.exp(-np.clip((tau - H @ w) / (0.5 * s_scale), -40, 40)))
            TH = mu + (H - mu_a) @ Amap.T
            return H + g[:, None] * (TH - H), g > 0.05
        return f
    out["gated_ot"] = [(f"p{q}", gated_ot(q)) for q in (25, 50, 90)]

    # contrastive second moment about the shared center, minus the assistant's (Xenos et al. 2026 style)
    c0 = (mu + mu_a) / 2
    Ac = cov(T["P_fit"], c0) - cov(T["A_fit"], c0)
    lam, U = np.linalg.eigh(Ac)
    W = U[:, np.argsort(lam)[::-1][:16]]
    PW = W @ W.T

    def ccs(b, q):
        gate = gate_fisher(q)

        def f(H):
            on = gate(H)
            return H + b * np.where(on[:, None], (mu - H) @ PW, 0), on
        return f
    out["ccs_pull"] = [(f"b={b},p{q}", ccs(b, q)) for b in (0.5, 1.0) for q in (50, 90)]

    # one-sided region over Fisher and identity scores, minimal data-metric edit back to the boundary
    if len(T["hold_ai_fit"]) >= 50 and len(T["hold_role_fit"]) >= 50:
        w_id = np.linalg.solve(S, T["hold_ai_fit"].mean(0) - T["hold_role_fit"].mean(0))
        B = np.stack([-w / np.linalg.norm(w), w_id / np.linalg.norm(w_id)])  # oriented so that larger = worse
        Q = T["P_fit"] @ B.T
        qm = Q.mean(0)
        Sq_i = np.linalg.inv(np.cov(Q.T))
        G = S @ B.T @ np.linalg.inv(B @ S @ B.T)  # maps a change in q to the minimal data-metric change in h
        dev = np.clip(Q - qm, 0, None)
        m_fit = np.sqrt(np.einsum("ij,jk,ik->i", dev, Sq_i, dev))

        def region(q):
            r = max(pct(m_fit, q), 1e-6)

            def f(H):
                dq = np.clip(H @ B.T - qm, 0, None)
                m = np.sqrt(np.einsum("ij,jk,ik->i", dq, Sq_i, dq))
                on = m > r
                scale = np.where(on, r / np.maximum(m, 1e-12), 1.0)
                delta_q = dq * scale[:, None] - dq
                return H + delta_q @ G.T, on
            return f
        out["region"] = [(f"p{q}", region(q)) for q in (50, 75, 90, 99)]
    return out, w


# ----------------------------------------------------------------------------- evaluation
def evaluate(F, T, ops, probes):
    vhat, mu = F["vhat"], F["mu"]
    s = lambda H: (H @ vhat - F["a0"]) / (F["p0"] - F["a0"])
    spread = float(np.sqrt(((T["single"] - mu) ** 2).sum(1).mean()))

    def off_spread(H):
        Y = H - mu
        Y = Y - np.outer(Y @ vhat, vhat)
        return float(np.sqrt((Y ** 2).sum(1).mean()))
    off_ref = off_spread(T["single"])
    jp, ip = probes["judge"], probes["identity"]
    base = {}
    if jp:
        base["p_in_out"] = float(jp(T["hold_out_ev"]).mean())
        base["p_in_in"] = float(jp(T["hold_in_ev"]).mean())
    if ip and len(T["hold_ai_ev"]):
        base["p_role_ai"] = float(ip(T["hold_ai_ev"]).mean())
        base["p_role_role"] = float(ip(T["hold_role_ev"]).mean())
    rows = []
    for name, settings in ops.items():
        for label, f in settings:
            Ho, on_o = f(T["hold_out_ev"])
            Hi, on_i = f(T["hold_in_ev"])
            r = {"method": name, "setting": label}
            so_b, so_a = s(T["hold_out_ev"]).mean(), s(Ho).mean()
            r["restore_axis"] = float((so_a - so_b) / (1 - so_b))
            r["distortion"] = float(np.sqrt(((Hi - T["hold_in_ev"]) ** 2).sum(1).mean()) / spread)
            r["off_spread_out"] = off_spread(Ho) / off_ref
            r["trigger_in"], r["trigger_out"] = float(on_i.mean()), float(on_o.mean())
            if jp:
                pa_, pi_ = float(jp(Ho).mean()), float(jp(Hi).mean())
                r["p_in_out_after"], r["p_in_in_after"] = pa_, pi_
                r["restore_probe"] = (pa_ - base["p_in_out"]) / max(base["p_in_in"] - base["p_in_out"], 1e-6)
                r["damage_probe"] = pi_ - base["p_in_in"]
            if ip and len(T["hold_ai_ev"]):
                Ha, _ = f(T["hold_ai_ev"])
                pr = float(ip(Ha).mean())
                r["p_role_ai_after"] = pr
                r["identity_fix"] = (pr - base["p_role_ai"]) / max(base["p_role_role"] - base["p_role_ai"], 1e-6)
            rows.append(r)
    return rows, base


def detection(F, T, w, probes):
    """How well each score separates real in-character hold tokens from drifted ones (evaluation conversations)."""
    Hi, Ho = T["hold_in_ev"], T["hold_out_ev"]
    y = np.r_[np.ones(len(Hi)), np.zeros(len(Ho))]
    out = {}
    scores = {"raw axis (cap)": lambda H: H @ F["vhat"], "Fisher (persona vs assistant)": lambda H: H @ w}
    if probes["judge"]:
        scores["judge probe (trained on hold labels)"] = probes["judge"]
    for name, fn in scores.items():
        si, so = fn(Hi), fn(Ho)
        sc = np.r_[si, so]
        auc = float(roc_auc_score(y, sc))
        res = {"auroc": auc}
        for fpr in (0.05, 0.25):
            thr = np.quantile(si, fpr)  # floor that edits this share of in-character tokens
            res[f"catch@{int(fpr * 100)}"] = float((so < thr).mean())
        out[name] = res
    return out


def science(F, T, w):
    """Where real drift goes, relative to the persona axis and the difference subspace."""
    vhat = F["vhat"]
    lam, U = eigh(F["C_d"])
    Ud = U[:, np.argsort(lam)[::-1][:8]]
    res = {}

    def decompose(delta):
        n2 = float(delta @ delta)
        along_v = float((delta @ vhat) ** 2 / n2)
        along_d8 = float(((Ud.T @ delta) ** 2).sum() / n2)
        return {"norm": float(np.sqrt(n2)), "cos_axis": float(delta @ vhat / np.sqrt(n2)), "share_axis": along_v,
                    "share_diff8": along_d8}
    if len(T["hold_out_ev"]) and len(T["hold_in_ev"]):
        res["drift"] = decompose(T["hold_out_ev"].mean(0) - T["hold_in_ev"].mean(0))
    if len(T["noprompt"]) and len(T["hold_in_ev"]):
        res["noprompt"] = decompose(T["noprompt"].mean(0) - T["hold_in_ev"].mean(0))
    if len(T["hold_ai_ev"]) and len(T["hold_role_ev"]):
        di = T["hold_ai_ev"].mean(0) - T["hold_role_ev"].mean(0)
        res["identity"] = decompose(di)
        if "drift" in res:
            dr = T["hold_out_ev"].mean(0) - T["hold_in_ev"].mean(0)
            res["identity"]["cos_drift"] = float(di @ dr / np.linalg.norm(di) / np.linalg.norm(dr))
    res["persona_axis_share_of_persona_variance"] = float(vhat @ F["Sp"] @ vhat / np.trace(F["Sp"]) * len(vhat))
    res["quota_C_t"], res["quota_C_d"] = quota(F["C_t"]), quota(F["C_d"])
    res["keep_axis_C_t"] = float(vhat @ F["C_t"] @ vhat)
    res["keep_axis_C_d"] = float(vhat @ F["C_d"] @ vhat)
    wn = w / np.linalg.norm(w)
    res["cos_fisher_axis"] = float(wn @ vhat)
    return res


def persona_space(rp, personas):
    """Geometry of the persona means at the steering layer: shared core, pairwise cosines, PCA of the means."""
    hid = rp.token_hidden[0]
    by_role = defaultdict(list)
    for i, r in enumerate(rp.rows):
        if r["kind"] == "answer":
            by_role[r["role"]].append(i)
    means = {r: rp.means(ix, hid).mean(0) for r, ix in by_role.items()}
    a = means["assistant"]
    V = {p: means[p] - a for p in personas if p in means}
    core = np.mean(list(V.values()), 0)
    cu = core / np.linalg.norm(core)
    names = list(V)
    cosm = [[float(V[x] @ V[y] / np.linalg.norm(V[x]) / np.linalg.norm(V[y])) for y in names] for x in names]
    share_core = {p: float((V[p] @ cu) ** 2 / (V[p] @ V[p])) for p in names}
    roles = [r for r in means if r in personas or r in ("assistant", "pirate_narrated")]
    M = np.stack([means[r] for r in roles])
    Mc = M - M.mean(0)
    _, sv, Vt = np.linalg.svd(Mc, full_matrices=False)
    coords = (Mc @ Vt[:3].T).tolist()
    return {"personas": names, "cos": cosm, "share_core": share_core, "pca_roles": roles, "pca_var": (sv ** 2 / (sv ** 2).sum()).tolist(),
                "pca_coords": coords, "cos_pc1_core": float(abs(Vt[0] @ cu)),
                "narrated_on_pirate_axis": float(((means.get("pirate_narrated", a) - a) @ V["pirate"]) / (V["pirate"] @ V["pirate"]))
                if "pirate" in V else None}


# ----------------------------------------------------------------------------- figures
def figures(results, space, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"none": "#6b6a64", "add": "#e2622d", "cap": "#2a78d6", "fisher_cap": "#6a4c93", "region": "#003057",
              "conceptor_tok": "#008f82", "ctok_tight": "#5fb3a9", "meanshift_tok": "#a9a59a", "diffpull": "#be1858",
              "gated_diffpull": "#e88aa9", "ctok_cap": "#7fa7d6", "gated_ot": "#b3a369", "ccs_pull": "#9a6f00"}
    personas = list(results)
    # 1: restore (judge probe) vs distortion, every setting, per persona
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False, sharey=True)
    for ax, p in zip(axes.flat, personas, strict=False):  # up to six panels
        for r in results[p]["ops"]:
            if "restore_probe" not in r:
                continue
            ax.scatter(r["distortion"], r["restore_probe"], s=26, color=colors.get(r["method"], "k"), alpha=.85,
                       label=r["method"])
        ax.axhline(0, color="#ccc", lw=.8)
        ax.axhline(1, color="#ccc", lw=.8, ls="--")
        ax.set_title(p)
        ax.set_xlabel("distortion of in-character tokens")
        ax.set_xscale("symlog", linthresh=0.05)
    axes[0, 0].set_ylabel("restore by judge probe (0 none, 1 = in-character level)")
    axes[1, 0].set_ylabel("restore by judge probe")
    h, lab = axes.flat[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, lab, strict=True):
        seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=7, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(out / "fig_restore_vs_distortion.png", dpi=140)
    plt.close(fig)
    # 2: detection AUROC per score per persona
    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = list(results[personas[0]]["detection"])
    x = np.arange(len(personas))
    for k, n in enumerate(names):
        ax.bar(x + (k - 1) * .27, [results[p]["detection"][n]["auroc"] for p in personas], .26, label=n)
    ax.set_xticks(x, personas)
    ax.set_ylim(.5, 1)
    ax.set_ylabel("AUROC: in-character vs drifted hold tokens")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_detection.png", dpi=140)
    plt.close(fig)
    # 3: where drift goes
    fig, ax = plt.subplots(figsize=(10, 4.2))
    keys = [("drift", "real drift (out minus in)"), ("noprompt", "no prompt minus in"), ("identity", "admits AI minus not")]
    for k, (key, lab) in enumerate(keys):
        ax.bar(x + (k - 1) * .27, [results[p]["science"].get(key, {}).get("share_axis", np.nan) for p in personas], .26, label=lab)
    ax.set_xticks(x, personas)
    ax.set_ylabel("share of the shift along the persona axis")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_drift_axis_share.png", dpi=140)
    plt.close(fig)
    # 4: persona space PCA of role means
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    C = np.array(space["pca_coords"])
    C = np.c_[C, np.zeros((len(C), max(0, 2 - C.shape[1])))]
    for (cx, cy), r in zip(C[:, :2], space["pca_roles"], strict=True):
        ax.scatter(cx, cy, s=60)
        ax.annotate(r, (cx, cy), textcoords="offset points", xytext=(6, 4))
    var = space["pca_var"] + [0.0, 0.0]
    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.set_title("Role means at the steering layer")
    fig.tight_layout()
    fig.savefig(out / "fig_persona_space.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    args = parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rp = Replay(Path(args.replay))
    judge_root = Path(args.judges)
    personas = [p for p in args.personas.split(",") if p]
    results = {}
    for p in personas:
        print(f"[{p}] fitting", flush=True)
        F, T = fit_persona(rp, p, args, judge_root)
        lam_rel, auc_fit = choose_shrink(F, T)
        if args.shrink is not None:
            lam_rel, auc_fit = args.shrink, float("nan")
        probes ={"judge": probe(T["hold_in_fit"], T["hold_out_fit"]),
                  "identity": probe(T["hold_role_fit"], T["hold_ai_fit"])}
        ops, w = operators(F, T, lam_rel)
        print(f"[{p}] evaluating {sum(len(v) for v in ops.values())} settings", flush=True)
        rows, base = evaluate(F, T, ops, probes)
        results[p] = {"counts": F["counts"] | {k: len(v) for k, v in T.items()}, "shrink": lam_rel,
                          "shrink_fit_auroc": auc_fit, "base": base, "detection": detection(F, T, w, probes),
                          "science": science(F, T, w), "ops": rows}
    space = persona_space(rp, personas)
    (out / "screen.json").write_text(json.dumps({"results": results, "persona_space": space}, indent=1))
    figures(results, space, out)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
