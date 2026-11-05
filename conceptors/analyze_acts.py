#!/usr/bin/env python3
"""
analyze_acts.py — turn persona_acts.py output into the numbers and 2-D projections the
conceptor explainer draws. Writes one JSON file (derived.json).

  python analyze_acts.py --run acts_qwen05                      # reads acts.npz, texts.jsonl, manifest.json,
                                                                # judge.jsonl if present; writes acts_qwen05/derived.json
  python analyze_acts.py --acts A.npz --texts T.jsonl --manifest M.json --out derived.json   # explicit paths

Conventions
  * hidden index i = output of decoder layer i-1 (0 = embeddings), as saved by persona_acts.py;
    arrays are indexed through manifest saved_hidden_idx, so thinned saves read the right layers
  * layers are fractions of n_layers: steer = n//2 + 1 (output of the phase-2 steering module n//2),
    read = 0.75 n, maps at 1/12 .. 1 n, logic at steer, read and 11/12 n (24 layers -> 13, 18, 22)
  * "standardized" = each dimension z-scored over the main role conditions; this removes the
    few massive-activation dimensions that otherwise dominate distances and PCA
  * persona score = projection onto the persona-minus-default mean difference, rescaled so the
    default (assistant) mean is 0 and the persona (pirate) mean is 1 (same layer and pooling)
  * with judge.jsonl, persona-side fits (directions, conceptors) use only rows the judge labels
    "fully" in character; the default role is not filtered
"""
import argparse, json, re
from fractions import Fraction
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PILOT_MAIN = ['assistant', 'pirate', 'doctor', 'teacher', 'bard', 'robot', 'consultant', 'ghost', 'pirate_doctor']
MAP_FRACS = [Fraction(1, 12), Fraction(1, 4), Fraction(5, 12), Fraction(13, 24), Fraction(2, 3), Fraction(5, 6), Fraction(1)]
READ_FRAC = Fraction(3, 4)
LOGIC_EXTRA_FRAC = Fraction(11, 12)
LOWRANK_KS = [4, 8, 16, 32]


def auc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def standardize(X, ref):
    mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def degenerate(text):
    w = re.findall(r"\w+", text.lower())
    if len(w) < 6:
        return False
    tri = list(zip(w, w[1:], w[2:]))
    return len(tri) > 0 and 1 - len(set(tri)) / len(tri) > 0.3


def frac_layer(f, n_layers):
    """round(f * n) with exact fractions (half rounds up), clipped to [0, n]."""
    return int(min(n_layers, max(0, int(f * n_layers + Fraction(1, 2)))))


def layer_plan(n_layers):
    steer = n_layers // 2 + 1
    read = frac_layer(READ_FRAC, n_layers)
    maps = sorted({frac_layer(f, n_layers) for f in MAP_FRACS})
    logic = sorted({steer, read, frac_layer(LOGIC_EXTRA_FRAC, n_layers)})
    return steer, read, maps, logic


def column_map(saved):
    """hidden index -> array column; requests for unsaved indices snap to the nearest saved one."""
    saved = list(saved)

    def col(h):
        return saved.index(h) if h in saved else int(np.argmin([abs(s - h) for s in saved]))
    return col


def load_judge(path, n1):
    """Phase-1 judge labels aligned with the npz rows (judge idx = texts.jsonl row index)."""
    labels = np.array([None] * n1, dtype=object)
    coh = np.full(n1, np.nan)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if j.get("phase", 1) == 1 and j["idx"] < n1:
            labels[j["idx"]] = j.get("roleplay_label")
            coh[j["idx"]] = j["coherence"] if j.get("coherence") is not None else np.nan
    return labels, coh


def lowrank_conceptor(X, k, aperture=None):
    """Rank-k conceptor C = U diag(keep) U^T on a Ledoit-Wolf-shrunk covariance of the centered rows of X.

    Shrinkage pulls every sample eigenvalue toward the mean eigenvalue m: lam' = (1 - delta) lam + delta m,
    and gives the null space lam' = delta m, so no direction is forced to exactly 0 or 1 by small n.
    The aperture (alpha^-2) defaults to the median of the kept shrunk eigenvalues.
    """
    mu = X.mean(0)
    Xc = X - mu
    n, d = Xc.shape
    delta = float(ledoit_wolf_shrinkage(Xc, assume_centered=True))
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = S ** 2 / n
    m = float(lam.sum() / d)
    k = min(k, len(lam))
    lam_k = (1 - delta) * lam[:k] + delta * m
    a = float(np.median(lam_k)) if aperture is None else aperture
    keep = lam_k / (lam_k + a)
    null_keep = delta * m / (delta * m + a)
    return dict(mu=mu, U=Vt[:k], keep=keep, delta=delta, aperture=a, null_keep=float(null_keep))


def kept_share(C, H, beta=None):
    """Mean share of each row's energy (relative to the conceptor's center) that C, or the interpolation
    (1 - beta) h + beta C h, keeps."""
    Hc = H - C["mu"]
    coef = Hc @ C["U"].T
    e_tot = (Hc ** 2).sum(1) + 1e-12
    if beta is None:
        return float((((coef * C["keep"]) ** 2).sum(1) / e_tot).mean())
    out = (1 - beta) * Hc + beta * ((coef * C["keep"]) @ C["U"])
    return float(((out ** 2).sum(1) / e_tot).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='run folder: reads acts.npz, texts.jsonl, manifest.json[, judge.jsonl]')
    ap.add_argument('--acts')
    ap.add_argument('--texts')
    ap.add_argument('--manifest')
    ap.add_argument('--judge', default=None, help='judge.jsonl (default: <run>/judge.jsonl if present)')
    ap.add_argument('--no_judge_filter', action='store_true', help='fit on all rows even if judge scores exist')
    ap.add_argument('--persona', default='pirate')
    ap.add_argument('--default', dest='default_role', default='assistant')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    run = Path(a.run) if a.run else None
    acts = a.acts or (run / 'acts.npz')
    texts = a.texts or (run / 'texts.jsonl')
    manifest = a.manifest or (run / 'manifest.json')
    out_path = a.out or (run / 'derived.json' if run else 'derived.json')
    judge_path = a.judge or (run / 'judge.jsonl' if run and (run / 'judge.jsonl').exists() else None)

    # load each array once: NpzFile decompresses the whole array on every z[key] access, ~80 times per run
    z = dict(np.load(acts))
    man = json.load(open(manifest))
    T = [json.loads(l) for l in open(texts)]
    p1t = [t for t in T if t.get('phase') != 2]
    p2t = [t for t in T if t.get('phase') == 2]
    role, q, task = z['p1_role'], z['p1_q_idx'], z['p1_task']
    n_layers = man['n_layers']
    STEER_HIDDEN, READ_HIDDEN, MAP_LAYERS, LOGIC_LAYERS = layer_plan(n_layers)
    saved = man.get('saved_hidden_idx', list(range(z['p1_resp_avg'].shape[1])))
    col = column_map(saved)

    present = list(dict.fromkeys(role.tolist()))
    MAIN = [r for r in PILOT_MAIN if r in present] + sorted(r for r in present if r not in PILOT_MAIN and not r.endswith('_narrated'))
    GROUPS = MAIN + [r for r in present if r.endswith('_narrated')]
    main_m = np.isin(role, MAIN)
    P, A = role == a.persona, role == a.default_role
    tr, te = q % 2 == 0, q % 2 == 1          # split by question: train on even, test on odd

    # judge filter: persona-side fits use only fully in-character rows; the default role stays unfiltered
    judge_labels = coh = None
    fit_ok = np.ones(len(role), dtype=bool)
    if judge_path is not None:
        judge_labels, coh = load_judge(judge_path, len(role))
        if not a.no_judge_filter:
            fit_ok = (judge_labels == 'fully') | (role == a.default_role)
    Pf = P & fit_ok                           # persona rows used for fitting

    out = dict(meta=dict(model=man['model'], d=man['d'], n_layers=n_layers,
                         counts={g: int((role == g).sum()) for g in GROUPS},
                         steer_hidden=STEER_HIDDEN, read_hidden=READ_HIDDEN, map_layers=MAP_LAYERS))

    # ------------------------------------------------------------------ layer sweep
    sweep = {}
    for key, name in [('p1_resp_avg', 'resp'), ('p1_prompt_last', 'last')]:
        rows = []
        for l in range(z[key].shape[1]):
            X = z[key][:, l, :].astype(np.float64)
            raw_e = (X[main_m] ** 2).mean(0)
            top3 = float(np.sort(raw_e)[::-1][:3].sum() / raw_e.sum())
            if X[main_m].std(0).max() < 1e-8:
                rows.append(dict(layer=int(saved[l]), md_auc=0.5, lr_auc=0.5, role_var=0.0, q_var=0.0, top3=top3)); continue
            Xs, _, _ = standardize(X, main_m)
            v = Xs[Pf & tr].mean(0) - Xs[A & tr].mean(0)
            md = auc(Xs[P & te] @ v, Xs[A & te] @ v)
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(
                np.r_[Xs[Pf & tr], Xs[A & tr]], np.r_[np.ones((Pf & tr).sum()), np.zeros((A & tr).sum())])
            lr = auc(clf.decision_function(Xs[P & te]), clf.decision_function(Xs[A & te]))
            Y = Xs[main_m] - Xs[main_m].mean(0); tot = (Y ** 2).sum()
            rm = np.array([Y[role[main_m] == r].mean(0) for r in role[main_m]])
            qm = np.array([Y[q[main_m] == k].mean(0) for k in q[main_m]])
            rows.append(dict(layer=int(saved[l]), md_auc=round(md, 4), lr_auc=round(lr, 4),
                             role_var=round(float((rm ** 2).sum() / tot), 4),
                             q_var=round(float((qm ** 2).sum() / tot), 4), top3=round(top3, 4)))
        sweep[name] = rows
    out['sweep'] = sweep

    # ------------------------------------------------------------------ 2-D maps
    def persona_axis(Xs):
        v = Xs[Pf].mean(0) - Xs[A].mean(0)
        u = v / np.linalg.norm(v)
        s = Xs @ u
        a0, p0 = s[A].mean(), s[Pf].mean()
        return u, (s - a0) / (p0 - a0)

    maps = {}
    for key, name in [('p1_resp_avg', 'resp'), ('p1_prompt_last', 'last')]:
        maps[name] = {}
        for l in MAP_LAYERS:
            X = z[key][:, col(l), :].astype(np.float64)
            if X[main_m].std(0).max() < 1e-8:  # e.g. identical answers: nothing to project
                maps[name][str(l)] = None
                continue
            Xs, _, _ = standardize(X, main_m)
            Xc = Xs - Xs[main_m].mean(0)
            # PCA view (what the data varies along most)
            _, S, Vt = np.linalg.svd(Xc[main_m], full_matrices=False)
            pca = Xc @ Vt[:2].T
            ev = (S ** 2) / (S ** 2).sum()
            # persona-plane view: x = persona score, y = top direction orthogonal to the persona axis
            u, score = persona_axis(Xs)
            R = Xc - np.outer(Xc @ u, u)
            _, _, Vt2 = np.linalg.svd(R[P | A], full_matrices=False)
            y = R @ Vt2[0]
            ys = y / (np.std(y[P | A]) + 1e-9)
            maps[name][str(l)] = dict(
                pca=np.round(pca / (np.std(pca[main_m], axis=0) + 1e-9), 3).tolist(),
                pca_ev=[round(float(ev[0]), 4), round(float(ev[1]), 4)],
                plane=np.round(np.c_[score, ys], 3).tolist())
    out['maps'] = maps

    # ------------------------------------------------------------------ points (shared metadata)
    pts = []
    for i, t in enumerate(p1t):
        p = dict(r=t['role'], t=t['task'], q=t['q_idx'], s=t['sp_idx'], lex=round(t['pirate_score'], 3),
                 brk=int(t['ai_break']), u=t['user'][:160], a=t['answer'][:260])
        if judge_labels is not None:
            p['j'] = judge_labels[i]
            p['coh'] = None if np.isnan(coh[i]) else round(float(coh[i]), 1)
        pts.append(p)
    out['points'] = pts

    # ------------------------------------------------------------------ hard negatives & groups on the persona axis
    groups = {}
    for key, name in [('p1_resp_avg', 'resp'), ('p1_prompt_last', 'last')]:
        X = z[key][:, col(STEER_HIDDEN), :].astype(np.float64)
        Xs, _, _ = standardize(X, main_m)
        _, score = persona_axis(Xs)
        groups[name] = {g: dict(mean=round(float(score[role == g].mean()), 3),
                                q10=round(float(np.percentile(score[role == g], 10)), 3),
                                q90=round(float(np.percentile(score[role == g], 90)), 3),
                                vals=np.round(score[role == g], 3).tolist(),
                                lex=round(float(z['p1_pirate_score'][role == g].mean()), 4))
                        for g in GROUPS}
    out['groups'] = groups

    # within-persona variation: share of the persona cloud's variance explained by task type
    X = z['p1_resp_avg'][:, col(STEER_HIDDEN), :].astype(np.float64)
    Xs, _, _ = standardize(X, main_m)
    within = {}
    for g in [a.persona, a.default_role]:
        Y = Xs[role == g]; Y = Y - Y.mean(0); tg = task[role == g]
        tm = np.array([Y[tg == k].mean(0) for k in tg]); qg = q[role == g]
        qm = np.array([Y[qg == k].mean(0) for k in qg])
        within[g] = dict(task_frac=round(float((tm ** 2).sum() / (Y ** 2).sum()), 3),
                         question_frac=round(float((qm ** 2).sum() / (Y ** 2).sum()), 3))
    out['within'] = within

    # ------------------------------------------------------------------ rank deficiency (raw space, steering layer)
    Xraw = z['p1_resp_avg'][:, col(STEER_HIDDEN), :].astype(np.float64)
    Pr = Xraw[Pf]; rng = np.random.default_rng(0)
    cover = []
    for n in [m for m in [2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512] if m < len(Pr) - 1] + [len(Pr) - 1]:
        fr = []
        for rep in range(20):
            idx = rng.permutation(len(Pr)); fit, hold = Pr[idx[:n]], Pr[idx[n:]] if n < len(Pr) else Pr[idx[-1:]]
            mu = fit.mean(0)
            _, S, Vt = np.linalg.svd(fit - mu, full_matrices=False)
            B = Vt[S > S.max() * 1e-8] if n > 1 else Vt[:0]
            H = hold - mu
            inside = ((H @ B.T) ** 2).sum(1) if len(B) else np.zeros(len(H))
            fr.append(float((inside / (H ** 2).sum(1)).mean()))
        cover.append(dict(n=n, inside=round(float(np.mean(fr)), 4)))
    # what each conceptor form keeps of a held-out default activation (centered at the persona mean)
    mu = Pr.mean(0)
    _, S, Vt = np.linalg.svd(Pr - mu, full_matrices=False)
    lam = S ** 2 / len(Pr); aa = float(np.median(lam[lam > lam.max() * 1e-10])); keep = lam / (lam + aa)
    H = Xraw[A] - mu
    coef = H @ Vt.T
    e_tot = (H ** 2).sum(1)
    e_C = ((coef * keep) ** 2).sum(1)
    e_notC = e_tot - (coef ** 2).sum(1) + ((coef * (1 - keep)) ** 2).sum(1)
    b = 0.3
    e_interp = (((1 - b) * H + b * ((coef * keep) @ Vt)) ** 2).sum(1)
    # raw (uncentered) fit, as in Postmus & Abreu: share of a held-out activation's energy C h keeps
    _, S2, Vt2 = np.linalg.svd(Pr, full_matrices=False)
    lam2 = S2 ** 2 / len(Pr); keep2 = lam2 / (lam2 + float(np.median(lam2)))
    Hr = Xraw[A]; c2 = Hr @ Vt2.T
    raw_keep_total = float((((c2 * keep2) ** 2).sum(1) / (Hr ** 2).sum(1)).mean())
    Hc = Hr - Hr.mean(0)
    cc = Hc @ Vt2.T
    raw_keep_content = float((((cc * keep2) ** 2).sum(1) / (Hc ** 2).sum(1)).mean())
    out['rank'] = dict(n=int(Pf.sum()), d=int(Xraw.shape[1]), rank=int((keep > 1e-6).sum()),
                       keep=np.round(np.sort(keep)[::-1], 4).tolist(), alpha_inv2=aa,
                       cover=cover,
                       kept_C=round(float((e_C / e_tot).mean()), 4),
                       kept_notC=round(float((e_notC / e_tot).mean()), 4),
                       kept_interp=round(float((e_interp / e_tot).mean()), 4),
                       raw_keep_total=round(raw_keep_total, 4), raw_keep_content=round(raw_keep_content, 4))

    # ------------------------------------------------------------------ conceptor logic on real data (reduced space)
    # Two fits, k=12 PCA dims, aperture alpha^-2 = median eigenvalue of the pooled correlation:
    #   "raw":  each role's own cloud (relative to the global mean) -> dominated by shared question content
    #   "diff": per-question differences from the default (role(q) - default(q)) -> content cancels
    def sim(C1, C2):
        return float(np.trace(C1 @ C2) / (np.linalg.norm(C1) * np.linalg.norm(C2)))
    k = 12
    qs = sorted(set(q.tolist()))
    if {'pirate', 'doctor', 'pirate_doctor', a.default_role} <= set(present):
        logic = {}
        others = [a.default_role, 'bard', 'ghost', 'robot', 'teacher', 'consultant']
        for L in LOGIC_LAYERS:
            X = z['p1_resp_avg'][:, col(L), :].astype(np.float64)
            Xs, _, _ = standardize(X, main_m)
            Xc = Xs - Xs[main_m].mean(0)
            res = {}
            for fit in ['raw', 'diff']:
                if fit == 'raw':
                    _, _, Vt = np.linalg.svd(Xc[main_m], full_matrices=False)
                    Z = {r: Xc[(role == r) & fit_ok] @ Vt[:k].T for r in MAIN}
                else:
                    base = np.array([Xs[A & (q == kk)].mean(0) for kk in qs])
                    D = {r: np.array([Xs[(role == r) & (q == kk)].mean(0) for kk in qs]) - base
                         for r in MAIN if r != a.default_role}
                    allD = np.vstack(list(D.values()))
                    _, _, Vt = np.linalg.svd(allD - allD.mean(0), full_matrices=False)
                    Z = {r: D[r] @ Vt[:k].T for r in D}
                pool = np.vstack(list(Z.values())); a_ = float(np.median(np.linalg.eigvalsh(pool.T @ pool / len(pool))))
                C = {}
                for r, Zg in Z.items():
                    R = Zg.T @ Zg / len(Zg); C[r] = R @ np.linalg.inv(R + a_ * np.eye(k))
                I = np.eye(k)
                AND = np.linalg.inv(np.linalg.inv(C['pirate']) + np.linalg.inv(C['doctor']) - I)
                OR = I - np.linalg.inv(np.linalg.inv(I - C['pirate']) + np.linalg.inv(I - C['doctor']) - I)
                t = C['pirate_doctor']
                sims = {'pirate AND doctor': sim(AND, t), 'pirate OR doctor': sim(OR, t),
                        'pirate': sim(C['pirate'], t), 'doctor': sim(C['doctor'], t)}
                for r in others:
                    if r in C:
                        sims[r] = sim(C[r], t)
                res[fit] = {n: round(v, 3) for n, v in sorted(sims.items(), key=lambda x: -x[1])}
            logic[str(L)] = res
        out['logic'] = dict(k=k, layers=LOGIC_LAYERS, sims=logic)

    # ------------------------------------------------------------------ judge-filter stand-in (keyword) and real judge filter
    lex = z['p1_pirate_score']
    keepP = P & (lex > 0)
    v_all = Xs[P].mean(0) - Xs[A].mean(0); v_f = Xs[keepP].mean(0) - Xs[A].mean(0)
    out['filter'] = dict(n_pirate=int(P.sum()), n_pass=int(keepP.sum()),
                         cos=round(float(v_all @ v_f / np.linalg.norm(v_all) / np.linalg.norm(v_f)), 3))
    if judge_labels is not None:
        jf = {}
        for g in GROUPS:
            m = role == g
            labs = judge_labels[m]
            jf[g] = dict(n=int(m.sum()), fully=int((labs == 'fully').sum()), somewhat=int((labs == 'somewhat').sum()),
                         not_=int((labs == 'not').sum()), unparsed=int(sum(x is None for x in labs)),
                         coherence=None if np.all(np.isnan(coh[m])) else round(float(np.nanmean(coh[m])), 1))
        v_j = Xs[Pf].mean(0) - Xs[A].mean(0) if Pf.any() else v_all
        out['judge'] = dict(source=str(judge_path), filtered=not a.no_judge_filter, n_fit=int(Pf.sum()),
                            by_role=jf, cos_filtered_vs_all=round(float(v_all @ v_j / np.linalg.norm(v_all) / np.linalg.norm(v_j)), 3))

    # ------------------------------------------------------------------ low-rank shrunk conceptors (raw space, steering layer)
    # raw fit: persona cloud on train questions; held-out keep for persona (test questions) vs default rows.
    # diff fit: per-question differences persona(q) - default(q) on train questions; held-out keep for
    #   the persona's test-question differences vs other roles' differences (content cancels in both).
    if Pf.sum() >= 4:
        b = 0.3
        lowrank = dict(beta=b, ks=[], raw={}, diff={})
        Ptr, Pte, Ate = Xraw[Pf & tr], Xraw[P & te], Xraw[A & te]
        def diffs(r, qset, mask=None):
            mask = (role == r) if mask is None else mask
            rows = [Xraw[mask & (q == kk)].mean(0) - Xraw[A & (q == kk)].mean(0)
                    for kk in qset if (mask & (q == kk)).any() and (A & (q == kk)).any()]
            return np.array(rows)
        qtr, qte = [kk for kk in qs if kk % 2 == 0], [kk for kk in qs if kk % 2 == 1]
        Dtr, Dte = diffs(a.persona, qtr, Pf), diffs(a.persona, qte)
        other_roles = [r for r in MAIN if r not in (a.persona, a.default_role)] + [r for r in GROUPS if r.endswith('_narrated')]
        for kk in LOWRANK_KS:
            if kk > len(Ptr):
                continue
            lowrank['ks'].append(kk)
            C = lowrank_conceptor(Ptr, kk)
            lowrank['raw'][str(kk)] = dict(
                delta=round(C['delta'], 4), aperture=C['aperture'], null_keep=round(C['null_keep'], 4),
                keep=np.round(C['keep'], 4).tolist(),
                kept_persona_heldout=round(kept_share(C, Pte), 4), kept_default=round(kept_share(C, Ate), 4),
                interp_persona_heldout=round(kept_share(C, Pte, b), 4), interp_default=round(kept_share(C, Ate, b), 4))
            if len(Dtr) >= 2 and kk <= len(Dtr):
                Cd = lowrank_conceptor(Dtr, kk)
                Cd['mu'] = np.zeros_like(Cd['mu'])  # differences are judged around 0 (no persona offset)
                row = dict(delta=round(Cd['delta'], 4), keep=np.round(Cd['keep'], 4).tolist(),
                           kept_persona_heldout=round(kept_share(Cd, Dte), 4) if len(Dte) else None)
                for r in other_roles:
                    Dr = diffs(r, qte)
                    if len(Dr):
                        row[f'kept_{r}'] = round(kept_share(Cd, Dr), 4)
                lowrank['diff'][str(kk)] = row
        out['lowrank'] = lowrank

    # ------------------------------------------------------------------ multi-turn (phase 2)
    if 'p2_resp_avg' in z:
        mt = []
        for read in [STEER_HIDDEN, READ_HIDDEN]:
            X1 = z['p1_resp_avg'][:, col(read), :].astype(np.float64)
            mu1, sd1 = X1[main_m].mean(0), X1[main_m].std(0) + 1e-6
            X1s = (X1 - mu1) / sd1
            v = X1s[Pf].mean(0) - X1s[A].mean(0); u = v / np.linalg.norm(v)
            a0, p0 = (X1s[A] @ u).mean(), (X1s[Pf] @ u).mean()
            X2s = (z['p2_resp_avg'][:, col(read), :].astype(np.float64) - mu1) / sd1
            mt.append((X2s @ u - a0) / (p0 - a0))
        rows = []
        for i, t in enumerate(p2t):
            rows.append(dict(c=t['cond'], v=t['conv'], k=t['turn'], s13=round(float(mt[0][i]), 3),
                             s18=round(float(mt[1][i]), 3), lex=round(t['pirate_score'], 3), brk=int(t['ai_break']),
                             deg=int(degenerate(t['answer'])), u=t['user'][:140], a=t['answer'][:300]))
        out['multiturn'] = dict(rows=rows, steer=man.get('phase2', {}))

    json.dump(out, open(out_path, 'w'), separators=(',', ':'))
    print('wrote', out_path)


if __name__ == '__main__':
    main()
