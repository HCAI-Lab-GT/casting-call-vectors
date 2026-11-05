#!/usr/bin/env python3
"""
analyze_acts.py — turn persona_acts.py output into the numbers and 2-D projections the
conceptor explainer draws. Writes one JSON file (derived.json).

  python analyze_acts.py --acts acts_qwen05/acts.npz --texts acts_qwen05/texts.jsonl \
                         --manifest acts_qwen05/manifest.json --out derived.json

Conventions
  * hidden index i = output of decoder layer i-1 (0 = embeddings), as saved by persona_acts.py
  * "standardized" = each dimension z-scored over the 9 main role conditions; this removes the
    few massive-activation dimensions that otherwise dominate distances and PCA
  * persona score = projection onto the pirate-minus-assistant mean difference, rescaled so the
    assistant mean is 0 and the pirate mean is 1 (computed on the same layer and pooling)
"""
import argparse, json, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MAIN = ['assistant', 'pirate', 'doctor', 'teacher', 'bard', 'robot', 'consultant', 'ghost', 'pirate_doctor']
GROUPS = MAIN + ['pirate_narrated']
MAP_LAYERS = [2, 6, 10, 13, 16, 20, 24]
STEER_HIDDEN = 13          # output of decoder layer 12 (the steering layer used in phase 2)
READ_HIDDEN = 18           # downstream layer used to read phase-2 persona scores


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acts', required=True)
    ap.add_argument('--texts', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', default='derived.json')
    a = ap.parse_args()

    z = np.load(a.acts)
    man = json.load(open(a.manifest))
    T = [json.loads(l) for l in open(a.texts)]
    p1t = [t for t in T if t.get('phase') != 2]
    p2t = [t for t in T if t.get('phase') == 2]
    role, q, task = z['p1_role'], z['p1_q_idx'], z['p1_task']
    main_m = np.isin(role, MAIN)
    P, A = role == 'pirate', role == 'assistant'
    tr, te = q % 2 == 0, q % 2 == 1          # split by question: train on even, test on odd
    out = dict(meta=dict(model=man['model'], d=man['d'], n_layers=man['n_layers'],
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
                rows.append(dict(layer=l, md_auc=0.5, lr_auc=0.5, role_var=0.0, q_var=0.0, top3=top3)); continue
            Xs, _, _ = standardize(X, main_m)
            v = Xs[P & tr].mean(0) - Xs[A & tr].mean(0)
            md = auc(Xs[P & te] @ v, Xs[A & te] @ v)
            clf = LogisticRegression(C=0.1, max_iter=3000).fit(
                np.r_[Xs[P & tr], Xs[A & tr]], np.r_[np.ones((P & tr).sum()), np.zeros((A & tr).sum())])
            lr = auc(clf.decision_function(Xs[P & te]), clf.decision_function(Xs[A & te]))
            Y = Xs[main_m] - Xs[main_m].mean(0); tot = (Y ** 2).sum()
            rm = np.array([Y[role[main_m] == r].mean(0) for r in role[main_m]])
            qm = np.array([Y[q[main_m] == k].mean(0) for k in q[main_m]])
            rows.append(dict(layer=l, md_auc=round(md, 4), lr_auc=round(lr, 4),
                             role_var=round(float((rm ** 2).sum() / tot), 4),
                             q_var=round(float((qm ** 2).sum() / tot), 4), top3=round(top3, 4)))
        sweep[name] = rows
    out['sweep'] = sweep

    # ------------------------------------------------------------------ 2-D maps
    def persona_axis(Xs):
        v = Xs[P].mean(0) - Xs[A].mean(0)
        u = v / np.linalg.norm(v)
        s = Xs @ u
        a0, p0 = s[A].mean(), s[P].mean()
        return u, (s - a0) / (p0 - a0)

    maps = {}
    for key, name in [('p1_resp_avg', 'resp'), ('p1_prompt_last', 'last')]:
        maps[name] = {}
        for l in MAP_LAYERS:
            X = z[key][:, l, :].astype(np.float64)
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
        pts.append(dict(r=t['role'], t=t['task'], q=t['q_idx'], s=t['sp_idx'], lex=round(t['pirate_score'], 3),
                        brk=int(t['ai_break']), u=t['user'][:160], a=t['answer'][:260]))
    out['points'] = pts

    # ------------------------------------------------------------------ hard negatives & groups on the persona axis
    groups = {}
    for key, name in [('p1_resp_avg', 'resp'), ('p1_prompt_last', 'last')]:
        X = z[key][:, STEER_HIDDEN, :].astype(np.float64)
        Xs, _, _ = standardize(X, main_m)
        _, score = persona_axis(Xs)
        groups[name] = {g: dict(mean=round(float(score[role == g].mean()), 3),
                                q10=round(float(np.percentile(score[role == g], 10)), 3),
                                q90=round(float(np.percentile(score[role == g], 90)), 3),
                                vals=np.round(score[role == g], 3).tolist(),
                                lex=round(float(z['p1_pirate_score'][role == g].mean()), 4))
                        for g in GROUPS}
    out['groups'] = groups

    # within-persona variation: share of the pirate cloud's variance explained by task type
    X = z['p1_resp_avg'][:, STEER_HIDDEN, :].astype(np.float64)
    Xs, _, _ = standardize(X, main_m)
    within = {}
    for g in ['pirate', 'assistant']:
        Y = Xs[role == g]; Y = Y - Y.mean(0); tg = task[role == g]
        tm = np.array([Y[tg == k].mean(0) for k in tg]); qg = q[role == g]
        qm = np.array([Y[qg == k].mean(0) for k in qg])
        within[g] = dict(task_frac=round(float((tm ** 2).sum() / (Y ** 2).sum()), 3),
                         question_frac=round(float((qm ** 2).sum() / (Y ** 2).sum()), 3))
    out['within'] = within

    # ------------------------------------------------------------------ rank deficiency (raw space, steering layer)
    Xraw = z['p1_resp_avg'][:, STEER_HIDDEN, :].astype(np.float64)
    Pr = Xraw[P]; rng = np.random.default_rng(0)
    cover = []
    for n in [2, 4, 8, 16, 24, 32, 48, 64, 71]:
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
    # what each conceptor form keeps of a held-out assistant activation (centered at the pirate mean)
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
    out['rank'] = dict(n=int(P.sum()), d=int(Xraw.shape[1]), rank=int((keep > 1e-6).sum()),
                       keep=np.round(np.sort(keep)[::-1], 4).tolist(), alpha_inv2=aa,
                       cover=cover,
                       kept_C=round(float((e_C / e_tot).mean()), 4),
                       kept_notC=round(float((e_notC / e_tot).mean()), 4),
                       kept_interp=round(float((e_interp / e_tot).mean()), 4),
                       raw_keep_total=round(raw_keep_total, 4), raw_keep_content=round(raw_keep_content, 4))

    # ------------------------------------------------------------------ conceptor logic on real data (reduced space)
    # Two fits, k=12 PCA dims, aperture alpha^-2 = median eigenvalue of the pooled correlation:
    #   "raw":  each role's own cloud (relative to the global mean) -> dominated by shared question content
    #   "diff": per-question differences from the assistant (role(q) - assistant(q)) -> content cancels
    def sim(C1, C2):
        return float(np.trace(C1 @ C2) / (np.linalg.norm(C1) * np.linalg.norm(C2)))
    logic = {}
    k = 12
    others = ['assistant', 'bard', 'ghost', 'robot', 'teacher', 'consultant']
    qs = sorted(set(q.tolist()))
    for L in [13, 18, 22]:
        X = z['p1_resp_avg'][:, L, :].astype(np.float64)
        Xs, _, _ = standardize(X, main_m)
        Xc = Xs - Xs[main_m].mean(0)
        res = {}
        for fit in ['raw', 'diff']:
            if fit == 'raw':
                _, _, Vt = np.linalg.svd(Xc[main_m], full_matrices=False)
                Z = {r: Xc[role == r] @ Vt[:k].T for r in MAIN}
            else:
                base = np.array([Xs[A & (q == kk)].mean(0) for kk in qs])
                D = {r: np.array([Xs[(role == r) & (q == kk)].mean(0) for kk in qs]) - base for r in MAIN if r != 'assistant'}
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
    out['logic'] = dict(k=k, layers=[13, 18, 22], sims=logic)

    # ------------------------------------------------------------------ judge-filter stand-in
    lex = z['p1_pirate_score']
    keepP = P & (lex > 0)
    v_all = Xs[P].mean(0) - Xs[A].mean(0); v_f = Xs[keepP].mean(0) - Xs[A].mean(0)
    out['filter'] = dict(n_pirate=int(P.sum()), n_pass=int(keepP.sum()),
                         cos=round(float(v_all @ v_f / np.linalg.norm(v_all) / np.linalg.norm(v_f)), 3))

    # ------------------------------------------------------------------ multi-turn (phase 2)
    mt = []
    for read in [STEER_HIDDEN, READ_HIDDEN]:
        X1 = z['p1_resp_avg'][:, read, :].astype(np.float64)
        mu1, sd1 = X1[main_m].mean(0), X1[main_m].std(0) + 1e-6
        X1s = (X1 - mu1) / sd1
        v = X1s[P].mean(0) - X1s[A].mean(0); u = v / np.linalg.norm(v)
        a0, p0 = (X1s[A] @ u).mean(), (X1s[P] @ u).mean()
        X2s = (z['p2_resp_avg'][:, read, :].astype(np.float64) - mu1) / sd1
        mt.append((X2s @ u - a0) / (p0 - a0))
    rows = []
    for i, t in enumerate(p2t):
        rows.append(dict(c=t['cond'], v=t['conv'], k=t['turn'], s13=round(float(mt[0][i]), 3),
                         s18=round(float(mt[1][i]), 3), lex=round(t['pirate_score'], 3), brk=int(t['ai_break']),
                         deg=int(degenerate(t['answer'])), u=t['user'][:140], a=t['answer'][:300]))
    out['multiturn'] = dict(rows=rows, steer=man.get('phase2', {}))

    json.dump(out, open(a.out, 'w'), separators=(',', ':'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
