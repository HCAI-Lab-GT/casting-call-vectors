"""Check the offline screen against the 20-turn holds.

For every method a hold ran, rebuild the exact operator that ran (the hold's own fits.npz and strength), apply it to the
replayed unsteered hold tokens, and score it with persona_screen's readouts. Then compare with what the four judges saw
in that method's hold (prompt context): in-character rate on ordinary turns, coherence, holding at the "drop the act"
pushes, and AI admissions at the "are you an AI?" probes. If the screen's numbers track the holds across methods and
personas, its ranking of operators that never ran (fisher_cap, region, gated_ot, ...) is worth acting on.

cap_aa is skipped: it acts on six later layers, which a single-layer replay cannot reproduce.

  python conceptors/validate_screen.py --replay DATA/replay_smollm3_v2 --data DATA \\
      --holds steer_hold_v2_{p}_prompt steer_hold_v3_{p} --out results/validate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import persona_screen as ps
from scipy.stats import spearmanr

JUDGES = ps.JUDGES
CAP_HOLD_PCT = 25
SKIP = {"cap_aa"}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replay", required=True)
    ap.add_argument("--data", required=True, help="folder with hold runs (fits.npz, manifest.json, texts.jsonl) and "
                                                   "their .judge-<name> folders")
    ap.add_argument("--holds", nargs="+", default=["steer_hold_v2_{p}_prompt", "steer_hold_v3_{p}"],
                    help="hold run name patterns, {p} = persona")
    ap.add_argument("--personas", default=",".join(ps.PERSONAS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=60000)
    ap.add_argument("--hold_prefix", default="steer_hold_v2_")
    return ap.parse_args(argv)


def deployed_operator(fits, method, strength):
    """f(H) -> (H', edited) reproducing steering.specs(fits, method, strength) at the steering module."""
    v, vhat = fits["v"], fits["vhat"]
    mu, C_t, C_tt, C_d = fits.get("tok_mu"), fits.get("tok_C"), fits.get("tok_C_tight"), fits.get("diff_C")
    proj = fits["proj_persona"]
    ones = lambda H: np.ones(len(H), bool)

    def cap(H, q):
        tau = float(np.percentile(proj, q))
        g = tau - H @ vhat
        on = g > 0
        return H + np.outer(np.where(on, g, 0), vhat), on

    def concept(H, C, b):
        return (1 - b) * H + b * (mu + (H - mu) @ C.T)
    if method == "none":
        return lambda H: (H, np.zeros(len(H), bool))
    if method == "add":
        return lambda H: (H + strength * v, ones(H))
    if method == "cap":
        return lambda H: cap(H, strength)
    if method == "conceptor_tok":
        return lambda H: (concept(H, C_t, strength), ones(H))
    if method == "meanshift_tok":
        return lambda H: ((1 - strength) * H + strength * mu, ones(H))
    if method == "ctok_tight":
        return lambda H: (concept(H, C_tt, strength), ones(H))
    if method == "diffpull":
        return lambda H: (H + strength * (mu - H) @ C_d.T, ones(H))
    if method == "add_ctok":
        return lambda H: (concept(H + strength * v, C_t, 1.0), ones(H))
    if method == "add_cap":
        return lambda H: (cap(H + strength * v, CAP_HOLD_PCT)[0], ones(H))
    if method == "ctok_cap":
        return lambda H: (cap(concept(H, C_t, 1.0), strength)[0], ones(H))
    raise ValueError(method)


def hold_outcomes(run: Path, judge_root: Path) -> dict:
    """method -> judged outcomes on the prompt-context turns of one hold run (mean over the four judges)."""
    rows = [json.loads(ln) for ln in (run / "texts.jsonl").read_text().splitlines() if ln.strip()]
    man = json.loads((run / "manifest.json").read_text())
    probes = {int(k): v for k, v in man.get("probes", {}).items()}
    push = {t for t, s in probes.items() if "drop the act" in s}
    ask = {t for t, s in probes.items() if "AI" in s and t not in push}
    per = {}
    for j in JUDGES:
        f = judge_root / f"{run.name}.judge-{j}" / "judge.jsonl"
        if not f.exists():
            continue
        for ln in f.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            t = rows[r["idx"]]
            if t.get("context") != "prompt":
                continue
            m = per.setdefault(t["method"], {k: [] for k in ("fully", "coh", "push", "ai")})
            idl = r.get("identity_r_label") or r.get("identity_label")
            if t["turn"] in ask:
                if idl:
                    m["ai"].append(idl == "ai")
            elif t["turn"] in push:
                m["push"].append(r.get("roleplay_label") == "fully")
            else:
                m["fully"].append(r.get("roleplay_label") == "fully")
                if r.get("coherence") is not None:
                    m["coh"].append(float(r["coherence"]))
    out = {}
    for meth, m in per.items():
        out[meth] = {k: (float(np.mean(v)) if v else None) for k, v in m.items()}
        out[meth]["strength"] = float(man.get("strengths", {}).get(meth, 0.0))
    return out


def main(argv=None):
    args = parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    rp = ps.Replay(Path(args.replay))
    table = []
    for p in [x for x in args.personas.split(",") if x]:
        sargs = argparse.Namespace(hold_prefix=args.hold_prefix, max_tokens=args.max_tokens)
        F, T = ps.fit_persona(rp, p, sargs, data)
        probes = {"judge": ps.probe(T["hold_in_fit"], T["hold_out_fit"]),
                  "identity": ps.probe(T["hold_role_fit"], T["hold_ai_fit"])}
        for pat in args.holds:
            run = data / pat.format(p=p)
            if not (run / "fits.npz").exists():
                print(f"[{p}] skip {run.name}: not available", flush=True)
                continue
            fits = dict(np.load(run / "fits.npz"))
            outcomes = hold_outcomes(run, data)
            if not outcomes:
                print(f"[{p}] skip {run.name}: no judge labels yet", flush=True)
                continue
            ops = {}
            for meth, oc in outcomes.items():
                if meth in SKIP or (meth != "none" and meth not in ("add", "cap", "conceptor_tok", "meanshift_tok",
                                                                     "ctok_tight", "diffpull", "add_ctok", "add_cap",
                                                                     "ctok_cap")):
                    continue
                ops[meth] = [(f"deployed {oc['strength']:g}", deployed_operator(fits, meth, oc["strength"]))]
            rows, _ = ps.evaluate(F, T, ops, probes)
            for r in rows:
                oc = outcomes[r["method"]]
                table.append(dict(persona=p, run=run.name, **r, **{f"hold_{k}": v for k, v in oc.items()}))
            print(f"[{p}] {run.name}: {len(rows)} methods scored", flush=True)
    (out / "validation.json").write_text(json.dumps(table, indent=1))

    # does the screen track the holds? rank correlations across every (persona, method) pair, and within persona
    def corr(xk, yk, rows):
        x = [r.get(xk) for r in rows]
        y = [r.get(yk) for r in rows]
        keep = [i for i in range(len(x)) if x[i] is not None and y[i] is not None]
        if len(keep) < 4:
            return None
        rho, pval = spearmanr([x[i] for i in keep], [y[i] for i in keep])
        return {"rho": float(rho), "p": float(pval), "n": len(keep)}
    pairs = [("restore_probe", "hold_fully"), ("restore_axis", "hold_fully"), ("distortion", "hold_coh"),
             ("restore_probe", "hold_push"), ("identity_fix", "hold_ai")]
    summary = {f"{a} vs {b}": corr(a, b, table) for a, b in pairs}
    within = {}
    for p in sorted({r["persona"] for r in table}):
        sub = [r for r in table if r["persona"] == p]
        within[p] = {f"{a} vs {b}": corr(a, b, sub) for a, b in pairs}
    (out / "validation_summary.json").write_text(json.dumps({"pooled": summary, "within_persona": within}, indent=1))
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (xk, yk, xl, yl) in zip(axes, [("restore_probe", "hold_fully", "screen: restore by judge probe",
                                            "hold: in-character rate, ordinary turns"),
                                           ("distortion", "hold_coh", "screen: distortion of in-character tokens",
                                            "hold: coherence (0-100)"),
                                           ("identity_fix", "hold_ai", "screen: identity fix",
                                            "hold: AI admissions at probes")], strict=True):
        for r in table:
            if r.get(xk) is None or r.get(yk) is None:
                continue
            ax.scatter(r[xk], r[yk], s=30, alpha=.85)
            ax.annotate(f"{r['persona'][:3]}:{r['method']}", (r[xk], r[yk]), fontsize=6, alpha=.7)
        s = summary.get(f"{xk} vs {yk}")
        ax.set_title(f"Spearman {s['rho']:.2f} (n={s['n']})" if s else "")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.tight_layout()
    fig.savefig(out / "fig_validation.png", dpi=140)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
