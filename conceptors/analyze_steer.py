#!/usr/bin/env python3
"""
analyze_steer.py — summarize a steer_eval.py run (judged by judge.py) into steer_derived.json.

  sweep     per (method, strength): judge fully/somewhat/not rates, mean coherence, read-out score quartiles,
            AI-break rate, pirate-word rate; plus 'selected': per method the smallest strength whose fully rate
            reaches --target with mean coherence >= --min_coherence (matched), else its strongest strength that
            keeps coherence >= --min_coherence (unmatched), else none. Without judge scores, the read-out stands in.
  hold      per (context, method): per-turn curves and T1/T5/T10 with drift T10 - T1 (GCAD protocol), split by
            probe vs ordinary turns.

  python analyze_steer.py --run RUNS/steer_sweep
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run: Path):
    rows = [json.loads(l) for l in (run / "texts.jsonl").read_text().splitlines() if l.strip()]
    judge = {}
    if (run / "judge.jsonl").exists():
        judge = {j["idx"]: j for j in map(json.loads, (run / "judge.jsonl").read_text().splitlines()) if j}
    for i, r in enumerate(rows):
        j = judge.get(i, {})
        r["label"], r["coherence"], r["voice"] = j.get("roleplay_label"), j.get("coherence"), j.get("voice_label")
    return rows, bool(judge), json.loads((run / "manifest.json").read_text())


def stats(rs: list[dict]) -> dict:
    labels = [r["label"] for r in rs]
    judged = [x for x in labels if x is not None]
    coh = [r["coherence"] for r in rs if r["coherence"] is not None]
    ro = np.array([r["readout"] for r in rs], dtype=float)
    out = dict(n=len(rs), n_judged=len(judged),
               fully_rate=round(judged.count("fully") / len(judged), 3) if judged else None,
               somewhat_rate=round(judged.count("somewhat") / len(judged), 3) if judged else None,
               coherence=round(float(np.mean(coh)), 1) if coh else None,
               readout=round(float(ro.mean()), 3), readout_q25=round(float(np.percentile(ro, 25)), 3),
               readout_q75=round(float(np.percentile(ro, 75)), 3),
               ai_break_rate=round(float(np.mean([r["ai_break"] for r in rs])), 3),
               lex=round(float(np.mean([r["pirate_score"] for r in rs])), 4))
    voices = [r.get("voice") for r in rs if r.get("voice") is not None]
    if voices:  # additive point-of-view rubric: speaks as / mixed / about the role
        out["voice_as_rate"] = round(voices.count("as") / len(voices), 3)
        out["voice_about_rate"] = round(voices.count("about") / len(voices), 3)
    return out


def select(table: list[dict], judged: bool, target: float, min_coh: float) -> dict:
    """Per method, the strength the hold test runs it at.

    Matched: the smallest strength with fully >= target and coherence >= min_coh, so matched methods set the persona
    up equally well. A method that never gets there runs at its strongest strength that keeps coherence >= min_coh
    (matched False): the most it can do without breaking the model. The hold test's prompt context can still show
    whether that keeps a prompted persona from drifting, which is the claim conceptors are meant to win.
    """
    by_method = defaultdict(list)
    for t in table:
        by_method[t["method"]].append(t)
    selected = {}
    for method, ts in by_method.items():
        if method == "none":
            continue
        ts = sorted(ts, key=lambda t: t["strength"])
        if judged:
            coherent = [t for t in ts if (t["coherence"] or 0) >= min_coh]
            ok = [t for t in coherent if (t["fully_rate"] or 0) >= target]
            if ok:
                pick, matched, why = ok[0], True, f"smallest strength with fully >= {target} and coherence >= {min_coh}"
            elif coherent:
                pick, matched, why = coherent[-1], False, f"never fully >= {target}; strongest with coherence >= {min_coh}"
            else:
                pick, matched, why = None, False, f"no strength kept coherence >= {min_coh}"
        else:
            ok = [t for t in ts if t["readout"] >= target]
            pick, matched, why = (ok[0], True, f"no judge: smallest strength with readout >= {target}") if ok else \
                (None, False, "none reached")
        selected[method] = None if pick is None else dict(
            strength=pick["strength"], matched=matched, fully_rate=pick["fully_rate"], coherence=pick["coherence"],
            readout=pick["readout"], rule=why)
    return selected


def examples(rs: list[dict], k: int = 3) -> list[dict]:
    return [dict(q=r["user"][:200], a=r["answer"][:500], label=r["label"], coh=r["coherence"], voice=r.get("voice"),
                 readout=round(r["readout"], 2)) for r in rs[:k]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--min_coherence", type=float, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    rows, judged, man = load(run)
    out = dict(meta=dict(stage=man["stage"], model=man["model"], fits=man["fits"], judged=judged,
                         n_rows=len(rows), n_think=man.get("n_answers_with_think_tags")))

    if man["stage"] == "sweep":
        groups = defaultdict(list)
        for r in rows:
            groups[(r["method"], r["strength"])].append(r)
        table = [dict(method=m, strength=s, **stats(rs), examples=examples(rs)) for (m, s), rs in groups.items()]
        out["sweep"] = table
        out["selected"] = select(table, judged, a.target, a.min_coherence)
    else:
        groups = defaultdict(list)
        for r in rows:
            groups[(r["context"], r["method"])].append(r)
        hold = []
        for (ctx, m), rs in groups.items():
            turns = sorted({r["turn"] for r in rs})
            per_turn = [dict(turn=t, probe=bool(rs_t[0].get("probe")), **stats(rs_t))
                        for t in turns for rs_t in [[r for r in rs if r["turn"] == t]]]
            pick = lambda t, key: next((p[key] for p in per_turn if p["turn"] == t), None)  # noqa: E731
            last = turns[-1]
            summary = {}
            for key in ("fully_rate", "coherence", "readout"):
                t1, t5, t10 = pick(0, key), pick(min(4, last), key), pick(last, key)
                summary[key] = dict(T1=t1, T5=t5, T10=t10,
                                    drift=None if t1 is None or t10 is None else round(t10 - t1, 3))
            hold.append(dict(context=ctx, method=m, strength=rs[0]["strength"], per_turn=per_turn, summary=summary,
                             ordinary=stats([r for r in rs if not r.get("probe")]),
                             probes=stats([r for r in rs if r.get("probe")]) if any(r.get("probe") for r in rs) else None,
                             examples=examples([r for r in rs if r["conv"] == 0], k=len(turns))))
        out["hold"] = hold

    path = Path(a.out) if a.out else run / "steer_derived.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    if "selected" in out:
        for m, s in out["selected"].items():
            print(f"  selected {m}: {s}")


if __name__ == "__main__":
    main()
