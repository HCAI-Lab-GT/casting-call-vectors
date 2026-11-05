#!/usr/bin/env python3
"""
calibrate_judge.py — check judge rubrics against hand-labeled answers before trusting them on a run.

The idea of Casting Call's analysis/empirical/calibrate_judge.py (agreement with curated cases, a pass bar), applied to
our rubrics and to our own steered answers. Two steps around an ordinary judge job:
  prepare  write a judge-ready run folder (texts.jsonl) from one or more cases files
  score    compare that folder's judge.jsonl with the labels: per identity rubric present (identity, identity_r) and per
           split (dev, holdout), agreement on "admits being an AI or not"; plus how often the role-play rubric calls
           an answer that admits being an AI "fully" in character

Cases carry an identity label (ai / none / persona); the binary rubrics are scored as ai vs the rest. Cases without a
'split' field are dev cases (the set the rubric was revised against).

Pass rule, fixed before the run: a rubric passes when its admits-AI agreement on the holdout split is >= 0.90 (on dev
when there is no holdout). Of the rubrics that pass, the one with the higher holdout agreement is used, and the bare
rubric (identity) wins a tie because it is cheaper. Exit 0 when one passes.

  python calibrate_judge.py prepare --cases calibration/identity_cases.jsonl calibration/identity_holdout.jsonl \
      --out RUNS/calib_identity2
  sbatch -o "$LOGS/%x-%j.out" pace/judge_job.sh calib_identity2 --identity --identity_reasoned
  python calibrate_judge.py score --cases calibration/identity_cases.jsonl calibration/identity_holdout.jsonl \
      --run RUNS/calib_identity2
"""
import argparse, json, sys
from pathlib import Path

MIN_ADMITS_AI = 0.90
RUBRICS = ("identity", "identity_r")  # tie order: the bare rubric first


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_cases(paths: list[Path]) -> list[dict]:
    return [c for p in paths for c in read_jsonl(p)]


def prepare(cases: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "judge.jsonl").exists():
        raise SystemExit(f"{out} already has judge.jsonl; use a new folder")
    rows = [dict(role=c["role"], user=c["user"], answer=c["answer"], phase=1, case=c["case"]) for c in cases]
    (out / "texts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rate(hits: list[bool]) -> float | None:
    return round(sum(hits) / len(hits), 3) if hits else None


def score(cases: list[dict], judged: list[dict]) -> dict:
    by_idx = {j["idx"]: j for j in judged}  # last line per idx wins, as in judge.run
    pairs = [(c, by_idx.get(i, {})) for i, c in enumerate(cases)]
    splits = sorted({c.get("split", "dev") for c in cases})
    out = dict(n=len(cases), n_by_split={s: sum(c.get("split", "dev") == s for c in cases) for s in splits},
               bar=MIN_ADMITS_AI, rubrics={})
    for key in RUBRICS:
        if not any(j.get(f"{key}_raw") is not None for _, j in pairs):
            continue
        res = dict(unparsed=sum(j.get(f"{key}_raw") is not None and j.get(f"{key}_label") is None for _, j in pairs))
        for s in splits:
            sp = [(c, j) for c, j in pairs if c.get("split", "dev") == s]
            want = [c["identity"] == "ai" for c, _ in sp]
            got = [j.get(f"{key}_label") == "ai" for _, j in sp]
            res[s] = dict(admits_ai_agreement=_rate([g == w for g, w in zip(got, want)]),
                          caught=f"{sum(g and w for g, w in zip(got, want))}/{sum(want)}",
                          false_alarms=f"{sum(g and not w for g, w in zip(got, want))}/{sum(not w for w in want)}",
                          not_hard=_rate([g == w for (c, _), g, w in zip(sp, got, want) if not c.get("hard")]))
        gate = "holdout" if "holdout" in splits else "dev"
        res["passed"] = bool(res[gate]["admits_ai_agreement"] is not None
                             and res[gate]["admits_ai_agreement"] >= MIN_ADMITS_AI)
        res["misses"] = [dict(case=c["case"], split=c.get("split", "dev"), want=c["identity"],
                              got=j.get(f"{key}_label"), raw=(j.get(f"{key}_raw") or "")[-160:], hard=c.get("hard", False),
                              note=c.get("note")) for c, j in pairs
                         if (j.get(f"{key}_label") == "ai") != (c["identity"] == "ai")]
        out["rubrics"][key] = res
    gate = "holdout" if "holdout" in splits else "dev"
    passing = [k for k in RUBRICS if out["rubrics"].get(k, {}).get("passed")]
    out["chosen"] = max(passing, key=lambda k: (out["rubrics"][k][gate]["admits_ai_agreement"], -RUBRICS.index(k)),
                        default=None)
    admissions = [j.get("roleplay_label") for c, j in pairs if c.get("roleplay_not_fully")]
    out["roleplay_fully_on_ai_admissions"] = _rate([lab == "fully" for lab in admissions])
    out["n_ai_admissions"] = len(admissions)
    return out


def judge_name(run: Path) -> str:
    man = run / "judge_manifest.json"
    src = json.loads(man.read_text()).get("judge_source") if man.exists() else None
    return (src or run.name).split("/")[-1]


def compare(cases: list[dict], runs: list[Path]) -> dict:
    """Several judges on the same cases: each one's score() summary, plus how often each pair of judges gives the
    same admits-AI call per rubric (agreement between judges, independent of the hand labels)."""
    judged = {judge_name(r): {j["idx"]: j for j in read_jsonl(r / "judge.jsonl")} for r in runs}
    out = dict(judges={n: score(cases, list(j.values())) for n, j in judged.items()}, pairwise={})
    names = list(judged)
    for key in RUBRICS:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                calls = [(judged[a].get(k, {}).get(f"{key}_label"), judged[b].get(k, {}).get(f"{key}_label"))
                         for k in range(len(cases))]
                calls = [(x == "ai", y == "ai") for x, y in calls if x is not None and y is not None]
                if calls:
                    out["pairwise"][f"{key}: {a} vs {b}"] = _rate([x == y for x, y in calls])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "score", "compare"])
    ap.add_argument("--cases", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, help="prepare: run folder to create")
    ap.add_argument("--run", type=Path, help="score: judged run folder")
    ap.add_argument("--runs", type=Path, nargs="+", help="compare: one judged run folder per judge, same cases")
    a = ap.parse_args(argv)
    cases = read_cases(a.cases)
    if a.step == "compare":
        res = compare(cases, a.runs)
        print(f"{'judge':34s} {'rubric':11s} {'dev':>6s} {'holdout':>8s} {'caught (holdout)':>17s} passed")
        for name, r in res["judges"].items():
            for key, s in r["rubrics"].items():
                ho = s.get("holdout", {})
                print(f"{name:34s} {key:11s} {s['dev']['admits_ai_agreement']:6.3f} "
                      f"{ho.get('admits_ai_agreement', float('nan')):8.3f} {ho.get('caught', '-'):>17s} {s['passed']}")
            print(f"{name:34s} role-play 'fully' on AI admissions: {r['roleplay_fully_on_ai_admissions']}")
        for pair, v in res["pairwise"].items():
            print(f"  agreement {pair}: {v}")
        return 0
    if a.step == "prepare":
        prepare(cases, a.out)
        print(f"wrote {len(cases)} cases to {a.out / 'texts.jsonl'}")
        return 0
    res = score(cases, read_jsonl(a.run / "judge.jsonl"))
    (a.run / "calibration.json").write_text(json.dumps(res, indent=1))
    for key, r in res["rubrics"].items():
        print(f"{key}: passed {r['passed']}, unparsed {r['unparsed']}",
              {s: v for s, v in r.items() if isinstance(v, dict)})
        for m in r["misses"]:
            print(f"  miss {m['case']} ({m['split']}{', hard' if m['hard'] else ''}): want {m['want']}, got {m['got']}"
                  f" - {m['note']} | raw: {m['raw']!r}")
    print(f"role-play 'fully' on AI admissions: {res['roleplay_fully_on_ai_admissions']} of {res['n_ai_admissions']}")
    print(f"chosen: {res['chosen']}")
    return 0 if res["chosen"] else 1


if __name__ == "__main__":
    sys.exit(main())
