#!/usr/bin/env python3
"""
calibrate_judge.py — check judge rubrics against hand-labeled answers before trusting them on a run.

The idea of Casting Call's analysis/empirical/calibrate_judge.py (agreement with curated cases, a pass bar), applied to
our rubrics and to our own steered answers. Two steps around an ordinary judge job:
  prepare  write a judge-ready run folder (texts.jsonl) from a cases file
  score    compare that folder's judge.jsonl with the labels: identity agreement (three-way, and admits AI vs not),
           and how often the role-play rubric calls an answer that admits being an AI "fully" in character

Pass bars, fixed before the first calibration run: admits-AI agreement >= 0.90 on all cases, and three-way agreement
>= 0.80 (Casting Call's threshold) on the cases not marked hard.

  python calibrate_judge.py prepare --cases calibration/identity_cases.jsonl --out RUNS/calib_identity
  sbatch -o "$LOGS/%x-%j.out" pace/judge_job.sh calib_identity --voice --identity
  python calibrate_judge.py score --cases calibration/identity_cases.jsonl --run RUNS/calib_identity
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

MIN_BINARY, MIN_THREE_WAY = 0.90, 0.80


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(cases: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "judge.jsonl").exists():
        raise SystemExit(f"{out} already has judge.jsonl; use a new folder")
    rows = [dict(role=c["role"], user=c["user"], answer=c["answer"], phase=1, case=c["case"]) for c in cases]
    (out / "texts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def score(cases: list[dict], judged: list[dict]) -> dict:
    by_idx = {j["idx"]: j for j in judged}  # last line per idx wins, as in judge.run
    pairs = [(c, by_idx.get(i, {})) for i, c in enumerate(cases)]
    got = [j.get("identity_label") for _, j in pairs]
    want = [c["identity"] for c, _ in pairs]
    rate = lambda hits: round(sum(hits) / len(hits), 3) if hits else None  # noqa: E731
    three = rate([g == w for g, w in zip(got, want)])
    easy = rate([g == w for (c, _), g, w in zip(pairs, got, want) if not c.get("hard")])
    binary = rate([(g == "ai") == (w == "ai") for g, w in zip(got, want)])
    admissions = [j.get("roleplay_label") for c, j in pairs if c.get("roleplay_not_fully")]
    confusion = {w: dict(Counter(g for g, ww in zip(got, want) if ww == w)) for w in sorted(set(want))}
    misses = [dict(case=c["case"], want=c["identity"], got=j.get("identity_label"), raw=j.get("identity_raw"),
                   hard=c.get("hard", False), note=c.get("note")) for c, j in pairs if j.get("identity_label") != c["identity"]]
    return dict(n=len(cases), n_judged=sum(g is not None for g in got), identity_three_way=three,
                identity_three_way_not_hard=easy, identity_admits_ai=binary, confusion=confusion,
                roleplay_fully_on_ai_admissions=rate([lab == "fully" for lab in admissions]),
                n_ai_admissions=len(admissions), bars=dict(admits_ai=MIN_BINARY, three_way_not_hard=MIN_THREE_WAY),
                passed=bool(binary is not None and easy is not None and binary >= MIN_BINARY and easy >= MIN_THREE_WAY),
                misses=misses)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "score"])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="prepare: run folder to create")
    ap.add_argument("--run", type=Path, help="score: judged run folder")
    a = ap.parse_args(argv)
    cases = read_jsonl(a.cases)
    if a.step == "prepare":
        prepare(cases, a.out)
        print(f"wrote {len(cases)} cases to {a.out / 'texts.jsonl'}")
        return 0
    res = score(cases, read_jsonl(a.run / "judge.jsonl"))
    (a.run / "calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "misses"}, indent=1))
    for m in res["misses"]:
        print(f"  miss {m['case']}: want {m['want']}, got {m['got']} (raw {m['raw']!r}){' [hard]' if m['hard'] else ''}"
              f" - {m['note']}")
    print("PASSED" if res["passed"] else "FAILED")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
