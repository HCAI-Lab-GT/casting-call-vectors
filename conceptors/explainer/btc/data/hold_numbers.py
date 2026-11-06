"""Per-persona 20-turn hold numbers for the explainer, from the parent session's pulled judge folders (read-only).

Same reductions as the parent's results_followups.py: mean over the judges present; prompt context; ordinary =
ordinary-turn fully-in-character rate; push = mean in character at turns 5 and 13 (index 4, 12); coherence = ordinary
coherence; ai = AI admissions at turns 8 and 16 (index 7, 15).
usage: python hold_numbers.py V2_DIR OUT_JSON
"""
import json
import statistics as st
import sys
from pathlib import Path

J = ("qwen", "gemma", "mistral", "llama")
PUSH, AIQ = (4, 12), (7, 15)
SMOL = ["pirate", "ghost", "bard", "consultant", "genie", "alien"]
OLMO = ["pirate", "genie", "consultant"]


def rows(v2: Path, run: str) -> dict:
    out = {}
    for j in J:
        f = v2 / f"{run}.judge-{j}" / "steer_derived.json"
        if f.exists():
            for h in json.loads(f.read_text())["hold"]:
                if h["context"] == "prompt":
                    out.setdefault(h["method"], []).append(h)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def turn(hs, t, key="fully_rate"):
    return mean(h["per_turn"][t].get(key) for h in hs if t < len(h["per_turn"]))


def cell(hs):
    if not hs:
        return None
    return dict(ord=mean(h["ordinary"]["fully_rate"] for h in hs), push=mean(turn(hs, t) for t in PUSH),
                coh=mean(h["ordinary"]["coherence"] for h in hs), ai=mean(turn(hs, t, "identity_ai_rate") for t in AIQ),
                late=mean(turn(hs, t) for t in range(16, 20)), judges=len(hs), strength=hs[0].get("strength"))


def main() -> None:
    v2, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = {"smol": {}, "olmo": {}}
    for model, ps, pre in (("smol", SMOL, "steer_hold"), ("olmo", OLMO, "olmo_hold")):
        for p in ps:
            runs = [f"{pre}_v2_{p}_prompt", f"{pre}_v3_{p}", f"{pre}_aperture_{p}",
                    f"{pre}_screen_{p}", f"{pre}_screen_region_{p}"]
            merged = {}
            for r in runs:
                for k, v in rows(v2, r).items():
                    tag = "region_fixed" if r.endswith(f"screen_region_{p}") and k == "region" else k
                    merged.setdefault(tag, v)
            res[model][p] = {k: cell(v) for k, v in merged.items()}
            nop = rows(v2, f"{pre}_nopush_{p}")
            res[model][p]["nopush"] = {k: cell(v) for k, v in nop.items()}
    out.write_text(json.dumps(res, indent=1))
    for model in res:
        for p, d in res[model].items():
            ks = {k: (v["judges"] if v and "judges" in v else len(v)) for k, v in d.items() if v}
            print(model, p, ks)


if __name__ == "__main__":
    main()
