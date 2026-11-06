"""Compact real-data block for the explainer: aperture per persona, method means, no-push late rates.

usage: python explainer_data.py HOLD_NUMBERS_JSON HOLDS_TABLE_JSON SPECTRUM_JSON OUT_JSON
SmolLM3 means use the five personas off the ceiling (pirate, ghost, bard, genie, alien); OLMo means use pirate and
genie (consultant sits at ceiling with the prompt alone on both models).
"""
import json
import statistics as st
import sys
from pathlib import Path

SMOL5 = ["pirate", "ghost", "bard", "genie", "alien"]
OLMO2 = ["pirate", "genie"]
AP = [(1, "conceptor_tok"), (3, "ctok_k3"), (10, "ctok_tight"), (30, "ctok_k30"), (100, "ctok_k100")]
METHODS = ["none", "add", "cap", "cap_aa", "conceptor_tok", "ctok_tight", "ctok_cap", "add_cap", "meanshift_tok",
           "diffpull", "fisher_cap", "region"]
KEYS = ("ord", "push", "coh", "ai")


def r(x, n=3):
    return None if x is None else round(x, n)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def main() -> None:
    hn = json.loads(Path(sys.argv[1]).read_text())
    table = json.loads(Path(sys.argv[2]).read_text())
    spec = json.loads(Path(sys.argv[3]).read_text())
    # fisher_cap and region (strongest region dose, radius percentile 25) from the overnight screen holds
    screen = {}
    for row in table:
        if row.get("context") == "prompt" and row["method"] in ("fisher_cap", "region"):
            if row["method"] == "region" and row["strength"] != 25.0:
                continue
            screen[(row["persona"], row["method"])] = dict(ord=row["fully"], push=row["push"], coh=row["coh"],
                                                           ai=row["ai"], judges=row["judges"], strength=row["strength"])

    def get(model, p, meth):
        if model == "smol" and meth in ("fisher_cap", "region"):
            return screen.get((p, meth))
        return hn[model].get(p, {}).get(meth)

    aperture = {}
    for model, ps in (("smol", SMOL5), ("olmo", OLMO2)):
        aperture[model] = {p: {str(k): ({kk: r(get(model, p, m)[kk]) for kk in ("ord", "push", "coh")}
                                        | {"judges": get(model, p, m)["judges"]}) if get(model, p, m) else None
                               for k, m in AP} for p in ps}
        for p in ps:
            aperture[model][p]["ref"] = {m: {kk: r(get(model, p, m)[kk]) for kk in ("ord", "push", "coh")}
                                         for m in ("none", "add", "cap", "meanshift_tok") if get(model, p, m)}
    means = {}
    for model, ps in (("smol", SMOL5), ("olmo", OLMO2)):
        means[model] = {}
        for meth in METHODS:
            cells = [get(model, p, meth) for p in ps]
            cells = [c for c in cells if c]
            if not cells:
                continue
            means[model][meth] = {k: r(mean(c.get(k) for c in cells)) for k in KEYS} | {
                "n_personas": len(cells), "min_judges": min(c["judges"] for c in cells)}
    nopush = {}
    for p in SMOL5:
        d = hn["smol"][p]
        nopush[p] = {m: {"std": r(d[m]["late"]) if d.get(m) else None,
                         "nopush": r(d["nopush"][m]["late"]) if d.get("nopush", {}).get(m) else None}
                     for m in ("none", "add", "cap", "ctok_tight")}
    out = dict(aperture=aperture, means=means, nopush=nopush, spectrum=spec)
    Path(sys.argv[4]).write_text(json.dumps(out, separators=(",", ":")))
    for model in means:
        for meth, v in means[model].items():
            print(model, f"{meth:14s}", {k: v[k] for k in KEYS}, "personas", v["n_personas"], "min judges", v["min_judges"])
    for model in aperture:
        for p, d in aperture[model].items():
            print(model, p, {k: (v["ord"], v["push"], v["coh"], v["judges"]) if v else None
                             for k, v in d.items() if k != "ref"})
    print("nopush", json.dumps(nopush))


if __name__ == "__main__":
    main()
