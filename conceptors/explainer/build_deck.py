#!/usr/bin/env python3
"""Assemble the lab-meeting deck: one self-contained, click-through HTML presentation.

  python explainer/build_deck.py --config deck_runs.json

The config names the run folders (paths are machine-specific, so it lives outside the repo):

  {"guide_data": "RUNS/smollm3_full", "samples": 1, "default_persona": "pirate", "out": "explainer/lab_deck.html",
   "calib": {"qwen": "RUNS/calib2_qwen", "gemma": "...", "mistral": "...", "llama": "..."},
   "personas": {"pirate": {"sweep": ["RUNS/steer_sweep_v2_pirate"],
                           "hold": {"qwen": ["RUNS/steer_hold_v2_pirate_prompt.judge-qwen",
                                             "RUNS/steer_hold_v2_pirate_steer_only.judge-qwen"], "gemma": [...], ...},
                           "fits": "RUNS/steer_hold_v2_pirate_prompt"}, ...}}

Sweep runs merge per method and hold runs per (context, method), the first run listed winning. "fits" is a hold run
folder holding fits.npz and manifest.json; it gives the steering-layer cap floors the trajectory slide draws.

Slides live in deck/slides.html; a line '{{plate:ID}}' pulls the field guide plate with that id from src/body.html,
and every plate the slides don't use becomes an appendix slide, so the field guide's scripts (which bind every plate
by id at load) find all their elements. Styles: the field guide's CSS, then deck/deck.css (HCAI Lab tokens, slide
layout). Scripts: deck/nav.js (navigation, no dependencies), then src/lib.js + toy.js + real.js + deck/results.js
in one function, as in build.py. Needs numpy only for the cap floors.
"""
import argparse, datetime, json, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import thin_points  # noqa: E402
from build_bench import load, merge  # noqa: E402

PLATE_RE = re.compile(r'<section class="(?:plate|zoo)[^"]*"(?: id="([^"]+)")?>.*?</section>', re.S)
TITLES = {"zoo-explorer": "The method zoo: twelve steering methods on one toy space"}  # sections without an h3
TRAJ = ("x_pre", "x_post", "off_pre", "off_post", "read_steered")
TURN_KEYS = ("turn", "probe", "n", "fully_rate", "coherence", "identity_ai_rate", "voice_as_rate", "readout") + TRAJ
SUMMARY_KEYS = ("n", "fully_rate", "coherence", "identity_ai_rate", "voice_as_rate", "readout") + TRAJ


def plates(body: str) -> dict:
    """Every plate section of the field guide, by id (plates without an id get one from their position)."""
    out = {}
    for i, m in enumerate(PLATE_RE.finditer(body)):
        pid = m.group(1) or f"plate{i}"
        html = m.group(0) if m.group(1) else re.sub(r'<section class="', f'<section id="{pid}" class="', m.group(0), count=1)
        out[pid] = html
    return out


def plate_title(pid: str, html: str) -> str:
    if pid in TITLES:
        return TITLES[pid]
    m = re.search(r"<h3>(.*?)</h3>", html, re.S) or re.search(r"<h4>(.*?)</h4>", html, re.S)
    if not m:
        return "A recipe for semantic conceptors on personas"
    t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return t if "<h3>" in html else "Open cells: " + t[0].lower() + t[1:]


def pick(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d and d[k] is not None}


def hold_rows(paths: list[str]) -> list[dict]:
    """Hold entries from several runs (e.g. one per context), first run winning per (context, method)."""
    seen, rows = set(), []
    for p in paths:
        d = load(p)
        for h in d["hold"]:
            key = (h["context"], h["method"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(context=h["context"], method=h["method"], strength=h["strength"],
                             per_turn=[pick(t, TURN_KEYS) for t in h["per_turn"]],
                             ordinary=pick(h["ordinary"], SUMMARY_KEYS),
                             examples=[dict(pick(e, TRAJ + ("readout",)), q=e["q"], a=e["a"], label=e.get("label"),
                                            coh=e.get("coh"), identity=e.get("identity")) for e in h["examples"]]))
    return rows


def floors(fits_dir: str | None) -> dict:
    """Steering-layer cap floors in default-0 / persona-1 units, for the methods that floor at that layer."""
    if not fits_dir:
        return {}
    d = pathlib.Path(fits_dir)
    if not (d / "fits.npz").exists() or not (d / "manifest.json").exists():
        return {}
    import numpy as np
    man = json.loads((d / "manifest.json").read_text())
    f, z = man["fits"], np.load(d / "fits.npz")
    a0, p0 = f.get("steer_a0"), f.get("steer_p0")
    if a0 is None or "proj_persona" not in z:
        return {}
    scale = lambda tau: round((tau - a0) / (p0 - a0), 4)  # noqa: E731
    st = man.get("strengths", {})
    out = {}
    if "cap" in st:
        out["cap"] = scale(float(np.percentile(z["proj_persona"], st["cap"])))
    if "add_cap" in st:
        out["add_cap"] = scale(float(np.percentile(z["proj_persona"], f.get("cap_hold_pct", 25))))
    return out


def calib_for(run: pathlib.Path) -> dict:
    c = json.loads((run / "calibration.json").read_text())
    man = run / "judge_manifest.json"
    model = json.loads(man.read_text()).get("judge_source") if man.exists() else None
    chosen = c["chosen"]
    r = c["rubrics"].get(chosen, {}) if chosen else {}
    return dict(model=model, chosen=chosen, holdout=r.get("holdout", {}).get("admits_ai_agreement"),
                dev=r.get("dev", {}).get("admits_ai_agreement"),
                roleplay_fully_on_ai=c.get("roleplay_fully_on_ai_admissions"), n_ai=c.get("n_ai_admissions"))


def persona_data(spec: dict) -> dict:
    sweep = merge(spec["sweep"], "sweep") if spec.get("sweep") else None
    out = dict(hold={j: hold_rows(paths) for j, paths in spec["hold"].items()}, floors=floors(spec.get("fits")))
    if sweep:
        out["sweep"] = [dict(pick(r, ("method", "strength", "n", "fully_rate", "coherence", "readout", "voice_as_rate")),
                             examples=[dict(q=e["q"], a=e["a"], label=e.get("label"), coh=e.get("coh"))
                                       for e in r["examples"][:2]]) for r in sweep["sweep"]]
        out["selected"] = sweep.get("selected", {})
        out["fits"] = {k: sweep["meta"]["fits"][k] for k in ("module", "read_hidden", "n_persona", "n_default", "n_tokens",
                                                            "source_model", "cap_aa_modules") if k in sweep["meta"]["fits"]}
    turns = [t["turn"] for rows in out["hold"].values() for h in rows for t in h["per_turn"]]
    out["n_turns"] = max(turns) + 1 if turns else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None, help="overrides the config's out")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    out_path = a.out or pathlib.Path(cfg.get("out", HERE / "lab_deck.html"))

    gd = pathlib.Path(cfg["guide_data"])
    gd = gd / "derived.json" if gd.is_dir() else gd
    samples = cfg.get("samples", 1)
    guide = thin_points(json.loads(gd.read_text()), samples) if samples else json.loads(gd.read_text())
    for k in ["resp", "last"]:
        for row in guide["sweep"][k]:
            row.pop("md_auc", None)

    personas = {p: persona_data(spec) for p, spec in cfg["personas"].items()}
    results = dict(personas=personas, default_persona=cfg.get("default_persona", next(iter(personas))),
                   calib={j: calib_for(pathlib.Path(p)) for j, p in cfg.get("calib", {}).items()})

    style = re.search(r"<style>(.*?)</style>", (HERE / "src/head.html").read_text(), re.S).group(1)
    body = (HERE / "src/body.html").read_text()
    all_plates = plates(body)
    slides = (HERE / "deck/slides.html").read_text()
    used = set(re.findall(r"\{\{plate:([\w-]+)\}\}", slides))
    missing = used - set(all_plates)
    if missing:
        raise SystemExit(f"slides name plates the field guide doesn't have: {sorted(missing)}")
    slides = re.sub(r"\{\{plate:([\w-]+)\}\}", lambda m: all_plates[m.group(1)], slides)
    appendix = "\n".join(
        f'<section class="slide appendix" data-title="{plate_title(pid, html)}" data-part="Appendix">\n'
        f'<header class="shead"><div class="kicker">Appendix</div><h2>{plate_title(pid, html)}</h2></header>\n{html}\n</section>'
        for pid, html in all_plates.items() if pid not in used)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    results["built"] = dict(date=datetime.date.today().isoformat(), commit=commit)

    def blob(x):
        return json.dumps(x, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")

    head = (HERE / "deck/head.html").read_text()
    css = (HERE / "deck/deck.css").read_text()
    nav = (HERE / "deck/nav.js").read_text()
    js = "\n".join((HERE / p).read_text() for p in ["src/lib.js", "src/toy.js", "src/real.js"])
    js += "\n{ /* deck slides in their own block: their helpers may shadow the field guide's names */\n" + \
          (HERE / "deck/results.js").read_text() + "\n}\n"
    html = (head + "<style>\n" + style + "\n" + css + "\n</style>\n"
            + '<div class="deck" id="deck">\n' + slides + "\n" + appendix + "\n</div>\n"
            + '<div class="tip" id="tip" hidden></div>\n'
            + '<script type="application/json" id="derived-data">' + blob(guide) + "</script>\n"
            + '<script type="application/json" id="deck-data">' + blob(results) + "</script>\n"
            + "<script>\n" + nav + "\n</script>\n"
            + '<script>\n(function(){\n"use strict";\n' + js + "\n})();\n</script>\n")

    ids = re.findall(r'\sid="([^"]+)"', html)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"duplicate element ids: {dup}")
    # the field guide's scripts bind elements by id at load and stop at the first missing one
    refs = set(re.findall(r"""(?:\$|Fig|Canvas|LineChart|onRange|onSeg|out|dataTable)\(\s*['"]([\w-]+)['"]""", js))
    gone = sorted(refs - set(ids) - {"zooOpt"})  # zooOpt is looked up optionally
    if gone:
        raise SystemExit(f"scripts reference ids the deck lacks: {gone}")
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(html.encode()) / 1024:.0f} KB; {len(used)} plates on slides, "
          f"{len(all_plates) - len(used)} in the appendix; personas {', '.join(personas)})")


if __name__ == "__main__":
    main()
