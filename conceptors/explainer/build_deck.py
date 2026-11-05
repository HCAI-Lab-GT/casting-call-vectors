#!/usr/bin/env python3
"""Assemble the lab-meeting deck: one self-contained, click-through HTML presentation.

  python explainer/build_deck.py --guide-data RUNS/smollm3_full --samples 1 \
      --sweep RUNS/steer_sweep_full RUNS/steer_sweep_combo \
      --hold qwen=RUNS/steer_hold_combo,RUNS/steer_hold_full gemma=RUNS/steer_hold_combo.judge-gemma,... \
      --calib qwen=RUNS/calib2_qwen gemma=RUNS/calib2_gemma ... --out explainer/lab_deck.html

Slides live in deck/slides.html; a line '{{plate:ID}}' pulls the field guide plate with that id from src/body.html,
and every plate the slides don't use becomes an appendix slide, so the field guide's scripts (which bind every plate
by id at load) find all their elements. Styles: the field guide's CSS, then deck/deck.css (HCAI Lab tokens, slide
layout). Scripts: deck/nav.js (navigation, no dependencies), then src/lib.js + toy.js + real.js + deck/results.js
in one function, as in build.py. Standard library only.
"""
import argparse, datetime, json, pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import thin_points  # noqa: E402
from build_bench import merge  # noqa: E402

PLATE_RE = re.compile(r'<section class="(?:plate|zoo)[^"]*"(?: id="([^"]+)")?>.*?</section>', re.S)
TITLES = {"zoo-explorer": "The method zoo: twelve steering methods on one toy space"}  # sections without an h3
TURN_KEYS = ("turn", "probe", "n", "fully_rate", "coherence", "identity_ai_rate", "voice_as_rate", "readout")
SUMMARY_KEYS = ("n", "fully_rate", "coherence", "identity_ai_rate", "voice_as_rate", "readout")


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


def hold_for(paths: list[str]) -> list[dict]:
    data = merge(paths, "hold")
    rows = []
    for h in data["hold"]:
        rows.append(dict(context=h["context"], method=h["method"], strength=h["strength"],
                         source=data["meta"]["sources"].get(h["method"]),
                         per_turn=[pick(t, TURN_KEYS) for t in h["per_turn"]],
                         ordinary=pick(h["ordinary"], SUMMARY_KEYS),
                         examples=[dict(q=e["q"], a=e["a"], label=e.get("label"), coh=e.get("coh"),
                                        identity=e.get("identity")) for e in h["examples"]]))
    return rows


def calib_for(run: pathlib.Path) -> dict:
    c = json.loads((run / "calibration.json").read_text())
    man = run / "judge_manifest.json"
    model = json.loads(man.read_text()).get("judge_source") if man.exists() else None
    chosen = c["chosen"]
    r = c["rubrics"].get(chosen, {}) if chosen else {}
    return dict(model=model, chosen=chosen, holdout=r.get("holdout", {}).get("admits_ai_agreement"),
                dev=r.get("dev", {}).get("admits_ai_agreement"),
                bare_holdout=c["rubrics"].get("identity", {}).get("holdout", {}).get("admits_ai_agreement"),
                reasoned_holdout=c["rubrics"].get("identity_r", {}).get("holdout", {}).get("admits_ai_agreement"),
                roleplay_fully_on_ai=c.get("roleplay_fully_on_ai_admissions"), n_ai=c.get("n_ai_admissions"))


def keyed(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        name, _, rest = p.partition("=")
        out[name] = rest.split(",")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide-data", type=pathlib.Path, required=True, help="derived.json or a run folder holding it")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--hold", nargs="+", required=True, help="judge=run,run ... (first run listed wins per method)")
    ap.add_argument("--calib", nargs="+", default=[], help="judge=calibration run folder")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "lab_deck.html")
    a = ap.parse_args()

    gd = a.guide_data / "derived.json" if a.guide_data.is_dir() else a.guide_data
    guide = thin_points(json.loads(gd.read_text()), a.samples) if a.samples else json.loads(gd.read_text())
    for k in ["resp", "last"]:
        for row in guide["sweep"][k]:
            row.pop("md_auc", None)

    sweep = merge(a.sweep, "sweep")
    results = dict(
        sweep=[dict(pick(r, ("method", "strength", "n", "fully_rate", "coherence", "readout", "voice_as_rate")),
                    examples=[dict(q=e["q"], a=e["a"], label=e.get("label"), coh=e.get("coh")) for e in r["examples"][:2]])
               for r in sweep["sweep"]],
        selected=sweep.get("selected", {}),
        hold={judge: hold_for(paths) for judge, paths in keyed(a.hold).items()},
        calib={judge: calib_for(pathlib.Path(p[0])) for judge, p in keyed(a.calib).items()},
        fits={k: sweep["meta"]["fits"][k] for k in ("module", "read_hidden", "n_persona", "n_default", "n_tokens",
                                                     "source_model") if k in sweep["meta"]["fits"]},
    )

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
    a.out.write_text(html)
    print(f"wrote {a.out} ({len(html.encode()) / 1024:.0f} KB; {len(used)} plates on slides, "
          f"{len(all_plates) - len(used)} in the appendix)")


if __name__ == "__main__":
    main()
