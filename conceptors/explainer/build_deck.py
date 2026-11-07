#!/usr/bin/env python3
"""Assemble the lab-meeting deck: one self-contained, click-through HTML presentation.

  python explainer/build_deck.py --config deck_runs.json

The config names the run folders (paths are machine-specific, so it lives outside the repo):

  {"guide_data": "RUNS/smollm3_full", "samples": 1, "default_persona": "pirate", "out": "explainer/lab_deck.html",
   "calib": {"qwen": "RUNS/calib2_qwen", "gemma": "...", "mistral": "...", "llama": "..."},
   "personas": {"pirate": {"sweep": ["RUNS/steer_sweep_v2_pirate"],
                           "hold": {"qwen": ["RUNS/steer_hold_v2_pirate_prompt.judge-qwen",
                                             "RUNS/steer_hold_v2_pirate_steer_only.judge-qwen"], "gemma": [...], ...},
                           "fits": "RUNS/steer_hold_v2_pirate_prompt"},
                "olmo_pirate": {"label": "OLMo · pirate", "model": "OLMo-3-7B", ...}}}

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

PLATE_RE = re.compile(r'<section class="(?:plate|zoo)[^"]*"(?: id="([^"]+)")?>')
TITLES = {"zoo-explorer": "The method zoo: twelve steering methods on one toy space"}  # sections without an h3
# plates the deck leaves out: the scripts still bind their ids, so they are kept in a hidden block, not as slides
PARKED = {"realturns"}  # the twelve-turn pilot; the 20-turn runs supersede it
TRAJ = ("x_pre", "x_post", "off_pre", "off_post", "read_steered")
TURN_KEYS = ("turn", "probe", "n", "fully_rate", "coherence", "identity_ai_rate", "voice_as_rate", "readout") + TRAJ
SUMMARY_KEYS = ("n", "fully_rate", "coherence", "identity_ai_rate", "voice_as_rate", "readout") + TRAJ


def section_end(s: str, start: int) -> int:
    """Index just past the </section> that closes the <section> opening at start, counting nested sections."""
    depth = 0
    for t in re.finditer(r"<section\b|</section>", s[start:]):
        depth += 1 if t.group(0) == "<section" else -1
        if depth == 0:
            return start + t.end()
    raise SystemExit(f"unclosed <section> at offset {start}")


def plates(body: str) -> dict:
    """Every plate section of the field guide, by id (plates without an id get one from their position).
    A plate can hold its own <section>s, so each one ends at its matching </section>, not the first one."""
    out, pos = {}, 0
    for i, m in enumerate(PLATE_RE.finditer(body)):
        if m.start() < pos:
            raise SystemExit(f"a plate opens inside another plate at offset {m.start()}")
        pos = section_end(body, m.start())
        whole = body[m.start():pos]
        pid = m.group(1) or f"plate{i}"
        out[pid] = whole if m.group(1) else re.sub(r'<section class="', f'<section id="{pid}" class="', whole, count=1)
    return out


def plate_title(pid: str, html: str) -> str:
    if pid in TITLES:
        return TITLES[pid]
    m = re.search(r"<h3>(.*?)</h3>", html, re.S) or re.search(r"<h4>(.*?)</h4>", html, re.S)
    if not m:
        return "A recipe for semantic conceptors on personas"
    t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    if "<h3>" in html:
        return t
    return "Open cells: three untested ideas" if t.lower().startswith("open cell") else "Open cells: " + t[0].lower() + t[1:]


def attr(s: str) -> str:
    """A title inside a double-quoted attribute: the judge plate's title quotes "in character"."""
    return s.replace("&", "&amp;").replace('"', "&quot;")


def pick(d: dict, keys) -> dict:
    return {k: d[k] for k in keys if k in d and d[k] is not None}


def probe_set(path: str) -> str:
    man = pathlib.Path(path) / "manifest.json"
    return json.loads(man.read_text()).get("probe_set", "standard") if man.exists() else "standard"


def hold_rows(paths: list[str]) -> list[dict]:
    """Hold entries from several runs (e.g. one per context), first run winning per (context, method). A run with
    other probes (steer_eval --probe_set none or unseen) becomes its own context, e.g. 'prompt_none'."""
    seen, rows = set(), []
    for p in paths:
        d, ps = load(p), probe_set(p)
        for h in d["hold"]:
            ctx = h["context"] if ps == "standard" else h["context"] + "_" + ps
            key = (ctx, h["method"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(context=ctx, method=h["method"], strength=h["strength"],
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
    for k in ("label", "model"):  # e.g. "OLMo · pirate" for a second model's persona on the same switch
        if spec.get(k):
            out[k] = spec[k]
    return out


def intern_texts(personas: dict) -> list[str]:
    """Replace each hold example's question and answer with an index into one shared list, which results.js
    expands at load. The four judges label the same answers, so this stores each string once instead of four times."""
    index, texts = {}, []
    for pd in personas.values():
        for rows in pd["hold"].values():
            for h in rows:
                for e in h["examples"]:
                    for k in ("q", "a"):
                        s = e.get(k)
                        if isinstance(s, str):
                            if s not in index:
                                index[s] = len(texts)
                                texts.append(s)
                            e[k] = index[s]
    return texts


def drop_missing(personas: dict) -> dict:
    """Keep only the run folders analyze_steer has written (for builds while judges are still landing), saying what
    was left out. A persona with no sweep and no hold left is dropped."""
    def keep(paths):
        return [p for p in paths if (pathlib.Path(p) / "steer_derived.json").exists()]
    out, gone = {}, []
    for name, spec in personas.items():
        sweep = keep(spec.get("sweep", []))
        hold = {j: keep(ps) for j, ps in spec.get("hold", {}).items()}
        gone += [p for p in spec.get("sweep", []) if p not in sweep]
        gone += [p for j, ps in spec.get("hold", {}).items() for p in ps if p not in hold[j]]
        hold = {j: ps for j, ps in hold.items() if ps}
        if sweep or hold:
            out[name] = dict(spec, sweep=sweep, hold=hold)
        else:
            gone.append(f"(persona {name})")
    if gone:
        print(f"--allow-missing: left out {len(gone)} not-yet-analyzed runs: "
              + ", ".join(pathlib.Path(p).name for p in gone), file=sys.stderr)
    return out


BTC_IDS = ("c1", "c3", "c4", "c5", "c6", "c8", "m3", "m4", "m5", "m8", "s2p")  # ids the field guide or deck also use
BTC_FONT = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;'
            '9..144,600&display=swap">\n')


def scope_css(css: str, scope: str) -> str:
    """Prefix every selector with scope, so Beyond the Cap's rules stay inside its plates (one level of @media)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    def one(sel: str) -> str:
        sel = sel.strip()
        if sel in (":root", "html", "body"):
            return scope
        return f"{sel} {scope}" if sel.startswith(":root") else f"{scope} {sel}"

    out, end = [], 0
    for m in re.finditer(r"([^{}]*)([{}])", css):
        text, br = m.groups()
        if br == "{" and not text.strip().startswith("@"):
            out.append(",".join(one(s) for s in text.split(",")) + "{")
        else:
            out.append(text + br)
        end = m.end()
    return "".join(out) + css[end:]


def btc_parts() -> tuple[dict, str, str]:
    """Beyond the Cap's sections by id (plus its colour legend), its scoped CSS and its script, ids de-clashed."""
    alt = "|".join(BTC_IDS)
    body = re.sub(rf'\b(id|for)="({alt})"', r'\1="bc-\2"', (HERE / "btc/body.html").read_text())
    js = re.sub(rf"#({alt})(['\"])", r"#bc-\1\2", (HERE / "btc/page.js").read_text())
    # its colour tokens live on .btc here, not on :root, so its canvases must read them there or they draw in black
    root = "getComputedStyle(document.documentElement)"
    if js.count(root) != 1:
        raise SystemExit(f"Beyond the Cap reads root styles {js.count(root)} times; expected the one colour loader")
    js = js.replace(root, "getComputedStyle(document.querySelector('.btc')||document.documentElement)")
    js = js.replace("getComputedStyle(document.body)", "getComputedStyle(document.querySelector('.btc')||document.body)")
    secs = {m.group(1): m.group(0) for m in
            re.finditer(r'<section class="(?:plate|part)[^"]*" id="([^"]+)">.*?</section>', body, re.S)}
    for m in re.finditer(r'</section>\s*(<div class="tablewrap".*?</table></div>)', body, re.S):  # tables between plates
        sid = re.findall(r'<section class="[^"]*" id="([^"]+)"', body[:m.start()])[-1]
        secs[sid] += "\n" + m.group(1)
    legend = re.search(r'<div class="legend" aria-label="Legend">.*?</div>', body, re.S)
    if legend:
        secs["legend"] = legend.group(0)
    return secs, scope_css((HERE / "btc/style.css").read_text(), ".btc"), js


def sec_title(html: str) -> str:
    m = re.search(r"<h[23][^>]*>(.*?)</h[23]>", html, re.S)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else "Beyond the cap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=None, help="overrides the config's out")
    ap.add_argument("--allow-missing", action="store_true",
                    help="build from the runs that exist so far instead of stopping at the first missing one")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.allow_missing:
        cfg["personas"] = drop_missing(cfg["personas"])
    out_path = a.out or pathlib.Path(cfg.get("out", HERE / "lab_deck.html"))

    gd = pathlib.Path(cfg["guide_data"])
    gd = gd / "derived.json" if gd.is_dir() else gd
    samples = cfg.get("samples", 1)
    guide = thin_points(json.loads(gd.read_text()), samples) if samples else json.loads(gd.read_text())
    for k in ["resp", "last"]:
        for row in guide["sweep"][k]:
            row.pop("md_auc", None)

    personas = {p: persona_data(spec) for p, spec in cfg["personas"].items()}
    default = cfg.get("default_persona")
    results = dict(personas=personas, texts=intern_texts(personas),
                   default_persona=default if default in personas else next(iter(personas)),
                   calib={j: calib_for(pathlib.Path(p)) for j, p in cfg.get("calib", {}).items()})
    if cfg.get("mechanism") and pathlib.Path(cfg["mechanism"]).exists():  # conceptor_spectrum.py output
        results["mechanism"] = json.loads(pathlib.Path(cfg["mechanism"]).read_text())

    # every <style> block: the later ones hold fixes (the method zoo's two columns among them)
    style = "\n".join(re.findall(r"<style>(.*?)</style>", (HERE / "src/head.html").read_text(), re.S))
    body = (HERE / "src/body.html").read_text()
    all_plates = plates(body)
    slides = (HERE / "deck/slides.html").read_text()
    used = set(re.findall(r"\{\{plate:([\w-]+)\}\}", slides))
    missing = used - set(all_plates)
    if missing:
        raise SystemExit(f"slides name plates the field guide doesn't have: {sorted(missing)}")
    slides = re.sub(r"\{\{plate:([\w-]+)\}\}", lambda m: all_plates[m.group(1)], slides)
    appendix = "\n".join(
        f'<section class="slide appendix" data-title="{attr(plate_title(pid, html))}" data-part="Appendix">\n'
        f'<header class="shead"><div class="kicker">Appendix</div><h2>{plate_title(pid, html)}</h2></header>\n{html}\n</section>'
        for pid, html in all_plates.items() if pid not in used and pid not in PARKED)
    parked = "\n".join(all_plates[pid] for pid in PARKED if pid in all_plates)
    # Beyond the Cap: '{{btc:ID}}' pulls one of its sections; the rest go to the appendix, since its script binds them all
    secs, btc_css, btc_js = btc_parts()
    used_btc = set(re.findall(r"\{\{btc:([\w-]+)\}\}", slides))
    if used_btc - set(secs):
        raise SystemExit(f"slides name Beyond the Cap sections it doesn't have: {sorted(used_btc - set(secs))}")
    # adjacent placeholders share one wrapper, so a section intro and its plate can be laid out together on the slide
    slides = re.sub(r"(?:\{\{btc:[\w-]+\}\})+", lambda m: '<div class="btc">' + "".join(
        secs[s] for s in re.findall(r"\{\{btc:([\w-]+)\}\}", m.group(0))) + "</div>", slides)
    # its one-paragraph part intros aren't slides: the slides they introduce say the same thing
    skip = used_btc | {"legend", "sec-start", "sec-move", "sec-ours", "sec-map"}
    appendix += "\n" + "\n".join(
        f'<section class="slide appendix" data-title="{attr(sec_title(h))}" data-part="Appendix">\n'
        f'<header class="shead"><div class="kicker">Appendix</div><h2>{sec_title(h)}</h2></header>\n'
        f'<div class="btc">{h}</div>\n</section>'
        for sid, h in secs.items() if sid not in skip)

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
    html = (head + BTC_FONT + "<style>\n" + style + "\n" + css + "\n</style>\n<style>\n" + btc_css + "\n</style>\n"
            + '<div class="deck" id="deck">\n' + slides + "\n" + appendix + "\n</div>\n"
            + '<div class="parked" hidden>\n' + parked + "\n</div>\n"
            + '<div class="tip" id="tip" hidden></div>\n'
            + '<script type="application/json" id="derived-data">' + blob(guide) + "</script>\n"
            + '<script type="application/json" id="deck-data">' + blob(results) + "</script>\n"
            + "<script>\n" + nav + "\n</script>\n"
            + '<script>\n(function(){\n"use strict";\n' + js + "\n})();\n</script>\n"
            + "<script>\n(function(){\n" + re.sub(r"</(script)", r"<\\/\1", btc_js, flags=re.I) + "\n})();\n</script>\n")

    ids = re.findall(r'\sid="([^"]+)"', html)
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"duplicate element ids: {dup}")
    # the field guide's scripts bind elements by id at load and stop at the first missing one
    refs = set(re.findall(r"""(?:\$|Fig|Canvas|LineChart|onRange|onSeg|out|dataTable)\(\s*['"]([\w-]+)['"]""", js))
    refs |= set(re.findall(r"""\$\(\s*['"]#([\w-]+)['"]""", btc_js))
    gone = sorted(refs - set(ids) - {"zooOpt"})  # zooOpt is looked up optionally
    if gone:
        raise SystemExit(f"scripts reference ids the deck lacks: {gone}")
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(html.encode()) / 1024:.0f} KB; {len(used)} plates on slides, "
          f"{len(all_plates) - len(used)} in the appendix; personas {', '.join(personas)})")


if __name__ == "__main__":
    main()
