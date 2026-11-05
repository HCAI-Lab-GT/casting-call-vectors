#!/usr/bin/env python3
"""Assemble the Persona Steering Bench page from steer_eval runs (after judge.py and analyze_steer.py).

  python explainer/build_bench.py --sweep RUNS/steer_sweep --hold RUNS/steer_hold --out explainer/steering_bench.html

--sweep and --hold take one or more run folders (or steer_derived.json paths); either may be omitted. With several
runs, each method comes from the first run listed that has it, and meta.sources records which run that was.
Standard library only.
"""
import argparse, datetime, json, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent


def load(p):
    p = pathlib.Path(p)
    return json.loads((p / 'steer_derived.json' if p.is_dir() else p).read_text())


def run_name(p):
    p = pathlib.Path(p)
    return p.name if p.is_dir() else p.parent.name


def merge(paths, key):
    """One stage's runs as one: rows of `key` ('sweep' or 'hold') and 'selected' entries per method, first run wins."""
    if not paths:
        return None
    runs = [load(p) for p in paths]
    out = dict(runs[0], **{key: []})
    if 'selected' in out:
        out['selected'] = {}
    sources = {}
    for p, d in zip(paths, runs):
        new = {r['method'] for r in d[key]} - set(sources)
        out[key] += [r for r in d[key] if r['method'] in new]
        if 'selected' in out:
            out['selected'].update({m: v for m, v in d.get('selected', {}).items() if m in new})
        sources.update({m: run_name(p) for m in sorted(new)})
    out['meta'] = dict(runs[0]['meta'], sources=sources)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', nargs='+', default=None)
    ap.add_argument('--hold', nargs='+', default=None)
    ap.add_argument('--out', type=pathlib.Path, default=HERE / 'steering_bench.html')
    a = ap.parse_args()
    data = dict(sweep=merge(a.sweep, 'sweep'), hold=merge(a.hold, 'hold'))
    if not data['sweep'] and not data['hold']:
        raise SystemExit('give --sweep and/or --hold')
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ''
    data['built'] = dict(date=datetime.date.today().isoformat(), commit=commit)
    blob = json.dumps(data, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
    html = (HERE / 'bench' / 'bench.html').read_text().replace('/*BENCH_DATA*/', blob)
    a.out.write_text(html)
    print(f'wrote {a.out} ({len(html.encode()) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
