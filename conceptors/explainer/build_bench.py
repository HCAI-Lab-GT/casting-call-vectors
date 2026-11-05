#!/usr/bin/env python3
"""Assemble the Persona Steering Bench page from steer_eval runs (after judge.py and analyze_steer.py).

  python explainer/build_bench.py --sweep RUNS/steer_sweep --hold RUNS/steer_hold --out explainer/steering_bench.html

--sweep and --hold take run folders (or steer_derived.json paths); either may be omitted. Standard library only.
"""
import argparse, datetime, json, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent


def load(p):
    if p is None:
        return None
    p = pathlib.Path(p)
    return json.loads((p / 'steer_derived.json' if p.is_dir() else p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default=None)
    ap.add_argument('--hold', default=None)
    ap.add_argument('--out', type=pathlib.Path, default=HERE / 'steering_bench.html')
    a = ap.parse_args()
    data = dict(sweep=load(a.sweep), hold=load(a.hold))
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
