#!/usr/bin/env python3
"""Assemble the Persona Steering Field Guide into one self-contained HTML page.

  python explainer/build.py                       # repo layout: reads ../acts_qwen05/derived.json
  python build.py --data derived.json --out page.html
  python build.py --data ../acts_smollm3_full     # a run folder: reads its derived.json

Parts: src/head.html (title, fonts, CSS), src/body.html (markup), derived.json (inlined as JSON),
src/lib.js + src/toy.js + src/real.js (wrapped in one function). Needs only the standard library."""
import argparse, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent


def default_data():
    for p in [HERE.parent / 'acts_qwen05' / 'derived.json', HERE / 'data' / 'derived.json']:
        if p.exists():
            return p
    return HERE.parent / 'acts_qwen05' / 'derived.json'


def thin_points(data, samples):
    """Keep the first `samples` answers per (role, system prompt, question): every question and prompt stays, repeat
    draws go. Points, map coordinates and each role's strip-plot values are filtered with the same mask, since the
    page pairs groups[pool][role].vals[j] with the j-th point of that role. Summary statistics stay as computed."""
    seen = {}
    keep = []
    for p in data['points']:
        key = (p['r'], p['s'], p['q'])               # s is the system-prompt index
        seen[key] = seen.get(key, 0) + 1
        keep.append(seen[key] <= samples)
    roles = [p['r'] for p in data['points']]
    data['points'] = [p for p, k in zip(data['points'], keep) if k]
    for pool in data['maps'].values():
        for m in pool.values():
            if m is not None:
                m['pca'] = [c for c, k in zip(m['pca'], keep) if k]
                m['plane'] = [c for c, k in zip(m['plane'], keep) if k]
    for pool in data['groups'].values():
        for role, g in pool.items():
            mask = [k for k, r in zip(keep, roles) if r == role]
            assert len(mask) == len(g['vals']), f'{role}: {len(mask)} points vs {len(g["vals"])} strip values'
            g['vals'] = [v for v, k in zip(g['vals'], mask) if k]
    data['meta']['thinned_samples'] = samples
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=pathlib.Path, default=default_data())
    ap.add_argument('--out', type=pathlib.Path, default=HERE / 'persona_steering_field_guide.html')
    ap.add_argument('--samples', type=int, default=None,
                    help='keep only the first N samples per prompt and question (large runs; default: all)')
    a = ap.parse_args()
    if a.data.is_dir():                             # a run folder: use its derived.json
        a.data = a.data / 'derived.json'
    head = (HERE / 'src/head.html').read_text()
    body = (HERE / 'src/body.html').read_text()
    js = '\n'.join((HERE / 'src' / f).read_text() for f in ['lib.js', 'toy.js', 'real.js'])
    data = json.load(open(a.data))
    if a.samples is not None:
        data = thin_points(data, a.samples)
    for k in ['resp', 'last']:                      # the page never reads these sweep fields
        for row in data['sweep'][k]:
            row.pop('md_auc', None)
    blob = json.dumps(data, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
    html = (head + '\n' + body + '\n<script type="application/json" id="derived-data">' + blob + '</script>\n'
            + '<script>\n(function(){\n"use strict";\n' + js + '\n})();\n</script>\n')
    a.out.write_text(html)
    print(f'wrote {a.out} ({len(html.encode()) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
