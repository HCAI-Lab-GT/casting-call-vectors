#!/usr/bin/env python3
"""Assemble the Persona Steering Field Guide into one self-contained HTML page.

  python explainer/build.py                       # repo layout: reads ../acts_qwen05/derived.json
  python build.py --data derived.json --out page.html

Parts: src/head.html (title, fonts, CSS), src/body.html (markup), derived.json (inlined as JSON),
src/lib.js + src/toy.js + src/real.js (wrapped in one function). Needs only the standard library."""
import argparse, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent


def default_data():
    for p in [HERE.parent / 'acts_qwen05' / 'derived.json', HERE / 'data' / 'derived.json']:
        if p.exists():
            return p
    return HERE.parent / 'acts_qwen05' / 'derived.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=pathlib.Path, default=default_data())
    ap.add_argument('--out', type=pathlib.Path, default=HERE / 'persona_steering_field_guide.html')
    a = ap.parse_args()
    head = (HERE / 'src/head.html').read_text()
    body = (HERE / 'src/body.html').read_text()
    js = '\n'.join((HERE / 'src' / f).read_text() for f in ['lib.js', 'toy.js', 'real.js'])
    data = json.load(open(a.data))
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
