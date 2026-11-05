"""Compare two derived.json files key by key: exact structure, numbers within a tolerance.

    python conceptors/tests/compare_derived.py OLD.json NEW.json [--tol 1e-3] [--only-shared]

--only-shared ignores top-level sections that exist in only one file (new sections added later).
Exits 1 if structure differs or any number differs by more than tol.
"""

import argparse
import json
import sys
from collections import defaultdict


def walk(a, b, path, out, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out["structure"].append(f"{path}/{k}: only in {'new' if k in b else 'old'}")
            else:
                walk(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["structure"].append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(a - b)
        section = (path.split("/")[1] if "/" in path else path).split("[")[0]
        out["maxdiff"][section] = max(out["maxdiff"][section], d)
        if d > tol:
            out["over"].append(f"{path}: {a} vs {b}")
    elif a != b:
        out["structure"].append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--only-shared", action="store_true")
    args = ap.parse_args()
    a, b = json.load(open(args.old)), json.load(open(args.new))
    if args.only_shared:
        keys = set(a) & set(b)
        a, b = {k: a[k] for k in keys}, {k: b[k] for k in keys}
    out = {"structure": [], "over": [], "maxdiff": defaultdict(float)}
    walk(a, b, "", out, args.tol)
    for section, d in sorted(out["maxdiff"].items()):
        print(f"  max |diff| {section:10s} {d:.3g}")
    for line in out["structure"][:20]:
        print("STRUCTURE", line)
    for line in out["over"][:20]:
        print("OVER", line)
    print(f"{len(out['structure'])} structural differences, {len(out['over'])} numbers over tol={args.tol}")
    sys.exit(1 if out["structure"] or out["over"] else 0)


if __name__ == "__main__":
    main()
