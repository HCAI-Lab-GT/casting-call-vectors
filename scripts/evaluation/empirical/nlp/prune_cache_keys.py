"""Remove specified keys from cached NLP experiment JSON files.

Usage:
    python prune_cache_keys.py self_bleu_mean self_bleu_std
    python prune_cache_keys.py --dry-run self_bleu_mean self_bleu_std
    python prune_cache_keys.py --glob "analysis/**/per_role/*.json" key1 key2
"""
import argparse
import json
import glob


def main():
    parser = argparse.ArgumentParser(description="Remove keys from cached NLP JSON summaries.")
    parser.add_argument("keys", nargs="+", help="Keys to remove from each method summary.")
    parser.add_argument(
        "--glob", default="analysis/empirical/nlp_experiments/per_role/*.json",
        dest="pattern", help="Glob pattern for JSON files (default: per_role cache).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        print(f"No files matched pattern: {args.pattern}")
        return

    keys_to_remove = set(args.keys)
    updated = 0

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        modified = False
        if "summaries" in data:
            for method, summary in data["summaries"].items():
                for key in keys_to_remove:
                    if key in summary:
                        if args.dry_run:
                            print(f"  {f}: would remove '{key}' from '{method}'")
                        else:
                            del summary[key]
                        modified = True

        if modified:
            updated += 1
            if not args.dry_run:
                with open(f, "w") as fh:
                    json.dump(data, fh, indent=2)

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}Updated {updated}/{len(files)} files")


if __name__ == "__main__":
    main()
