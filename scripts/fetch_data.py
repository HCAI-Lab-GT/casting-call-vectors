"""Download the released data from Hugging Face into ./data (about 860 MB)."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from pvx.release_data import HF_REPO, HF_REVISION, REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=HF_REVISION)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data")
    ap.add_argument(
        "--include",
        nargs="*",
        help="glob patterns to fetch only part of the dataset, e.g. 'vectors/*' 'battery/*'",
    )
    args = ap.parse_args()
    path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.out,
        allow_patterns=[*args.include, "manifest.json"] if args.include else None,
    )
    print(f"Release data at {path}")


if __name__ == "__main__":
    main()
