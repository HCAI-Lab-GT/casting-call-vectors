#!/usr/bin/env python
"""
Download models for the Marin personality vector experiments.

Usage:
  python scripts/marin/download_models.py                    # download Llama 3.2 1B (prototype)
  python scripts/marin/download_models.py --include-marin    # also download Marin 8B
  python scripts/marin/download_models.py --models marin-8b  # only Marin 8B
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

MODEL_MAP = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "marin-8b": "marin-community/marin-8b-instruct",
}

DEFAULT_MODELS = ["llama-1b"]


def main():
    parser = argparse.ArgumentParser(description="Download models for Marin experiments.")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODEL_MAP.keys()),
        default=None,
        help=f"Model aliases to download. Choices: {list(MODEL_MAP.keys())}",
    )
    parser.add_argument(
        "--include-marin",
        action="store_true",
        help="Also download Marin 8B (~16 GB).",
    )
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    models = set(args.models) if args.models else set(DEFAULT_MODELS)
    if args.include_marin:
        models.add("marin-8b")

    for alias in sorted(models):
        repo_id = MODEL_MAP[alias]
        print(f"[download] {alias} -> {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            token=token,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True,
        )
        print(f"[ok] cached {repo_id}")

    print("All requested models are cached.")


if __name__ == "__main__":
    main()
