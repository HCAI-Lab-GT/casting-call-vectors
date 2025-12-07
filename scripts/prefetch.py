#!/usr/bin/env python
"""
Prefetch models and datasets used in this project so first runs don't stall on downloads.

Usage examples:
  python scripts/prefetch.py                     # download default set (Qwen 1.7B + Boardgame-QA)
  python scripts/prefetch.py --include-large     # also cache the big GPT-OSS 20B model
  python scripts/prefetch.py --models qwen20b    # pick specific aliases
  python scripts/prefetch.py --datasets boardgameqa bigbenchhard

Environment:
  - Reads .env (HF_TOKEN, HF_HOME, TRANSFORMERS_CACHE, HF_HUB_ENABLE_HF_TRANSFER).
  - If HF_TOKEN is set, passes it to snapshot_download.
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


# Model aliases to repo_ids
MODEL_MAP = {
    "qwen1.7b": "rd211/Qwen3-1.7B-Instruct",
    "qwen2b": "Qwen/Qwen2-1.5B-Instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-7b": "openai/gpt-oss-7b",
}

# Dataset aliases to HF ids (tasksource datasets are script-based but we rely on cached conversion)
DATASET_MAP = {
    "boardgameqa": ("tasksource/Boardgame-QA", "test"),
    "bigbenchhard": ("maveriq/bigbenchhard", "train"),
}


def ensure_transfer_enabled():
    # Enable fast transfer if not set
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def load_env():
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    return hf_token, cache_dir


def prefetch_models(model_ids, token, cache_dir, include_large):
    for mid in model_ids:
        repo_id = MODEL_MAP[mid]
        if not include_large and "20b" in mid:
            print(f"[skip] {repo_id} (large). Use --include-large to fetch.")
            continue
        print(f"[model] downloading {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            token=token,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True,
        )
        print(f"[ok] cached {repo_id}")


def prefetch_datasets(dataset_ids, token, cache_dir):
    for did in dataset_ids:
        repo_id, split = DATASET_MAP[did]
        print(f"[dataset] caching {repo_id} split={split} ...")
        load_dataset(
            repo_id,
            split=split,
            download_mode="reuse_dataset_if_exists",
            use_auth_token=token,
            cache_dir=cache_dir,
        )
        print(f"[ok] cached {repo_id}:{split}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prefetch HF models/datasets for this repo.")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODEL_MAP.keys()),
        default=["qwen1.7b"],
        help="Which model aliases to download (default: qwen1.7b).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASET_MAP.keys()),
        default=["boardgameqa"],
        help="Which datasets to cache (default: boardgameqa).",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also download large models (e.g., gpt-oss-20b).",
    )
    return parser.parse_args()


def main():
    ensure_transfer_enabled()
    token, cache_dir = load_env()
    args = parse_args()
    # Expand model choices if user passed include-large without explicitly listing
    models = set(args.models)
    if args.include_large:
        models.add("gpt-oss-20b")
    prefetch_models(sorted(models), token, cache_dir, include_large=args.include_large)
    prefetch_datasets(args.datasets, token, cache_dir)
    print("All requested artifacts are cached.")


if __name__ == "__main__":
    main()
