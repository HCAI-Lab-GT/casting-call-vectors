#!/usr/bin/env python3
"""
Lightweight runner to launch Inspect AI evals using YAML presets.

Defaults:
  - Model presets: configs/models.yaml
  - Run presets:   configs/runs.yaml

Examples:
  # Use run preset
  python scripts/run_eval.py --run bbeh-mini-qwen3-1.7b

  # Override model and limit
  python scripts/run_eval.py --run bbeh-mini-qwen3-1.7b --limit 5 --model qwen1.5b

  # Direct task/model without presets
  python scripts/run_eval.py --task inspect_evals/bbeh_mini --model-id hf/Qwen/Qwen3-1.7B --limit 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "configs" / "models.yaml"
DEFAULT_RUNS = ROOT / "configs" / "runs.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def lookup_model(models: Dict[str, Any], name: str) -> Tuple[str, Dict[str, Any]]:
    for entry in models.get("models", []):
        if entry.get("name") == name:
            return entry["model"], entry.get("generate", {}) or {}
    raise SystemExit(f"Unknown model preset: {name}")


def lookup_run(runs: Dict[str, Any], name: str) -> Dict[str, Any]:
    for entry in runs.get("runs", []):
        if entry.get("name") == name:
            return entry
    raise SystemExit(f"Unknown run preset: {name}")


def parse_kv_list(pairs: list[str]) -> Dict[str, str]:
    kv = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Expected key=value, got: {item}")
        k, v = item.split("=", 1)
        kv[k] = v
    return kv


def build_command(
    task: str,
    model_id: str,
    limit: int | None,
    log_dir: str | None,
    solver_args: Dict[str, Any],
    model_args: Dict[str, Any],
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--python",
        ".venv/bin/python",
        "inspect",
        "eval",
        task,
        "--model",
        model_id,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if log_dir:
        cmd += ["--log-dir", log_dir]
    for k, v in solver_args.items():
        cmd += ["-S", f"{k}={v}"]
    for k, v in model_args.items():
        cmd += ["-M", f"{k}={v}"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Run preset name from runs.yaml")
    ap.add_argument("--task", help="Override task (inspect task path)")
    ap.add_argument("--model", help="Model preset name from models.yaml")
    ap.add_argument("--model-id", help="Explicit model id (hf/... or api)")
    ap.add_argument("--limit", type=int, help="Override sample limit")
    ap.add_argument("--log-dir", help="Override log dir")
    ap.add_argument("--model-config", default=str(DEFAULT_MODELS), help="Path to models.yaml")
    ap.add_argument("--run-config", default=str(DEFAULT_RUNS), help="Path to runs.yaml")
    ap.add_argument(
        "--model-arg", action="append", default=[], help="Extra model arg key=value (repeatable)"
    )
    ap.add_argument(
        "--solver-arg", action="append", default=[], help="Extra solver arg key=value (repeatable)"
    )
    ap.add_argument("--dry-run", action="store_true", help="Print command only")
    args = ap.parse_args()

    models_cfg = load_yaml(Path(args.model_config))
    runs_cfg = load_yaml(Path(args.run_config))

    if args.run:
        run = lookup_run(runs_cfg, args.run)
        task = args.task or run["task"]
        model_name = args.model or run["model_ref"]
        limit = args.limit if args.limit is not None else run.get("limit")
        log_dir = args.log_dir or run.get("log_dir")
    else:
        task = args.task
        model_name = args.model
        limit = args.limit
        log_dir = args.log_dir
    if not task:
        raise SystemExit("task is required (via --task or run preset)")

    if args.model_id:
        model_id = args.model_id
        default_generate = {}
    else:
        if not model_name:
            raise SystemExit("model is required (via --model, --model-id, or run preset)")
        model_id, default_generate = lookup_model(models_cfg, model_name)

    solver_args = dict(default_generate)
    solver_args.update(parse_kv_list(args.solver_arg))
    model_args = parse_kv_list(args.model_arg)

    cmd = build_command(task, model_id, limit, log_dir, solver_args, model_args)
    print(" ".join(cmd))
    if args.dry_run:
        return
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
