#!/usr/bin/env python3
"""Run Inspect AI evals from YAML presets with optional WandB and heartbeats."""

from __future__ import annotations

# ruff: noqa: I001

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from pvx import Heartbeat, setup_logging
from run_eval_helpers import (
    DEFAULT_MODELS,
    DEFAULT_RUNS,
    build_command,
    default_log_dir,
    load_yaml,
    lookup_model,
    lookup_run,
    parse_kv_list,
)


def main() -> None:
    load_dotenv()
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
    ap.add_argument(
        "--task-arg",
        action="append",
        default=[],
        help="Task arg key=value (repeatable) passed with -T",
    )
    ap.add_argument(
        "--cot",
        action="store_true",
        help="Force prompt_type=chain_of_thought when supported (e.g. BBH)",
    )
    ap.add_argument(
        "--temperature", type=float, help="Override generation temperature (passed to inspect eval)"
    )
    ap.add_argument("--no-wandb", action="store_true", help="Disable WandB (enabled by default)")
    ap.add_argument("--wandb-project", help="Override WANDB_PROJECT")
    ap.add_argument("--wandb-entity", help="Override WANDB_ENTITY")
    ap.add_argument(
        "--wandb-tag",
        action="append",
        default=[],
        help="Tag to add to WandB run (repeatable)",
    )
    ap.add_argument(
        "--heartbeat-interval",
        type=int,
        default=30,
        help="Seconds between heartbeat logs (0 disables)",
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
        log_dir = args.log_dir or run.get("log_dir") or default_log_dir(task)
    else:
        task = args.task
        model_name = args.model
        limit = args.limit
        log_dir = args.log_dir or (default_log_dir(task) if task else None)
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
    task_args = parse_kv_list(args.task_arg)

    if args.run:
        run_task_args = run.get("task_args", {})
        task_args = {**run_task_args, **task_args}

    if args.cot and "prompt_type" not in task_args:
        task_args["prompt_type"] = "chain_of_thought"

    cmd = build_command(
        task,
        model_id,
        limit,
        log_dir,
        solver_args,
        model_args,
        task_args,
        args.temperature,
    )
    env = os.environ.copy()
    wandb_enabled = not args.no_wandb
    if wandb_enabled:
        if not env.get("WANDB_API_KEY"):
            print(
                "WANDB_API_KEY is required because WandB is enabled by default. Set it or pass --no-wandb.",
                file=sys.stderr,
            )
            sys.exit(1)
        env["INSPECT_WANDB_ENABLED"] = "1"
        if args.wandb_project:
            env["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            env["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_tag:
            env["WANDB_TAGS"] = ",".join(args.wandb_tag)
    else:
        env.pop("INSPECT_WANDB_ENABLED", None)
    print(" ".join(cmd))
    if args.dry_run:
        return
    logger = setup_logging(name="run-eval")
    if args.heartbeat_interval > 0:
        with Heartbeat(logger, f"running inspect eval {task}", interval=args.heartbeat_interval):
            result = subprocess.run(cmd, env=env)
    else:
        result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
