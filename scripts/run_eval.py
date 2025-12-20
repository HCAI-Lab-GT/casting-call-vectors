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
from pvx import Heartbeat, setup_logging

def main() -> None:
    '''
    Main function to run Inspect AI evals based on command-line arguments and presets.
    Supports WandB integration and heartbeat logging.
    '''
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Run preset name from runs.yaml")
    ap.add_argument("--task", help="Override task (inspect task path)")
    ap.add_argument("--model", help="Model preset name from models.yaml")
    ap.add_argument("--model-id", help="Explicit model id (hf/... or api)")
    ap.add_argument("--log-dir", help="Override log dir")
    ap.add_argument("--limit", type=int, help="Override sample limit")

    # generated configs
    ap.add_argument("--max_tokens", type=int, help="Override max_token limit")
    ap.add_argument(
        "--temperature", type=float, help="Override generation temperature (passed to inspect eval)"
    )

    # config files
    ap.add_argument("--model-config", default=str(DEFAULT_MODELS), help="Path to models.yaml")
    ap.add_argument("--run-config", default=str(DEFAULT_RUNS), help="Path to runs.yaml")
    ap.add_argument(
        "-M",
        "--model-arg", 
        action="append",
        default=[],
        help="Extra model arg key=value (repeatable) passed with -M"
    )
    ap.add_argument(
        "-T",
        "--task-arg",
        action="append",
        default=[],
        help="Task arg key=value (repeatable) passed with -T",
    )
    ap.add_argument(
        "-S",
        "--solver-arg",
        action="append",
        default=[],
        help="Extra solver arg key=value (repeatable) passed with -S"
    )
    ap.add_argument(
        "--cot",
        action="store_true",
        help="Force prompt_type=chain_of_thought when supported (e.g. BBH)",
    )
    ap.add_argument("--no-display", action="store_true", help="Disable Inspect UI (enabled by default)")
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

    # load configs
    models_cfg = load_yaml(Path(args.model_config))
    runs_cfg = load_yaml(Path(args.run_config))

    # Load run if specified
    run = lookup_run(runs_cfg, args.run) if args.run else {}

    # Param Priorities: CLI > Run > None
    task        =   args.task       or run.get('task')
    model_name  =   args.model      or run.get('model_ref')
    limit       =   args.limit      or run.get('limit')
    log_dir     =   args.log_dir    or run.get('log_dir', default_log_dir(task) if task else None)

    # Task is required
    if not task:
        raise SystemExit("task is required (via --task or run preset)")

    # One of model_id or model_name must be specified
    if not args.model_id and not model_name:
        raise SystemExit("model is required (via --model, --model-id, or run preset)")

    # Load task and solver configs
        # run.yaml
    task_args = run.get('task_args', {})
    solver_args = run.get('solver_args', {})
    
        # CLI
    task_args.update(parse_kv_list(args.task_arg))
    solver_args.update(parse_kv_list(args.solver_arg))

    # Load generate config (will be unpacked later in build_command)
    generate_configs = run.get('generate_args', {})
    generate_fields = ['max_tokens', 'temperature', 'top_p', 'top_k']
    generate_configs.update({field: getattr(args, field) for field in generate_fields
                             if hasattr(args, field) and getattr(args, field) is not None})

    # Load model via if specified
    model = lookup_model(models_cfg, model_name) if model_name is not None else {}
    
    # Param Priorities: Arguments > Run > None, args
    model_id = args.model_id or model.get('model')
    model_args = model.get('args', {}) # models.yaml: args
    model_args.update(run.get('model_args_override', {})) # runs.yaml: model_args_override
    model_args.update(parse_kv_list(args.model_arg)) # CLI -M args

    # chain-of-thought prompt if requested
    if args.cot and "prompt_type" not in task_args:
        task_args["prompt_type"] = "chain_of_thought"

    # build uv run python command
    cmd = build_command(
        task=task,
        model_id=model_id,
        limit=limit,
        log_dir=log_dir,
        model_args=model_args,
        solver_args=solver_args,
        task_args=task_args,
        gen_config=generate_configs,
        no_display=args.no_display
    )

    env = os.environ.copy()

    # configure WandB
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

    logger = setup_logging(name="run-eval")
    logger.info("Running `%s`", " ".join(cmd))

    if args.dry_run:
        return

    # run with heartbeats if requested, otherwise normal run via subprocess
    if args.heartbeat_interval > 0:
        with Heartbeat(logger, f"running inspect eval {task}", interval=args.heartbeat_interval):
            result = subprocess.run(cmd, env=env)
    else:
        result = subprocess.run(cmd, env=env)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
