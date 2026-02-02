#!/usr/bin/env python3
"""Run Inspect AI evals from YAML presets with optional WandB and heartbeats."""

from __future__ import annotations

# ruff: noqa: I001

import os
import sys
import json
import uuid
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote
import argparse

from dotenv import load_dotenv

from inspect_ai.log import list_eval_logs, read_eval_log, convert_eval_logs
import wandb

from pvx.utils.wandb_utils import create_table

from helpers import (
    DEFAULT_MODELS,
    DEFAULT_RUNS,
    build_command,
    default_log_dir,
    load_yaml,
    lookup_model,
    lookup_run,
    parse_kv_list,
)
from pvx.utils.logging_utils import Heartbeat, setup_logging, format_object

logger = setup_logging(name="run-eval")


def main() -> None:
    """
    Main function to run Inspect AI evals based on command-line arguments and presets.
    Supports WandB integration and heartbeat logging.
    """
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--run", help="Run preset name from runs.yaml")
    ap.add_argument("-t", "--task", help="Override task (inspect task path)")
    ap.add_argument("-m", "--model", help="Model preset name from models.yaml")
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
        help="Extra model arg key=value (repeatable) passed with -M",
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
        help="Extra solver arg key=value (repeatable) passed with -S",
    )
    ap.add_argument(
        "--cot",
        action="store_true",
        help="Force prompt_type=chain_of_thought when supported (e.g. BBH)",
    )
    ap.add_argument(
        "--no-display", action="store_true", help="Disable Inspect UI (enabled by default)"
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
        "--wandb-save-method",
        choices=["file", "artifact", "none"],
        default="file",
        help="How to save logs to WandB: file (default), artifact, or none",
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

    # Convert flat task to one-item chain task
    if "tasks" not in run:
        run = {"name": run["name"], "tasks": [{k: v for k, v in run.items() if k != "name"}]}

    logger.info(format_object(run, "Run Details: "))

    # create launch id (8 char hex) to group batched runs
    batch_id = uuid.uuid4().hex[:8]
    logger.info("Batch ID: %s", batch_id)

    # Run all tasks in run
    for task in run["tasks"]:
        run_task(args, task, models_cfg, batch_id)


def run_task(args, task_cfg, models_cfg, batch_id=None) -> None:
    """
    Runs a task with each of its traits in Inspect AI.

    Args:
        args (argparse.Namespace): CLI args
        task_cfg (dict): task configs, effectively params inside runs.yaml
        models_cfg (dict): all model configs, used to lookup configs per model per task
    """
    # Param Priorities: CLI > task > None
    task = args.task or task_cfg.get("task")
    model_name = args.model or task_cfg.get("model_ref")
    limit = args.limit or task_cfg.get("limit")
    log_dir = args.log_dir or task_cfg.get("log_dir", default_log_dir(task) if task else None)

    # Task is required
    if not task:
        raise SystemExit("task is required (via --task or task preset)")

    # One of model_id or model_name must be specified
    if not args.model_id and not model_name:
        raise SystemExit("model is required (via --model, --model-id, or task preset)")

    # Load task and solver configs
    # run.yaml
    task_args = task_cfg.get("task_args", {})
    solver_args = task_cfg.get("solver_args", {})

    # CLI
    task_args.update(parse_kv_list(args.task_arg))
    solver_args.update(parse_kv_list(args.solver_arg))

    # Load generate config (will be unpacked later in build_command)
    generate_configs = task_cfg.get("generate_args", {})
    generate_fields = ["max_tokens", "temperature", "top_p", "top_k"]
    generate_configs.update(
        {
            field: getattr(args, field)
            for field in generate_fields
            if hasattr(args, field) and getattr(args, field) is not None
        }
    )

    # Load model via if specified
    model = lookup_model(models_cfg, model_name) if model_name is not None else {}

    # Param Priorities: Arguments > task > None, args
    model_id = args.model_id or model.get("model")
    if model_id is None:
        logger.error("No model_id specified. Use --model-id or configure in models.yaml")
        sys.exit(1)
    model_args = model.get("args", {})  # models.yaml: args
    model_args.update(task_cfg.get("model_args_override", {}))  # runs.yaml: model_args_override
    model_args.update(parse_kv_list(args.model_arg))  # CLI -M args

    # chain-of-thought prompt if requested
    if args.cot and "prompt_type" not in task_args:
        task_args["prompt_type"] = "chain_of_thought"

    if "trait" not in model_args or isinstance(model_args["trait"], str):
        model_args["trait"] = [model_args.get("trait")]

    for trait in model_args["trait"]:
        single_trait_model_args = model_args.copy()

        if trait:
            single_trait_model_args["trait"] = trait
        else:
            single_trait_model_args.pop("trait")

        # build uv run python command
        cmd = build_command(
            task=task,
            model_id=model_id,  # type: ignore[arg-type]  # validated above
            limit=limit,
            log_dir=log_dir,
            model_args=single_trait_model_args,
            solver_args=solver_args,
            task_args=task_args,
            gen_config=generate_configs,
            no_display=args.no_display,
        )

        env = os.environ.copy()

        ### WandB Configurations
        wandb_enabled = not args.no_wandb
        if wandb_enabled:
            if not env.get("WANDB_API_KEY"):
                logger.error(
                    "WANDB_API_KEY is required because WandB is enabled by default. Set it or pass --no-wandb."
                )
                sys.exit(1)
            env["INSPECT_WANDB_ENABLED"] = "1"
            if args.wandb_project:
                env["INSPECT_WANDB_PROJECT"] = args.wandb_project
                env["WANDB_PROJECT"] = args.wandb_project
            if args.wandb_entity:
                env["INSPECT_WANDB_ENTITY"] = args.wandb_entity
                env["WANDB_ENTITY"] = args.wandb_entity

            # Set WandB tags and configs
            short_task_name = task.split("@")[-1]
            model_tags = [
                f"model:{model_name}",
                f"benchmark:{short_task_name}",
                f"trait:{trait}" if trait else None,
                f"limit:{limit}" if limit else None,
                f"batch_id:{batch_id}" if batch_id else None,
            ]
            model_configs = {
                "model_id": model_name,
                "benchmark_suite": task,
                "trait": trait,
                "limit": limit,
                "temperature": generate_configs.get("temperature", None),
                "top_p": generate_configs.get("top_p", None),
                "max_tokens": generate_configs.get("max_tokens", None),
            }

            env["INSPECT_WANDB_MODELS_TAGS"] = json.dumps(
                args.wandb_tag + [t for t in model_tags if t is not None]
            )
            env["INSPECT_WANDB_MODELS_CONFIG"] = json.dumps(model_configs)

            # Rename WandB run name
            env["WANDB_NAME"] = f"{short_task_name}__{model_name}__{trait}__{batch_id}"

        else:
            env.pop("INSPECT_WANDB_ENABLED", None)

        logger.info("Running `%s`", " ".join(cmd))
        logger.info("Task: %s", task)
        logger.info(format_object(single_trait_model_args, "Model Args: "))
        logger.info(format_object(solver_args, "Solver Args: "))
        logger.info(format_object(task_args, "Task Args: "))
        logger.info(format_object(generate_configs, "Generatin Args: "))

        if args.dry_run:
            return

        # run with heartbeats if requested, otherwise normal run via subprocess
        if args.heartbeat_interval > 0:
            with Heartbeat(
                logger, f"running inspect eval {task}", interval=args.heartbeat_interval
            ):
                result = subprocess.run(cmd, env=env)
        else:
            result = subprocess.run(cmd, env=env)

        if result.returncode == 0 and wandb_enabled:
            # newest log first by default, mild hack
            logs = list_eval_logs(log_dir)
            latest = logs[0]
            latest_path = Path(log_dir) / os.path.basename(unquote(urlparse(latest.name).path))

            # read header to get Inspect run_id
            hdr = read_eval_log(latest_path, header_only=True)
            run_id = hdr.eval.run_id  # run_id identifies the W&B Run

            ## Code if .json in separate location not same as .eval
            # out_dir = Path(...)
            # out_dir.mkdir(parents=True, exist_ok=True)
            # convert_eval_logs(str(latest_path), to="json", output_dir=str(out_dir), stream=True)
            # json_path = out_dir / latest_path.with_suffix(".json").name

            convert_eval_logs(str(latest_path), to="json", output_dir=log_dir, stream=True)

            json_path = os.path.join(log_dir, latest_path.with_suffix(".json").name)

            with wandb.init(  # type: ignore[attr-defined]  # wandb stubs incomplete
                project=env.get("WANDB_PROJECT"),
                entity=env.get("WANDB_ENTITY"),
                id=run_id,
                resume="must",
            ) as run:
                # rehydrate table with samples from json and upload to the run as artifact
                data_table = create_table(json_path)
                run.log({"samples": data_table})

                ## Save InspectAI logs to WandB
                if args.wandb_save_method == "artifact":
                    art = wandb.Artifact(name="results", type="eval")
                    art.add_file(str(json_path))
                    run.log_artifact(art)
                elif args.wandb_save_method == "file":
                    run.save(json_path, log_dir)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
