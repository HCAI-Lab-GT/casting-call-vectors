from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ROOT / "configs" / "models.yaml"
DEFAULT_RUNS = ROOT / "configs" / "runs.yaml"

def build_command(
    task: str,
    model_id: str,
    limit: int | None,
    log_dir: str | None,
    solver_args: Dict[str, Any],
    model_args: Dict[str, Any],
    task_args: Dict[str, Any],
    gen_config: Dict[str, Any],
    no_display: bool = False
) -> list[str]:
    '''
    Build uv command for eval.
    Ran by subprocess in run_eval.py.
    
    Args:
        task (str): Task name.
        model_id (str): Model identifier.
        limit (int | None): Sample limit.
        log_dir (str | None): Log directory.
        solver_args (Dict[str, Any]) : Solver arguments.
        model_args (Dict[str, Any]): Model arguments.
        task_args (Dict[str, Any]): Task arguments.
        gen_config (Dict[str, Any]): Generate arguments.
        no_display (bool = False): Disable Inspect UI
        
    Returns:
        List of command words for command line.
    '''
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
        cmd.extend(["--limit", str(limit)])
    if log_dir:
        cmd.extend(["--log-dir", log_dir])
        
    # temperature, max_token, etc
    for k, v in gen_config.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
        
    # pass args
    for k, v in model_args.items():
        cmd.extend(["-M", f"{k}={v}"])
    for k, v in solver_args.items():
        cmd.extend(["-S", f"{k}={v}"])
    for k, v in task_args.items():
        cmd.extend(["-T", f"{k}={v}"])
        
    if no_display:
        cmd.extend(["--display", "none"])
        
    return cmd

def load_yaml(path: Path) -> Dict[str, Any]:
    '''
    Load YAML file from path.
    
    Args:
        path (Path): Path to YAML file.
        
    Returns:
        Dictionary representation of the YAML file.
    '''
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def lookup_model(models: Dict[str, Any], name: str) -> Dict[str, Any]:
    '''
    Lookup model preset by name.
    
    Args:
        models (Dict[str, Any]): Models configuration dictionary.
        name (str): Name of the model preset to look up.
        
    Returns:
        Tuple of model id and generation arguments dictionary.
        
    Raises:
        SystemExit: If the model preset is not found.
    '''
    for entry in models.get("models", []):
        if entry.get("name") == name:
            return entry
            # return entry["model"], entry.get("generate", {}) or {}
    raise SystemExit(f"Unknown model preset: {name}")


def lookup_run(runs: Dict[str, Any], name: str) -> Dict[str, Any]:
    '''
    Lookup run preset by name.
    
    Args:
        runs (Dict[str, Any]): Runs configuration dictionary.
        name (str): Name of the run preset to look up.
        
    Returns:
        Dictionary of the run preset.
        
    Raises:
        SystemExit: If the run preset is not found.
    '''
    for entry in runs.get("runs", []):
        if entry.get("name") == name:
            return entry
    raise SystemExit(f"Unknown run preset: {name}")


def parse_kv_list(pairs: list[str]) -> Dict[str, str]:
    '''
    Parse list of key=value strings into dictionary.
    IF no pairs, return empty dict.
    
    Args:
        pairs (list[str] | None): List of strings in key=value format.
        
    Returns:
        Dictionary of parsed key-value pairs.
    '''
    if pairs is None:
        return {}
    
    kv: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Expected key=value, got: {item}")
        k, v = item.split("=", 1)
        kv[k] = v
    return kv


def default_log_dir(task: str) -> str:
    '''
    Get default log directory for a task.
    
    Args:
        task (str): Task name.
        
    Returns:
        Default log directory path as string.
    '''
    base = task.split(":", 1)[0].replace("/", "_")
    return str(ROOT / "logs" / base)