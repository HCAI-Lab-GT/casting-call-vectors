from __future__ import annotations

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
    kv: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Expected key=value, got: {item}")
        k, v = item.split("=", 1)
        kv[k] = v
    return kv


def default_log_dir(task: str) -> str:
    base = task.split(":", 1)[0].replace("/", "_")
    return str(ROOT / "logs" / base)


def build_command(
    task: str,
    model_id: str,
    limit: int | None,
    log_dir: str | None,
    solver_args: Dict[str, Any],
    model_args: Dict[str, Any],
    task_args: Dict[str, Any],
    temperature: float | None,
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
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    for k, v in solver_args.items():
        cmd += ["-S", f"{k}={v}"]
    for k, v in model_args.items():
        cmd += ["-M", f"{k}={v}"]
    for k, v in task_args.items():
        cmd += ["-T", f"{k}={v}"]
    return cmd
