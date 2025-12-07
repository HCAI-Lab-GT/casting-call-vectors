import os
from typing import Any, Dict, Iterable, Optional

import wandb
from dotenv import load_dotenv

_WANDB_ENV_VARS = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RUN_GROUP"]


def _load_env():
    # Load .env once to populate environment for wandb.init
    load_dotenv()


def init_wandb(
    run_name: str,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Initialize a Weights & Biases run using env defaults.

    Env vars:
      WANDB_API_KEY   (required)
      WANDB_PROJECT   (default project if not passed)
      WANDB_ENTITY    (your wandb entity/user)
      WANDB_RUN_GROUP (optional grouping key)
    """
    _load_env()
    project = project or os.environ.get("WANDB_PROJECT")
    entity = entity or os.environ.get("WANDB_ENTITY")
    group = os.environ.get("WANDB_RUN_GROUP")

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set. Add it to .env or your environment.")

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=list(tags) if tags else None,
        config=config or {},
        settings=wandb.Settings(console="auto"),
    )
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    wandb.log(metrics, step=step)


def finish():
    wandb.finish()
