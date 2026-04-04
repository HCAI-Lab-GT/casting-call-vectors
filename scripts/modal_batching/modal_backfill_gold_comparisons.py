"""
Modal wrapper for backfill_assistant_axis_gold_comparisons.py

Runs the backfill judge scoring script on Modal with CPU only (API requests only).

Usage:
    # Dry run
    modal run scripts/modal_batching/modal_backfill_gold_comparisons.py --dry-run

    # Fire-and-forget
    modal run --detach scripts/modal_batching/modal_backfill_gold_comparisons.py \
        --roles "soldier,scholar"

    # Overwrite all assistant_axis scores
    modal run --detach scripts/modal_batching/modal_backfill_gold_comparisons.py \
        --roles "soldier,scholar" --overwrite-assistant-axis --max-concurrency 500

    # Overwrite everything
    modal run --detach scripts/modal_batching/modal_backfill_gold_comparisons.py \
        --overwrite --max-concurrency 500

    # Check status / logs
    modal app list
    modal app logs <app-id>
"""

import modal

# ---------------------------------------------------------------------------
# 1) IMAGE — lightweight container for API requests (no GPU dependencies).
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2.0",
        "openai>=1.53.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.0",
        "torch>=2.3.0",
        "transformers>=4.44.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.66.0",
        "datasets>=3.1.0",
        "anthropic>=0.75.0",
        "sentence-transformers>=5.2.0",
        "hf-transfer>=0.1.9",
    )
    .add_local_python_source("pvx")
    .add_local_file(
        "scripts/patch_scripts/backfill_assistant_axis_gold_comparisons.py",
        remote_path="/root/backfill_assistant_axis_gold_comparisons.py",
    )
)

# ---------------------------------------------------------------------------
# 2) VOLUME — persistent shared storage for experiment CSVs.
#
#    First-time setup (run once from your local machine):
#        modal volume create personality-vectors-data
#        modal volume put personality-vectors-data \
#            ./experiment_data/gold_prompt_experiments /gold_prompt_experiments
#        modal volume put personality-vectors-data \
#            ./persona_data /persona_data
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("personality-vectors-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# 3) APP
# ---------------------------------------------------------------------------
app = modal.App("backfill-gold-comparisons")


# ---------------------------------------------------------------------------
# 4) FUNCTION — runs in the cloud with CPU only (API requests).
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def backfill_gold_comparisons(
    input_dir: str = "/data/gold_prompt_experiments",
    gold_prompts_dir: str = "/data/persona_data/gold_labels_prompts_dataset",
    roles: list[str] | None = None,
    overwrite: bool = False,
    overwrite_assistant_axis: bool = False,
    skip_steered_refresh: bool = False,
    dry_run: bool = False,
    backend: str = "openai",
    judge_model: str = "openai/gpt-4.1-mini",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
    max_concurrency: int = 50,
):
    """
    Runs inside a Modal container in the cloud (CPU only, API requests).
    """
    import sys

    sys.argv = [
        "backfill_assistant_axis_gold_comparisons.py",
        "--input_dir", input_dir,
        "--gold_prompts_dir", gold_prompts_dir,
        "--backend", backend,
        "--judge_model", judge_model,
        "--base_url", base_url,
        "--api_key_env", api_key_env,
        "--max_concurrency", str(max_concurrency),
        *(["--overwrite"] if overwrite else []),
        *(["--overwrite_assistant_axis"] if overwrite_assistant_axis else []),
        *(["--skip_steered_refresh"] if skip_steered_refresh else []),
        *(["--dry_run"] if dry_run else []),
        *(["--roles"] + roles if roles else []),
    ]

    sys.path.insert(0, "/root")

    from backfill_assistant_axis_gold_comparisons import main

    main()

    volume.commit()


# ---------------------------------------------------------------------------
# 5) LOCAL ENTRYPOINT — CLI interface that runs on your machine.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input_dir: str = "/data/gold_prompt_experiments",
    gold_prompts_dir: str = "/data/persona_data/gold_labels_prompts_dataset",
    roles: str = "",
    overwrite: bool = False,
    overwrite_assistant_axis: bool = False,
    skip_steered_refresh: bool = False,
    dry_run: bool = False,
    backend: str = "openai",
    judge_model: str = "openai/gpt-4.1-mini",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
    max_concurrency: int = 50,
):
    role_list = [r.strip() for r in roles.split(",") if r.strip()] or None

    backfill_gold_comparisons.remote(
        input_dir=input_dir,
        gold_prompts_dir=gold_prompts_dir,
        roles=role_list,
        overwrite=overwrite,
        overwrite_assistant_axis=overwrite_assistant_axis,
        skip_steered_refresh=skip_steered_refresh,
        dry_run=dry_run,
        backend=backend,
        judge_model=judge_model,
        base_url=base_url,
        api_key_env=api_key_env,
        max_concurrency=max_concurrency,
    )
