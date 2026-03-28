"""
Modal wrapper for regenerate_malformed_responses.py

Runs the malformed-response regeneration script on Modal with 2× H100 GPUs.

Usage:
    # Dry run
    modal run scripts/modal_batching/modal_regenerate_malformed_responses.py --dry-run

    # Fire-and-forget
    modal run --detach scripts/modal_batching/modal_regenerate_malformed_responses.py \
        --roles "soldier,scholar"

    # Specific roles, alphas, columns
    modal run --detach scripts/modal_batching/modal_regenerate_malformed_responses.py \
        --roles "soldier,scholar" --alphas "1.0,2.5" --columns "steered,assistant_axis"

    # Check status / logs
    modal app list
    modal app logs <app-id>
"""

import modal

# ---------------------------------------------------------------------------
# 1) IMAGE — GPU-capable container with torch, transformers, and pvx.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.3.0",
        "transformers>=4.44.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "pandas>=2.0",
        "pyyaml>=6.0.2",
        "hf-transfer>=0.1.9",
        "openai>=1.53.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "datasets>=3.1.0",
        "anthropic>=0.75.0",
        "sentence-transformers>=5.2.0",
        "tenacity>=8.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/hf_cache"})
    .add_local_python_source("pvx")
    .add_local_file(
        "scripts/evaluation/regenerate_malformed_responses.py",
        remote_path="/root/regenerate_malformed_responses.py",
    )
)

# ---------------------------------------------------------------------------
# 2) VOLUME — persistent shared storage for experiment CSVs and model files.
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
app = modal.App("regenerate-malformed-responses")


# ---------------------------------------------------------------------------
# 4) FUNCTION — runs in the cloud with 2× H100 GPUs for model inference.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100:2",
    cpu=4,
    memory=16384,
    timeout=21600,  # 6 hours
    volumes={"/data": volume},
)
def regenerate_malformed(
    input_dir: str = "/data/gold_prompt_experiments",
    roles: list[str] | None = None,
    columns: list[str] | None = None,
    alphas: list[float] | None = None,
    model: str = "allenai/Olmo-3-7B-Instruct",
    safetensors_dir: str = "/data/persona_data/model_layer_inits/",
    pt_dir: str = "/data/persona_data/assistant-axis/olmo-3-7b-instruct/vectors/",
    max_new_tokens: int = 2000,
    dry_run: bool = False,
):
    """
    Runs inside a Modal container in the cloud with GPU access.
    """
    import sys

    sys.argv = [
        "regenerate_malformed_responses.py",
        "--input_dir", input_dir,
        "--model", model,
        "--safetensors_dir", safetensors_dir,
        "--pt_dir", pt_dir,
        "--max_new_tokens", str(max_new_tokens),
        *(["--dry_run"] if dry_run else []),
        *(["--roles"] + roles if roles else []),
        *(["--columns"] + columns if columns else []),
        *(["--alphas"] + [str(a) for a in alphas] if alphas else []),
    ]

    sys.path.insert(0, "/root")

    from regenerate_malformed_responses import main

    main()

    volume.commit()


# ---------------------------------------------------------------------------
# 5) LOCAL ENTRYPOINT — CLI interface that runs on your machine.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input_dir: str = "/data/experiment_data/gold_prompt_experiments",
    roles: str = "",
    columns: str = "",
    alphas: str = "",
    model: str = "allenai/Olmo-3-7B-Instruct",
    safetensors_dir: str = "/data/persona_data/model_layer_inits/",
    pt_dir: str = "/data/persona_data/assistant-axis/olmo-3-7b-instruct/vectors/",
    max_new_tokens: int = 2000,
    dry_run: bool = False,
):
    role_list = [r.strip() for r in roles.split(",") if r.strip()] or None
    column_list = [c.strip() for c in columns.split(",") if c.strip()] or None
    alpha_list = [float(a.strip()) for a in alphas.split(",") if a.strip()] or None

    regenerate_malformed.remote(
        input_dir=input_dir,
        roles=role_list,
        columns=column_list,
        alphas=alpha_list,
        model=model,
        safetensors_dir=safetensors_dir,
        pt_dir=pt_dir,
        max_new_tokens=max_new_tokens,
        dry_run=dry_run,
    )
