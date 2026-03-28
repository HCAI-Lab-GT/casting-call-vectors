"""
Modal wrapper for regenerate_assistant_axis_by_alpha.py

Runs the regeneration script on Modal's cloud infrastructure with GPU support.

Usage:
    # Dry run (preview changes)
    modal run scripts/evaluation/modal_regenerate_assistant_axis.py --dry-run

    # Fire-and-forget (like sbatch)
    modal run --detach scripts/evaluation/modal_regenerate_assistant_axis.py --alphas "2.5"

    # Specific roles and alphas
    modal run --detach scripts/evaluation/modal_regenerate_assistant_axis.py \
        --roles "soldier,scholar,whale" --alphas "1.0,2.5,5.0"

    # Custom model / layer
    modal run --detach scripts/evaluation/modal_regenerate_assistant_axis.py \
        --alphas "2.5" --assistant-axis-layer 14

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
    # Copy the local pvx package into the image so `from pvx import ...` works.
    .add_local_python_source("pvx")
    # Copy the script itself so we can import it inside the container.
    .add_local_file(
        "scripts/evaluation/regenerate_assistant_axis_by_alpha.py",
        remote_path="/root/regenerate_assistant_axis_by_alpha.py",
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
app = modal.App("regenerate-assistant-axis")


# ---------------------------------------------------------------------------
# 4) FUNCTION — runs in the cloud with a GPU for model inference.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=7200,
    volumes={"/data": volume},
)
def regenerate_assistant_axis(
    input_dir: str = "/data/gold_prompt_experiments",
    roles: list[str] | None = None,
    alphas: list[float] | None = None,
    model: str = "allenai/Olmo-3-7B-Instruct",
    json_filepath: str = "/data/persona_data/persona_models.json",
    pt_dir: str = "/data/persona_data/assistant-axis/olmo-3-7b-instruct/vectors/",
    assistant_axis_layer: int | None = None,
    max_new_tokens: int = 2000,
    temperature_mode: str = "row",
    fixed_temperature: float = 0.2,
    dry_run: bool = False,
):
    """
    Runs inside a Modal container in the cloud with GPU access.
    """
    import sys

    sys.argv = [
        "regenerate_assistant_axis_by_alpha.py",
        "--input_dir", input_dir,
        "--model", model,
        "--json_filepath", json_filepath,
        "--pt_dir", pt_dir,
        "--max_new_tokens", str(max_new_tokens),
        "--temperature_mode", temperature_mode,
        "--fixed_temperature", str(fixed_temperature),
        *(["--dry_run"] if dry_run else []),
        *(["--roles"] + roles if roles else []),
        *(["--alphas"] + [str(a) for a in alphas] if alphas else []),
        *(["--assistant_axis_layer", str(assistant_axis_layer)] if assistant_axis_layer is not None else []),
    ]

    sys.path.insert(0, "/root")

    from regenerate_assistant_axis_by_alpha import main

    main()

    volume.commit()


# ---------------------------------------------------------------------------
# 5) LOCAL ENTRYPOINT — CLI interface that runs on your machine.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input_dir: str = "/data/experiment_data/gold_prompt_experiments",
    roles: str = "",
    alphas: str = "",
    model: str = "allenai/Olmo-3-7B-Instruct",
    json_filepath: str = "/data/persona_data/persona_models.json",
    pt_dir: str = "/data/persona_data/assistant-axis/olmo-3-7b-instruct/vectors/",
    assistant_axis_layer: int = 16,
    max_new_tokens: int = 2000,
    temperature_mode: str = "row",
    fixed_temperature: float = 0.2,
    dry_run: bool = False,
):
    role_list = [r.strip() for r in roles.split(",") if r.strip()] or None
    alpha_list = [float(a.strip()) for a in alphas.split(",") if a.strip()] or None
    layer = assistant_axis_layer if assistant_axis_layer >= 0 else None

    regenerate_assistant_axis.remote(
        input_dir=input_dir,
        roles=role_list,
        alphas=alpha_list,
        model=model,
        json_filepath=json_filepath,
        pt_dir=pt_dir,
        assistant_axis_layer=layer,
        max_new_tokens=max_new_tokens,
        temperature_mode=temperature_mode,
        fixed_temperature=fixed_temperature,
        dry_run=dry_run,
    )
