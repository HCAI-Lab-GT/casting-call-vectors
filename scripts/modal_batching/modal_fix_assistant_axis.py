"""
Modal wrapper for fix_assistant_axis_alpha_response.py

Runs the repair script on Modal's cloud infrastructure, similar to SLURM sbatch.

Usage:
    # Dry run (preview changes)
    modal run scripts/modal_batching/modal_fix_assistant_axis.py --dry-run

    # Fire-and-forget (like sbatch)
    modal run --detach scripts/modal_batching/modal_fix_assistant_axis.py --alpha 2.5

    # Specific roles
    modal run --detach scripts/modal_batching/modal_fix_assistant_axis.py \
        --roles "soldier,scholar,whale" --alpha 2.5

    # Check status / logs
    modal app list
    modal app logs <app-id>
"""

import modal

# ---------------------------------------------------------------------------
# 1) IMAGE — the container environment your code runs inside.
#    Think of this like a Dockerfile or conda environment that Modal builds
#    once and caches. Every function invocation starts from this snapshot.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Install only what this script actually needs (pandas + pvx's logging).
    # The full pyproject.toml pulls torch/vllm/transformers which are huge
    # and unnecessary for a pandas-only repair job.
    .pip_install("pandas>=2.0")
    # Copy the local pvx package into the image so `from pvx import ...` works.
    # Modal snapshots src/pvx/ at build time and bakes it into the container.
    .add_local_python_source("pvx", remote_path="/root/pvx")
    # Copy the script itself so we can import it inside the container.
    .add_local_file(
        "scripts/patch_scripts/fix_assistant_axis_alpha_response.py",
        remote_path="/root/fix_assistant_axis_alpha_response.py",
    )
)

# ---------------------------------------------------------------------------
# 2) VOLUME — persistent shared storage, like a SLURM scratch filesystem.
#    Data here survives across runs. You upload your CSVs once, then every
#    function invocation can read/write them.
#
#    First-time setup (run once from your local machine):
#        modal volume create personality-vectors-data
#        modal volume put personality-vectors-data \
#            ./experiment_data/gold_prompt_experiments /gold_prompt_experiments
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("personality-vectors-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# 3) APP — a named Modal application. Groups your functions together.
#    When you `modal run --detach`, this name shows up in `modal app list`.
# ---------------------------------------------------------------------------
app = modal.App("fix-assistant-axis")


# ---------------------------------------------------------------------------
# 4) FUNCTION — the actual compute unit. Decorating with @app.function tells
#    Modal: "run this in the cloud with these resources."
#
#    SLURM equivalent of the #SBATCH directives at the top of a .sh script.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    # gpu="T4",                # Uncomment if you need a GPU. This script doesn't.
    cpu=2,                     # Like: #SBATCH --cpus-per-task=2
    memory=4096,               # Like: #SBATCH --mem=4G  (in MiB)
    timeout=1800,              # Like: #SBATCH --time=00:30:00  (in seconds)
    volumes={"/data": volume}, # Mount the volume at /data inside the container
)
def fix_assistant_axis(
    input_dir: str = "/data/gold_prompt_experiments",
    roles: list[str] | None = None,
    alpha: float = 2.5,
    dry_run: bool = False,
):
    """
    Runs inside a Modal container in the cloud.

    The function body executes remotely — it has access to the Volume at /data
    and the packages baked into the image, but NOT your local filesystem.
    """
    import sys

    # Build an argv list as if we called the script from the command line.
    sys.argv = [
        "fix_assistant_axis_alpha_response.py",
        "--input_dir", input_dir,
        "--alpha", str(alpha),
        *(["--dry_run"] if dry_run else []),
        *(["--roles"] + roles if roles else []),
    ]

    # Add the root directory to the path so we can import the script and pvx.
    sys.path.insert(0, "/root")

    from fix_assistant_axis_alpha_response import main

    main()

    # After writing, tell Modal to persist any changes back to the Volume.
    volume.commit()


# ---------------------------------------------------------------------------
# 5) LOCAL ENTRYPOINT — the CLI interface that runs on YOUR machine.
#    This parses args locally, then calls .remote() to ship the work to Modal.
#
#    This is the sbatch equivalent: you type a command locally, it submits
#    the job to run elsewhere, and you get back a job ID.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    input_dir: str = "/data/gold_prompt_experiments",
    roles: str = "",
    alpha: float = 2.5,
    dry_run: bool = False,
):
    role_list = [r.strip() for r in roles.split(",") if r.strip()] or None

    # .remote() sends this function call to Modal's cloud. Your terminal
    # streams the logs back (stdout/stderr) just like watching a SLURM job.
    # With `modal run --detach`, it returns immediately instead.
    fix_assistant_axis.remote(
        input_dir=input_dir,
        roles=role_list,
        alpha=alpha,
        dry_run=dry_run,
    )
