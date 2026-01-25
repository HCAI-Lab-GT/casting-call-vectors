"""SLURM job generation utilities for cluster execution.

This module generates SLURM batch scripts for running persona vector
extraction and analysis jobs on GPU clusters (e.g., MATS).

Typical usage:
    >>> script = generate_extraction_job(
    ...     persona_dir="persona_data/vocational_personas/instructions",
    ...     riasec_filter="R",
    ...     limit=25,
    ...     wandb_project="pvx-phase1",
    ... )
    >>> with open("jobs/extract_R.sh", "w") as f:
    ...     f.write(script)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default SLURM configurations for different job types
DEFAULT_EXTRACTION_CONFIG = {
    "partition": "gpu",
    "gpus": 1,
    "cpus_per_task": 4,
    "mem": "32G",
    "time": "8:00:00",
}

DEFAULT_ANALYSIS_CONFIG = {
    "partition": "cpu",
    "gpus": 0,
    "cpus_per_task": 8,
    "mem": "16G",
    "time": "2:00:00",
}


@dataclass
class SLURMConfig:
    """Configuration for SLURM job submission.

    Attributes:
        partition: SLURM partition name
        gpus: Number of GPUs requested
        cpus_per_task: Number of CPU cores
        mem: Memory allocation (e.g., "32G")
        time: Wall time limit (e.g., "8:00:00")
        job_name: Name for the job
        output_dir: Directory for SLURM logs
        account: Optional account/allocation name
        constraint: Optional node constraint (e.g., "a100")
        extra_directives: Additional SLURM directives
    """

    partition: str = "gpu"
    gpus: int = 1
    cpus_per_task: int = 4
    mem: str = "32G"
    time: str = "8:00:00"
    job_name: str = "pvx-job"
    output_dir: str = "logs/slurm"
    account: str | None = None
    constraint: str | None = None
    extra_directives: dict[str, str] = field(default_factory=dict)


def _generate_slurm_header(config: SLURMConfig) -> str:
    """Generate SLURM header directives from config."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={config.job_name}",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
        f"#SBATCH --mem={config.mem}",
        f"#SBATCH --time={config.time}",
        f"#SBATCH --output={config.output_dir}/%x_%j.out",
        f"#SBATCH --error={config.output_dir}/%x_%j.err",
    ]

    if config.gpus > 0:
        lines.append(f"#SBATCH --gpus={config.gpus}")

    if config.account:
        lines.append(f"#SBATCH --account={config.account}")

    if config.constraint:
        lines.append(f"#SBATCH --constraint={config.constraint}")

    for key, value in config.extra_directives.items():
        lines.append(f"#SBATCH --{key}={value}")

    return "\n".join(lines)


def _generate_environment_setup(
    conda_env: str | None = None,
    use_uv: bool = True,
    extra_env_vars: dict[str, str] | None = None,
) -> str:
    """Generate environment setup commands."""
    lines = [
        "",
        "# Environment setup",
        "set -euo pipefail",  # Exit on error, undefined vars, pipe failures
        "",
    ]

    # Load modules (common on HPC clusters)
    lines.extend(
        [
            "# Load modules (adjust for your cluster)",
            "module purge 2>/dev/null || true",
            "module load cuda/12.1 2>/dev/null || true",
            "",
        ]
    )

    # Conda environment
    if conda_env:
        lines.extend(
            [
                "# Activate conda environment",
                "source $(conda info --base)/etc/profile.d/conda.sh",
                f"conda activate {conda_env}",
                "",
            ]
        )

    # Extra environment variables
    if extra_env_vars:
        lines.append("# Custom environment variables")
        for key, value in extra_env_vars.items():
            lines.append(f"export {key}={value}")
        lines.append("")

    # UV setup
    if use_uv:
        lines.extend(
            [
                "# Ensure uv is available",
                'export PATH="$HOME/.cargo/bin:$PATH"',
                "",
            ]
        )

    # Working directory
    lines.extend(
        [
            "# Change to project directory",
            "cd $SLURM_SUBMIT_DIR",
            "",
        ]
    )

    return "\n".join(lines)


def generate_extraction_job(
    persona_dir: str = "persona_data/vocational_personas/instructions",
    model_id: str = "allenai/OLMo-7B-Instruct",
    layer: int = 14,
    output_dir: str = "outputs/vectors",
    riasec_filter: str | None = None,
    limit: int | None = None,
    num_questions: int = 50,
    wandb_project: str | None = "pvx-phase1",
    wandb_run_name: str | None = None,
    resume_from: str | None = None,
    config: SLURMConfig | None = None,
    conda_env: str | None = None,
    use_uv: bool = True,
) -> str:
    """Generate SLURM batch script for persona vector extraction.

    Args:
        persona_dir: Directory containing persona JSON files
        model_id: HuggingFace model identifier
        layer: Layer for activation extraction
        output_dir: Directory for saving extracted vectors
        riasec_filter: Filter personas by RIASEC type (R/I/A/S/E/C)
        limit: Maximum number of personas to process
        num_questions: Number of questions per persona
        wandb_project: W&B project for logging (None to disable)
        wandb_run_name: Custom W&B run name
        resume_from: Resume from checkpoint directory
        config: SLURM configuration (uses defaults if None)
        conda_env: Conda environment name
        use_uv: Use uv for running Python

    Returns:
        Complete SLURM batch script as string
    """
    if config is None:
        job_name = f"pvx-extract-{riasec_filter or 'all'}"
        config = SLURMConfig(
            partition="gpu",
            gpus=1,
            cpus_per_task=4,
            mem="32G",
            time="8:00:00",
            job_name=job_name,
        )

    # Build the command
    runner = "uv run" if use_uv else "python"
    cmd_parts = [
        f"{runner} scripts/run_extraction.py",
        f"--model {model_id}",
        f"--layer {layer}",
        f"--persona-dir {persona_dir}",
        f"--output-dir {output_dir}",
        f"--num-questions {num_questions}",
    ]

    if riasec_filter:
        cmd_parts.append(f"--riasec {riasec_filter}")

    if limit:
        cmd_parts.append(f"--limit {limit}")

    if wandb_project:
        cmd_parts.append(f"--wandb-project {wandb_project}")

    if wandb_run_name:
        cmd_parts.append(f"--wandb-run-name {wandb_run_name}")

    if resume_from:
        cmd_parts.append(f"--resume {resume_from}")

    command = " \\\n    ".join(cmd_parts)

    # Assemble script
    script_parts = [
        _generate_slurm_header(config),
        _generate_environment_setup(conda_env=conda_env, use_uv=use_uv),
        "# Create output directories",
        f"mkdir -p {output_dir}",
        f"mkdir -p {config.output_dir}",
        "",
        "# Run extraction",
        'echo "Starting extraction job: $SLURM_JOB_ID"',
        'echo "Node: $(hostname)"',
        'echo "GPUs: $CUDA_VISIBLE_DEVICES"',
        "",
        command,
        "",
        'echo "Extraction complete: $SLURM_JOB_ID"',
    ]

    return "\n".join(script_parts)


def generate_analysis_job(
    vectors_dir: str = "outputs/vectors",
    output_dir: str = "outputs/analysis",
    n_pcs: int = 10,
    wandb_project: str | None = "pvx-phase1",
    wandb_run_name: str | None = None,
    config: SLURMConfig | None = None,
    conda_env: str | None = None,
    use_uv: bool = True,
) -> str:
    """Generate SLURM batch script for geometry analysis.

    Args:
        vectors_dir: Directory containing extracted vectors
        output_dir: Directory for analysis outputs
        n_pcs: Number of principal components to compute
        wandb_project: W&B project for logging
        wandb_run_name: Custom W&B run name
        config: SLURM configuration (uses defaults if None)
        conda_env: Conda environment name
        use_uv: Use uv for running Python

    Returns:
        Complete SLURM batch script as string
    """
    if config is None:
        config = SLURMConfig(
            partition="cpu",
            gpus=0,
            cpus_per_task=8,
            mem="16G",
            time="2:00:00",
            job_name="pvx-analysis",
        )

    # Build the command
    runner = "uv run" if use_uv else "python"
    cmd_parts = [
        f"{runner} scripts/run_analysis.py",
        f"--vectors-dir {vectors_dir}",
        f"--output-dir {output_dir}",
        f"--n-pcs {n_pcs}",
    ]

    if wandb_project:
        cmd_parts.append(f"--wandb-project {wandb_project}")

    if wandb_run_name:
        cmd_parts.append(f"--wandb-run-name {wandb_run_name}")

    command = " \\\n    ".join(cmd_parts)

    # Assemble script
    script_parts = [
        _generate_slurm_header(config),
        _generate_environment_setup(conda_env=conda_env, use_uv=use_uv),
        "# Create output directories",
        f"mkdir -p {output_dir}",
        f"mkdir -p {config.output_dir}",
        "",
        "# Run analysis",
        'echo "Starting analysis job: $SLURM_JOB_ID"',
        'echo "Node: $(hostname)"',
        "",
        command,
        "",
        'echo "Analysis complete: $SLURM_JOB_ID"',
    ]

    return "\n".join(script_parts)


def generate_batch_extraction_jobs(
    persona_dir: str = "persona_data/vocational_personas/instructions",
    model_id: str = "allenai/OLMo-7B-Instruct",
    output_base: str = "outputs/vectors",
    personas_per_job: int = 25,
    wandb_project: str | None = "pvx-phase1",
    config: SLURMConfig | None = None,
    conda_env: str | None = None,
    use_uv: bool = True,
) -> dict[str, str]:
    """Generate separate SLURM scripts for each RIASEC type.

    This creates 6 independent jobs that can run in parallel on
    different nodes, one per RIASEC dimension.

    Args:
        persona_dir: Directory containing persona JSON files
        model_id: HuggingFace model identifier
        output_base: Base directory for outputs (will create subdirs)
        personas_per_job: Number of personas per RIASEC type
        wandb_project: W&B project for logging
        config: Base SLURM configuration
        conda_env: Conda environment name
        use_uv: Use uv for running Python

    Returns:
        Dict mapping RIASEC letter -> script content
    """
    riasec_types = ["R", "I", "A", "S", "E", "C"]
    scripts = {}

    for riasec in riasec_types:
        scripts[riasec] = generate_extraction_job(
            persona_dir=persona_dir,
            model_id=model_id,
            output_dir=f"{output_base}/{riasec}",
            riasec_filter=riasec,
            limit=personas_per_job,
            wandb_project=wandb_project,
            wandb_run_name=f"extract-{riasec}",
            config=config,
            conda_env=conda_env,
            use_uv=use_uv,
        )

    return scripts


def write_job_scripts(
    output_dir: str = "jobs",
    **kwargs,
) -> list[Path]:
    """Write batch extraction jobs to files.

    Args:
        output_dir: Directory to write job scripts
        **kwargs: Arguments passed to generate_batch_extraction_jobs

    Returns:
        List of paths to created job scripts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scripts = generate_batch_extraction_jobs(**kwargs)
    created_files = []

    for riasec, script in scripts.items():
        filepath = output_path / f"extract_{riasec}.sh"
        filepath.write_text(script)
        filepath.chmod(0o755)  # Make executable
        created_files.append(filepath)
        logger.info(f"Created job script: {filepath}")

    # Also create analysis job
    analysis_script = generate_analysis_job(
        wandb_project=kwargs.get("wandb_project", "pvx-phase1"),
        conda_env=kwargs.get("conda_env"),
        use_uv=kwargs.get("use_uv", True),
    )
    analysis_path = output_path / "run_analysis.sh"
    analysis_path.write_text(analysis_script)
    analysis_path.chmod(0o755)
    created_files.append(analysis_path)
    logger.info(f"Created analysis script: {analysis_path}")

    # Create submission helper script
    submit_all = [
        "#!/bin/bash",
        "# Submit all extraction jobs",
        "set -e",
        "",
        "cd $(dirname $0)",
        "",
        "echo 'Submitting RIASEC extraction jobs...'",
    ]

    for riasec in ["R", "I", "A", "S", "E", "C"]:
        submit_all.append(f"JOB_{riasec}=$(sbatch --parsable extract_{riasec}.sh)")
        submit_all.append(f'echo "Submitted {riasec}: $JOB_{riasec}"')

    submit_all.extend(
        [
            "",
            "# Submit analysis job with dependency on all extraction jobs",
            'DEPS="afterok:$JOB_R:$JOB_I:$JOB_A:$JOB_S:$JOB_E:$JOB_C"',
            "JOB_ANALYSIS=$(sbatch --parsable --dependency=$DEPS run_analysis.sh)",
            'echo "Submitted analysis (depends on extraction): $JOB_ANALYSIS"',
            "",
            "echo 'All jobs submitted!'",
            "squeue -u $USER",
        ]
    )

    submit_path = output_path / "submit_all.sh"
    submit_path.write_text("\n".join(submit_all))
    submit_path.chmod(0o755)
    created_files.append(submit_path)
    logger.info(f"Created submission script: {submit_path}")

    return created_files
