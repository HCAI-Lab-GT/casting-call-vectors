#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

echo "===CLI ARGS=================="
echo "ROLE: $ROLE"  # role to generate role vector from
echo "LOCAL_MODEL: $LOCAL_MODEL"
echo "==========================="

# module load anaconda3
module load uv
source .venv/bin/activate
uv sync

srun uv run python src/pvx/implementations/roles/role_persona_model.py \
        --role "$ROLE" \
        --model_name "$LOCAL_MODEL";
SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    # Clean up output directory
    rm -rf "$OUTPUT_DIR"
    exit "$SLURM_RETURN_CODE"
fi