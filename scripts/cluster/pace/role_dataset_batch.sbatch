#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

ROLE_LIST=("$@")

# echo $ROLES_NL
echo $ROLE_LIST

echo "===CLI ARGS=================="
echo "BACKEND: $BACKEND"
echo "MODEL: $MODEL"
echo "ROLES:"  # role to generate persona vector from
printf '  %s\n' "${ROLE_LIST[@]}"
echo "==========================="

# module load anaconda3
module load uv
source .venv/bin/activate
uv sync

srun uv run python src/pvx/implementations/roles/role_dataset.py \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --roles "${ROLE_LIST[@]}" 

SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    # Clean up output directory
    rm -rf "$OUTPUT_DIR"
    exit "$SLURM_RETURN_CODE"
fi