#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

ROLE_LIST=("$@")

echo "===CLI ARGS=================="
echo "MODEL: $MODEL"
echo "ALPHAS: $ALPHAS_STR"
echo "ROLES:"
printf '  %s\n' "${ROLE_LIST[@]}"
echo "==========================="

module load uv
source .venv/bin/activate
uv sync

# Build optional --alphas flag
ALPHAS_ARG=()
if [[ -n "$ALPHAS_STR" ]]; then
    read -ra ALPHAS_ARG <<< "--alphas $ALPHAS_STR"
fi

srun uv run python scripts/evaluation/regenerate_assistant_axis_by_alpha.py \
        --model "$MODEL" \
        "${ALPHAS_ARG[@]}" \
        --roles "${ROLE_LIST[@]}"

SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    exit "$SLURM_RETURN_CODE"
fi
