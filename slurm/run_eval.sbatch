#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

echo "===CLI ARGS=================="
echo "RUN_NAME: $RUN_NAME"  # name of run
echo "==========================="

# module load anaconda3
module load uv
source .venv/bin/activate
uv sync

srun uv run python scripts/run_eval.py \
        --run "$RUN_NAME" \
        --no-display;
SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    # Clean up output directory
    rm -rf "$OUTPUT_DIR"
    exit "$SLURM_RETURN_CODE"
fi