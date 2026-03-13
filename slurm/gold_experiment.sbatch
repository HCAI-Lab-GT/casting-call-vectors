#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

# ===CLI ARGS==================
# MODEL=""
# ROLES=()
# LAYERS=()

# Parse args: roles first, optional --model and --layers anywhere

# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     -m|--model)     MODEL="$2"; shift 2;;
#     -r|--roles)     shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ROLES+=("$1"); shift; done;;
#     -l|--layers)    shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do LAYERS+=("$1"); shift; done;;
#     *)              echo "Unknown argument: $1" >&2; exit 1;;
#   esac
# done

echo "===CLI ARGS=================="
echo "MODEL: $MODEL"
# echo "ROLES:"  # role to generate persona vector from
# printf '  %s\n' "${ROLES[@]}"
# echo "LAYERS:"  # layer to extract activations from
# printf '  %s\n' "${LAYERS[@]}"
echo "==========================="

# module load anaconda3
module load uv
source .venv/bin/activate
uv sync

srun uv run python src/pvx/experiments/gold_prompt_experiments.py \
        --model "$MODEL" \
        --roles "Lawyer" "Actor" "Chemist" \
        --layers 1 5 10 14 20 \
        --alphas 1 2 3 4 5
        # --roles "${ROLE_LIST[@]}"
        # --layers "${LAYERS[@]}"
        
SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    # Clean up output directory
    rm -rf "$OUTPUT_DIR"
    exit "$SLURM_RETURN_CODE"
fi