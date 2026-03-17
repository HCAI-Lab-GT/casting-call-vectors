#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

# ===CLI ARGS==================
MODEL=""
TRAITS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)     MODEL="$2"; shift 2;;
    -t|--traits)    shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do TRAITS+=("$1"); shift; done;;
    *)              echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

echo "===CLI ARGS=================="
echo "MODEL: $MODEL"
echo "TRAITS:"  # traits to generate trait coverage vectors from
printf '  %s\n' "${TRAITS[@]}"
echo "==========================="

# module load anaconda3
module load uv
source .venv/bin/activate
uv sync

srun uv run python src/pvx/implementations/trait_coverage/trait_coverage_persona_model.py \
        --model "$MODEL" \
        --traits "${TRAITS[@]}"

SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    exit "$SLURM_RETURN_CODE"
fi
