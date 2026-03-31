#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

# ===CLI ARGS==================
MODEL=""
ALPHA=""
ROLES=()
LAYERS=()
SAMPLE_COUNTS=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2;;
    -a|--alpha) ALPHA="$2"; shift 2;;
    -r|--roles) shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ROLES+=("$1"); shift; done;;
    -l|--layers) shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do LAYERS+=("$1"); shift; done;;
    -N|--sample_counts|--sample-counts) shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do SAMPLE_COUNTS+=("$1"); shift; done;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
  
done

echo "MODEL: $MODEL"
echo "ALPHA: $ALPHA"
echo "ROLES: ${ROLES[*]}"
echo "LAYERS: ${LAYERS[*]}"
echo "SAMPLE_COUNTS: ${SAMPLE_COUNTS[*]}"

# Optional env setup
if command -v module >/dev/null 2>&1; then
  module load uv || true
fi
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
uv sync

srun uv run python src/pvx/implementations/roles_layers/role_layers_persona_model.py \
  --model "$MODEL" \
  --alpha "$ALPHA" \
  --roles "${ROLES[@]}" \
  --layers "${LAYERS[@]}" \
  --sample_counts "${SAMPLE_COUNTS[@]}"

SLURM_RETURN_CODE=$?
if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    exit "$SLURM_RETURN_CODE"
fi
