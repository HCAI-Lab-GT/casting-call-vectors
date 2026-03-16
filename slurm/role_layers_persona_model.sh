#!/bin/bash
#SBATCH --job-name=role_layers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/role_layers_second_batch_%j.out
#SBATCH --error=logs/slurm/role_layers_second_batch_%j.err

set -euo pipefail

# Defaults
MODEL="allenai/Olmo-3-7B-Instruct"
ALPHA="1.0"
ROLES=("Actors")
LAYERS=(14)
SAMPLE_COUNTS=(40)

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL="$2"; shift 2
      ;;
    -a|--alpha)
      ALPHA="$2"; shift 2
      ;;
    -r|--roles)
      shift
      ROLES=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ROLES+=("$1")
        shift
      done
      ;;
    -l|--layers)
      shift
      LAYERS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        LAYERS+=("$1")
        shift
      done
      ;;
    -N|--sample_counts|--sample-counts)
      shift
      SAMPLE_COUNTS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        SAMPLE_COUNTS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "MODEL: $MODEL"
echo "ALPHA: $ALPHA"
echo "ROLES: ${ROLES[*]}"
echo "LAYERS: ${LAYERS[*]}"
echo "SAMPLE_COUNTS: ${SAMPLE_COUNTS[*]}"

cd /workspace/personality-vectors

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