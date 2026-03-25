#!/bin/bash
#SBATCH --job-name=regenerate_assistant_axis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/regenerate_assistant_axis/regenerate_assistant_axis_%j.out
#SBATCH --error=logs/slurm/regenerate_assistant_axis/regenerate_assistant_axis_%j.err

set -euo pipefail

###
# scripts/evaluation/regenerate_assistant_axis_by_alpha.py
#
# Regenerates assistant_axis responses in Comparison_GoldStandard CSVs
# using each row's alpha value (matching steered alpha).
#
# Usage:
#   sbatch slurm/regenerate_assistant_axis.sh [--model MODEL] [--alphas A1 A2 ...] [--roles ROLE1 ROLE2 ...]
#
# Author: iiisong
###

# Defaults
MODEL="allenai/Olmo-3-7B-Instruct"
ROLES=()
ALPHAS=()

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL="$2"; shift 2
      ;;
    -r|--roles)
      shift
      ROLES=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ROLES+=("$1")
        shift
      done
      ;;
    -a|--alphas)
      shift
      ALPHAS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ALPHAS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

LOG_DIR="logs/slurm/regenerate_assistant_axis"

# Creates log dir if not exists
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

echo "MODEL: $MODEL"
echo "ROLES: ${ROLES[*]}"
echo "ALPHAS: ${ALPHAS[*]}"

cd /workspace/personality-vectors

if command -v module >/dev/null 2>&1; then
  module load uv || true
fi
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

uv sync

# Build optional --alphas flag
ALPHAS_ARG=()
if [[ ${#ALPHAS[@]} -gt 0 ]]; then
  ALPHAS_ARG=(--alphas "${ALPHAS[@]}")
fi

# Build optional --roles flag
ROLES_ARG=()
if [[ ${#ROLES[@]} -gt 0 ]]; then
  ROLES_ARG=(--roles "${ROLES[@]}")
fi

srun uv run python scripts/evaluation/regenerate_assistant_axis_by_alpha.py \
  --model "$MODEL" \
  "${ROLES_ARG[@]}" \
  "${ALPHAS_ARG[@]}"
