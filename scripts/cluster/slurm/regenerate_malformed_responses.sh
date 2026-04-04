#!/bin/bash
#SBATCH --job-name=regenerate_malformed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/regenerate_malformed/regenerate_malformed_%j.out
#SBATCH --error=logs/slurm/regenerate_malformed/regenerate_malformed_%j.err
#SBATCH --account=gts-schava6-fy20phase3
#SBATCH --qos=inferno
#SBATCH --mem 512G

set -euo pipefail

# --gres=gpu:1
# --time=08:00:00


###
# scripts/patch_scripts/regenerate_malformed_responses.py
#
# Regenerates malformed steered and assistant_axis responses
# in Comparison_GoldStandard CSVs.
#
# Usage:
#   sbatch scripts/cluster/slurm/regenerate_malformed_responses.sh [--model MODEL] [--alphas A1 A2 ...] [--columns COL1 COL2 ...] [--roles ROLE1 ROLE2 ...]
###

# Defaults
MODEL="allenai/Olmo-3-7B-Instruct"
ROLES=()
ALPHAS=()
COLUMNS=()

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
    -c|--columns)
      shift
      COLUMNS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        COLUMNS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

LOG_DIR="logs/slurm/regenerate_malformed"

# Creates log dir if not exists
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

echo "MODEL: $MODEL"
echo "ROLES: ${ROLES[*]}"
echo "ALPHAS: ${ALPHAS[*]}"
echo "COLUMNS: ${COLUMNS[*]}"

# cd /workspace/personality-vectors

if command -v module >/dev/null 2>&1; then
  module load uv || true
fi
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

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

# Build optional --columns flag
COLUMNS_ARG=()
if [[ ${#COLUMNS[@]} -gt 0 ]]; then
  COLUMNS_ARG=(--columns "${COLUMNS[@]}")
fi

srun python scripts/patch_scripts/regenerate_malformed_responses.py \
  --model "$MODEL" \
  "${ROLES_ARG[@]}" \
  "${ALPHAS_ARG[@]}" \
  "${COLUMNS_ARG[@]}"
