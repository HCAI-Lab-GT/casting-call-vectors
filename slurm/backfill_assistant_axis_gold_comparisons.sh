#!/bin/bash
#SBATCH --job-name=backfill_assistant_axis_gold_comparisons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/backfill_assistant_axis_gold_comparisons/backfill_assistant_axis_gold_comparisons_%j.out
#SBATCH --error=logs/slurm/backfill_assistant_axis_gold_comparisons/backfill_assistant_axis_gold_comparisons_%j.err

set -euo pipefail

# Wrapper for scripts/evaluation/backfill_assistant_axis_gold_comparisons.py
# Usage:
#   sbatch slurm/backfill_assistant_axis_gold_comparisons.sh [--roles ROLE1 ROLE2 ...] [--overwrite] [--dry_run] [other options]

ROLES=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--roles)
      shift
      ROLES=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ROLES+=("$1")
        shift
      done
      ;;
    --overwrite|--dry_run|--skip_steered_refresh)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --input_dir|--gold_prompts_dir|--backend|--judge_model|--base_url|--api_key_env)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

LOG_DIR="logs/slurm/backfill_assistant_axis_gold_comparisons"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

cd /workspace/personality-vectors

if command -v module >/dev/null 2>&1; then
  module load uv || true
fi
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
uv sync

ROLES_ARG=()
if [[ ${#ROLES[@]} -gt 0 ]]; then
  ROLES_ARG=(--roles "${ROLES[@]}")
fi

srun uv run python scripts/evaluation/backfill_assistant_axis_gold_comparisons.py \
  "${ROLES_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
