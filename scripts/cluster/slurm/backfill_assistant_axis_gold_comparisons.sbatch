#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

# Parse --roles and pass remaining args through as extra args
ROLES=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run|--overwrite|--overwrite_assistant_axis|--skip_steered_refresh|--verbose)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --input_dir|--gold_prompts_dir|--backend|--judge_model|--base_url|--api_key_env|--max_concurrency|--checkpoint_every|--roles_file)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      ROLES+=("$1")
      shift
      ;;
  esac
done

echo "===CLI ARGS=================="
echo "ROLES:"
printf '  %s\n' "${ROLES[@]}"
echo "EXTRA_ARGS: ${EXTRA_ARGS[*]}"
echo "==========================="

module load uv
source .venv/bin/activate

ROLES_ARG=()
if [[ ${#ROLES[@]} -gt 0 ]]; then
  ROLES_ARG=(--roles "${ROLES[@]}")
fi

srun uv run python scripts/patch_scripts/backfill_assistant_axis_gold_comparisons.py \
  "${ROLES_ARG[@]}" \
  "${EXTRA_ARGS[@]}"

SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    exit "$SLURM_RETURN_CODE"
fi
