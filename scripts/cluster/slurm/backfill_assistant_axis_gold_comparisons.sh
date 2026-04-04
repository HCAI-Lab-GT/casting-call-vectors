#!/bin/bash
source .env

NAME="backfill_assistant_axis_gold_comparisons"

# Usage:
#   ./scripts/cluster/slurm/backfill_assistant_axis_gold_comparisons.sh [--roles ROLE1 ROLE2 ...] [--roles_file path.json]
#     [--judge_model MODEL] [--base_url URL] [--api_key_env VAR]
#     [--max_concurrency N] [--verbose] [--dry_run] [--overwrite] [other options]

# defaults
ROLES=()
ROLES_FILE=""
JUDGE_MODEL="openai/gpt-4.1-mini"
BASE_URL="https://openrouter.ai/api/v1"
API_KEY_ENV="OPENROUTER_API_KEY"
MAX_CONCURRENCY=100
VERBOSE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--roles)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ROLES+=("$1")
        shift
      done
      ;;
    --roles_file)
      ROLES_FILE="$2"
      shift 2
      ;;
    --judge_model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --base_url)
      BASE_URL="$2"
      shift 2
      ;;
    --api_key_env)
      API_KEY_ENV="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --max_concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --dry_run|--overwrite|--overwrite_assistant_axis|--skip_steered_refresh)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --input_dir|--gold_prompts_dir|--backend|--checkpoint_every)
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

export ROLES
export ROLES_FILE
export JUDGE_MODEL
export BASE_URL
export API_KEY_ENV
export MAX_CONCURRENCY
export VERBOSE
export EXTRA_ARGS

### SLURM SETTINGS ###
NUM_CPUS=4
MEM=16G
TIME=08:00:00

ACCOUNT="$PACE_ACCOUNT"
QOS=embers
LOG_DIR="logs/slurm/$NAME"

### VALIDATION ###
if [ -z "$ACCOUNT" ]; then
    echo "Error: PACE_ACCOUNT is not set in .env"
    exit 1
fi

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

### SUBMIT ###
sbatch  --ntasks=1 --cpus-per-task "$NUM_CPUS" --mem "$MEM" \
        --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
        --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
        "scripts/cluster/slurm/$NAME.sbatch" \
        "${ROLES[@]}" \
        ${ROLES_FILE:+--roles_file "$ROLES_FILE"} \
        --judge_model "$JUDGE_MODEL" \
        --base_url "$BASE_URL" \
        --api_key_env "$API_KEY_ENV" \
        --max_concurrency "$MAX_CONCURRENCY" \
        $( [[ "$VERBOSE" == "true" ]] && echo "--verbose" ) \
        "${EXTRA_ARGS[@]}"
