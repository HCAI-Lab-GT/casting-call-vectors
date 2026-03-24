#!/bin/bash
source .env

NAME="create_gold_standard_prompts_cpu"

# Defaults
BACKEND="openai"
MODEL="openai/gpt-4.1-mini"
NUM_DIALOGUE_PROMPTS=5
TEMPERATURE=0.7
MAX_NEW_TOKENS=2048
OUTPUT_DIR=""
ROLE=""
PARTITION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--role)
      ROLE="$2"
      shift 2
      ;;
    -b|--backend)
      BACKEND="$2"
      shift 2
      ;;
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    --num-dialogue-prompts)
      NUM_DIALOGUE_PROMPTS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    *)
      if [ -z "$ROLE" ]; then
        ROLE="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [ -z "$ROLE" ]; then
  echo "Usage: ./slurm/create_gold_standard_prompts_cpu.sh --role <role_name>"
  echo "   or: ./slurm/create_gold_standard_prompts_cpu.sh <role_name>"
  exit 1
fi

ACCOUNT="${PACE_ACCOUNT:-}"
QOS=""
LOG_DIR="slurm/create_gold_standard_prompts"

if [ -z "$ACCOUNT" ]; then
  echo "Error: PACE_ACCOUNT is not set (source .env)."
  exit 1
fi

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  --account "$ACCOUNT"
  --gres "none"
  "slurm/create_gold_standard_prompts_cpu.sbatch"
  --role "$ROLE"
  --backend "$BACKEND"
  --model "$MODEL"
  --num-dialogue-prompts "$NUM_DIALOGUE_PROMPTS"
  --temperature "$TEMPERATURE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [ -n "$QOS" ]; then
  SBATCH_ARGS+=(--qos "$QOS")
fi

if [ -n "$PARTITION" ]; then
  SBATCH_ARGS+=(--partition "$PARTITION")
fi

if [ -n "$OUTPUT_DIR" ]; then
  SBATCH_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

sbatch "${SBATCH_ARGS[@]}"
