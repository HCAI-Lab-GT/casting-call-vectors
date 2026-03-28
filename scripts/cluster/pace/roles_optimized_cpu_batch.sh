#!/bin/bash
source .env

NAME="roles_optimized_cpu_batch"

###
# Submit a CPU-only SLURM job to generate Q/A responses for a single role
# via src/pvx/implementations/roles_optimized/roles_optimized_cpu.py.
#
# On successful completion (Q/A JSON exists), the sbatch automatically
# chains to roles_optimized_gpu_batch.sh for persona vector extraction.
#
# Usage:
#   bash scripts/cluster/pace/roles_optimized_cpu_batch.sh --role <role> --model <model> \
#       --layer <layer> --missing-counts <c1> [c2 ...] [options...]
#
# Author: iiisong
# Date: 2026-03-23
###

# ===CLI ARGS==================
# defaults
ROLE=""
MODEL="allenai/Olmo-3-7B-Instruct"
LAYER=16
MISSING_COUNTS=()
BIN_CONFIG=""  # JSON file with [{role, layer, missing_counts}, ...]; overrides ROLE/LAYER/MISSING_COUNTS
SAFETENSORS_DIR="./persona_data/model_layer_inits/"
QA_RESPONSES_DIR="./persona_data/model_qa_responses"
DATASET_DIR="./persona_data/role_datasets/"
BACKEND="openai"
BASE_URL="https://openrouter.ai/api/v1"
API_KEY_ENV="OPENROUTER_API_KEY"
# Gold experiment params (forwarded through to GPU job)
GOLD_SAVE_DIR="./experiment_data/gold_prompt_experiments/"
ALPHAS=(1 1.5 2 2.5)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
QUESTIONS_FILE="./configs/validation_questions.jsonl"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"
EXPECTED_ROWS=228
NO_CHAIN=false  # if true, skip gold experiment chaining (propagated through to GPU phase)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)                ROLE="$2"; shift 2;;
    -m|--model)            MODEL="$2"; shift 2;;
    -l|--layer)            LAYER="$2"; shift 2;;
    --missing-counts)      shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do MISSING_COUNTS+=("$1"); shift; done;;
    --safetensors-dir)     SAFETENSORS_DIR="$2"; shift 2;;
    --qa-responses-dir)    QA_RESPONSES_DIR="$2"; shift 2;;
    --dataset-dir)         DATASET_DIR="$2"; shift 2;;
    --backend)             BACKEND="$2"; shift 2;;
    --base-url)            BASE_URL="$2"; shift 2;;
    --api-key-env)         API_KEY_ENV="$2"; shift 2;;
    --gold-save-dir)       GOLD_SAVE_DIR="$2"; shift 2;;
    -a|--alphas)           shift; ALPHAS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ALPHAS+=("$1"); shift; done;;
    -t|--temperatures)     shift; TEMPERATURES=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do TEMPERATURES+=("$1"); shift; done;;
    -N|--max-new-tokens)   MAX_NEW_TOKENS="$2"; shift 2;;
    -q|--questions-file)   QUESTIONS_FILE="$2"; shift 2;;
    -g|--gold-prompts-dir) GOLD_PROMPTS_DIR="$2"; shift 2;;
    --expected-rows)       EXPECTED_ROWS="$2"; shift 2;;
    --bin-config)          BIN_CONFIG="$2"; shift 2;;
    --no-chain)            NO_CHAIN=true; shift;;
    *)                     echo "Unknown argument: $1" >&2; exit 1;;
  esac
done
# ===========================

### SLURM SETTINGS ###
# CPU-only job: no GPU needed for OpenRouter/OpenAI backend
NUM_NODES=1
NUM_WORKERS=8
MEM_PER_NODE=32G
TIME=08:00:00

# job settings (typically do not change)
ACCOUNT="$PACE_ACCOUNT" # set from .env
QOS=embers
LOG_DIR="logs/slurm/$NAME"

### CLI ARGS VALIDATION ###
if [ -z "$ACCOUNT" ]; then
    echo "Error: Account is required for PACE access"
    exit 1
fi
if [ -z "$BIN_CONFIG" ]; then
    if [ -z "$ROLE" ]; then
        echo "Error: --role is required (or use --bin-config for multi-role bin mode)"
        exit 1
    fi
    if [[ ${#MISSING_COUNTS[@]} -eq 0 ]]; then
        echo "Error: --missing-counts requires at least one value"
        exit 1
    fi
elif [ ! -f "$BIN_CONFIG" ]; then
    echo "Error: --bin-config file not found: $BIN_CONFIG"
    exit 1
fi

# Creates log dir if not exists
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

### PRE-JOB ###
# None

### JOB CALL (CPU-only, no --gres) ###
if [ -n "$BIN_CONFIG" ]; then
    # Bin mode: pass config file; TARGET_PAIRS computed inside the sbatch
    sbatch  --ntasks-per-node 1 --cpus-per-task "$NUM_WORKERS" --mem "$MEM_PER_NODE" \
            --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
            --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
            "scripts/cluster/pace/$NAME.sbatch" \
            --model "$MODEL" \
            --bin-config "$BIN_CONFIG" \
            --safetensors-dir "$SAFETENSORS_DIR" \
            --qa-responses-dir "$QA_RESPONSES_DIR" \
            --dataset-dir "$DATASET_DIR" \
            --backend "$BACKEND" \
            --base-url "$BASE_URL" \
            --api-key-env "$API_KEY_ENV" \
            --gold-save-dir "$GOLD_SAVE_DIR" \
            --alphas "${ALPHAS[@]}" \
            --temperatures "${TEMPERATURES[@]}" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --questions-file "$QUESTIONS_FILE" \
            --gold-prompts-dir "$GOLD_PROMPTS_DIR" \
            --expected-rows "$EXPECTED_ROWS" \
            $( [[ "$NO_CHAIN" == "true" ]] && echo "--no-chain" )
else
    # Single-role mode (default)
    # target_pairs = max of missing counts (generate enough Q/A for the largest needed extraction)
    TARGET_PAIRS=$(printf '%s\n' "${MISSING_COUNTS[@]}" | sort -n | tail -1)

    sbatch  --ntasks-per-node 1 --cpus-per-task "$NUM_WORKERS" --mem "$MEM_PER_NODE" \
            --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
            --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
            "scripts/cluster/pace/$NAME.sbatch" \
            --role "$ROLE" \
            --model "$MODEL" \
            --layer "$LAYER" \
            --missing-counts "${MISSING_COUNTS[@]}" \
            --target-pairs "$TARGET_PAIRS" \
            --safetensors-dir "$SAFETENSORS_DIR" \
            --qa-responses-dir "$QA_RESPONSES_DIR" \
            --dataset-dir "$DATASET_DIR" \
            --backend "$BACKEND" \
            --base-url "$BASE_URL" \
            --api-key-env "$API_KEY_ENV" \
            --gold-save-dir "$GOLD_SAVE_DIR" \
            --alphas "${ALPHAS[@]}" \
            --temperatures "${TEMPERATURES[@]}" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --questions-file "$QUESTIONS_FILE" \
            --gold-prompts-dir "$GOLD_PROMPTS_DIR" \
            --expected-rows "$EXPECTED_ROWS" \
            $( [[ "$NO_CHAIN" == "true" ]] && echo "--no-chain" )
fi

### POST JOB ###
# None (chaining handled inside the sbatch on success)
