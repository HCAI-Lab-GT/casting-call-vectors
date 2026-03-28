#!/bin/bash
source .env

NAME="roles_optimized_gpu_batch"

###
# Submit a GPU SLURM job to extract persona vectors (model inits) for a single
# role using src/pvx/implementations/roles_optimized/roles_optimized_gpu.py,
# once per missing sample count.
#
# On successful extraction of each count's safetensors file, the sbatch
# automatically submits slurm/gold_experiment.sh for that (role, count) pairing
# if it has not yet been fully run (checked via CSV completion logic from
# gold_experiment_autobatch.sh).
#
# Usage:
#   bash slurm/roles_optimized_gpu_batch.sh --role <role> --model <model> \
#       --layer <layer> --counts <c1> [c2 ...] [options...]
#
# Author: iiisong
# Date: 2026-03-23
###

# ===CLI ARGS==================
# defaults
ROLE=""
MODEL="allenai/Olmo-3-7B-Instruct"
ANSWER_MODEL=""  # model used for Q/A generation; defaults to MODEL if empty
LAYER=16
COUNTS=()
SAFETENSORS_DIR="./persona_data/model_layer_inits/"
QA_RESPONSES_DIR="./persona_data/model_qa_responses"
DATASET_DIR="./persona_data/role_datasets/"
# Gold experiment defaults
GOLD_SAVE_DIR="./experiment_data/gold_prompt_experiments/"
ALPHAS=(1 1.5 2 2.5)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
QUESTIONS_FILE="./configs/validation_questions.jsonl"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"
EXPECTED_ROWS=228
NO_CHAIN=false  # if true, skip gold experiment chaining after GPU extraction

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)                ROLE="$2"; shift 2;;
    -m|--model)            MODEL="$2"; shift 2;;
    --answer-model)        ANSWER_MODEL="$2"; shift 2;;
    -l|--layer)            LAYER="$2"; shift 2;;
    --counts)              shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do COUNTS+=("$1"); shift; done;;
    --safetensors-dir)     SAFETENSORS_DIR="$2"; shift 2;;
    --qa-responses-dir)    QA_RESPONSES_DIR="$2"; shift 2;;
    --dataset-dir)         DATASET_DIR="$2"; shift 2;;
    --gold-save-dir)       GOLD_SAVE_DIR="$2"; shift 2;;
    -a|--alphas)           shift; ALPHAS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ALPHAS+=("$1"); shift; done;;
    -t|--temperatures)     shift; TEMPERATURES=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do TEMPERATURES+=("$1"); shift; done;;
    -N|--max-new-tokens)   MAX_NEW_TOKENS="$2"; shift 2;;
    -q|--questions-file)   QUESTIONS_FILE="$2"; shift 2;;
    -g|--gold-prompts-dir) GOLD_PROMPTS_DIR="$2"; shift 2;;
    --expected-rows)       EXPECTED_ROWS="$2"; shift 2;;
    --no-chain)            NO_CHAIN=true; shift;;
    *)                     echo "Unknown argument: $1" >&2; exit 1;;
  esac
done
# ===========================

### SLURM SETTINGS ###
# runtime settings
GPU_CONFIG=gpu:1
NUM_NODES=1
NUM_WORKERS=8
MEM_PER_NODE=512G
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
if [ -z "$ROLE" ]; then
    echo "Error: --role is required"
    exit 1
fi
if [[ ${#COUNTS[@]} -eq 0 ]]; then
    echo "Error: --counts requires at least one value"
    exit 1
fi

# Creates log dir if not exists
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

### PRE-JOB ###
# None

### JOB CALL ###
sbatch  --gres "$GPU_CONFIG" --ntasks-per-node 1 --cpus-per-task "$NUM_WORKERS" --mem "$MEM_PER_NODE" \
        --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
        --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
        "slurm/$NAME.sbatch" \
        --role "$ROLE" \
        --model "$MODEL" \
        $( [[ -n "$ANSWER_MODEL" ]] && echo "--answer-model $ANSWER_MODEL" ) \
        --layer "$LAYER" \
        --counts "${COUNTS[@]}" \
        --safetensors-dir "$SAFETENSORS_DIR" \
        --qa-responses-dir "$QA_RESPONSES_DIR" \
        --dataset-dir "$DATASET_DIR" \
        --gold-save-dir "$GOLD_SAVE_DIR" \
        --alphas "${ALPHAS[@]}" \
        --temperatures "${TEMPERATURES[@]}" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --questions-file "$QUESTIONS_FILE" \
        --gold-prompts-dir "$GOLD_PROMPTS_DIR" \
        --expected-rows "$EXPECTED_ROWS" \
        $( [[ "$NO_CHAIN" == "true" ]] && echo "--no-chain" )

### POST JOB ###
# None (chaining handled inside the sbatch on success)
