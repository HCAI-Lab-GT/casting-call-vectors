#!/bin/bash
source .env

NAME="gold_experiment" # name of the script (mostly for logging purposes)

###
# src/pvx/experiments/gold_prompt_experiments.py
#
# This script is used to run gold prompt experiments for persona steering.
# It calls src/pvx/experiments/gold_prompt_experiments.py with specified parameters.
#
# Usage:
# ./gold_prompt_experiments_batch.sh --roles <role1> <role2> ... --alphas <alpha1> <alpha2> ...
#
# Author: iiisong
# Date: 2026-03-21
###

# ===CLI ARGS==================
# defaults
MODEL="allenai/Olmo-3-7B-Instruct"
ROLES=()
ALPHAS=(1 1.5 2 2.5)
LAYERS=(16)
SAMPLE_COUNTS=(50)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
QUESTIONS_FILE="./configs/validation_questions.jsonl"
SAVE_DIR="./experiment_data/gold_prompt_experiments/"
SAFETENSORS_DIR="./persona_data/model_inits/"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)            MODEL="$2"; shift 2;;
    -r|--roles)            shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ROLES+=("$1"); shift; done;;
    -a|--alphas)           shift; ALPHAS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ALPHAS+=("$1"); shift; done;;
    -l|--layers)           shift; LAYERS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do LAYERS+=("$1"); shift; done;;
    -s|--sample-counts)    shift; SAMPLE_COUNTS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do SAMPLE_COUNTS+=("$1"); shift; done;;
    -t|--temperatures)     shift; TEMPERATURES=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do TEMPERATURES+=("$1"); shift; done;;
    -N|--max-new-tokens)   MAX_NEW_TOKENS="$2"; shift 2;;
    -q|--questions-file)   QUESTIONS_FILE="$2"; shift 2;;
    -d|--save-dir)         SAVE_DIR="$2"; shift 2;;
    --safetensors-dir)     SAFETENSORS_DIR="$2"; shift 2;;
    -g|--gold-prompts-dir) GOLD_PROMPTS_DIR="$2"; shift 2;;
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

# Assert account exists
if [ -z "$ACCOUNT" ]; then
    echo "Error: Account is required for PACE access"
    exit 1
fi

# Assert roles exist
if [[ ${#ROLES[@]} -eq 0 ]]; then
  echo "Error: At least one role is required"
  exit 1
fi

# Creates log dir if not exists
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

### PRE-JOB ###
# None

### JOB CALL (add '-C amd' for amd) ###
sbatch  --gres "$GPU_CONFIG" --ntasks-per-node 1 --cpus-per-task "$NUM_WORKERS" --mem "$MEM_PER_NODE" \
        --account "$ACCOUNT" --time "$TIME" \
        --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
        "scripts/cluster/slurm/$NAME.sbatch" \
        --model "$MODEL" \
        --roles "${ROLES[@]}" \
        --alphas "${ALPHAS[@]}" \
        --layers "${LAYERS[@]}" \
        --sample-counts "${SAMPLE_COUNTS[@]}" \
        --temperatures "${TEMPERATURES[@]}" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --questions-file "$QUESTIONS_FILE" \
        --save-dir "$SAVE_DIR" \
        --safetensors-dir "$SAFETENSORS_DIR" \
        --gold-prompts-dir "$GOLD_PROMPTS_DIR"

### POST JOB ###
# None