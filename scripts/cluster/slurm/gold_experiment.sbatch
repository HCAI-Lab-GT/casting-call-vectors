#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

# ===CLI ARGS==================
MODEL=""
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

echo "===CLI ARGS=================="
echo "MODEL: $MODEL"
echo "ROLES:"
printf '  %s\n' "${ROLES[@]}"
echo "ALPHAS: ${ALPHAS[*]}"
echo "LAYERS: ${LAYERS[*]}"
echo "SAMPLE_COUNTS: ${SAMPLE_COUNTS[*]}"
echo "TEMPERATURES: ${TEMPERATURES[*]}"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "QUESTIONS_FILE: $QUESTIONS_FILE"
echo "SAVE_DIR: $SAVE_DIR"
echo "SAFETENSORS_DIR: $SAFETENSORS_DIR"
echo "GOLD_PROMPTS_DIR: $GOLD_PROMPTS_DIR"
echo "==========================="

# module load anaconda3
module load uv

if [ ! -d ".venv" ]; then
    uv sync
fi

source .venv/bin/activate

srun uv run --no-sync python src/pvx/experiments/gold_prompt_experiments.py \
        --model "$MODEL" \
        --roles "${ROLES[@]}" \
        --alphas "${ALPHAS[@]}" \
        --layers "${LAYERS[@]}" \
        --sample_counts "${SAMPLE_COUNTS[@]}" \
        --temperatures "${TEMPERATURES[@]}" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --questions_file "$QUESTIONS_FILE" \
        --save_dir "$SAVE_DIR" \
        --safetensors_dir "$SAFETENSORS_DIR" \
        --gold_prompts_dir "$GOLD_PROMPTS_DIR"

SLURM_RETURN_CODE=$?

if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
    echo "Error: failed with exit code $SLURM_RETURN_CODE"
    exit "$SLURM_RETURN_CODE"
fi