#!/bin/bash
source .env

###
# Reads incomplete role inits from a JSON file (e.g. configs/role_init_incomplete.json),
# re-verifies which entries are still incomplete by checking for the expected safetensors
# files, and submits a CPU job for each remaining role.
#
# Each CPU job chains automatically:
#   CPU (Q/A generation) → GPU (init extraction per count) → gold_experiment.sh (if needed)
#
# Usage:
#   bash slurm/roles_optimized_autobatch.sh [-f <json_file>] [options...]
#
# Options:
#   -f, --file              Path to incomplete inits JSON (default: configs/role_init_incomplete.json)
#   -m, --model             Model ID override (default: reads per-entry from JSON)
#   --safetensors-dir       Dir to check/write safetensors (default: ./persona_data/model_layer_inits/)
#   --qa-responses-dir      Dir for generated Q/A JSON (default: ./persona_data/model_qa_responses)
#   --dataset-dir           Role dataset directory (default: ./persona_data/role_datasets/)
#   --backend               API backend for CPU phase (default: openai)
#   --base-url              API base URL (default: https://openrouter.ai/api/v1)
#   --api-key-env           Env var name for API key (default: OPENROUTER_API_KEY)
#   --gold-save-dir         Gold experiment results dir (default: ./experiment_data/gold_prompt_experiments/)
#   -a, --alphas            Alphas for gold experiment (default: 1 1.5 2 2.5)
#   -t, --temperatures      Temperatures for gold experiment (default: 0.2)
#   -N, --max-new-tokens    Max tokens for gold experiment (default: 2000)
#   -q, --questions-file    Questions file for gold experiment
#   -g, --gold-prompts-dir  Gold prompts dataset dir
#   --expected-rows         Expected CSV rows per combination for completion check (default: 228)
#
# Author: iiisong
# Date: 2026-03-23
###

# ===CLI ARGS==================
# defaults
JSON_FILE="configs/role_init_incomplete.json"
MODEL=""   # empty = use model_id from each JSON entry
SAFETENSORS_DIR="./persona_data/model_layer_inits/"
QA_RESPONSES_DIR="./persona_data/model_qa_responses"
DATASET_DIR="./persona_data/role_datasets/"
BACKEND="openai"
BASE_URL="https://openrouter.ai/api/v1"
API_KEY_ENV="OPENROUTER_API_KEY"
# Gold experiment defaults (forwarded through the chain)
GOLD_SAVE_DIR="./experiment_data/gold_prompt_experiments/"
ALPHAS=(1 1.5 2 2.5)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
QUESTIONS_FILE="./configs/validation_questions.jsonl"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"
EXPECTED_ROWS=228

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)             JSON_FILE="$2"; shift 2;;
    -m|--model)            MODEL="$2"; shift 2;;
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
    *)                     echo "Unknown argument: $1" >&2; exit 1;;
  esac
done
# ===========================

### VALIDATION ###
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found: $JSON_FILE"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required but not found in PATH."
    exit 1
fi

TOTAL=$(jq 'length' "$JSON_FILE")
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No entries found in $JSON_FILE. Nothing to do."
    exit 0
fi

echo "Found $TOTAL entries in $JSON_FILE. Verifying which are still incomplete..."

submitted=0
skipped=0

for ((i=0; i<TOTAL; i++)); do
    role=$(jq -r ".[$i].role" "$JSON_FILE")
    model_id=$(jq -r ".[$i].model_id" "$JSON_FILE")
    layer=$(jq -r ".[$i].layer" "$JSON_FILE")
    readarray -t missing_counts < <(jq -r ".[$i].missing_counts[]" "$JSON_FILE")

    # Allow --model to override the per-entry model_id
    effective_model="${MODEL:-$model_id}"
    SAFE_MODEL="${effective_model//\//__}"

    # Re-verify: check which counts are still actually missing on disk
    still_missing=()
    for count in "${missing_counts[@]}"; do
        expected="${SAFETENSORS_DIR}/${role}_persona_initialization/${SAFE_MODEL}_layer${layer}_count${count}.safetensors"
        if [ ! -f "$expected" ]; then
            still_missing+=("$count")
        fi
    done

    if [[ ${#still_missing[@]} -eq 0 ]]; then
        echo "  Skipping $role: all counts (${missing_counts[*]}) already present on disk"
        ((skipped++))
        continue
    fi

    echo "  Submitting CPU job for role=$role model=$effective_model layer=$layer missing_counts=(${still_missing[*]})"
    bash slurm/roles_optimized_cpu_batch.sh \
        --role "$role" \
        --model "$effective_model" \
        --layer "$layer" \
        --missing-counts "${still_missing[@]}" \
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
        --expected-rows "$EXPECTED_ROWS"

    ((submitted++))
done

echo ""
echo "Done. Submitted: $submitted CPU jobs, Skipped: $skipped (already complete)."
