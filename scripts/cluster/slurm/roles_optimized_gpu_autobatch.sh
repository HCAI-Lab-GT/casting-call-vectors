#!/bin/bash
source .env

###
# Reads incomplete role inits from a JSON file (e.g. configs/role_init_incomplete.json),
# re-verifies which entries are still incomplete by checking for the expected safetensors
# files AND the required Q/A responses, and submits a GPU job for each remaining role.
#
# Unlike roles_optimized_autobatch.sh, this script submits GPU jobs directly without
# a preceding CPU phase. Roles missing Q/A responses are skipped (run CPU phase first).
#
# Each GPU job optionally chains to gold_experiment.sh on success. Pass --no-chain
# to disable this (e.g. during testing when the full pipeline is not needed).
#
# Usage:
#   bash scripts/cluster/slurm/roles_optimized_gpu_autobatch.sh [-f <json_file>] [options...]
#
# Options:
#   -f, --file              Path to incomplete inits JSON (default: configs/role_init_incomplete.json)
#   -m, --model             Model ID override (default: reads per-entry from JSON)
#   --safetensors-dir       Dir to check/write safetensors (default: ./persona_data/model_inits/)
#   --qa-responses-dir      Dir for Q/A responses JSON (default: ./persona_data/model_qa_responses)
#   --dataset-dir           Role dataset directory (default: ./persona_data/role_datasets/)
#   --no-chain              Disable gold experiment chaining after GPU extraction (e.g. for testing)
#   --gold-save-dir         Gold experiment results dir (default: ./experiment_data/gold_prompt_experiments/)
#   -a, --alphas            Alphas for gold experiment (default: 1 1.5 2 2.5)
#   -t, --temperatures      Temperatures for gold experiment (default: 0.2)
#   -N, --max-new-tokens    Max tokens for gold experiment (default: 2000)
#   -q, --questions-file    Questions file for gold experiment
#   -g, --gold-prompts-dir  Gold prompts dataset dir
#   --expected-rows         Expected CSV rows per combination for completion check (default: 228)
#   -b, --num-bins          Number of SLURM jobs to group roles into (default: 0 = one job per role).
#                           Each bin processes its roles sequentially.
#
# Author: iiisong
# Date: 2026-03-23
###

# ===CLI ARGS==================
# defaults
JSON_FILE="configs/role_init_incomplete.json"
MODEL=""        # empty = use model_id from each JSON entry
ANSWER_MODEL="" # model used for Q/A generation; defaults to MODEL if empty
SAFETENSORS_DIR="./persona_data/model_inits/"
QA_RESPONSES_DIR="./persona_data/model_qa_responses"
DATASET_DIR="./persona_data/role_datasets/"
NO_CHAIN=false
# Gold experiment defaults (forwarded to GPU batch)
GOLD_SAVE_DIR="./experiment_data/gold_prompt_experiments/"
ALPHAS=(1 1.5 2 2.5)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
QUESTIONS_FILE="./configs/validation_questions.jsonl"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"
EXPECTED_ROWS=228
NUM_BINS=0  # 0 = one job per role (default); N > 0 = group into N bins

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)             JSON_FILE="$2"; shift 2;;
    -m|--model)            MODEL="$2"; shift 2;;
    --answer-model)        ANSWER_MODEL="$2"; shift 2;;
    --safetensors-dir)     SAFETENSORS_DIR="$2"; shift 2;;
    --qa-responses-dir)    QA_RESPONSES_DIR="$2"; shift 2;;
    --dataset-dir)         DATASET_DIR="$2"; shift 2;;
    --no-chain)            NO_CHAIN=true; shift;;
    --gold-save-dir)       GOLD_SAVE_DIR="$2"; shift 2;;
    -a|--alphas)           shift; ALPHAS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ALPHAS+=("$1"); shift; done;;
    -t|--temperatures)     shift; TEMPERATURES=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do TEMPERATURES+=("$1"); shift; done;;
    -N|--max-new-tokens)   MAX_NEW_TOKENS="$2"; shift 2;;
    -q|--questions-file)   QUESTIONS_FILE="$2"; shift 2;;
    -g|--gold-prompts-dir) GOLD_PROMPTS_DIR="$2"; shift 2;;
    --expected-rows)       EXPECTED_ROWS="$2"; shift 2;;
    -b|--num-bins)         NUM_BINS="$2"; shift 2;;
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
skipped_no_qa=0

# Collect all still-incomplete entries before submission
declare -a PENDING_ROLES
declare -a PENDING_MODELS
declare -a PENDING_ANSWER_MODELS
declare -a PENDING_LAYERS
declare -a PENDING_MISSING  # each element is a space-joined string of missing counts

for ((i=0; i<TOTAL; i++)); do
    role=$(jq -r ".[$i].role" "$JSON_FILE")
    model_id=$(jq -r ".[$i].model_id" "$JSON_FILE")
    layer=$(jq -r ".[$i].layer" "$JSON_FILE")
    readarray -t counts < <(jq -r ".[$i].missing_counts[]" "$JSON_FILE")

    # Allow --model to override the per-entry model_id
    effective_model="${MODEL:-$model_id}"
    SAFE_MODEL="${effective_model//\//__}"
    effective_answer_model="${ANSWER_MODEL:-$effective_model}"
    SAFE_ANSWER_MODEL="${effective_answer_model//\//__}"

    # Check if Q/A responses exist for this role (required for GPU extraction)
    QA_PATH="${QA_RESPONSES_DIR}/${SAFE_ANSWER_MODEL}/${role}.json"
    if [ ! -f "$QA_PATH" ]; then
        echo "  Skipping $role: Q/A responses not found at $QA_PATH (run CPU phase first)"
        ((skipped_no_qa++))
        continue
    fi

    # Re-verify: check which counts are still actually missing on disk
    still_missing=()
    for count in "${counts[@]}"; do
        expected="${SAFETENSORS_DIR}/${role}_persona_initialization/${SAFE_MODEL}_layer${layer}_count${count}.safetensors"
        if [ ! -f "$expected" ]; then
            still_missing+=("$count")
        fi
    done

    if [[ ${#still_missing[@]} -eq 0 ]]; then
        echo "  Skipping $role: all counts (${counts[*]}) already present on disk"
        ((skipped++))
        continue
    fi

    PENDING_ROLES+=("$role")
    PENDING_MODELS+=("$effective_model")
    PENDING_ANSWER_MODELS+=("$effective_answer_model")
    PENDING_LAYERS+=("$layer")
    PENDING_MISSING+=("${still_missing[*]}")
done

TOTAL_PENDING=${#PENDING_ROLES[@]}

# ===SUBMIT JOBS==================
# Common args forwarded to every gpu_batch call
COMMON_ARGS=(
    --safetensors-dir "$SAFETENSORS_DIR"
    --qa-responses-dir "$QA_RESPONSES_DIR"
    --dataset-dir "$DATASET_DIR"
    --gold-save-dir "$GOLD_SAVE_DIR"
    --alphas "${ALPHAS[@]}"
    --temperatures "${TEMPERATURES[@]}"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --questions-file "$QUESTIONS_FILE"
    --gold-prompts-dir "$GOLD_PROMPTS_DIR"
    --expected-rows "$EXPECTED_ROWS"
)
[[ "$NO_CHAIN" == "true" ]] && COMMON_ARGS+=(--no-chain)

if [[ "$NUM_BINS" -le 0 || "$TOTAL_PENDING" -eq 0 ]]; then
    # Default: one job per role
    for ((j=0; j<TOTAL_PENDING; j++)); do
        role="${PENDING_ROLES[$j]}"
        effective_model="${PENDING_MODELS[$j]}"
        effective_answer_model="${PENDING_ANSWER_MODELS[$j]}"
        layer="${PENDING_LAYERS[$j]}"
        read -ra still_missing <<< "${PENDING_MISSING[$j]}"

        echo "  Submitting GPU job for role=$role model=$effective_model answer_model=$effective_answer_model layer=$layer counts=(${still_missing[*]})"
        bash scripts/cluster/slurm/roles_optimized_gpu_batch.sh \
            --role "$role" \
            --model "$effective_model" \
            --answer-model "$effective_answer_model" \
            --layer "$layer" \
            --counts "${still_missing[@]}" \
            "${COMMON_ARGS[@]}"

        ((submitted++))
    done
else
    # Bin mode: group roles into NUM_BINS jobs so each job processes its roles sequentially
    bin_size=$(( (TOTAL_PENDING + NUM_BINS - 1) / NUM_BINS ))
    echo "Bin mode: $TOTAL_PENDING pending roles → $NUM_BINS bins (~$bin_size roles/bin)"

    for ((bin=0; bin<NUM_BINS; bin++)); do
        start=$((bin * bin_size))
        [[ $start -ge $TOTAL_PENDING ]] && break
        end=$((start + bin_size))
        [[ $end -gt $TOTAL_PENDING ]] && end=$TOTAL_PENDING

        # Submit one GPU job per role in this bin (GPU jobs are independent)
        for ((j=start; j<end; j++)); do
            role="${PENDING_ROLES[$j]}"
            effective_model="${PENDING_MODELS[$j]}"
            effective_answer_model="${PENDING_ANSWER_MODELS[$j]}"
            layer="${PENDING_LAYERS[$j]}"
            read -ra still_missing <<< "${PENDING_MISSING[$j]}"

            echo "  [Bin $((bin+1))/$NUM_BINS] Submitting GPU job for role=$role model=$effective_model answer_model=$effective_answer_model layer=$layer counts=(${still_missing[*]})"
            bash scripts/cluster/slurm/roles_optimized_gpu_batch.sh \
                --role "$role" \
                --model "$effective_model" \
                --answer-model "$effective_answer_model" \
                --layer "$layer" \
                --counts "${still_missing[@]}" \
                "${COMMON_ARGS[@]}"

            ((submitted++))
        done
    done
fi

echo ""
echo "Done. Submitted: $submitted GPU jobs, Skipped: $skipped (already complete), Skipped (no Q/A): $skipped_no_qa."
