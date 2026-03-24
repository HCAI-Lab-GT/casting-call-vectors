#!/bin/bash

if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script must be submitted to SLURM for execution"
    exit 1
fi

# Ensure we run from the repo root (where the job was submitted)
cd "$SLURM_SUBMIT_DIR"

# ===CLI ARGS==================
ROLE=""
MODEL="allenai/Olmo-3-7B-Instruct"
LAYER=16
COUNTS=()
SAFETENSORS_DIR="./persona_data/model_layer_inits/"
QA_RESPONSES_DIR="./persona_data/model_qa_responses"
DATASET_DIR="./persona_data/role_datasets/"
# Gold experiment params
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

echo "===CLI ARGS=================="
echo "ROLE: $ROLE"
echo "MODEL: $MODEL"
echo "LAYER: $LAYER"
echo "COUNTS: ${COUNTS[*]}"
echo "SAFETENSORS_DIR: $SAFETENSORS_DIR"
echo "QA_RESPONSES_DIR: $QA_RESPONSES_DIR"
echo "GOLD_SAVE_DIR: $GOLD_SAVE_DIR"
echo "ALPHAS: ${ALPHAS[*]}"
echo "TEMPERATURES: ${TEMPERATURES[*]}"
echo "==========================="

module load uv
source .venv/bin/activate
uv sync

SAFE_MODEL="${MODEL//\//__}"

# Helper: check if the gold experiment CSV has all required parameter combinations
# for a specific (layer, sample_count) pairing complete.
# Returns 0 (success/complete) or 1 (incomplete/missing).
check_csv_complete() {
    local csv_file="$1"
    local expected="$2"
    local layer="$3"
    local sample_count="$4"
    local alphas_py="$5"
    local temperatures_py="$6"

    if [ ! -f "$csv_file" ]; then
        return 1
    fi

    python3 -c "
import csv, sys

csv_file = '$csv_file'
expected = $expected
layer = $layer
sample_count = $sample_count
alphas = [$alphas_py]
temperatures = [$temperatures_py]

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
except Exception:
    sys.exit(1)

for alpha in alphas:
    for temperature in temperatures:
        count = sum(
            1 for r in rows
            if (int(r['layer']) == layer and
                int(r['sample_count']) == sample_count and
                float(r['alpha']) == alpha and
                float(r['temperature']) == temperature)
        )
        if count < expected:
            sys.exit(1)

sys.exit(0)
" 2>/dev/null
    return $?
}

ALPHAS_PY=$(IFS=,; echo "${ALPHAS[*]}")
TEMPERATURES_PY=$(IFS=,; echo "${TEMPERATURES[*]}")

### GPU PHASE: Extract persona vectors for all counts in one pass ###
# Collect which counts are still missing on disk
COUNTS_FOR_PYTHON=()
for count in "${COUNTS[@]}"; do
    SAFETENSORS_PATH="${SAFETENSORS_DIR}/${ROLE}_persona_initialization/${SAFE_MODEL}_layer${LAYER}_count${count}.safetensors"
    if [ -f "$SAFETENSORS_PATH" ]; then
        echo "Safetensors already exists at $SAFETENSORS_PATH, skipping extraction for count=$count"
    else
        COUNTS_FOR_PYTHON+=("$count")
    fi
done

if [[ ${#COUNTS_FOR_PYTHON[@]} -gt 0 ]]; then
    echo "--- Extracting role=$ROLE layer=$LAYER counts=(${COUNTS_FOR_PYTHON[*]}) in one pass ---"
    srun uv run --no-sync python src/pvx/implementations/roles_optimized/roles_optimized_gpu.py \
        --activation_extraction_model "$MODEL" \
        --roles "$ROLE" \
        --layer "$LAYER" \
        --counts "${COUNTS_FOR_PYTHON[@]}" \
        --qa_responses_dir "$QA_RESPONSES_DIR" \
        --dataset_dir "$DATASET_DIR" \
        --output_dir "$SAFETENSORS_DIR"

    GPU_RETURN_CODE=$?
    if [ "$GPU_RETURN_CODE" -ne 0 ]; then
        echo "Error: GPU extraction failed for $ROLE (exit code $GPU_RETURN_CODE)"
        exit "$GPU_RETURN_CODE"
    fi
fi

### CHAIN: Submit gold experiment per count if not yet complete ###
any_success=false
for count in "${COUNTS[@]}"; do
    SAFETENSORS_PATH="${SAFETENSORS_DIR}/${ROLE}_persona_initialization/${SAFE_MODEL}_layer${LAYER}_count${count}.safetensors"

    if [ ! -f "$SAFETENSORS_PATH" ]; then
        echo "Warning: Safetensors not found at $SAFETENSORS_PATH after extraction. Skipping gold experiment for count=$count."
        continue
    fi

    any_success=true

    if [[ "$NO_CHAIN" == "true" ]]; then
        echo "Skipping gold experiment chain for $ROLE (layer=$LAYER, count=$count) [--no-chain]"
    else
        CSV_FILE="${GOLD_SAVE_DIR}/Comparison_GoldStandard_${ROLE}.csv"

        if check_csv_complete "$CSV_FILE" "$EXPECTED_ROWS" "$LAYER" "$count" "$ALPHAS_PY" "$TEMPERATURES_PY"; then
            echo "Gold experiment for $ROLE (layer=$LAYER, count=$count) already complete. Skipping."
        else
            echo "Submitting gold experiment for $ROLE (layer=$LAYER, count=$count)..."
            source .env  # reload for PACE_ACCOUNT on compute node

            bash slurm/gold_experiment.sh \
                --model "$MODEL" \
                --roles "$ROLE" \
                --layers "$LAYER" \
                --sample-counts "$count" \
                --alphas "${ALPHAS[@]}" \
                --temperatures "${TEMPERATURES[@]}" \
                --max-new-tokens "$MAX_NEW_TOKENS" \
                --questions-file "$QUESTIONS_FILE" \
                --save-dir "$GOLD_SAVE_DIR" \
                --safetensors-dir "$SAFETENSORS_DIR" \
                --gold-prompts-dir "$GOLD_PROMPTS_DIR"
        fi
    fi
done

if [ "$any_success" = false ]; then
    echo "Error: No safetensors produced for $ROLE."
    exit 1
fi

echo "GPU extraction phase complete for $ROLE."
