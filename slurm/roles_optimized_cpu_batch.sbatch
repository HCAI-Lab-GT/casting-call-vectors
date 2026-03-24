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
TARGET_PAIRS=60
BIN_CONFIG=""  # JSON file with [{role, layer, counts}, ...] for multi-role bin mode
SAFETENSORS_DIR="./persona_data/model_layer_inits/"
QA_RESPONSES_DIR="./persona_data/model_qa_responses"
DATASET_DIR="./persona_data/role_datasets/"
BACKEND="openai"
BASE_URL="https://openrouter.ai/api/v1"
API_KEY_ENV="OPENROUTER_API_KEY"
# Gold experiment params (forwarded to GPU job)
GOLD_SAVE_DIR="./experiment_data/gold_prompt_experiments/"
ALPHAS=(1 1.5 2 2.5)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
QUESTIONS_FILE="./configs/validation_questions.jsonl"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"
EXPECTED_ROWS=228
NO_CHAIN=false  # if true, skip gold experiment chaining (propagated to GPU phase)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)                ROLE="$2"; shift 2;;
    -m|--model)            MODEL="$2"; shift 2;;
    -l|--layer)            LAYER="$2"; shift 2;;
    --counts)              shift; while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do COUNTS+=("$1"); shift; done;;
    --target-pairs)        TARGET_PAIRS="$2"; shift 2;;
    --bin-config)          BIN_CONFIG="$2"; shift 2;;
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
    --no-chain)            NO_CHAIN=true; shift;;
    *)                     echo "Unknown argument: $1" >&2; exit 1;;
  esac
done
# ===========================

echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') | INFO | .sbatch | Processed CLI"

module load uv

if [ ! -d ".venv" ]; then
    uv sync
fi

echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') | INFO | .sbatch | Loaded uv"

source .venv/bin/activate

echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') | INFO | .sbatch | Loaded .venv environment"

if [ -n "$BIN_CONFIG" ]; then
    ### BIN MODE: process multiple roles sequentially ###
    echo "===BIN MODE=================="
    echo "BIN_CONFIG: $BIN_CONFIG"
    echo "MODEL: $MODEL"
    echo "BACKEND: $BACKEND"
    echo "QA_RESPONSES_DIR: $QA_RESPONSES_DIR"
    echo "DATASET_DIR: $DATASET_DIR"
    echo "NO_CHAIN: ${NO_CHAIN:-false}"
    echo "Roles in bin: $(python3 -c "import json; print(' '.join(e['role'] for e in json.load(open('$BIN_CONFIG'))))")"
    echo "==========================="

    # Compute TARGET_PAIRS = max of all counts across all bin entries
    TARGET_PAIRS=$(python3 -c "
import json
entries = json.load(open('$BIN_CONFIG'))
print(max(max(e['counts']) for e in entries))
")

    # Build roles list for Python
    mapfile -t BIN_ROLES < <(python3 -c "import json; [print(e['role']) for e in json.load(open('$BIN_CONFIG'))]")

    echo "$(date '+%Y-%m-%d %H:%M:%S,%3N') | INFO | .sbatch | Calling srun..."

    ### CPU PHASE: process all bin roles sequentially ###
    srun uv run --no-sync python src/pvx/implementations/roles_optimized/roles_optimized_cpu.py \
        --roles "${BIN_ROLES[@]}" \
        --answer_model "$MODEL" \
        --target_pairs "$TARGET_PAIRS" \
        --backend "$BACKEND" \
        --base_url "$BASE_URL" \
        --api_key_env "$API_KEY_ENV" \
        --output_dir "$QA_RESPONSES_DIR" \
        --dataset_dir "$DATASET_DIR"

    SLURM_RETURN_CODE=$?
    if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
        echo "Error: CPU Q/A generation failed with exit code $SLURM_RETURN_CODE"
        exit "$SLURM_RETURN_CODE"
    fi

    ### CHAIN: Submit GPU extraction job per role in the bin ###
    source .env  # reload for PACE_ACCOUNT in case compute node env doesn't have it
    SAFE_MODEL="${MODEL//\//__}"

    while IFS='|' read -r role layer counts; do
        QA_PATH="${QA_RESPONSES_DIR}/${SAFE_MODEL}/${role}.json"
        if [ ! -f "$QA_PATH" ]; then
            echo "Warning: Q/A responses not found at $QA_PATH for role=$role. Skipping GPU submission."
            continue
        fi

        QA_COUNT=$(python3 -c "import json; d=json.load(open('$QA_PATH')); print(len(d.get('positive', [])))" 2>/dev/null || echo "0")
        echo "Q/A for $role: $QA_COUNT positive pairs at $QA_PATH"

        read -ra counts_arr <<< "$counts"
        echo "Chaining to GPU extraction job for $role (layer=$layer, counts=${counts_arr[*]})..."

        bash slurm/roles_optimized_gpu_batch.sh \
            --role "$role" \
            --model "$MODEL" \
            --layer "$layer" \
            --counts "${counts_arr[@]}" \
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
    done < <(python3 -c "
import json
entries = json.load(open('$BIN_CONFIG'))
for e in entries:
    print('{}|{}|{}'.format(e['role'], e['layer'], ' '.join(map(str, e['counts']))))
")

else
    ### SINGLE-ROLE MODE (default) ###
    echo "===CLI ARGS=================="
    echo "ROLE: $ROLE"
    echo "MODEL: $MODEL"
    echo "LAYER: $LAYER"
    echo "COUNTS: ${COUNTS[*]}"
    echo "TARGET_PAIRS: $TARGET_PAIRS"
    echo "BACKEND: $BACKEND"
    echo "QA_RESPONSES_DIR: $QA_RESPONSES_DIR"
    echo "DATASET_DIR: $DATASET_DIR"
    echo "==========================="

    ### CPU PHASE: Generate and judge Q/A responses ###
    srun uv run --no-sync python src/pvx/implementations/roles_optimized/roles_optimized_cpu.py \
        --roles "$ROLE" \
        --answer_model "$MODEL" \
        --target_pairs "$TARGET_PAIRS" \
        --backend "$BACKEND" \
        --base_url "$BASE_URL" \
        --api_key_env "$API_KEY_ENV" \
        --output_dir "$QA_RESPONSES_DIR" \
        --dataset_dir "$DATASET_DIR"

    SLURM_RETURN_CODE=$?
    if [ "$SLURM_RETURN_CODE" -ne 0 ]; then
        echo "Error: CPU Q/A generation failed with exit code $SLURM_RETURN_CODE"
        exit "$SLURM_RETURN_CODE"
    fi

    ### CHECK: Verify Q/A JSON was successfully generated ###
    SAFE_MODEL="${MODEL//\//__}"
    QA_PATH="${QA_RESPONSES_DIR}/${SAFE_MODEL}/${ROLE}.json"

    if [ ! -f "$QA_PATH" ]; then
        echo "Error: Q/A responses not found at $QA_PATH after CPU job. Skipping GPU submission."
        exit 1
    fi

    QA_COUNT=$(python3 -c "import json; d=json.load(open('$QA_PATH')); print(len(d.get('positive', [])))" 2>/dev/null || echo "0")
    echo "Q/A responses generated at $QA_PATH ($QA_COUNT positive pairs)"

    if [[ "$QA_COUNT" -lt "$TARGET_PAIRS" ]]; then
        echo "Warning: Only $QA_COUNT/$TARGET_PAIRS pairs generated. GPU extraction may produce fewer counts than expected."
    fi

    ### CHAIN: Submit GPU extraction job ###
    echo "Chaining to GPU extraction job for $ROLE (layer=$LAYER, counts=${COUNTS[*]})..."
    source .env  # reload for PACE_ACCOUNT in case compute node env doesn't have it

    bash slurm/roles_optimized_gpu_batch.sh \
        --role "$ROLE" \
        --model "$MODEL" \
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
fi
