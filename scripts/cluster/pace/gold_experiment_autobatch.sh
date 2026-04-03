#!/bin/bash
source .env

###
# Reads role keys from a JSON file, divides them into bins,
# and submits a gold_prompt_experiments_batch.sh job for each bin.
#
# Usage:
# ./gold_prompt_experiments_autobatch.sh [-f <json_filepath>] [-m <model>] [-n <num_bins>] [-b <bin_size>]
#
# Options:
#   -n, --bins       Number of bins to divide roles into (round-robin distribution)
#   -b, --bin-size   Number of roles per bin (contiguous batching, takes priority over -n)
#
# If neither -n nor -b is specified, defaults to 20 bins.
#
# Author: iiisong
# Date: 2026-03-21
###

# ===CLI ARGS==================
# defaults
JSON_FILE="configs/role_list.json"
MODEL="allenai/Olmo-3-7B-Instruct"
N=""           # number of bins (mutually exclusive with BIN_SIZE)
BIN_SIZE=""    # number of roles per bin (takes priority over N)
SAVE_DIR="./experiment_data/gold_prompt_experiments/"
LAYERS=(16)
SAMPLE_COUNTS=(50)
ALPHAS=(1 1.5 2 2.5)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
QUESTIONS_FILE="./configs/validation_questions.jsonl"
SAFETENSORS_DIR="./persona_data/model_inits/"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"
EXPECTED_ROWS=228

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)             JSON_FILE="$2"; shift 2;;
    -m|--model)            MODEL="$2"; shift 2;;
    -n|--bins)             N="$2"; shift 2;;
    -b|--bin-size)         BIN_SIZE="$2"; shift 2;;
    -d|--save-dir)         SAVE_DIR="$2"; shift 2;;
    -l|--layers)           shift; LAYERS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do LAYERS+=("$1"); shift; done;;
    -s|--sample-counts)    shift; SAMPLE_COUNTS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do SAMPLE_COUNTS+=("$1"); shift; done;;
    -a|--alphas)           shift; ALPHAS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do ALPHAS+=("$1"); shift; done;;
    -t|--temperatures)     shift; TEMPERATURES=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do TEMPERATURES+=("$1"); shift; done;;
    -N|--max-new-tokens)   MAX_NEW_TOKENS="$2"; shift 2;;
    -q|--questions-file)   QUESTIONS_FILE="$2"; shift 2;;
    --safetensors-dir)     SAFETENSORS_DIR="$2"; shift 2;;
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

# Extract keys from JSON
ALL_ROLES=($(python3 -c "import json,sys; data=json.load(open('$JSON_FILE')); print('\n'.join(data.keys()))"))

if [[ ${#ALL_ROLES[@]} -eq 0 ]]; then
    echo "Error: No roles found in $JSON_FILE"
    exit 1
fi

# Helper function to check if a CSV file has all required parameter combinations complete
# Arguments: csv_file, expected_rows, layers, sample_counts, alphas, temperatures
check_csv_complete() {
    local csv_file="$1"
    local expected="$2"
    shift 2
    
    # Remaining args are: layers... -- sample_counts... -- alphas... -- temperatures...
    local layers=()
    local sample_counts=()
    local alphas=()
    local temperatures=()
    local current_array="layers"
    
    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            case "$current_array" in
                layers) current_array="sample_counts";;
                sample_counts) current_array="alphas";;
                alphas) current_array="temperatures";;
            esac
        else
            case "$current_array" in
                layers) layers+=("$arg");;
                sample_counts) sample_counts+=("$arg");;
                alphas) alphas+=("$arg");;
                temperatures) temperatures+=("$arg");;
            esac
        fi
    done
    
    if [ ! -f "$csv_file" ]; then
        return 1
    fi
    
    # Convert bash arrays to comma-separated strings for Python
    local layers_py=$(IFS=,; echo "${layers[*]}")
    local sample_counts_py=$(IFS=,; echo "${sample_counts[*]}")
    local alphas_py=$(IFS=,; echo "${alphas[*]}")
    local temperatures_py=$(IFS=,; echo "${temperatures[*]}")
    
    # Use Python to check all combinations have expected rows
    python3 -c "
import csv
import sys

csv_file = '$csv_file'
expected = $expected
layers = [$layers_py]
sample_counts = [$sample_counts_py]
alphas = [$alphas_py]
temperatures = [$temperatures_py]

try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
except Exception:
    sys.exit(1)

# Check each combination
for layer in layers:
    for sample_count in sample_counts:
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

# Filter out roles where output CSV has all required combinations complete
ROLES=()
for role in "${ALL_ROLES[@]}"; do
    csv_file="$SAVE_DIR/Comparison_GoldStandard_${role}.csv"
    if check_csv_complete "$csv_file" "$EXPECTED_ROWS" \
        "${LAYERS[@]}" -- "${SAMPLE_COUNTS[@]}" -- "${ALPHAS[@]}" -- "${TEMPERATURES[@]}"; then
        continue  # skip completed role
    fi
    ROLES+=("$role")
done

SKIPPED=$(( ${#ALL_ROLES[@]} - ${#ROLES[@]} ))
echo "Skipping $SKIPPED already-completed roles (all layer/sample_count/alpha/temperature combinations have $EXPECTED_ROWS+ rows)."

TOTAL=${#ROLES[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "All roles already completed. Nothing to do."
    exit 0
fi

# Determine number of bins: BIN_SIZE takes priority, then N, then default 20
if [[ -n "$BIN_SIZE" ]]; then
    # Calculate number of bins from bin size (ceiling division)
    N=$(( (TOTAL + BIN_SIZE - 1) / BIN_SIZE ))
    echo "Found $TOTAL remaining roles, using bin size $BIN_SIZE -> $N bins..."
elif [[ -n "$N" ]]; then
    echo "Found $TOTAL remaining roles, dividing into $N bins..."
else
    N=20
    echo "Found $TOTAL remaining roles, dividing into $N bins (default)..."
fi

### SUBMIT BINS ###
for ((i=0; i<N; i++)); do
    BIN_ROLES=()
    
    if [[ -n "$BIN_SIZE" ]]; then
        # Contiguous batching: each bin gets up to BIN_SIZE consecutive roles
        START=$((i * BIN_SIZE))
        for ((j=START; j<START+BIN_SIZE && j<TOTAL; j++)); do
            BIN_ROLES+=("${ROLES[$j]}")
        done
    else
        # Round-robin batching: distribute roles evenly across bins
        for ((j=i; j<TOTAL; j+=N)); do
            BIN_ROLES+=("${ROLES[$j]}")
        done
    fi

    if [[ ${#BIN_ROLES[@]} -eq 0 ]]; then
        continue
    fi

    echo "Bin $((i+1))/$N (${#BIN_ROLES[@]} roles): ${BIN_ROLES[*]}"
    bash scripts/cluster/pace/gold_experiment.sh \
        --model "$MODEL" \
        --roles "${BIN_ROLES[@]}" \
        --layers "${LAYERS[@]}" \
        --sample-counts "${SAMPLE_COUNTS[@]}" \
        --alphas "${ALPHAS[@]}" \
        --temperatures "${TEMPERATURES[@]}" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --questions-file "$QUESTIONS_FILE" \
        --save-dir "$SAVE_DIR" \
        --safetensors-dir "$SAFETENSORS_DIR" \
        --gold-prompts-dir "$GOLD_PROMPTS_DIR"

done