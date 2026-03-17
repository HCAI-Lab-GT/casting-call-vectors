#!/bin/bash
source .env

###
# Reads trait keys from a JSON file, divides them into N bins,
# and submits a trait_coverage_batch.sh job for each bin.
#
# Usage:
# ./trait_coverage_autobatch.sh [-f <json_filepath>] [-m <model>] [-n <num_bins>]
#
# Author: iiisong
# Date: 2026-03-16
###

# ===CLI ARGS==================
# defaults
JSON_FILE="configs/trait_coverage_list.json"
MODEL="allenai/Olmo-3-7B-Instruct"
N=20
INITS_DIR="persona_data/trait_coverage_inits"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)    JSON_FILE="$2"; shift 2;;
    -m|--model)   MODEL="$2"; shift 2;;
    -n|--bins)    N="$2"; shift 2;;
    -d|--inits-dir) INITS_DIR="$2"; shift 2;;
    *)            echo "Unknown argument: $1" >&2; exit 1;;
  esac
done
# ===========================

### VALIDATION ###
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found: $JSON_FILE"
    exit 1
fi

# Extract keys from JSON
ALL_TRAITS=($(python3 -c "import json,sys; data=json.load(open('$JSON_FILE')); print('\n'.join(data.keys()))"))

if [[ ${#ALL_TRAITS[@]} -eq 0 ]]; then
    echo "Error: No traits found in $JSON_FILE"
    exit 1
fi

# Filter out traits that already have a safetensors file for the given model
SAFE_MODEL="${MODEL//\//__}"
TRAITS=()
for trait in "${ALL_TRAITS[@]}"; do
    if [ ! -f "$INITS_DIR/${trait}_persona_initialization/${SAFE_MODEL}.safetensors" ]; then
        TRAITS+=("$trait")
    fi
done

SKIPPED=$(( ${#ALL_TRAITS[@]} - ${#TRAITS[@]} ))
echo "Skipping $SKIPPED already-completed traits."

TOTAL=${#TRAITS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "All traits already completed. Nothing to do."
    exit 0
fi

echo "Found $TOTAL remaining traits, dividing into $N bins..."

### SUBMIT BINS ###
for ((i=0; i<N; i++)); do
    BIN_TRAITS=()
    for ((j=i; j<TOTAL; j+=N)); do
        BIN_TRAITS+=("${TRAITS[$j]}")
    done

    if [[ ${#BIN_TRAITS[@]} -eq 0 ]]; then
        continue
    fi

    echo "Bin $((i+1))/$N: ${BIN_TRAITS[*]}"
    bash slurm/trait_coverage_batch.sh --model "$MODEL" --traits "${BIN_TRAITS[@]}"

done
