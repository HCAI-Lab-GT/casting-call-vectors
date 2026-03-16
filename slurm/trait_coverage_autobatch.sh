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

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)    JSON_FILE="$2"; shift 2;;
    -m|--model)   MODEL="$2"; shift 2;;
    -n|--bins)    N="$2"; shift 2;;
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
TRAITS=($(python3 -c "import json,sys; data=json.load(open('$JSON_FILE')); print('\n'.join(data.keys()))"))

TOTAL=${#TRAITS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "Error: No traits found in $JSON_FILE"
    exit 1
fi

echo "Found $TOTAL traits, dividing into $N bins..."

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
