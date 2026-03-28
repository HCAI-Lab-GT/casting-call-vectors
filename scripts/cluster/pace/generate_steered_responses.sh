#!/bin/bash
source .env

NAME="generate_steered_responses" # name of the script (mostly for logging purposes)

###
# src/pvx/implementations/roles/persona_dataset.py
#
# This script is used to generate new persona vectors.
# It calls src/pvx/implementations/roles/persona_dataset.py with specified run.
#
# Usage:
# ./role_dataset_batch.sh <roles>
#
# Author: iiisong
# Date: 2026-01-21
###
# ===CLI ARGS==================

# defaults
MODEL="allenai/Olmo-3-7B-Instruct" # "Qwen/Qwen2.5-7B-Instruct"
ALPHA=2.5 # "hf-local"

ROLES=()

# parse args: roles first, optional --model anywhere
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2 ;;
    -a|--alpha) ALPHA="$2"; shift 2 ;;
    --) shift; while [[ $# -gt 0 ]]; do ROLES+=("$1"); shift; done ;;
    *) ROLES+=("$1"); shift ;;
  esac
done

export MODEL
export ALPHA
# ===========================

### SLURM SETTINGS ###
# runtime settings
GPU_CONFIG=gpu:h200:2
# GPU_CONFIG=gpu:a100:1
NUM_NODES=32
NUM_WORKERS=8
# MEM_PER_NODE=256G
MEM_PER_NODE=512G
TIME=08:00:00
# TIME=00:01:00


# job settings (typically do not change)
ACCOUNT="$PACE_ACCOUNT" # set from .env
QOS=embers
LOG_DIR="logs/slurm/$NAME"


### CLI ARGS VALIDATION ###

# Assert run name exists
if [ -z "$ACCOUNT" ]; then
    echo "Error: Account is required for PACE access"
    exit 1
fi

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
# sbatch  --gres "$GPU_CONFIG" --ntasks-per-node 1 --cpus-per-task "$NUM_WORKERS" --mem "$MEM_PER_NODE" \
#         --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
#         --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
#         "scripts/cluster/pace/$NAME.sbatch" \
#         "${ROLES[@]}"

sbatch  --gres "$GPU_CONFIG" --ntasks-per-node 1 --cpus-per-task "$NUM_WORKERS" --mem "$MEM_PER_NODE" \
        --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
        --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
        "scripts/cluster/pace/$NAME.sbatch" \
        "${ROLES[@]}"

### POST JOB ###
# None