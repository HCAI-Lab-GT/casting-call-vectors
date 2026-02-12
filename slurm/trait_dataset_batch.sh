#!/bin/bash
source .env

NAME="trait_dataset_batch" # name of the script (mostly for logging purposes)

###
# src/pvx/implementations/base/persona_dataset.py
#
# This script is used to generate new persona vectors.
# It calls src/pvx/implementations/base/persona_dataset.py with specified run.
#
# Usage:
# ./trait_dataset_batch.sh <traits>
#
# Author: mparwani
# Date: 2026-01-21
###

# ===CLI ARGS==================
export TRAIT="$1" # trait
export LOCAL_MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}" # local_model, default if not provided
# ===========================


### SLURM SETTINGS ###
# runtime settings
GPU_CONFIG=gpu:h100:1
NUM_NODES=32
NUM_WORKERS=8
MEM_PER_NODE=512G
TIME=04:00:00

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

# Assert run name exists
if [ -z "$TRAIT" ]; then
    echo "Error: Trait is required"
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
        --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
        --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
        "slurm/$NAME.sbatch"

### POST JOB ###
# None