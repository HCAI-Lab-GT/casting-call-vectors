#!/bin/bash
source .env

NAME="run_eval" # name of the script (mostly for logging purposes)

###
# scripts/run_evals.jsons.sh
#
# This script is used to run Inspect AI tasks on batched slurm.
# It calls scripts/run_evals.py with specified run.
#
# Usage:
# ./run_evals.sh <run>
#
# Author: iiisong
# Date: 2025-012-29
###

# ===CLI ARGS==================
export RUN_NAME="$1" # run name to select (.json.gz directory)
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
if [ -z "$RUN_NAME" ]; then
    echo "Error: Run name is required"
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
        "scripts/cluster/pace/$NAME.sbatch"

### POST JOB ###
# None