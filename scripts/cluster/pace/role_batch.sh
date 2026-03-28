#!/bin/bash
source .env

NAME="role_batch" # name of the script (mostly for logging purposes)

###
# src/pvx/implementations/roles/roles_persona_model.py
#
# This script is used to generate new role vectors.
# It calls src/pvx/implementations/roles/roles_persona_model.py with specified run.
#
# Usage:
# ./role_batch.sh <role>
#
# Author: iiisong
# Date: 2026-01-21
###

# ===CLI ARGS==================
# export ROLE="$1" # role
# export LOCAL_MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}" # local_model, default if not provided

# defaults
MODEL="allenai/Olmo-3-7B-Instruct"

ROLES=()

# parse args: roles first, optional --model anywhere
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2 ;;
    --) shift; while [[ $# -gt 0 ]]; do ROLES+=("$1"); shift; done ;;
    *) ROLES+=("$1"); shift ;;
  esac
done

export MODEL
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

# Assert run name exists
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
sbatch  --gres "$GPU_CONFIG" --ntasks-per-node 1 --cpus-per-task "$NUM_WORKERS" --mem "$MEM_PER_NODE" \
        --account "$ACCOUNT" --qos "$QOS" --time "$TIME" \
        --output "$LOG_DIR/$NAME-%j.out" --error "$LOG_DIR/$NAME-%j.err" \
        "scripts/cluster/pace/$NAME.sbatch" \
        "${ROLES[@]}"

### POST JOB ###
# None