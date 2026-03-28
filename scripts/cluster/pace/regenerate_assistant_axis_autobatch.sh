#!/bin/bash

###
# Reads roles from a JSON file (or explicit --roles), filters out ignored roles,
# and submits regenerate_assistant_axis.sh sbatch jobs.
#
# No completeness checks are performed: all non-ignored roles are assumed pending.
#
# Usage:
#   bash slurm/regenerate_assistant_axis_autobatch.sh [options...]
#
# Options:
#   -f, --file         Path to roles JSON (default: configs/role_list.json)
#   -r, --roles        Optional explicit list of roles (overrides --file)
#   -m, --model        Model ID (default: allenai/Olmo-3-7B-Instruct)
#   -a, --alphas       Optional alphas list passed through to regenerate job
#   -b, --num-bins     Number of sbatch jobs to split roles into (default: 0 = one job per role)
#   --dry-run          Print planned submissions without calling sbatch
#
# Author: iiisong
# Date: 2026-03-25
###

set -euo pipefail

# Defaults
JSON_FILE="configs/role_list.json"
EXPLICIT_ROLES=()
MODEL="allenai/Olmo-3-7B-Instruct"
ALPHAS=(1.0 1.5 2.0 2.5)
NUM_BINS=0
DRY_RUN=false


while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      JSON_FILE="$2"; shift 2
      ;;
    -r|--roles)
      shift
      EXPLICIT_ROLES=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        EXPLICIT_ROLES+=("$1")
        shift
      done
      ;;
    -m|--model)
      MODEL="$2"; shift 2
      ;;
    -a|--alphas)
      shift
      ALPHAS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ALPHAS+=("$1")
        shift
      done
      ;;
    -b|--num-bins)
      NUM_BINS="$2"; shift 2
      ;;
    --dry-run)
      DRY_RUN=true; shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ${#EXPLICIT_ROLES[@]} -gt 0 ]]; then
  ALL_ROLES=("${EXPLICIT_ROLES[@]}")
  echo "Using explicitly provided roles: ${ALL_ROLES[*]}"
else
  if [[ ! -f "$JSON_FILE" ]]; then
    echo "Error: JSON file not found: $JSON_FILE"
    exit 1
  fi
  mapfile -t ALL_ROLES < <(python3 -c "import json; data=json.load(open('$JSON_FILE')); print('\\n'.join(data.keys()))")
  echo "Loaded ${#ALL_ROLES[@]} roles from $JSON_FILE"
fi

if [[ ${#ALL_ROLES[@]} -eq 0 ]]; then
  echo "No roles found. Nothing to do."
  exit 0
fi


# No ignore logic: process all roles
ROLES=("${ALL_ROLES[@]}")
echo "Submitting ${#ROLES[@]} role(s) with no completeness checks."

auto_submit() {
  local -a job_roles=("$@")
  local -a cmd=(sbatch slurm/regenerate_assistant_axis.sh --model "$MODEL" --roles "${job_roles[@]}")

  if [[ ${#ALPHAS[@]} -gt 0 ]]; then
    cmd+=(--alphas "${ALPHAS[@]}")
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [DRY RUN] ${cmd[*]}"
  else
    echo "  Submitting: roles=(${job_roles[*]})"
    "${cmd[@]}"
  fi
}

if [[ "$NUM_BINS" -le 0 ]]; then
  # Default: one job per role
  for role in "${ROLES[@]}"; do
    auto_submit "$role"
  done
else
  # Group into NUM_BINS jobs
  total=${#ROLES[@]}
  bin_size=$(( (total + NUM_BINS - 1) / NUM_BINS ))
  echo "Bin mode: ${total} roles -> ${NUM_BINS} bins (~${bin_size} roles/bin)"

  for ((bin=0; bin<NUM_BINS; bin++)); do
    start=$((bin * bin_size))
    [[ $start -ge $total ]] && break
    end=$((start + bin_size))
    [[ $end -gt $total ]] && end=$total

    bin_roles=("${ROLES[@]:$start:$((end - start))}")
    auto_submit "${bin_roles[@]}"
  done
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run complete."
else
  echo "Done."
fi
