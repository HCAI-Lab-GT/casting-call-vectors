#!/bin/bash

###
# Reads roles from a JSON file (or explicit --roles), and submits backfill_assistant_axis_gold_comparisons.sh sbatch jobs.
# No ignore logic: all roles are processed.
#
# Usage:
#   bash scripts/cluster/pace/backfill_assistant_axis_gold_comparisons_autobatch.sh [options...]
#
# Options:
#   -f, --file         Path to roles JSON (default: configs/role_list.json)
#   -r, --roles        Optional explicit list of roles (overrides --file)
#   -b, --num-bins     Number of sbatch jobs to split roles into (default: 0 = one job per role)
#   --overwrite        Passes --overwrite to backfill script
#   --dry-run          Print planned submissions without calling sbatch
#   [other options]    Passed through to sbatch script
#
# Author: Copilot
# Date: 2026-03-25
###

set -euo pipefail

JSON_FILE="configs/role_list.json"
EXPLICIT_ROLES=()
NUM_BINS=0
DRY_RUN=false
OVERWRITE=false
EXTRA_ARGS=()

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
    -b|--num-bins)
      NUM_BINS="$2"; shift 2
      ;;
    --overwrite)
      OVERWRITE=true; EXTRA_ARGS+=("--overwrite"); shift
      ;;
    --dry-run)
      DRY_RUN=true; shift
      ;;
    *)
      EXTRA_ARGS+=("$1"); shift
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

ROLES=("${ALL_ROLES[@]}")
echo "Submitting ${#ROLES[@]} role(s) with no ignore logic."

auto_submit() {
  local -a job_roles=("$@")
  local -a cmd=(sbatch scripts/cluster/pace/backfill_assistant_axis_gold_comparisons.sh --roles "${job_roles[@]}" "${EXTRA_ARGS[@]}")
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [DRY RUN] ${cmd[*]}"
  else
    echo "  Submitting: roles=(${job_roles[*]})"
    "${cmd[@]}"
  fi
}

if [[ "$NUM_BINS" -le 0 ]]; then
  for role in "${ROLES[@]}"; do
    auto_submit "$role"
  done
else
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
