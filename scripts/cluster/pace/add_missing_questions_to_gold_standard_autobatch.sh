#!/bin/bash

###
# Reads roles from a JSON file (or explicit --roles), filters out ignored roles,
# and runs add_missing_questions_to_gold_standard.py in batches.
#
# No completeness checks are performed: all non-ignored roles are processed.
#
# Usage:
#   bash slurm/add_missing_questions_to_gold_standard_autobatch.sh [options...]
#
# Options:
#   -f, --file         Path to roles JSON (default: configs/role_list.json)
#   -r, --roles        Optional explicit list of roles (overrides --file)
#   -a, --alphas       Required list of alpha values
#   -l, --layer        Required layer value
#   -b, --num-bins     Number of batches to split roles into (default: 0 = one run per role)
#   --dry-run          Print planned commands without executing
#
# Author: iiisong
# Date: 2026-03-25
###

set -euo pipefail

# Defaults
JSON_FILE="configs/role_list.json"
EXPLICIT_ROLES=()
ALPHAS=()
LAYER=""
NUM_BINS=0
DRY_RUN=false

# Keep this list aligned with regenerate_assistant_axis_autobatch.sh
IGNORE_ROLES=(
  toddler
  optimist
  novelist
  cynic
  traditionalist
  rebel
  geographer
  immigrant
)

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
    -a|--alphas)
      shift
      ALPHAS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ALPHAS+=("$1")
        shift
      done
      ;;
    -l|--layer)
      LAYER="$2"; shift 2
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

if [[ ${#ALPHAS[@]} -eq 0 ]]; then
  echo "Error: --alphas is required"
  exit 1
fi

if [[ -z "$LAYER" ]]; then
  echo "Error: --layer is required"
  exit 1
fi

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

IGNORE_CSV=","$(IFS=,; echo "${IGNORE_ROLES[*]}")","

ROLES=()
IGNORED_COUNT=0
for role in "${ALL_ROLES[@]}"; do
  if [[ "$IGNORE_CSV" == *",$role,"* ]]; then
    ((IGNORED_COUNT+=1))
    continue
  fi
  ROLES+=("$role")
done

if [[ ${#ROLES[@]} -eq 0 ]]; then
  echo "All input roles were ignored. Nothing to run."
  exit 0
fi

echo "Ignored ${IGNORED_COUNT} role(s): ${IGNORE_ROLES[*]}"
echo "Processing ${#ROLES[@]} role(s) with no completeness checks."

run_batch() {
  local -a batch_roles=("$@")
  local -a cmd=(
    uv run python scripts/evaluation/add_missing_questions_to_gold_standard.py
    --roles "${batch_roles[@]}"
    --alphas "${ALPHAS[@]}"
    --layer "$LAYER"
  )

  if [[ "$DRY_RUN" == "true" ]]; then
    cmd+=(--dry_run)
    echo "  [DRY RUN] ${cmd[*]}"
  else
    echo "  Running: roles=(${batch_roles[*]})"
    "${cmd[@]}"
  fi
}

if [[ "$NUM_BINS" -le 0 ]]; then
  for role in "${ROLES[@]}"; do
    run_batch "$role"
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

    batch_roles=("${ROLES[@]:$start:$((end - start))}")
    run_batch "${batch_roles[@]}"
  done
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run complete."
else
  echo "Done."
fi
