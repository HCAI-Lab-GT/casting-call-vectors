#!/bin/bash
source .env

###
# Reads roles from a JSON file (e.g. configs/role_list.json), re-verifies which
# roles still have missing (layer, count) safetensors combinations on disk, and
# submits an sbatch job (role_layers_persona_model.sh) for each pending role.
#
# A role is considered "pending" if at least one of the requested (layer, count)
# combinations is missing from disk. Already-complete roles are skipped.
#
# Usage:
#   bash slurm/role_layers_persona_model_autobatch.sh [options...]
#
# Options:
#   -f, --file              Path to roles JSON (default: configs/role_list.json)
#                           Expected format: {"role_name": "description", ...}
#   -r, --roles             Optional explicit list of roles to process (overrides --file)
#   -m, --model             Model ID (default: allenai/Olmo-3-7B-Instruct)
#   -a, --alpha             Alpha for steering (default: 1.0)
#   -l, --layers            Layers to check/generate (default: 16)
#   -N, --sample-counts     Sample counts to check/generate (default: 40)
#   --safetensors-dir       Dir to check/write safetensors (default: ./persona_data/model_layer_inits/)
#   -b, --num-bins          Number of sbatch jobs to group roles into (default: 0 = one job per role).
#                           In bin mode each sbatch job receives multiple roles and processes
#                           them sequentially via the --roles flag.
#   --dry-run               Print what would be submitted without actually calling sbatch.
#
# Author: iiisong
# Date: 2026-03-24
###

# ===CLI ARGS==================
# defaults
JSON_FILE="configs/role_list.json"
EXPLICIT_ROLES=()
MODEL="allenai/Olmo-3-7B-Instruct"
ALPHA="1.0"
LAYERS=(16)
SAMPLE_COUNTS=(40)
SAFETENSORS_DIR="./persona_data/model_layer_inits/"
NUM_BINS=0   # 0 = one job per role (default); N > 0 = group into N bins
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)          JSON_FILE="$2"; shift 2;;
    -r|--roles)         shift; EXPLICIT_ROLES=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do EXPLICIT_ROLES+=("$1"); shift; done;;
    -m|--model)         MODEL="$2"; shift 2;;
    -a|--alpha)         ALPHA="$2"; shift 2;;
    -l|--layers)        shift; LAYERS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do LAYERS+=("$1"); shift; done;;
    -N|--sample-counts) shift; SAMPLE_COUNTS=(); while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do SAMPLE_COUNTS+=("$1"); shift; done;;
    --safetensors-dir)  SAFETENSORS_DIR="$2"; shift 2;;
    -b|--num-bins)      NUM_BINS="$2"; shift 2;;
    --dry-run)          DRY_RUN=true; shift;;
    *)                  echo "Unknown argument: $1" >&2; exit 1;;
  esac
done
# ===========================

if [[ ${#EXPLICIT_ROLES[@]} -gt 0 ]]; then
    ALL_ROLES=("${EXPLICIT_ROLES[@]}")
    echo "Using explicitly provided roles: ${ALL_ROLES[*]}"
else
    if [ ! -f "$JSON_FILE" ]; then
        echo "Error: JSON file not found: $JSON_FILE"
        exit 1
    fi
    mapfile -t ALL_ROLES < <(python3 -c "import json,sys; data=json.load(open('$JSON_FILE')); print('\n'.join(data.keys()))")
    echo "Loaded ${#ALL_ROLES[@]} roles from $JSON_FILE"
fi

if [[ ${#ALL_ROLES[@]} -eq 0 ]]; then
    echo "No roles found. Nothing to do."
    exit 0
fi

echo "Checking layers: ${LAYERS[*]}"
echo "Checking sample counts: ${SAMPLE_COUNTS[*]}"
echo ""

# ===VERIFY MISSING===
SAFE_MODEL="${MODEL//\//__}"

declare -a PENDING_ROLES     # roles that have at least one missing combo
declare -a PENDING_MISSING   # per role: space-joined "layer:count" pairs that are still missing

skipped=0

for role in "${ALL_ROLES[@]}"; do
    missing_combos=()
    for layer in "${LAYERS[@]}"; do
        for count in "${SAMPLE_COUNTS[@]}"; do
            expected="${SAFETENSORS_DIR}/${role}_persona_initialization/${SAFE_MODEL}_layer${layer}_count${count}.safetensors"
            if [ ! -f "$expected" ]; then
                missing_combos+=("layer${layer}_count${count}")
            fi
        done
    done

    if [[ ${#missing_combos[@]} -eq 0 ]]; then
        echo "  Skipping $role: all (layer, count) combinations already present on disk"
        ((skipped++))
    else
        echo "  Pending  $role: missing ${missing_combos[*]}"
        PENDING_ROLES+=("$role")
        PENDING_MISSING+=("${missing_combos[*]}")
    fi
done

TOTAL_PENDING=${#PENDING_ROLES[@]}
echo ""
echo "Summary: ${TOTAL_PENDING} roles pending, ${skipped} already complete."

if [[ "$TOTAL_PENDING" -eq 0 ]]; then
    echo "Nothing to submit."
    exit 0
fi

# ===SUBMIT JOBS==================
submitted=0

# Helper: derive minimal layers/counts needed from a space-joined "layerX_countY" string
get_layers_for_missing() {
    local missing_str="$1"
    echo "$missing_str" | tr ' ' '\n' | grep -oP 'layer\K[0-9]+' | sort -un | tr '\n' ' '
}
get_counts_for_missing() {
    local missing_str="$1"
    echo "$missing_str" | tr ' ' '\n' | grep -oP 'count\K[0-9]+' | sort -un | tr '\n' ' '
}

if [[ "$NUM_BINS" -le 0 ]]; then
    # Default: one sbatch job per role, passing only the layers/counts that are missing
    for ((j=0; j<TOTAL_PENDING; j++)); do
        role="${PENDING_ROLES[$j]}"
        missing_str="${PENDING_MISSING[$j]}"

        read -ra job_layers  <<< "$(get_layers_for_missing "$missing_str")"
        read -ra job_counts  <<< "$(get_counts_for_missing "$missing_str")"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY RUN] sbatch slurm/role_layers_persona_model.sh \\"
            echo "              --model \"$MODEL\" --alpha \"$ALPHA\" \\"
            echo "              --roles \"$role\" \\"
            echo "              --layers ${job_layers[*]} \\"
            echo "              --sample-counts ${job_counts[*]}"
            echo "              # missing: $missing_str"
        else
            echo "  Submitting job for role=$role layers=(${job_layers[*]}) counts=(${job_counts[*]})"
            sbatch slurm/role_layers_persona_model.sh \
                --model "$MODEL" \
                --alpha "$ALPHA" \
                --roles "$role" \
                --layers "${job_layers[@]}" \
                --sample_counts "${job_counts[@]}"
            ((submitted++))
        fi
    done
else
    # Bin mode: group roles into NUM_BINS jobs; each job processes its roles sequentially.
    # All requested layers/counts are passed per bin; load_or_create skips existing files.
    bin_size=$(( (TOTAL_PENDING + NUM_BINS - 1) / NUM_BINS ))
    echo "Bin mode: $TOTAL_PENDING pending roles → $NUM_BINS bins (~$bin_size roles/bin)"
    echo ""

    for ((bin=0; bin<NUM_BINS; bin++)); do
        start=$((bin * bin_size))
        [[ $start -ge $TOTAL_PENDING ]] && break
        end=$((start + bin_size))
        [[ $end -gt $TOTAL_PENDING ]] && end=$TOTAL_PENDING

        bin_roles=("${PENDING_ROLES[@]:$start:$((end - start))}")

        # Compute the union of missing layers/counts across all roles in this bin
        all_missing_in_bin=""
        for ((j=start; j<end; j++)); do
            all_missing_in_bin+=" ${PENDING_MISSING[$j]}"
        done
        read -ra bin_layers <<< "$(get_layers_for_missing "$all_missing_in_bin")"
        read -ra bin_counts <<< "$(get_counts_for_missing "$all_missing_in_bin")"

        bin_roles_str="${bin_roles[*]}"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY RUN] Bin $((bin+1))/$NUM_BINS — roles: ($bin_roles_str)"
            echo "            sbatch slurm/role_layers_persona_model.sh \\"
            echo "              --model \"$MODEL\" --alpha \"$ALPHA\" \\"
            echo "              --roles ${bin_roles[*]} \\"
            echo "              --layers ${bin_layers[*]} \\"
            echo "              --sample-counts ${bin_counts[*]}"
        else
            echo "  Submitting bin $((bin+1))/$NUM_BINS: roles=(${bin_roles_str}) layers=(${bin_layers[*]}) counts=(${bin_counts[*]})"
            sbatch slurm/role_layers_persona_model.sh \
                --model "$MODEL" \
                --alpha "$ALPHA" \
                --roles "${bin_roles[@]}" \
                --layers "${bin_layers[@]}" \
                --sample_counts "${bin_counts[@]}"
            ((submitted++))
        fi
    done
fi

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete. $TOTAL_PENDING roles would be submitted (${skipped} already complete)."
else
    echo "Done. Submitted: $submitted sbatch job(s), Skipped: $skipped (already complete)."
fi
