#!/usr/bin/env bash
# check_init_incomplete.sh
#
# Identify (role, model_id, layer) triples that are missing one or more
# target_pairs counts in the safetensors init folder, and write the results
# to configs/role_init_incomplete.json.
#
# Usage:
#   ./scripts/check_init_incomplete.sh \
#     --roles_list   configs/role_list.json \
#     --model_id     allenai/Olmo-3-7B-Instruct \
#     --layer        16 \
#     --target_pairs 20 40 60 \
#     --folder_dir   persona_data/model_inits \
#     --output       configs/role_init_incomplete.json
#
# Requires: jq

set -euo pipefail

# ---------- defaults ----------
ROLES_LIST="configs/role_list.json"
MODEL_ID=""
LAYER=""
TARGET_PAIRS=()
FOLDER_DIR="persona_data/model_inits"
OUTPUT="configs/role_init_incomplete.json"

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --roles_list)   ROLES_LIST="$2";   shift 2 ;;
        --model_id)     MODEL_ID="$2";     shift 2 ;;
        --layer)        LAYER="$2";        shift 2 ;;
        --folder_dir)   FOLDER_DIR="$2";   shift 2 ;;
        --output)       OUTPUT="$2";       shift 2 ;;
        --target_pairs)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TARGET_PAIRS+=("$1")
                shift
            done
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------- validation ----------
if [[ -z "$ROLES_LIST" || -z "$MODEL_ID" || -z "$LAYER" || ${#TARGET_PAIRS[@]} -eq 0 ]]; then
    echo "Usage: $0 --roles_list FILE --model_id ID --layer N --target_pairs N [N ...] [--folder_dir DIR] [--output FILE]" >&2
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required but not found in PATH." >&2
    exit 1
fi

if [[ ! -f "$ROLES_LIST" ]]; then
    echo "Error: roles list file not found: $ROLES_LIST" >&2
    exit 1
fi

# ---------- derived values ----------
# allenai/Olmo-3-7B-Instruct  ->  allenai__Olmo-3-7B-Instruct
SAFE_MODEL_ID="${MODEL_ID//\//__}"

# ---------- main loop ----------
incomplete_json="[]"
total_roles=0
incomplete_count=0

while IFS= read -r role; do
    [[ -z "$role" ]] && continue
    total_roles=$((total_roles + 1))

    role_dir="${FOLDER_DIR}/${role}_persona_initialization"
    missing_counts="[]"

    for count in "${TARGET_PAIRS[@]}"; do
        expected="${role_dir}/${SAFE_MODEL_ID}_layer${LAYER}_count${count}.safetensors"
        if [[ ! -f "$expected" ]]; then
            missing_counts=$(jq -n --argjson arr "$missing_counts" --argjson c "$count" '$arr + [$c]')
        fi
    done

    if [[ "$missing_counts" != "[]" ]]; then
        incomplete_count=$((incomplete_count + 1))
        entry=$(jq -n \
            --arg role "$role" \
            --arg model_id "$MODEL_ID" \
            --argjson layer "$LAYER" \
            --argjson missing "$missing_counts" \
            '{role: $role, model_id: $model_id, layer: $layer, counts: $missing}')
        incomplete_json=$(jq -n --argjson arr "$incomplete_json" --argjson e "$entry" '$arr + [$e]')
    fi
done < <(jq -r 'keys[]' "$ROLES_LIST")

# ---------- write output ----------
mkdir -p "$(dirname "$OUTPUT")"
echo "$incomplete_json" | jq '.' > "$OUTPUT"

echo "Checked $total_roles roles."
echo "Incomplete: $incomplete_count  ->  $OUTPUT"
