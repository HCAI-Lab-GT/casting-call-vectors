#!/usr/bin/env bash
# Checks which roles in configs/role_list.json lack a corresponding JSON file
# in persona_data/role_inits/ and writes missing roles to configs/role_json_incomplete.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ROLE_LIST="$REPO_ROOT/configs/role_list.json"
ROLE_INITS_DIR="$REPO_ROOT/persona_data/gold_standard_prompts"
OUTPUT="$REPO_ROOT/configs/role_prompt_json_incomplete.json"

if [ ! -f "$ROLE_LIST" ]; then
    echo "Error: $ROLE_LIST not found" >&2
    exit 1
fi

python3 - <<EOF
import json, os, sys

role_list_path = "$ROLE_LIST"
role_inits_dir = "$ROLE_INITS_DIR"
output_path = "$OUTPUT"

with open(role_list_path) as f:
    roles = list(json.load(f).keys())

missing = [
    role for role in roles
    if not os.path.isfile(os.path.join(role_inits_dir, f"{role}.json"))
]

with open(output_path, "w") as f:
    json.dump(missing, f, indent=2)

print(f"{len(missing)}/{len(roles)} roles missing from {role_inits_dir}")
print(f"Written to {output_path}")
EOF
