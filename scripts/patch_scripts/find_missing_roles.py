#!/usr/bin/env python3
"""
Find roles in role_list.json that are not in gold_standard_prompts directory.
"""

import json
import os
from pathlib import Path


def main():
    # Define paths
    role_list_path = Path(__file__).parent.parent / "configs" / "role_list.json"
    gold_standard_dir = Path(__file__).parent.parent / "persona_data" / "role_datasets"
    
    # Load role_list.json
    with open(role_list_path, 'r') as f:
        role_list = json.load(f)
    
    # Get all roles from role_list
    roles_in_list = set(role_list.keys())
    
    # Get all roles from gold_standard_prompts directory
    gold_standard_files = set()
    if gold_standard_dir.exists():
        for file in gold_standard_dir.iterdir():
            if file.is_file() and file.suffix == '.json':
                # Remove .json extension to get role name
                role_name = file.stem
                gold_standard_files.add(role_name)
    
    # Find missing roles
    missing_roles = roles_in_list - gold_standard_files
    
    # Print results
    print(f"Total roles in role_list.json: {len(roles_in_list)}")
    print(f"Total roles in gold_standard_prompts: {len(gold_standard_files)}")
    print(f"\nRoles missing from gold_standard_prompts: {len(missing_roles)}")
    
    if missing_roles:
        print("\nMissing roles:")
        for role in sorted(missing_roles):
            print(f"  - {role}")
    else:
        print("\nAll roles from role_list.json are present in gold_standard_prompts!")


if __name__ == "__main__":
    main()
