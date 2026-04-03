#!/usr/bin/env python3
"""
Check which roles don't have 20, 40, or 50 sample count inits in model_inits.
"""
import json
from pathlib import Path


def main():
    # Load roles from role_list.json
    role_list_path = Path("configs/role_list.json")
    with open(role_list_path) as f:
        roles = list(json.load(f).keys())
    
    # Path to model_inits
    model_inits_path = Path("persona_data/model_inits")
    
    # Track roles and missing sample counts.
    missing_roles = []
    incomplete_roles = []
    missing_20_roles = []
    missing_40_roles = []
    missing_50_roles = []
    
    for role in roles:
        role_dir = model_inits_path / f"{role}_persona_initialization"
        
        if not role_dir.exists():
            missing_roles.append(role)
            continue
        
        # Find all sample counts for this role
        sample_counts = set()
        for file in role_dir.glob("*count*.safetensors"):
            # Extract count from filename like "...count20.safetensors"
            filename = file.name
            if "count" in filename:
                # Parse the count number
                count_str = filename.split("count")[1].split(".")[0]
                try:
                    count = int(count_str)
                    sample_counts.add(count)
                except ValueError:
                    pass
        
        required_counts = {20, 40, 50}
        missing = required_counts - sample_counts
        
        if missing:
            incomplete_roles.append(role)
            if 20 in missing:
                missing_20_roles.append(role)
            if 40 in missing:
                missing_40_roles.append(role)
            if 50 in missing:
                missing_50_roles.append(role)
            print(f"❌ {role}")
            print(f"   Has: {sorted(sample_counts)}")
            print(f"   Missing: {sorted(missing)}")
        else:
            print(f"✓ {role}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nTotal roles: {len(roles)}")
    print(f"Roles without folder: {len(missing_roles)}")
    print(f"Roles missing 20/40/50 samples: {len(incomplete_roles)}")
    
    if incomplete_roles:
        print(f"\nIncomplete roles:")
        for role in sorted(incomplete_roles):
            print(f"  - {role}")

    print(f"\nMissing 50 sample count ({len(missing_50_roles)}):")
    for role in sorted(missing_50_roles):
        print(f"  - {role}")

    print(f"\nMissing 40 sample count ({len(missing_40_roles)}):")
    for role in sorted(missing_40_roles):
        print(f"  - {role}")

    print(f"\nMissing 20 sample count ({len(missing_20_roles)}):")
    for role in sorted(missing_20_roles):
        print(f"  - {role}")
    
    if missing_roles:
        print(f"\nRoles without any model_inits folder:")
        for role in sorted(missing_roles):
            print(f"  - {role}")


if __name__ == "__main__":
    main()
