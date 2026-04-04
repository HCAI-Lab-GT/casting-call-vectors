import os
import json
import glob
import subprocess
import argparse

LOG_DIR = "logs/slurm/regenerate_assistant_axis_migration/"
CSV_DIR = "experiment_data/gold_prompt_experiments/"
ROLE_LIST = "configs/role_list.json"
FIX_SCRIPT = "scripts/patch_scripts/fix_baseline_responses.py"


def get_roles(role_list_path):
    with open(role_list_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


def log_has_completed(log_path):
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Check last 20 lines for completion
        for line in lines[-20:]:
            if "Completed regeneration" in line:
                return True
    except Exception:
        pass
    return False


def csv_has_dtype(csv_path):
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                if "dtype" in line:
                    return True
    except Exception:
        pass
    return False


def main(dry_run=False):
    import re
    # Use a custom role list file if provided
    role_list_path = main.role_list_file if hasattr(main, 'role_list_file') and main.role_list_file else ROLE_LIST
    roles = set(get_roles(role_list_path))
    # Collect roles that would be processed
    roles_to_process = []
    log_files = glob.glob(os.path.join(LOG_DIR, '*.out'))
    log_info = []  # (role, log_file, csv_path)
    for log_file in log_files:
        # Try to extract the role from the log file (look for 'ROLES: <role>' in the first 50 lines)
        role = None
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    m = re.match(r'ROLES:\s*([\w_\-]+)', line)
                    if m:
                        role = m.group(1)
                        print(role)
                        break
        except Exception:
            continue
        if not role or role not in roles:
            continue
        if not log_has_completed(log_file):
            continue
        csv_path = os.path.join(CSV_DIR, f"Comparison_GoldStandard_{role}.csv")
        if not os.path.exists(csv_path):
            continue
        if not csv_has_dtype(csv_path):
            continue
        roles_to_process.append(role)
        log_info.append((role, log_file, csv_path))

    if dry_run:
        if roles_to_process:
            print("[DRY RUN] Roles that would be processed:")
            for r in sorted(set(roles_to_process)):
                print(f"  - {r}")
        else:
            print("[DRY RUN] No roles would be processed.")

    for role, log_file, csv_path in log_info:
        cmd = [
            "python", FIX_SCRIPT, csv_path, "--output", csv_path, "--role", role
        ]
        if dry_run:
            cmd.append("--dry-run")
        print(f"[{'DRY RUN' if dry_run else 'RUN'}] Would run: {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check completed roles and fix baselines if needed.")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions, do not run fix script.")
    parser.add_argument("--role-list", type=str, help="Path to role list JSON file (default: configs/role_list.json)")
    args = parser.parse_args()
    main.role_list_file = args.role_list
    main(dry_run=args.dry_run)
