import csv
import json
import argparse
from pathlib import Path


def is_malformed(response):
    """Check if the response contains 'dtype' (malformed)."""
    return 'dtype' in response


def process_csv(input_path: Path, dry_run: bool = True) -> int:
    """
    Fix malformed baseline responses in a single CSV by replacing them with
    a good response for the same question found elsewhere in the file.

    Returns the number of rows fixed (or that would be fixed in dry_run mode).
    """
    with open(input_path, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    if not reader:
        return 0

    # First pass: build cache of good baseline responses keyed by question
    cache: dict[str, str] = {}
    for row in reader:
        question = row['question']
        baseline = row.get('baseline', '')
        if baseline and not is_malformed(baseline):
            cache[question] = baseline
    cache_count = len(cache)
    print(f"Cache count: {cache_count}")
    # Second pass: replace malformed entries
    fixed = 0
    for row in reader:
        question = row['question']
        baseline = row.get('baseline', '')
        if (not baseline or is_malformed(baseline)) and question in cache:
            if dry_run:
                print(
                    f"Would fix  | question: {question[:50]!r}\n"
                    f"  Old: {baseline[:80]}\n"
                    f"  New: {cache[question][:80]}\n"
                )
            else:
                row['baseline'] = cache[question]
            fixed += 1

    if not dry_run and fixed:
        with open(input_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=reader[0].keys())
            writer.writeheader()
            writer.writerows(reader)

    return fixed


def load_roles(role_list_path: Path) -> list[str]:
    with open(role_list_path, encoding='utf-8') as f:
        data = json.load(f)
    # Support both a list of role names and a dict keyed by role name
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def main():
    parser = argparse.ArgumentParser(
        description="Fix malformed baseline responses across all roles in role_list.json."
    )
    parser.add_argument(
        '--input_dir',
        default='./experiment_data/regenerated_modal/gold_prompt_experiments',
        help='Directory containing Comparison_GoldStandard_<role>.csv files',
    )
    parser.add_argument(
        '--role_list',
        default='./configs/role_list.json',
        help='Path to role_list.json',
    )
    parser.add_argument(
        '--roles',
        nargs='+',
        default=None,
        help='Restrict to these roles (default: all roles in role_list.json)',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would change without writing output',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    role_list_path = Path(args.role_list)

    if args.roles:
        roles = args.roles
    else:
        roles = load_roles(role_list_path)

    if not roles:
        print("No roles to process.")
        return

    total_fixed = 0
    missing_files = []

    for role in roles:
        csv_path = input_dir / f"Comparison_GoldStandard_{role}.csv"
        print(f"CSV Path: {csv_path}")
        if not csv_path.exists():
            missing_files.append(role)
            continue

        fixed = process_csv(csv_path, dry_run=args.dry_run)
        action = "would fix" if args.dry_run else "fixed"
        print(f"[{role}] {action} {fixed} row(s) in {csv_path.name}")
        total_fixed += fixed

    if missing_files:
        print(f"\nNo CSV found for {len(missing_files)} role(s): {', '.join(missing_files)}")

    action = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{action} {total_fixed} total row(s) across {len(roles) - len(missing_files)} file(s).")


if __name__ == '__main__':
    main()
