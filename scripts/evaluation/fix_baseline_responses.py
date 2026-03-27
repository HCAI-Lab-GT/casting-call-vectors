import csv
import argparse
import os
from collections import defaultdict


def is_malformed(response):
    """Check if the response contains 'dtype' (malformed)."""
    return 'dtype' in response


def process_csv(input_path, output_path=None, dry_run=True):
    # Cache: {question: non-malformed baseline response}
    cache = {}
    rows = []
    def role_match(row, roles):
        if not roles:
            return True
        return row.get('role') in roles

    with open(input_path, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        # First pass: build cache of good responses (for all roles)
        for row in reader:
            question = row['question']
            baseline = row['baseline']
            if not is_malformed(baseline):
                cache[question] = baseline
        # Second pass: replace malformed responses for selected roles
        for row in reader:
            question = row['question']
            baseline = row['baseline']
            if is_malformed(baseline) and question in cache and role_match(row, process_csv.roles):
                if dry_run:
                    print(f"Would replace baseline for role: {row.get('role')} | question: {question[:40]}...\n  Old: {baseline[:60]}\n  New: {cache[question][:60]}\n")
                else:
                    row['baseline'] = cache[question]
            rows.append(row)
    # Write output if not dry run
    if not dry_run and output_path:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Fix malformed baseline responses in CSV.")
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (required if not dry run)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change, do not write output')
    parser.add_argument('--role', nargs='*', help='Role(s) to process (if omitted, process all)')
    args = parser.parse_args()
    if not args.dry_run and not args.output:
        parser.error('Output file must be specified if not dry run.')
    # Pass roles to process_csv via function attribute
    process_csv.roles = set(args.role) if args.role else None
    process_csv(args.input_csv, args.output, args.dry_run)


if __name__ == '__main__':
    main()
