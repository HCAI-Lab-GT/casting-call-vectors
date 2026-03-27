import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Delete all .malformed_report.csv files with dry run option.")
    parser.add_argument("--dir", default="./experiment_data/gold_prompt_experiments", help="Directory to search for report files.")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be deleted without deleting.")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    files = list(base_dir.glob("**/*.malformed_report.csv"))
    if not files:
        print("No .malformed_report.csv files found.")
        return
    print(f"Found {len(files)} .malformed_report.csv files:")
    for f in files:
        print(f"  {f}")
    if args.dry_run:
        print("Dry run: No files deleted.")
    else:
        for f in files:
            try:
                f.unlink()
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

if __name__ == "__main__":
    main()
