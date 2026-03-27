import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Concatenate all missing_data CSVs into one with a role column.")
    parser.add_argument("--input_dir", default="./experiment_data/experiment_missing_data", help="Directory with *_missing_data.csv files.")
    parser.add_argument("--output", default="./experiment_data/experiment_missing_data/all_roles_missing_data.csv", help="Output CSV file.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*_missing_data.csv"))
    if not files:
        print(f"No *_missing_data.csv files found in {input_dir}")
        return

    dfs = []
    for f in files:
        role = f.stem.replace('_missing_data', '')
        df = pd.read_csv(f)
        df.insert(0, 'role', role)
        dfs.append(df)
    big_df = pd.concat(dfs, ignore_index=True)
    big_df.to_csv(args.output, index=False)
    print(f"Concatenated {len(files)} files into {args.output}")

if __name__ == "__main__":
    main()
