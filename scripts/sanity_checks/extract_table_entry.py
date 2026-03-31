import pandas as pd
import argparse

def extract_entry(csv_path, alpha, question, column):
    df = pd.read_csv(csv_path)
    # Find the row matching both alpha and question
    match = df[(df['alpha'] == alpha) & (df['question'] == question)]
    if match.empty:
        print("No matching entry found.")
        return
    # Print the requested column value
    value = match.iloc[0][column]
    print(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a specific table entry from a CSV.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value to match")
    parser.add_argument("--question", required=True, help="Exact question string to match")
    parser.add_argument("--column", required=True, help="Column name for the response (e.g., baseline, steered)")
    args = parser.parse_args()
    extract_entry(args.csv, args.alpha, args.question, args.column)