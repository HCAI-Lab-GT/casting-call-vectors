import argparse
from pathlib import Path
import pandas as pd


def load_all_roles(input_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(input_dir.glob("*_missing_data.csv")):
        role = csv_path.stem.replace("_missing_data", "")
        df = pd.read_csv(csv_path)
        df.insert(0, "role", role)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_missing_data.csv files found in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def get_roles(df: pd.DataFrame, col: str) -> set[str]:
    return set(df.loc[df[col] > 0, "role"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise missing/malformed data across all role reports.")
    parser.add_argument("--input_dir", default="experiment_data/experiment_missing_data_modal")
    args = parser.parse_args()

    df = load_all_roles(Path(args.input_dir))

    steering_cols = [
        ("steered_missing", "steered_missing_data"),
        ("steered_malformed", "steered_malformed_data"),
        ("steered_duplicate", "steered_duplicate_data"),
    ]
    assistant_cols = [
        ("assistant_axis_missing", "assistant_missing_data"),
        ("assistant_axis_malformed", "assistant_malformed_data"),
        ("assistant_axis_duplicate", "assistant_duplicate_data"),
    ]

    if "questions_missing" in df.columns:
        q_roles = get_roles(df, "questions_missing")
        print(f"questions_missing: {len(q_roles)} roles")
        print(sorted(q_roles))
        print()

    steering_roles = {}
    for col, label in steering_cols:
        roles = get_roles(df, col)
        steering_roles[label] = roles
        print(f"{label}: {len(roles)} roles")
        print(sorted(roles))
        print()

    assistant_roles = {}
    for col, label in assistant_cols:
        roles = get_roles(df, col)
        assistant_roles[label] = roles
        print(f"{label}: {len(roles)} roles")
        print(sorted(roles))
        print()

    all_steering_roles = set().union(*steering_roles.values())
    print(f"Unique roles with any steering issue: {len(all_steering_roles)}")
    print(sorted(all_steering_roles))
    print()

    all_assistant_roles = set().union(*assistant_roles.values())
    print(f"Unique roles with any assistant issue: {len(all_assistant_roles)}")
    print(sorted(all_assistant_roles))
    print()

    all_roles = all_steering_roles.union(all_assistant_roles)
    if "questions_missing" in df.columns:
        all_roles |= q_roles
    print(f"Total unique roles with any issue: {len(all_roles)}")
    print(sorted(all_roles))


if __name__ == "__main__":
    main()
