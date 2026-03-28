import pandas as pd

def main():
    # Load the CSV file
    df = pd.read_csv("experiment_data/experiment_missing_data/all_roles_missing_data.csv")

    # Helper to get unique roles for a given column where value > 0
    def get_roles(col):
        return set(df.loc[df[col] > 0, "role"])

    # Columns of interest
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

    # Collect unique roles for each type
    steering_roles = {}
    for col, label in steering_cols:
        roles = get_roles(col)
        steering_roles[label] = roles
        print(f"{label}: {len(roles)} roles")
        print(sorted(roles))
        print()

    assistant_roles = {}
    for col, label in assistant_cols:
        roles = get_roles(col)
        assistant_roles[label] = roles
        print(f"{label}: {len(roles)} roles")
        print(sorted(roles))
        print()

    # Unique set and count for steering
    all_steering_roles = set().union(*steering_roles.values())
    print(f"Unique roles with any steering issue: {len(all_steering_roles)}")
    print(sorted(all_steering_roles))
    print()

    # Unique set and count for assistant
    all_assistant_roles = set().union(*assistant_roles.values())
    print(f"Unique roles with any assistant issue: {len(all_assistant_roles)}")
    print(sorted(all_assistant_roles))
    print()

    # Total unique roles with any issue
    all_roles = all_steering_roles.union(all_assistant_roles)
    print(f"Total unique roles with any issue: {len(all_roles)}")
    print(sorted(all_roles))

if __name__ == "__main__":
    main()
