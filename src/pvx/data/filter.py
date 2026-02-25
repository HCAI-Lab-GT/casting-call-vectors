import json
import os

import pandas as pd


def load_data(json_path: str) -> pd.DataFrame:
    """Load the JSON files and clean the null values."""
    json_files = os.listdir(json_path)
    contents = []
    for json_ in json_files:
        item = dict()
        with open(f"{json_path}/{json_}", "r") as f:
            data = json.load(f)
        item = {"name": json_.replace(".json", ""), **data["work_styles"]}
        contents.append(item)
    df = pd.DataFrame(contents)
    df.set_index("name", inplace=True)
    df.dropna(inplace=True)
    return df


def get_summaries(df):
    df = df / 10  # DR is in between 0-10, scaling it to 0-1
    df["non_zeros"] = 21 - (df == 0).sum(axis=1)

    # Avoiding extreme peaks in the distinctiveness by filtering out those that don't have median diversity of distinctiveness

    median_diversity = df["non_zeros"].median()
    df = df[df.non_zeros >= median_diversity]
    df.drop("non_zeros", axis=1, inplace=True)

    # now we needed items that are maximally spread-out across the 21-dimensions we have (each occupation has a 21-dimensiaonal vector)
    # A simple top-n selection across 21 dimensions should suffice now.

    num_needed = 250
    min_items_per_dim = round(num_needed / 21)
    highly_distinctive = set()
    for dim in df.columns:
        top_occ = df[dim].nlargest(min_items_per_dim).index

        highly_distinctive.update(top_occ)

    # Fill the rest based on their overall distinctiveness
    remaining = df.drop(highly_distinctive)
    num_needed = num_needed - len(highly_distinctive)
    rest = remaining.sum(axis=1).nlargest(num_needed)
    rest = set(rest)
    occ_filtered = highly_distinctive.union(rest)

    return occ_filtered, highly_distinctive


def main():
    json_path = "data/occupation_profiles"
    df = load_data(json_path)
    occ_filtered, highly_distinctive = get_summaries(df)
    with open("data/top_distinct.txt", "w") as f:
        for occ in highly_distinctive:
            f.write(f"{occ}\n")

    with open("data/filtered_occ.txt", "w") as f:
        for occ in occ_filtered:
            f.write(f"{occ}\n")


if __name__ == "__main__":
    main()
