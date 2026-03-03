import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "logs/hexaco_results/olmo-7b_response"
OUT_DIR = "analysis/hexaco_barplots_mean"
DIFF_OUT_DIR = "analysis/hexaco_barplots_mean_difference"

def extract_facets(text, fname=""):
    match = re.search(r"===Hexaco Counts===\s*({.*?})\s*===Hexaco Answers===", text, re.DOTALL)
    if match:
        block = match.group(1)
        facets_match = re.search(r'"facets"\s*:\s*({.*?})\s*,', block, re.DOTALL)
        if facets_match:
            facets_str = facets_match.group(1)
            # Remove trailing commas
            while re.search(r",\s*}", facets_str):
                facets_str = re.sub(r",\s*}", "}", facets_str)
            try:
                facets = ast.literal_eval(facets_str)
                return facets
            except Exception as e:
                print(f"Literal eval error in {fname}: {e}")
    return {}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DIFF_OUT_DIR, exist_ok=True)
    facet_values = {}
    file_facets = []

    # Collect facet values from all files
    for fname in os.listdir(LOG_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(LOG_DIR, fname)
            with open(path, "r") as f:
                content = f.read()
            facets = extract_facets(content, fname)
            if facets:
                file_facets.append((fname, facets))
                for k, v in facets.items():
                    facet_values.setdefault(k, []).append(v)

    # Compute mean for each facet
    mean_facets = {k: np.mean(vs) for k, vs in facet_values.items()}

    # Plot side-by-side bar plots and difference bar plots for each file
    for fname, facets in file_facets:
        labels = []
        file_vals = []
        mean_vals = []
        for k in mean_facets:
            if k in facets:
                labels.append(k)
                file_vals.append(facets[k])
                mean_vals.append(mean_facets[k])
        x = np.arange(len(labels))
        width = 0.35

        # Side-by-side bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, file_vals, width, label=f"{fname} Facet Value")
        plt.bar(x + width/2, mean_vals, width, label="Mean Facet Value")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("HEXACO Facet Value")
        plt.title(f"HEXACO Facets: {fname} vs Mean")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, fname.replace(".txt", ".png"))
        plt.savefig(out_path)
        plt.close()

        # Difference bar plot
        diff_vals = [file_val - mean_val for file_val, mean_val in zip(file_vals, mean_vals)]
        plt.figure(figsize=(12, 6))
        plt.bar(x, diff_vals, width=0.6)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Difference from Mean HEXACO Facet Value")
        plt.title(f"HEXACO Facet Differences: {fname} - Mean")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.tight_layout()
        diff_out_path = os.path.join(DIFF_OUT_DIR, fname.replace(".txt", ".png"))
        plt.savefig(diff_out_path)
        plt.close()

        # Count number of facets that are different
        num_diff = sum(1 for file_val, mean_val in zip(file_vals, mean_vals) if file_val != mean_val)
        print(f"{fname}: {num_diff} facets differ from the mean.")

if __name__ == "__main__":
    main()