import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "logs/hexaco_results/olmo-7b_response"
OUT_DIR = "analysis/hexaco_barplots_mean"
DIFF_OUT_DIR = "analysis/hexaco_barplots_mean_difference"
DOMAIN_OUT_DIR = "analysis/hexaco_domain_barplots_mean"
DOMAIN_DIFF_OUT_DIR = "analysis/hexaco_domain_barplots_mean_difference"

def extract_facets_and_domains(text, fname=""):
    match = re.search(r"===Hexaco Counts===\s*({.*?})\s*===Hexaco Answers===", text, re.DOTALL)
    if match:
        block = match.group(1)
        facets_match = re.search(r'"facets"\s*:\s*({.*?})\s*,', block, re.DOTALL)
        domains_match = re.search(r'"domains"\s*:\s*({.*?})\s*[,}]', block, re.DOTALL)
        facets, domains = {}, {}
        if facets_match:
            facets_str = facets_match.group(1)
            while re.search(r",\s*}", facets_str):
                facets_str = re.sub(r",\s*}", "}", facets_str)
            try:
                facets = ast.literal_eval(facets_str)
            except Exception as e:
                print(f"Literal eval error in facets {fname}: {e}")
        if domains_match:
            domains_str = domains_match.group(1)
            while re.search(r",\s*}", domains_str):
                domains_str = re.sub(r",\s*}", "}", domains_str)
            try:
                domains = ast.literal_eval(domains_str)
            except Exception as e:
                print(f"Literal eval error in domains {fname}: {e}")
        return facets, domains
    return {}, {}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DIFF_OUT_DIR, exist_ok=True)
    os.makedirs(DOMAIN_OUT_DIR, exist_ok=True)
    os.makedirs(DOMAIN_DIFF_OUT_DIR, exist_ok=True)
    facet_values = {}
    domain_values = {}
    file_facets_domains = []

    # Collect facet and domain values from all files
    for fname in os.listdir(LOG_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(LOG_DIR, fname)
            with open(path, "r") as f:
                content = f.read()
            facets, domains = extract_facets_and_domains(content, fname)
            if facets or domains:
                file_facets_domains.append((fname, facets, domains))
                for k, v in facets.items():
                    facet_values.setdefault(k, []).append(v)
                for k, v in domains.items():
                    domain_values.setdefault(k, []).append(v)

    # Compute mean for each facet and domain
    mean_facets = {k: np.mean(vs) for k, vs in facet_values.items()}
    mean_domains = {k: np.mean(vs) for k, vs in domain_values.items()}

    # Plot facet and domain bar plots and difference plots for each file
    for fname, facets, domains in file_facets_domains:
        # Facet plots
        facet_labels = []
        file_facet_vals = []
        mean_facet_vals = []
        for k in mean_facets:
            if k in facets:
                facet_labels.append(k)
                file_facet_vals.append(facets[k])
                mean_facet_vals.append(mean_facets[k])
        x_facet = np.arange(len(facet_labels))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x_facet - width/2, file_facet_vals, width, label=f"{fname} Facet Value")
        plt.bar(x_facet + width/2, mean_facet_vals, width, label="Mean Facet Value")
        plt.xticks(x_facet, facet_labels, rotation=45, ha="right")
        plt.ylabel("HEXACO Facet Value")
        plt.title(f"HEXACO Facets: {fname} vs Mean")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, fname.replace(".txt", ".png"))
        plt.savefig(out_path)
        plt.close()

        diff_facet_vals = [file_val - mean_val for file_val, mean_val in zip(file_facet_vals, mean_facet_vals)]
        plt.figure(figsize=(12, 6))
        plt.bar(x_facet, diff_facet_vals, width=0.6)
        plt.xticks(x_facet, facet_labels, rotation=45, ha="right")
        plt.ylabel("Difference from Mean HEXACO Facet Value")
        plt.title(f"HEXACO Facet Differences: {fname} - Mean")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.tight_layout()
        diff_out_path = os.path.join(DIFF_OUT_DIR, fname.replace(".txt", ".png"))
        plt.savefig(diff_out_path)
        plt.close()

        # Domain plots
        domain_labels = []
        file_domain_vals = []
        mean_domain_vals = []
        for k in mean_domains:
            if k in domains:
                domain_labels.append(k)
                file_domain_vals.append(domains[k])
                mean_domain_vals.append(mean_domains[k])
        x_domain = np.arange(len(domain_labels))

        plt.figure(figsize=(8, 4))
        plt.bar(x_domain - width/2, file_domain_vals, width, label=f"{fname} Domain Value")
        plt.bar(x_domain + width/2, mean_domain_vals, width, label="Mean Domain Value")
        plt.xticks(x_domain, domain_labels, rotation=30, ha="right")
        plt.ylabel("HEXACO Domain Value")
        plt.title(f"HEXACO Domains: {fname} vs Mean")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        domain_out_path = os.path.join(DOMAIN_OUT_DIR, fname.replace(".txt", ".png"))
        plt.savefig(domain_out_path)
        plt.close()

        diff_domain_vals = [file_val - mean_val for file_val, mean_val in zip(file_domain_vals, mean_domain_vals)]
        plt.figure(figsize=(8, 4))
        plt.bar(x_domain, diff_domain_vals, width=0.6)
        plt.xticks(x_domain, domain_labels, rotation=30, ha="right")
        plt.ylabel("Difference from Mean HEXACO Domain Value")
        plt.title(f"HEXACO Domain Differences: {fname} - Mean")
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.tight_layout()
        domain_diff_out_path = os.path.join(DOMAIN_DIFF_OUT_DIR, fname.replace(".txt", ".png"))
        plt.savefig(domain_diff_out_path)
        plt.close()

        num_diff_facets = sum(1 for file_val, mean_val in zip(file_facet_vals, mean_facet_vals) if file_val != mean_val)
        num_diff_domains = sum(1 for file_val, mean_val in zip(file_domain_vals, mean_domain_vals) if file_val != mean_val)
        print(f"{fname}: {num_diff_facets} facets differ from the mean, {num_diff_domains} domains differ from the mean.")

if __name__ == "__main__":
    main()