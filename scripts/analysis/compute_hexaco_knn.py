import re
import ast
import numpy as np
from pathlib import Path

LOG_DIR = Path("logs/hexaco_results/olmo-7b_response")
OUT_DIR = Path("analysis/hexaco_knn")
K = 5

def extract_facets(text):
    match = re.search(r"===Hexaco Counts===\s*({.*?})\s*===Hexaco Answers===", text, re.DOTALL)
    if match:
        block = match.group(1)
        facets_match = re.search(r'"facets"\s*:\s*({.*?})\s*,', block, re.DOTALL)
        if facets_match:
            facets_str = facets_match.group(1)
            while re.search(r",\s*}", facets_str):
                facets_str = re.sub(r",\s*}", "}", facets_str)
            try:
                return ast.literal_eval(facets_str)
            except Exception:
                return {}
    return {}

def compute_knn(facet_matrix, idx, k):
    target = facet_matrix[idx]
    dists = np.linalg.norm(facet_matrix - target, axis=1)
    sorted_indices = np.argsort(dists)
    knn_indices = [i for i in sorted_indices if i != idx][:k]
    knn_dists = dists[knn_indices]
    return knn_indices, knn_dists

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for f in LOG_DIR.iterdir() if f.name.endswith(".txt")]
    facet_dicts = []
    file_names = []

    for f in files:
        with f.open("r") as fh:
            content = fh.read()
        facets = extract_facets(content)
        if facets:
            facet_dicts.append(facets)
            file_names.append(f.name)

    if not facet_dicts:
        return

    facet_keys = list(facet_dicts[0].keys())
    facet_matrix = np.array([[fd.get(k, 0.0) for k in facet_keys] for fd in facet_dicts])

    for idx, fname in enumerate(file_names):
        knn_indices, knn_dists = compute_knn(facet_matrix, idx, K)
        out_path = OUT_DIR / fname.replace(".txt", "_knn.txt")
        with out_path.open("w") as out_f:
            out_f.write(f"Role: {fname}\n")
            out_f.write(f"Top {K} nearest roles (by facet distance):\n")
            for i, dist in zip(knn_indices, knn_dists):
                out_f.write(f"{file_names[i]}: {dist:.4f}\n")

if __name__ == "__main__":
    main()