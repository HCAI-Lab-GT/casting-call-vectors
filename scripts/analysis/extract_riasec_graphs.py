from pathlib import Path
import re
import ast
import matplotlib.pyplot as plt

# Baseline dictionary (replace with your actual baseline)
baseline_counts = {
    "Realistic": 9,
    "Investigative": 6,
    "Artistic": 8,
    "Social": 10,
    "Enterprising": 10,
    "Conventional": 9,
}

input_dir = Path("logs/riasec_results/olmo-7b_response")
output_dir = Path("analysis/riasec_barplots_mean")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_counts_from_log(log_path):
    pattern = re.compile(r"===RIASEC Counts===.*?\n.*?(\{.*\})", re.DOTALL)
    text = log_path.read_text()
    match = pattern.search(text)
    if not match:
        return None
    try:
        counts = ast.literal_eval(match.group(1))
        return counts
    except Exception as e:
        print(f"⚠️ Failed to parse counts in {log_path}: {e}")
        return None

for log_file in input_dir.glob("*.txt"):
    counts = extract_counts_from_log(log_file)
    if counts is None:
        print(f"⚠️ No counts found in {log_file}")
        continue
    concept = log_file.stem.split("_alpha")[0].replace("_", " ")
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(baseline_counts.keys())
    values = [counts.get(k, 0) for k in keys]
    baseline_values = [baseline_counts[k] for k in keys]
    ax.bar(keys, values, alpha=0.7, label="Olmo-7B Response")
    ax.bar(keys, baseline_values, alpha=0.4, label="Baseline")
    ax.set_title(f"RIASEC Counts: {concept}")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / f"{log_file.stem}_barplot.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

# What was changed and why:
# - Script iterates through olmo-7b_response logs, extracts RIASEC counts, compares to baseline, and saves bar plots.
# - Used pathlib for file handling, matplotlib for plotting, ast for safe dict parsing, and re for log parsing.
# - Output folder is created if it does not exist.
# - Prints warnings for missing or unparsable logs.