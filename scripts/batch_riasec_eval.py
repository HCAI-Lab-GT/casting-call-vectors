import subprocess
import yaml
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model_type",
    type=str,
    default="olmo-7b",
    help="Model type for all runs (default: olmo-7b)"
)
args = parser.parse_args()

yaml_path = Path("configs/riasec_runs.yaml")
with open(yaml_path) as f:
    config = yaml.safe_load(f)

for run in config["runs"]:
    role = run["role"]
    model = run["model"]
    alpha = run["alpha"]
    safe_model = model.replace("/", "_")
    outfile = f"logs/riasec_results/{args.model_type}/{role.replace(' ', '_')}_alpha{alpha}_{safe_model}_riasec.txt"
    Path("logs/riasec_results").mkdir(parents=True, exist_ok=True)
    print(f"Running: {role} {model} {alpha}")
    cmd = [
        "uv", "run", "python", "src/pvx/implementations/judges/riasec_judge.py",
        "-c", role, "-m", model, "-a", str(alpha)
    ]
    with open(outfile, "w") as out:
        subprocess.run(cmd, stdout=out, check=True)