import subprocess
import yaml
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--model_type",
    type=str,
    default="marin-8b_prompt",
    help="Model type for all runs"
)
parser.add_argument("-m", "--model_name",
    type=str,
    default="marin-community/marin-8b-instruct",
    help="Model name for steering vector to be applied on"
)
parser.add_argument("-a", "--alpha",
    type=float,
    default=2.0,
    help="Alpha for steering"
)
args = parser.parse_args()

yaml_path = Path("configs/riasec_runs.yaml")
with open(yaml_path) as f:
    config = yaml.safe_load(f)

for role in config["roles"]:
    model = args.model_name
    alpha = args.alpha
    safe_model = model.replace("/", "_")
    outfile = f"logs/riasec_results/{args.model_type}/{role.replace(' ', '_')}_alpha{alpha}_{safe_model}_riasec.txt"
    Path("logs/riasec_results").mkdir(parents=True, exist_ok=True)
    print(f"Running: {role} {model} {alpha}")
    cmd = [
        "uv", "run", "python", "src/pvx/implementations/judges/riasec_judge.py",
        "-c", role, "-m", model, "-a", str(alpha), "-r"
    ]
    with open(outfile, "w") as out:
        subprocess.run(cmd, stdout=out, check=True)