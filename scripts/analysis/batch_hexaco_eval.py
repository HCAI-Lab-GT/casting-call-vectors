import json
import argparse
from pathlib import Path

import torch
import yaml

from pvx import setup_logging
from pvx.implementations.roles.role_persona_model import RolePersonaModel
from pvx.implementations.judges.hexaco_judge import HexacoJudge


logger = setup_logging(name="batch-riasec-eval")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--model_type",
    type=str,
    default="olmo-7b_response",
    help="Model type for all runs"
)
parser.add_argument("-m", "--model_name",
    type=str,
    default="allenai/Olmo-3-7B-Instruct",
    help="Model name for steering vector to be applied on"
)
parser.add_argument("-a", "--alpha",
    type=float,
    default=1.0,
    help="Alpha for steering"
)
parser.add_argument("-r", "--roles_run",
    type=str,
    default="all_roles",
    help="Key for roles to run in config file"
)
parser.add_argument("-y", "--config_path",
    type=str,
    default="./configs/scoring/hexaco_100.json",
    help="Path to interest profiler config"
)
args = parser.parse_args()

yaml_path = Path("configs/psychometric_runs.yaml")
with open(yaml_path) as f:
    config = yaml.safe_load(f)

logger.info("Loading model: %s", args.model_name)
pvx = RolePersonaModel.base_model(target_model_id=args.model_name, layer=16)
hexaco_judge = HexacoJudge(hexaco_path=args.config_path)

safe_model = args.model_name.replace("/", "_")
Path(f"logs/hexaco_results/{args.model_type}").mkdir(parents=True, exist_ok=True)

for role in config[args.roles_run]:
    json_path = Path(f"./persona_data/model_inits/{role}_persona_initialization/{args.model_name}.json")
    if not json_path.exists():
        logger.warning("No saved vectors for role '%s', skipping", role)
        continue

    with open(json_path) as f:
        data = json.load(f)

    pvx.prompt_persona_vector = torch.tensor(data["prompt_persona_vector"])
    pvx.response_persona_vector = torch.tensor(data["response_persona_vector"])
    pvx._persona_base = None
    pvx._persona_base_key = None

    logger.info("Running: %s | alpha=%s", role, args.alpha)
    responses, answers, scores = hexaco_judge(pvx, args.alpha, 10, 0.1)

    outfile = f"logs/hexaco_results/{args.model_type}/{role.replace(' ', '_')}_alpha{args.alpha}_{safe_model}_hexaco.txt"
    with open(outfile, "w") as f:
        f.write(f"===CONCEPT===\n{role}\n")
        f.write(f"===Hexaco Counts===\n{json.dumps(scores, indent=2)}\n")
        f.write(f"===Hexaco Answers===\n{json.dumps(answers, indent=2)}\n")

    logger.info("Scores for '%s': %s", role, scores)
