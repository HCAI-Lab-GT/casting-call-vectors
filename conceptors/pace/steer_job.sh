#!/bin/bash
#SBATCH -J conceptors-steer
#SBATCH -q coe-ice
#SBATCH --gres=gpu:H100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
# Run steer_eval.py on one GPU. --fit_run, --out and --select_from are run names under $CONCEPTORS_DATA/runs;
# write @AA_ROLES for the Assistant Axis roles folder. Submit from the release root:
#   sbatch -o "$LOGS/%x-%j.out" conceptors/pace/steer_job.sh --preset smollm3-3b --fit_run smollm3_quick \
#       --stage sweep --out steer_sweep_smollm3 --roles_dir @AA_ROLES
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source conceptors/pace/env.sh
export HF_HUB_OFFLINE=1
RUNS="$CONCEPTORS_DATA/runs"
AA_ROLES="$AA_DIR/data/roles/instructions"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fit_run|--out|--select_from|--default_run|--identity_replay) args+=("$1" "$RUNS/$2"); shift 2 ;;
    *) args+=("${1//@AA_ROLES/$AA_ROLES}"); shift ;;
  esac
done
echo "release $(git rev-parse HEAD) on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd conceptors
"$PY" steer_eval.py "${args[@]}"
