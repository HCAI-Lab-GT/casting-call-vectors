#!/bin/bash
#SBATCH -J conceptors-replay
#SBATCH -q coe-ice
#SBATCH --gres=gpu:H100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
# Run replay_acts.py on one GPU: teacher-force saved hold turns and answers, save their activations.
# Run names and --out resolve against $CONCEPTORS_DATA/runs. Submit from the release root:
#   sbatch -o "$LOGS/%x-%j.out" conceptors/pace/replay_job.sh --preset smollm3-3b --out replay_smollm3 \
#       --holds steer_hold_v2_pirate_prompt --answers smollm3_full --roles assistant pirate
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source conceptors/pace/env.sh
export HF_HUB_OFFLINE=1  # models are fetched beforehand by fetch_job.sh
echo "release $(git rev-parse HEAD) on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd conceptors
"$PY" replay_acts.py --runs_root "$CONCEPTORS_DATA/runs" "$@"
