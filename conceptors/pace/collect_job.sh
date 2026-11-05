#!/bin/bash
#SBATCH -J conceptors-collect
#SBATCH -q coe-ice
#SBATCH --gres=gpu:H100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
# Run persona_acts.py on one GPU. Pass persona_acts.py arguments; --out is relative to $CONCEPTORS_DATA/runs.
# Submit from the release root:
#   sbatch -o "$LOGS/%x-%j.out" conceptors/pace/collect_job.sh --preset smollm3-3b --out smollm3_quick \
#       --roles_dir @AA_ROLES --questions_file @AA_QUESTIONS --roles pirate assistant --n_questions 20
# Write @AA_ROLES / @AA_QUESTIONS for the Assistant Axis paths; they are replaced here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source conceptors/pace/env.sh
export HF_HUB_OFFLINE=1  # models are fetched beforehand by fetch_job.sh
AA_ROLES="$AA_DIR/data/roles/instructions"
AA_QUESTIONS="$AA_DIR/data/extraction_questions.jsonl"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) args+=(--out "$CONCEPTORS_DATA/runs/$2"); shift 2 ;;
    *) a="${1//@AA_ROLES/$AA_ROLES}"; args+=("${a//@AA_QUESTIONS/$AA_QUESTIONS}"); shift ;;
  esac
done
echo "release $(git rev-parse HEAD) on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd conceptors
"$PY" persona_acts.py "${args[@]}"
