#!/bin/bash
#SBATCH -J conceptors-fetch
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
# Download model snapshots into $HF_HOME and clone the pinned Assistant Axis data. CPU only.
# Submit from the release root:  sbatch -o "$LOGS/%x-%j.out" conceptors/pace/fetch_job.sh MODEL [MODEL ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source conceptors/pace/env.sh
if [ ! -d "$AA_DIR/.git" ]; then
  git clone -q https://github.com/safety-research/assistant-axis "$AA_DIR"
fi
git -C "$AA_DIR" fetch -q origin "$AA_COMMIT" 2>/dev/null || true
git -C "$AA_DIR" checkout -q "$AA_COMMIT"
echo "assistant-axis at $(git -C "$AA_DIR" rev-parse HEAD)"
for model in "$@"; do
  "$UV_PROJECT_ENVIRONMENT/bin/hf" download "$model" --exclude "*.pth" "original/*" > /dev/null
  echo "fetched $model"
done
