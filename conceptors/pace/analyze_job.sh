#!/bin/bash
#SBATCH -J conceptors-analyze
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
# Run analyze_acts.py on one or more run folders (CPU only; login nodes kill long CPU work).
# Submit from the release root:  sbatch -o "$LOGS/%x-%j.out" conceptors/pace/analyze_job.sh RUN [RUN ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source conceptors/pace/env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "release $(git rev-parse HEAD) on $(hostname)"
for run in "$@"; do
  start=$(date +%s)
  "$PY" conceptors/analyze_acts.py --run "$CONCEPTORS_DATA/runs/$run"
  echo "$run analyzed in $(( $(date +%s) - start ))s"
done
