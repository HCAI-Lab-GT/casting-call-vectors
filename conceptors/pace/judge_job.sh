#!/bin/bash
#SBATCH -J conceptors-judge
#SBATCH -q coe-ice
#SBATCH --gres=gpu:H100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
# Serve $JUDGE_MODEL with vLLM on this node, judge one run folder, then stop the server.
# Runs after the subject-model job (chain with --dependency=afterok:<collect job id>).
# Submit from the release root:  sbatch -o "$LOGS/%x-%j.out" conceptors/pace/judge_job.sh RUN_NAME [judge.py args]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source conceptors/pace/env.sh
export HF_HUB_OFFLINE=1
RUN="$CONCEPTORS_DATA/runs/$1"; shift
PORT="${JUDGE_PORT:-$((20000 + RANDOM % 20000))}"
LOG="$RUN/vllm_judge.log"
"$UV_PROJECT_ENVIRONMENT/bin/vllm" serve "$JUDGE_MODEL" --port "$PORT" --host 127.0.0.1 --served-model-name judge \
  --max-model-len 8192 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then break; fi
  if ! kill -0 "$SERVER" 2>/dev/null; then echo "vLLM exited early; see $LOG"; tail -30 "$LOG"; exit 1; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "vLLM not ready after 15 min"; tail -30 "$LOG"; exit 1; }
echo "judge $JUDGE_MODEL ready on port $PORT; release $(git rev-parse HEAD)"
"$PY" conceptors/judge.py --run "$RUN" --roles_dir "$AA_DIR/data/roles/instructions" --backend local \
  --base_url "http://127.0.0.1:$PORT/v1" --judge_model judge --workers 32 "$@"
