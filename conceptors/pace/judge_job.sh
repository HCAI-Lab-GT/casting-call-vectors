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
# per-job compile caches on node-local disk: concurrent judge jobs sharing the NFS-home cache hit
# "OSError: [Errno 116] Stale file handle" inside torch inductor
JOB_TMP="${TMPDIR:-/tmp}/conceptors-judge-${SLURM_JOB_ID:-$$}"
export VLLM_CACHE_ROOT="$JOB_TMP/vllm" TORCHINDUCTOR_CACHE_DIR="$JOB_TMP/inductor" TRITON_CACHE_DIR="$JOB_TMP/triton"
mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
PORT="${JUDGE_PORT:-$((20000 + RANDOM % 20000))}"
LOG="$RUN/vllm_judge.log"
# Shared node: a per-job key (env vars and stdin only, never argv) keeps other users out of the server,
# and the readiness check requires that the socket on $PORT belongs to our own vLLM process tree.
export VLLM_API_KEY="$(openssl rand -hex 24)"
export LOCAL_JUDGE_API_KEY="$VLLM_API_KEY"
# serve the local snapshot folder: transformers 4.57 makes a hub call for repo ids even when offline
JUDGE_PATH="$("$PY" -c 'import os; from huggingface_hub import snapshot_download as s; print(s(os.environ["JUDGE_MODEL"], local_files_only=True))')"
echo "judge snapshot $(basename "$JUDGE_PATH")"
"$UV_PROJECT_ENVIRONMENT/bin/vllm" serve "$JUDGE_PATH" --port "$PORT" --host 127.0.0.1 --served-model-name judge \
  --max-model-len 8192 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
owned() {  # is the listener on $PORT our server or one of its children? (ss shows pids only for our own processes)
  local p
  for p in $(ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2); do
    if [ "$p" = "$SERVER" ] || pgrep -P "$SERVER" | grep -qx "$p"; then return 0; fi
  done
  return 1
}
ready() {
  owned && printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY" |
    curl -sf -H @- "http://127.0.0.1:$PORT/v1/models" > /dev/null
}
for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER" 2>/dev/null; then echo "vLLM exited early; see $LOG"; tail -30 "$LOG"; exit 1; fi
  if ready; then break; fi
  sleep 5
done
ready || { echo "vLLM not ready after 15 min"; tail -30 "$LOG"; exit 1; }
echo "judge $JUDGE_MODEL ready on port $PORT; release $(git rev-parse HEAD)"
"$PY" conceptors/judge.py --run "$RUN" --roles_dir "$AA_DIR/data/roles/instructions" --backend local \
  --base_url "http://127.0.0.1:$PORT/v1" --judge_model judge --workers 32 "$@"
