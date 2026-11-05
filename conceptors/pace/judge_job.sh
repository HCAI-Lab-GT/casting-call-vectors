#!/bin/bash
#SBATCH -J conceptors-judge
#SBATCH -q coe-ice
#SBATCH --gres=gpu:H100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
# Serve $JUDGE_MODEL with vLLM on this node, judge one run folder, then stop the server.
# Runs after the subject-model job (chain with --dependency=afterok:<collect job id>).
# Submit from the release root:  sbatch -o "$LOGS/%x-%j.out" conceptors/pace/judge_job.sh RUN_NAME[,RUN_NAME...] [judge.py args]
# Several comma-separated runs share one judge server (one model load, one queue slot); they are judged in order.
# Another judge: sbatch --export=ALL,JUDGE_MODEL=google/gemma-3-27b-it ... (snapshot must be in $HF_HOME already)
# ANALYZE_KEY=identity|identity_r in --export also runs analyze_steer.py on each steer run after judging it.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source conceptors/pace/env.sh
export HF_HUB_OFFLINE=1
IFS=, read -r -a RUN_NAMES <<< "$1"; shift
RUN="$CONCEPTORS_DATA/runs/${RUN_NAMES[0]}"  # the server log goes in the first run's folder
# per-job compile caches on node-local disk: concurrent judge jobs sharing the NFS-home cache hit
# "OSError: [Errno 116] Stale file handle" inside torch inductor
# (random name, owner-only, ownership checked: the caches hold compiled code, so no predictable shared path)
JOB_TMP="$(mktemp -d "${TMPDIR:-/tmp}/conceptors-judge-${SLURM_JOB_ID:-$$}.XXXXXXXX")"
chmod 700 "$JOB_TMP"
if [ -L "$JOB_TMP" ] || [ "$(stat -c %u "$JOB_TMP")" != "$(id -u)" ]; then echo "unsafe cache dir $JOB_TMP"; exit 1; fi
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
# per-family serve flags: Mistral's repos ship their tokenizer only as tekken.json, so serve them in Mistral format
EXTRA=()
case "$JUDGE_MODEL" in mistralai/*) EXTRA=(--tokenizer-mode mistral --config-format mistral --load-format mistral) ;; esac
# JUDGE_TP > 1 for judges that need several GPUs (submit with a matching --gres, e.g. gpu:H200:2 for a 70B in bf16)
"$UV_PROJECT_ENVIRONMENT/bin/vllm" serve "$JUDGE_PATH" --port "$PORT" --host 127.0.0.1 --served-model-name judge \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --tensor-parallel-size "${JUDGE_TP:-1}" "${EXTRA[@]}" > "$LOG" 2>&1 &
SERVER=$!
# On exit, stop the server and never wait on it forever: a vLLM still loading can ignore SIGTERM, and a plain
# `wait` then held a failed job until its time limit (job 6826242, 5 Nov 2026).
cleanup() {
  if kill -0 "$SERVER" 2>/dev/null; then
    kill "$SERVER" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER" 2>/dev/null || break; sleep 2; done
    pkill -9 -P "$SERVER" 2>/dev/null || true
    kill -9 "$SERVER" 2>/dev/null || true
  fi
  rm -rf "$JOB_TMP"
}
trap cleanup EXIT
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
# Up to 45 min: with several judge jobs reading weights from the shared volume at once, loading took 18 min.
for _ in $(seq 1 540); do
  if ! kill -0 "$SERVER" 2>/dev/null; then echo "vLLM exited early; see $LOG"; tail -30 "$LOG"; exit 1; fi
  if ready; then break; fi
  sleep 5
done
ready || { echo "vLLM not ready after 45 min"; tail -30 "$LOG"; exit 1; }
echo "judge $JUDGE_MODEL ready on port $PORT; release $(git rev-parse HEAD)"
for name in "${RUN_NAMES[@]}"; do
  R="$CONCEPTORS_DATA/runs/$name"
  echo "judging $name"
  "$PY" conceptors/judge.py --run "$R" --roles_dir "$AA_DIR/data/roles/instructions" --backend local \
    --base_url "http://127.0.0.1:$PORT/v1" --judge_model judge --workers 32 "$@"
  if [ -n "${ANALYZE_KEY:-}" ]; then "$PY" -B conceptors/analyze_steer.py --run "$R" --identity_key "$ANALYZE_KEY"; fi
done
