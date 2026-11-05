# Paths for conceptors jobs on PACE ICE. Source it; override any variable before sourcing.
# Everything this project writes on PACE lives on the CS 7634 course volume: runs, logs, models, the venv, the uv
# cache and the code releases. Nothing goes to scratch (300 GB / 1M-file quota) or home (30 GB). See AGENTS.md.
export CONCEPTORS_DATA="${CONCEPTORS_DATA:-/storage/ice-shared/cs7634/staff/conceptors}"
export HF_HOME="${HF_HOME:-$CONCEPTORS_DATA/hf_cache}"
# the token stays in the private home dir; the shared course volume is readable by other users
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CONCEPTORS_DATA/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$CONCEPTORS_DATA/venvs/casting-call-vectors}"
export AA_DIR="${AA_DIR:-$CONCEPTORS_DATA/assistant-axis}"
export AA_COMMIT="${AA_COMMIT:-a98961956072224eaf244eb289d6c01700b63795}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

# The course volume's top level is world-writable with no sticky bit, so another user could rename staff/ aside and
# plant a look-alike tree. Before a job runs code or loads weights from the volume, every folder from just below the
# volume root down to the venv, the HF cache and the release it runs from must be ours and closed to other writers.
CONCEPTORS_VOLUME="${CONCEPTORS_VOLUME:-/storage/ice-shared/cs7634}"
conceptors_trusted() {
  local p d
  for p in "$UV_PROJECT_ENVIRONMENT" "$HF_HOME" "$PWD"; do
    case "$p" in "$CONCEPTORS_VOLUME"/*) ;; *) continue ;; esac
    d=$p
    while [ ! -e "$d" ]; do d=$(dirname "$d"); done
    while [ "$d" != "$CONCEPTORS_VOLUME" ] && [ "$d" != / ]; do
      if [ ! -O "$d" ] || [ -n "$(find "$d" -maxdepth 0 -perm /022)" ]; then
        echo "env.sh: refusing to run, $d is not owned by $(id -un) or is writable by others" >&2
        return 1
      fi
      d=$(dirname "$d")
    done
  done
}
conceptors_trusted || return 1
