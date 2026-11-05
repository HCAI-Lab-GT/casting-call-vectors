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
# volume root down to the venv, the HF cache and the release it runs from must be a real directory (not a symlink),
# ours, and closed to other writers. Paths are compared physically, and any stat or resolve failure refuses. This narrows but cannot close the window between the
# check and exec; only a sticky bit on the volume root (its owner's call) stops staff/ being renamed.
CONCEPTORS_VOLUME="${CONCEPTORS_VOLUME:-/storage/ice-shared/cs7634}"
conceptors_trusted() {
  local p d info owner mode kind phys inside
  if [ "$(realpath -- "$CONCEPTORS_VOLUME")" != "$CONCEPTORS_VOLUME" ]; then
    echo "env.sh: refusing to run, CONCEPTORS_VOLUME=$CONCEPTORS_VOLUME is not a physical path" >&2; return 1
  fi
  case "$CONCEPTORS_DATA" in "$CONCEPTORS_VOLUME"/*)
    if [ ! -d "$CONCEPTORS_DATA" ] || [ -L "$CONCEPTORS_DATA" ]; then
      echo "env.sh: refusing to run, $CONCEPTORS_DATA is missing or a symlink" >&2; return 1
    fi ;;
  esac
  for p in "$CONCEPTORS_DATA" "$UV_PROJECT_ENVIRONMENT" "$HF_HOME" "$PWD"; do
    # the prefix test is textual, so a path that reaches the volume through a symlink, .. or // must not slip past it
    phys=$(realpath -m -- "$p") || { echo "env.sh: refusing to run, cannot resolve $p" >&2; return 1; }
    inside=
    case "$p" in "$CONCEPTORS_VOLUME"/*) inside=1 ;; esac
    case "$phys" in "$CONCEPTORS_VOLUME"/*) inside=1 ;; esac
    [ -n "$inside" ] || continue
    if [ "$p" != "$phys" ]; then
      echo "env.sh: refusing to run, $p resolves to $phys; use the physical path" >&2; return 1
    fi
    d=$p
    while [ ! -e "$d" ] && [ ! -L "$d" ]; do d=$(dirname "$d"); done
    while [ "$d" != "$CONCEPTORS_VOLUME" ] && [ "$d" != / ]; do
      info=$(stat -c '%u %a %F' "$d") || { echo "env.sh: refusing to run, cannot stat $d" >&2; return 1; }
      read -r owner mode kind <<< "$info"
      if [ "$owner" != "$(id -u)" ] || [ "$kind" != directory ] || (( 8#$mode & 8#022 )); then
        echo "env.sh: refusing to run, $d is not a directory owned by $(id -un) and closed to other writers" >&2
        return 1
      fi
      d=$(dirname "$d")
    done
  done
}
conceptors_trusted || return 1
