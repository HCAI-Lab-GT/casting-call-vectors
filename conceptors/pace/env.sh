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
