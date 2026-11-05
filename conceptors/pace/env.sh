# Paths for conceptors jobs on PACE ICE. Source it; override any variable before sourcing.
# Data (models, AA clone, runs, logs) lives on the shared course volume; scratch holds only envs and caches.
export CONCEPTORS_DATA="${CONCEPTORS_DATA:-/storage/ice-shared/cs7634/staff/conceptors}"
export CONCEPTORS_SCRATCH="${CONCEPTORS_SCRATCH:-$HOME/scratch}"
export HF_HOME="${HF_HOME:-$CONCEPTORS_DATA/hf_cache}"
# the token stays in the private home dir; the shared course volume is readable by other users
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CONCEPTORS_SCRATCH/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$CONCEPTORS_SCRATCH/venvs/casting-call-vectors}"
export AA_DIR="${AA_DIR:-$CONCEPTORS_DATA/assistant-axis}"
export AA_COMMIT="${AA_COMMIT:-a98961956072224eaf244eb289d6c01700b63795}"
export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
