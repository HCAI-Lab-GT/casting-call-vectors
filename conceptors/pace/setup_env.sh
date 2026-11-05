#!/bin/bash
# Build the shared venv for a deployed release (run once per release, from its root, on a login node).
#   bash conceptors/pace/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$UV_CACHE_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$CONCEPTORS_DATA"/{hf_cache,runs,logs}
uv sync --frozen --no-dev
"$PY" -c "import torch, transformers, vllm; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__)"
