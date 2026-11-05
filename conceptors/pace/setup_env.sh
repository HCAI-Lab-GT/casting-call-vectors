#!/bin/bash
# Build the shared venv for a deployed release (from its root, on a login node). Only needed when uv.lock changes.
#   bash conceptors/pace/setup_env.sh
# --no-install-project: conceptors/ runs as scripts and never imports pvx, so the venv doesn't point at any one
# release folder (an editable install would break when that release is pruned).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$UV_CACHE_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$CONCEPTORS_DATA"/{hf_cache,runs,logs,releases}
uv sync --frozen --no-dev --no-install-project
"$PY" -c "import torch, transformers, vllm; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__)"
