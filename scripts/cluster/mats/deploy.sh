#!/bin/bash
set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-mats-cluster}"
CLUSTER_DIR="${CLUSTER_DIR:-~/persona-vectors}"

echo "=========================================="
echo "Deploying to cluster: $CLUSTER_HOST"
echo "=========================================="

echo ""
echo ">>> Running local pre-flight checks..."
uv run pytest tests/ -q --tb=no || {
    echo "ERROR: Local tests failed. Fix before deploying."
    exit 1
}
echo "Local tests: PASSED"

echo ""
echo ">>> Syncing code to cluster via git..."
# Push local changes first
echo "Pushing local changes..."
jj git push --bookmark safetensors-migration 2>/dev/null || git push origin safetensors-migration

# Pull on cluster (uses SSH agent forwarding)
echo "Pulling on cluster..."
ssh -A "$CLUSTER_HOST" "cd $CLUSTER_DIR && source ~/.local/bin/env && git pull"

echo ""
echo ">>> Creating logs directory..."
ssh "$CLUSTER_HOST" "mkdir -p $CLUSTER_DIR/logs/slurm"

echo ""
echo ">>> Verifying cluster environment..."
ssh "$CLUSTER_HOST" "cd $CLUSTER_DIR && source ~/.local/bin/env && source .venv/bin/activate && python -c 'from pvx.pvx_models.abstract_persona_model import AbstractPersonaModel; print(\"Import check: OK\")'"

echo ""
echo "=========================================="
echo "Deploy complete!"
echo ""
echo "To submit job:"
echo "  ssh $CLUSTER_HOST"
echo "  cd $CLUSTER_DIR"
echo "  sbatch scripts/cluster/mats/generate_vectors.slurm"
echo ""
echo "Or quick submit:"
echo "  ssh $CLUSTER_HOST 'cd $CLUSTER_DIR && sbatch scripts/cluster/mats/generate_vectors.slurm'"
echo "=========================================="
