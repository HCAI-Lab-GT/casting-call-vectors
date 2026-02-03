#!/bin/bash
set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-mats-cluster}"
CLUSTER_DIR="${CLUSTER_DIR:-~/persona-vectors}"
LOCAL_DIR="${LOCAL_DIR:-.}"

echo "=========================================="
echo "Syncing results from cluster"
echo "=========================================="

echo ""
echo ">>> Checking for generated safetensors..."
ssh "$CLUSTER_HOST" "find $CLUSTER_DIR/persona_data/model_inits -name '*.safetensors' 2>/dev/null" || {
    echo "No safetensors files found on cluster yet."
    exit 0
}

echo ""
echo ">>> Syncing safetensors files..."
rsync -avz --progress \
    "$CLUSTER_HOST:$CLUSTER_DIR/persona_data/model_inits/" \
    "$LOCAL_DIR/persona_data/model_inits/"

echo ""
echo ">>> Syncing manifest.json..."
rsync -avz --progress \
    "$CLUSTER_HOST:$CLUSTER_DIR/persona_data/model_inits/manifest.json" \
    "$LOCAL_DIR/persona_data/model_inits/" 2>/dev/null || echo "No manifest.json yet"

echo ""
echo ">>> Syncing logs..."
mkdir -p logs/slurm
rsync -avz --progress \
    "$CLUSTER_HOST:$CLUSTER_DIR/logs/slurm/" \
    "$LOCAL_DIR/logs/slurm/" 2>/dev/null || echo "No logs yet"

echo ""
echo ">>> Local safetensors files:"
find "$LOCAL_DIR/persona_data/model_inits" -name "*.safetensors" -exec ls -lh {} \; 2>/dev/null || echo "None yet"

echo ""
echo "=========================================="
echo "Sync complete!"
echo ""
echo "Next steps:"
echo "  1. Review the generated files"
echo "  2. Commit with: jj commit -m 'feat: add generated persona vectors'"
echo "  3. Push with: jj git push --bookmark safetensors-migration"
echo "=========================================="
