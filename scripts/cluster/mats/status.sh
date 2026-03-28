#!/bin/bash
set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-mats-cluster}"
CLUSTER_DIR="${CLUSTER_DIR:-~/persona-vectors}"

echo "=========================================="
echo "Cluster Job Status"
echo "=========================================="

echo ""
echo ">>> Your jobs:"
ssh "$CLUSTER_HOST" "squeue -u \$USER -o '%.10i %.20j %.8T %.10M %.6D %R'"

echo ""
echo ">>> Recent job logs:"
ssh "$CLUSTER_HOST" "ls -lt $CLUSTER_DIR/logs/slurm/*.log 2>/dev/null | head -5" || echo "No logs yet"

echo ""
echo ">>> Tail of most recent log:"
LATEST=$(ssh "$CLUSTER_HOST" "ls -t $CLUSTER_DIR/logs/slurm/*.log 2>/dev/null | head -1")
if [ -n "$LATEST" ]; then
    echo "File: $LATEST"
    ssh "$CLUSTER_HOST" "tail -30 $LATEST"
else
    echo "No logs found"
fi

echo ""
echo ">>> Generated safetensors:"
ssh "$CLUSTER_HOST" "find $CLUSTER_DIR/persona_data/model_inits -name '*.safetensors' -exec ls -lh {} \; 2>/dev/null" || echo "None yet"

echo ""
echo "=========================================="
