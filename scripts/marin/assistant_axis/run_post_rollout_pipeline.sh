#!/bin/bash
# Run after Marin 8B rollouts complete on all GPUs.
# Extracts activations, computes PCA, runs comparison.
#
# Usage:
#   bash scripts/marin/assistant_axis/run_post_rollout_pipeline.sh [model_id] [gpu_id]
#   bash scripts/marin/assistant_axis/run_post_rollout_pipeline.sh marin-community/marin-8b-instruct 0

set -e

MODEL_ID="${1:-marin-community/marin-8b-instruct}"
GPU_ID="${2:-0}"
VENV=".venv/bin/python"

echo "=========================================="
echo "Post-Rollout Pipeline: ${MODEL_ID}"
echo "GPU: ${GPU_ID}"
echo "=========================================="

# Step 1: Count rollout files
SAFE_MODEL=$(echo "$MODEL_ID" | sed 's/\//__/g')
ROLLOUT_DIR="data/assistant_axis/rollouts/${SAFE_MODEL}"
N_FILES=$(ls "$ROLLOUT_DIR"/*.json 2>/dev/null | wc -l)
echo "Found ${N_FILES} rollout files in ${ROLLOUT_DIR}"

if [ "$N_FILES" -lt 10 ]; then
    echo "ERROR: Too few rollout files. Expected 51+. Aborting."
    exit 1
fi

# Step 2: Extract activations
echo ""
echo "Step 2: Extracting activations..."
CUDA_VISIBLE_DEVICES=$GPU_ID $VENV scripts/marin/assistant_axis/extract_activations.py \
    --model_id "$MODEL_ID" \
    --device cuda:0

# Step 3: Compute PCA
echo ""
echo "Step 3: Computing PCA..."
$VENV scripts/marin/assistant_axis/compute_pca.py \
    --model_id "$MODEL_ID"

# Step 4: Calibrate steering
echo ""
echo "Step 4: Calibrating steering magnitude..."
CUDA_VISIBLE_DEVICES=$GPU_ID $VENV scripts/marin/assistant_axis/calibrate_steering.py \
    --model_id "$MODEL_ID" \
    --device cuda:0

# Step 5: Compare RIASEC vs PCA
echo ""
echo "Step 5: Comparing approaches..."
$VENV scripts/marin/analysis/compare_approaches.py \
    --model_id "$MODEL_ID"

# Step 6: Generate visualizations
echo ""
echo "Step 6: Generating visualizations..."
$VENV scripts/marin/analysis/visualize.py \
    --model_id "$MODEL_ID"

# Step 7: Comprehensive figure
echo ""
echo "Step 7: Generating comprehensive figure..."
$VENV scripts/marin/analysis/comprehensive_figure.py

echo ""
echo "=========================================="
echo "Pipeline complete for ${MODEL_ID}"
echo "=========================================="
