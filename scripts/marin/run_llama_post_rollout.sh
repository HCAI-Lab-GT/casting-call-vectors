#!/bin/bash
# Post-rollout pipeline for Llama 1B expanded roles
# Run after generate_rollouts.py completes for all 49 roles
set -e

MODEL="meta-llama/Llama-3.2-1B-Instruct"
DEVICE="${1:-cuda:0}"

echo "=== Post-rollout pipeline for $MODEL on $DEVICE ==="

ROLLOUT_DIR="data/assistant_axis/rollouts/meta-llama__Llama-3.2-1B-Instruct"
N_ROLES=$(ls "$ROLLOUT_DIR"/*.json 2>/dev/null | wc -l)
echo "Found $N_ROLES role rollout files"

if [ "$N_ROLES" -lt 40 ]; then
    echo "ERROR: Only $N_ROLES roles found. Expected ~49. Rollouts may still be running."
    exit 1
fi

echo ""
echo "--- Step 1: Extract activations ---"
uv run python scripts/marin/assistant_axis/extract_activations.py \
    --model_id "$MODEL" --device "$DEVICE"

echo ""
echo "--- Step 2: Compute PCA ---"
uv run python scripts/marin/assistant_axis/compute_pca.py \
    --model_id "$MODEL" --n_components 20

echo ""
echo "--- Step 3: Compare RIASEC vs PCA ---"
uv run python scripts/marin/analysis/compare_approaches.py \
    --model_id "$MODEL" --n_pca_components 10

echo ""
echo "--- Step 4: Regenerate figures ---"
uv run python scripts/marin/analysis/paper_figures.py

echo ""
echo "=== Pipeline complete ==="
