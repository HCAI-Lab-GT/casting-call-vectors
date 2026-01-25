#!/usr/bin/env bash
# Phase 1: Full 150-persona extraction
#
# This script runs the full extraction pipeline for Phase 1.
# Requires a GPU with at least 16GB VRAM for OLMo-7B.
#
# Usage:
#   # Run locally with GPU
#   ./experiments/phase1/run_full_extraction.sh
#
#   # Run on SLURM cluster
#   sbatch experiments/phase1/run_full_extraction.sh
#
#SBATCH --job-name=pvx-phase1
#SBATCH --output=logs/phase1_%j.log
#SBATCH --error=logs/phase1_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

# Configuration
MODEL="allenai/OLMo-7B-Instruct"
LAYER=14
NUM_QUESTIONS=50
WANDB_PROJECT="pvx-phase1"
PERSONA_DIR="persona_data/vocational_personas/instructions"

echo "=================================================="
echo "PHASE 1: Full Persona Vector Extraction"
echo "=================================================="
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "Questions per persona: $NUM_QUESTIONS"
echo "Output: outputs/runs/{model_slug}/{run_id}/ (auto-generated)"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
elif [[ $(uname -m) == "arm64" ]]; then
    echo "Apple Silicon detected (MPS)"
else
    echo "WARNING: No GPU detected, extraction will be very slow"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Count personas
PERSONA_COUNT=$(ls -1 "$PERSONA_DIR"/*.json 2>/dev/null | grep -v default | wc -l | tr -d ' ')
echo "Found $PERSONA_COUNT personas in $PERSONA_DIR"
echo ""

# Verify RIASEC distribution
echo "RIASEC distribution:"
python3 -c "
import json
from pathlib import Path
from collections import Counter
counts = Counter()
for f in Path('$PERSONA_DIR').glob('*.json'):
    if f.stem == 'default': continue
    try:
        data = json.load(open(f))
        r = data.get('_metadata', {}).get('riasec_primary', '?')
        counts[r] += 1
    except: pass
for letter in 'RIASEC':
    print(f'  {letter}: {counts.get(letter, 0)}')
"
echo ""

# Check for existing runs (note: resume requires specifying a run directory)
echo "Run tracking enabled - outputs saved to outputs/runs/olmo-7b-instruct/{run_id}/"
echo "To resume a specific run, use: --resume outputs/runs/olmo-7b-instruct/{run_id}/vectors"
RESUME_FLAG=""
echo ""

# Run extraction
echo "Starting extraction..."
echo "=================================================="

uv run python scripts/run_extraction.py \
    --persona-dir "$PERSONA_DIR" \
    --model "$MODEL" \
    --layer "$LAYER" \
    --num-questions "$NUM_QUESTIONS" \
    --wandb-project "$WANDB_PROJECT" \
    --track \
    $RESUME_FLAG

echo ""
echo "=================================================="
echo "Extraction complete!"
echo ""
echo "Vectors saved to: outputs/runs/olmo-7b-instruct/{run_id}/"
echo ""
echo "Next: Run analysis on the latest run:"
echo "  uv run python -c \"from pvx.outputs import OutputManager; print(OutputManager().get_latest('olmo-7b-instruct').output_dir())\""
echo ""
echo "Or use the analysis script:"
echo "  uv run python experiments/phase1/run_riasec_analysis.py \\"
echo "    --vectors-dir outputs/runs/olmo-7b-instruct/{run_id}/vectors \\"
echo "    --wandb-project $WANDB_PROJECT"
