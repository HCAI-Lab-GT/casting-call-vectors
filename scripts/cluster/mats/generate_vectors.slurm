#!/bin/bash
#SBATCH --job-name=pvx-generate
#SBATCH --output=logs/slurm/pvx-generate_%j.log
#SBATCH --error=logs/slurm/pvx-generate_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

cd ~/persona-vectors
source ~/.local/bin/env
source .venv/bin/activate

export HF_HOME=/mnt/nw/teams/team_krier_shavit_c9/.cache/huggingface

echo "Python: $(which python)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

echo ""
echo "========== VALIDATION =========="
python -c "
import torch
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel
from pvx.pvx_models.persona_dataset import PersonaDataset

assert torch.cuda.is_available(), 'CUDA not available!'
print('CUDA check: PASSED')

for trait in ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional']:
    ds = PersonaDataset.from_json(trait=trait)
    print(f'Dataset {trait}: {ds.num_questions} questions')
print('Dataset check: PASSED')
"

echo ""
echo "========== GENERATION =========="
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
TRAITS="${TRAITS:-realistic investigative artistic social enterprising conventional}"

echo "Model: $MODEL_ID"
echo "Traits: $TRAITS"

for trait in $TRAITS; do
    echo ""
    echo ">>> Generating vectors for: $trait"
    python -c "
import torch
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel

model = RIASECPersonaModel(
    target_model_id='$MODEL_ID',
    trait='$trait',
)
model.extract_persona_vector()
print(f'Vectors extracted for $trait')
print(f'  prompt_persona_vector shape: {model.prompt_persona_vector.shape}')
print(f'  response_persona_vector shape: {model.response_persona_vector.shape}')
# Safetensors saved automatically via __init__
"
    echo ">>> Completed: $trait"
done

echo ""
echo "========== RESULTS =========="
find persona_data/model_inits -name "*.safetensors" -exec ls -lh {} \;

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
