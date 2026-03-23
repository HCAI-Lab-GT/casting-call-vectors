#!/bin/bash
# Example SLURM job submission scripts for optimized roles pipeline

# ==============================================================================
# Single-Job Example: Run both CPU and GPU sequentially on same job
# ==============================================================================

# single_job.sh
#!/bin/bash
#SBATCH --job-name=roles-optimized-all
#SBATCH --output=logs/roles_optimized_%j.log
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

set -e

cd /path/to/persona-vectors

# Phase 1: CPU task (generates and judges Q/A)
# Even though we have a GPU, the CPU phase doesn't use it
python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --model "qwen2.5:7b-instruct" \
  --roles "Lawyers" "Doctor" "Engineer" \
  --target_pairs 40 \
  --temperature 0.9 \
  --max_new_tokens 2048 \
  --output_dir "./persona_data/model_qa_responses"

echo "✅ CPU phase completed at $(date)"

# Phase 2: GPU task (extracts activations)
# Now use the GPU that has been waiting
python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --model "qwen2.5:7b-instruct" \
  --roles "Lawyers" "Doctor" "Engineer" \
  --layer 14 \
  --qa_responses_dir "./persona_data/model_qa_responses" \
  --output_dir "./persona_data/model_inits"

echo "✅ GPU phase completed at $(date)"


# ==============================================================================
# Multi-Job Example: Run CPU and GPU in parallel (SLURM dependency)
# ==============================================================================

# submit_parallel_jobs.sh
#!/bin/bash

cd /path/to/persona-vectors

# Submit CPU job (no GPU needed)
CPU_JOB=$(sbatch --parsable \
  --job-name=roles-cpu \
  --output=logs/cpu_%j.log \
  --time=2:00:00 \
  --partition=cpu \
  --cpus-per-task=4 \
  --mem=16GB \
  cpu_job.sh)

echo "Submitted CPU job: $CPU_JOB"

# Submit GPU job (depends on CPU completion)
GPU_JOB=$(sbatch --parsable \
  --job-name=roles-gpu \
  --output=logs/gpu_%j.log \
  --time=2:00:00 \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=16GB \
  --dependency=afterok:$CPU_JOB \
  gpu_job.sh)

echo "Submitted GPU job: $GPU_JOB (depends on $CPU_JOB)"
echo ""
echo "Monitor progress with:"
echo "  squeue -j $CPU_JOB"
echo "  squeue -j $GPU_JOB"


# ==============================================================================
# cpu_job.sh: CPU-only task
# ==============================================================================

#!/bin/bash
#SBATCH --job-name=roles-cpu
#SBATCH --output=logs/cpu_%j.log
#SBATCH --time=2:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

set -e

cd /path/to/persona-vectors

echo "Starting CPU task at $(date)"

python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --model "qwen2.5:7b-instruct" \
  --roles "Lawyers" "Doctor" "Engineer" "Nurse" "Teacher" \
  --target_pairs 40 \
  --temperature 0.9 \
  --max_new_tokens 2048 \
  --output_dir "./persona_data/model_qa_responses"

echo "Completed CPU task at $(date)"


# ==============================================================================
# gpu_job.sh: GPU-only task
# ==============================================================================

#!/bin/bash
#SBATCH --job-name=roles-gpu
#SBATCH --output=logs/gpu_%j.log
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB

set -e

cd /path/to/persona-vectors

echo "Starting GPU task at $(date)"

python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --model "qwen2.5:7b-instruct" \
  --roles "Lawyers" "Doctor" "Engineer" "Nurse" "Teacher" \
  --layer 14 \
  --qa_responses_dir "./persona_data/model_qa_responses" \
  --output_dir "./persona_data/model_inits"

echo "Completed GPU task at $(date)"


# ==============================================================================
# Advanced: Multi-layer extraction (parallel GPU jobs)
# ==============================================================================

# submit_multilayer_jobs.sh
#!/bin/bash

cd /path/to/persona-vectors

# CPU job: Generate and judge Q/A
CPU_JOB=$(sbatch --parsable \
  --job-name=roles-cpu-multilayer \
  --output=logs/cpu_multilayer_%j.log \
  --partition=cpu \
  --cpus-per-task=4 \
  --mem=16GB \
  --time=2:00:00 \
  cpu_job.sh)

echo "Submitted CPU job: $CPU_JOB"

# Submit GPU jobs for different layers (all depend on CPU)
for LAYER in 12 14 16 18 20; do
  GPU_JOB=$(sbatch --parsable \
    --job-name=roles-gpu-layer$LAYER \
    --output=logs/gpu_layer${LAYER}_%j.log \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=16GB \
    --time=2:00:00 \
    --dependency=afterok:$CPU_JOB \
    bash -c "python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
      --model 'qwen2.5:7b-instruct' \
      --roles 'Lawyers' 'Doctor' 'Engineer' \
      --layer $LAYER \
      --qa_responses_dir './persona_data/model_qa_responses' \
      --output_dir './persona_data/model_inits'")

  echo "Submitted GPU job for layer $LAYER: $GPU_JOB"
done


# ==============================================================================
# Array Job: Process many roles in batches
# ==============================================================================

# submit_array_jobs.sh
#!/bin/bash

cd /path/to/persona-vectors

# Create list of roles
ROLES=(
  "Lawyers"
  "Doctor"
  "Engineer"
  "Nurse"
  "Teacher"
  "Accountant"
  "Architect"
  "Consultant"
  "Analyst"
  "Manager"
)

# Submit array for CPU (process roles in parallel)
CPU_JOB=$(sbatch --parsable \
  --job-name=roles-cpu-array \
  --output=logs/cpu_array_%a_%j.log \
  --partition=cpu \
  --cpus-per-task=4 \
  --mem=16GB \
  --time=2:00:00 \
  --array=0-$((${#ROLES[@]}-1)) \
  bash -c "python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
    --model 'qwen2.5:7b-instruct' \
    --roles '${ROLES[$SLURM_ARRAY_TASK_ID]}' \
    --target_pairs 40")

echo "Submitted CPU array job: $CPU_JOB with ${#ROLES[@]} tasks"

# Submit array for GPU (process roles in parallel, depends on CPU)
GPU_JOB=$(sbatch --parsable \
  --job-name=roles-gpu-array \
  --output=logs/gpu_array_%a_%j.log \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=16GB \
  --time=2:00:00 \
  --array=0-$((${#ROLES[@]}-1)) \
  --dependency=afterok:$CPU_JOB \
  bash -c "python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
    --model 'qwen2.5:7b-instruct' \
    --roles '${ROLES[$SLURM_ARRAY_TASK_ID]}' \
    --layer 14")

echo "Submitted GPU array job: $GPU_JOB with ${#ROLES[@]} tasks"


# ==============================================================================
# Rolling submission: Balance CPU and GPU load
# ==============================================================================

# rolling_submission.sh
# Idea: Don't submit all CPU jobs at once, submit in waves
# as GPUs become free

#!/bin/bash

cd /path/to/persona-vectors

ROLES=(
  "Lawyers" "Doctor" "Engineer" "Nurse" "Teacher"
  "Accountant" "Architect" "Consultant" "Analyst" "Manager"
)

ROLES_PER_BATCH=5
NUM_BATCHES=$(( (${#ROLES[@]} + ROLES_PER_BATCH - 1) / ROLES_PER_BATCH ))

for BATCH in $(seq 0 $((NUM_BATCHES - 1))); do
  START=$((BATCH * ROLES_PER_BATCH))
  END=$((START + ROLES_PER_BATCH))
  BATCH_ROLES=("${ROLES[@]:$START:$ROLES_PER_BATCH}")

  echo "Submitting batch $BATCH with roles: ${BATCH_ROLES[@]}"

  # Submit CPU job for this batch
  CPU_JOB=$(sbatch --parsable \
    --job-name=roles-cpu-batch$BATCH \
    --partition=cpu \
    --cpus-per-task=4 \
    --mem=16GB \
    --time=2:00:00 \
    bash -c "python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
      --model 'qwen2.5:7b-instruct' \
      --roles $(printf '%s ' "${BATCH_ROLES[@]}")")

  # Submit GPU job for this batch (depends on its CPU job)
  sbatch \
    --job-name=roles-gpu-batch$BATCH \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=2 \
    --mem=16GB \
    --time=2:00:00 \
    --dependency=afterok:$CPU_JOB \
    bash -c "python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
      --model 'qwen2.5:7b-instruct' \
      --roles $(printf '%s ' "${BATCH_ROLES[@]}")"

  echo "  CPU job: $CPU_JOB"
done

echo "All batches submitted!"


# ==============================================================================
# Monitoring script
# ==============================================================================

# monitor.sh
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== SLURM Job Status ===${NC}"
echo ""

echo -e "${GREEN}Running jobs:${NC}"
squeue --states=RUNNING --format="%.8i %.30j %.8T %.10M %.10L"
echo ""

echo -e "${GREEN}Pending jobs:${NC}"
squeue --states=PENDING --format="%.8i %.30j %.8T %.10M"
echo ""

echo -e "${GREEN}Recent completions (past 1 hour):${NC}"
sacct --starttime="1 hour ago" --format=JobID,JobName,State,Elapsed%20 \
  | grep -E "(COMPLETED|FAILED|CANCELLED)"


# ==============================================================================
# Utility: Check Q/A generation status
# ==============================================================================

# check_qa_status.sh
#!/bin/bash

cd /path/to/persona-vectors

echo "=== Q/A Response Generation Status ==="
echo ""

QA_DIR="./persona_data/model_qa_responses"
if [ ! -d "$QA_DIR" ]; then
  echo "No Q/A responses generated yet"
  exit 0
fi

for MODEL_DIR in "$QA_DIR"/*; do
  if [ -d "$MODEL_DIR" ]; then
    MODEL=$(basename "$MODEL_DIR")
    echo "Model: $MODEL"

    for ROLE_FILE in "$MODEL_DIR"/*.json; do
      if [ -f "$ROLE_FILE" ]; then
        ROLE=$(basename "$ROLE_FILE" .json)
        POSITIVE_COUNT=$(jq '.positive | length' "$ROLE_FILE")
        BASE_COUNT=$(jq '.base | length' "$ROLE_FILE")
        echo "  $ROLE: $POSITIVE_COUNT positive, $BASE_COUNT base"
      fi
    done

    echo ""
  fi
done
