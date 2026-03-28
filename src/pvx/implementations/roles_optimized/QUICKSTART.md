# Quick Start Guide: Roles Optimized (OpenRouter + GPU)

## What is roles_optimized?

A refactored persona vector extraction pipeline that:
- Uses **OpenRouter API** for generating answers (CPU-only, no GPU)
- Uses **GPU-loaded model** for extracting activations (GPU only when needed)
- Maximizes GPU utilization by eliminating idle time

## Key Architecture

```
Answer Model: OpenRouter API (no GPU needed)
     ↓
Judges responses (CPU)
     ↓
Saves Q/A to JSON
     ↓
Extraction Model: Loads on GPU (GPU optimized)
     ↓
Extracts activations from Q/A
     ↓
Saves persona vectors
```

## Setup

### 1. Set OpenRouter API Key

```bash
export OPENROUTER_API_KEY="your-api-key"
```

Or pass via command line (see examples below).

### 2. Install Dependencies

```bash
pip install openai  # For OpenRouter API client
```

## Basic Usage

### Phase 1: CPU (Generate Q/A via OpenRouter)

```bash
python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --answer_model "allenai/Olmo-3-7B-Instruct" \
  --roles "Lawyers" "Doctor" \
  --target_pairs 40
```

**Output**: `persona_data/model_qa_responses/{model}/{role}.json`
**GPU Used**: None ✓ (remote API call)

### Phase 2: GPU (Extract Activations)

```bash
python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --activation_extraction_model "allenai/Olmo-3-7B-Instruct" \
  --answer_model "allenai/Olmo-3-7B-Instruct" \
  --roles "Lawyers" "Doctor" \
  --layer 14
```

**Output**: `persona_data/model_inits/{role}/.../` (safetensors)
**GPU Used**: Yes ✓ (100% utilization)

## SLURM Usage

### Submit Both Jobs (Recommended)

```bash
# CPU job (no GPU needed)
CPU_JOB=$(sbatch --job-name=roles-cpu --partition=cpu cpu_job.sh | awk '{print $4}')

# GPU job (waits for CPU, gets Q/A ready)
sbatch --job-name=roles-gpu --partition=gpu --gres=gpu:1 \
  --dependency=afterok:$CPU_JOB \
  gpu_job.sh
```

**Benefit**: GPU starts immediately with Q/A ready (zero idle time)

### Single Job (Simple)

```bash
sbatch --partition=gpu --gres=gpu:1 single_job.sh
```

(Runs CPU then GPU sequentially on same job)

## Features

✅ **Different Answer & Extraction Models** - Can use different models for each phase
✅ **OpenRouter API** - No local GPU for response generation
✅ **GPU-Only Activation** - Load model on GPU only when needed
✅ **Skip Missing Roles** - Doesn't fail if a role dataset doesn't exist
✅ **Q/A Caching** - Saves responses to JSON (debug, resume)
✅ **Pre-tokenization** - All messages tokenized before GPU loop
✅ **Random Shuffling** - Ensures variance in samples
✅ **Incremental Extraction** - Can resume from partial results
✅ **SLURM-Friendly** - Job scheduling and dependencies

## Common Tasks

### Process Multiple Roles

```bash
# CPU phase
python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --roles "Lawyers" "Doctor" "Engineer" "Nurse" "Teacher"

# GPU phase
python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --roles "Lawyers" "Doctor" "Engineer" "Nurse" "Teacher"
```

### Different Answer and Extraction Models

```bash
# CPU: Use fast small model for generation via API
python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --answer_model "meta-llama/Llama-2-7b"

# GPU: Use better model for extraction
python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --activation_extraction_model "allenai/Olmo-3-7B-Instruct" \
  --answer_model "meta-llama/Llama-2-7b"
```

### Extract from Multiple Layers

```bash
for LAYER in 12 14 16 18 20; do
  python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
    --layer $LAYER \
    --roles "Lawyers"
done
```

### Inspect Generated Q/A

```bash
# View number of Q/A pairs
jq '.positive | length' \
  persona_data/model_qa_responses/allenai__Olmo-3-7B-Instruct/Lawyers.json

# Extract specific Q/A
jq '.positive[0]' \
  persona_data/model_qa_responses/allenai__Olmo-3-7B-Instruct/Lawyers.json
```

### Resume Interrupted GPU Job

If GPU extraction was interrupted, re-run the same command:

```bash
python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --activation_extraction_model "allenai/Olmo-3-7B-Instruct" \
  --answer_model "allenai/Olmo-3-7B-Instruct" \
  --roles "Lawyers"
```

Will resume from saved Q/A and recompute (doesn't need to regenerate).

## Troubleshooting

### "OPENROUTER_API_KEY must be provided"
```
Error: OPENROUTER_API_KEY must be provided or set as environment variable
Solution: Export API key or pass --api_key
  export OPENROUTER_API_KEY="sk-..."
  # or
  python ... --api_key "sk-..."
```

### "Q/A responses not found"
```
Error: Q/A responses not found at {path}
Solution: Run CPU task first
  python -m roles_optimized_cpu --roles "Lawyers"
```

### "Role dataset not found"
```
Error: Role dataset not found for {role}
Solution: Check persona_data/role_datasets/{role}.json exists
          Or skip missing roles (pipeline continues)
```

### GPU Out of Memory
```
Error: CUDA out of memory
Solutions:
  1. Use smaller model
  2. Extract fewer roles per job (split across jobs)
  3. Reduce target_pairs
```

## Architecture Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Answer Model | Loaded locally | OpenRouter API |
| GPU for Answers | Yes (wasteful) | No ✓ |
| Extraction Model | Same process | Separate phase |
| GPU for Extract | Yes | Yes ✓ |
| GPU Idle Time | High | None ✓ |
| SLURM-Friendly | No | Yes ✓ |
| Debuggable | No | Yes ✓ |

## File Locations

```
src/pvx/implementations/roles_optimized/
├── roles_optimized_cpu.py      # CPU task (OpenRouter)
├── roles_optimized_gpu.py      # GPU task (Extraction)
├── README.md                   # Full documentation
├── QUICKSTART.md              # This file
├── SLURM_EXAMPLES.sh          # Job submission examples
└── example.py                 # Usage examples
```

## Next Steps

1. **Read full docs**: `README.md` for comprehensive guide
2. **Review SLURM**: `SLURM_EXAMPLES.sh` for job submission patterns
3. **See examples**: `example.py` for usage walkthrough
4. **Submit jobs**: Use templates from SLURM_EXAMPLES.sh

## OpenRouter Models

Common models available on OpenRouter:

```
allenai/Olmo-3-7B-Instruct          (default)
allenai/Olmo-7B-Instruct
meta-llama/Llama-2-70b-chat
meta-llama/Llama-3-70b-instruct
mistralai/Mistral-large
openai/gpt-4
```

See [OpenRouter models](https://openrouter.ai/models) for full list.

## Support

**Questions?** See `README.md`
**Architecture?** See `DESIGN.md`
**SLURM Jobs?** See `SLURM_EXAMPLES.sh`
**Example Code?** Run `example.py`
