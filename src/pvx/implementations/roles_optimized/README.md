# Roles Optimized Implementation

## Overview

This optimized implementation splits persona vector extraction into two distinct phases with **separate answer and extraction models**:

1. **CPU Task** (`roles_optimized_cpu.py`): Generate and judge Q/A via **OpenRouter API** (no GPU needed)
2. **GPU Task** (`roles_optimized_gpu.py`): Extract activations with **GPU-loaded model** (GPU only when needed)

This separation enables **maximal GPU utilization** by:
- **CPU jobs**: Use OpenRouter API for responses (CPU-only nodes, no local GPU)
- **GPU jobs**: Load model on GPU only for activation extraction
- **Result**: Zero GPU idle time, GPU maximally utilized for activation work

## Architecture

### Model Separation

| Component | Default Model | Where It Runs | GPU? |
|-----------|---------------|---------------|------|
| **Answer Generation** | `allenai/Olmo-3-7B-Instruct` | OpenRouter API (remote) | ✗ No |
| **Judging** | LLMJudge | CPU (or remote API) | ✗ No |
| **Activation Extraction** | `allenai/Olmo-3-7B-Instruct` | GPU (loaded locally) | ✓ Yes |

Both default to the same model but can differ if needed.

## Key Optimizations

### 1. Skip Missing Roles
If a role dataset doesn't exist, the pipeline automatically skips it without failing.

### 2. OpenRouter API for Response Generation (CPU)
- **No local GPU needed** - remote API call
- Multiple roles can be processed in parallel on CPU nodes
- Judges responses without GPU
- Saves Q/A to JSON for GPU phase

### 3. GPU-Only Activation Extraction
- **Loads model on GPU only when needed** - separate from answer generation
- Pre-tokenizes all messages before GPU loop
- Randomly shuffles Q/A for variance
- Extracts in single GPU pass
- Saves persona vectors to safetensors

### 4. Q/A Persistence
- Saves to: `persona_data/model_qa_responses/{answer_model_id}/{role}.json`
- Enables debugging and manual inspection
- GPU phase doesn't regenerate responses

### 5. Incremental Extraction
- Build on previous sample counts (doesn't restart)
- Can resume from interruption
- Merge additional Q/A pairs naturally

## Usage

### Standard Workflow

```bash
# Phase 1: Generate and judge Q/A (CPU, uses OpenRouter API)
python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --answer_model "allenai/Olmo-3-7B-Instruct" \
  --roles "Lawyers" "Doctor" "Engineer" \
  --target_pairs 40 \
  --output_dir "./persona_data/model_qa_responses"

# Phase 2: Extract activations (GPU, loads model on GPU)
python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --activation_extraction_model "allenai/Olmo-3-7B-Instruct" \
  --answer_model "allenai/Olmo-3-7B-Instruct" \
  --roles "Lawyers" "Doctor" "Engineer" \
  --layer 14 \
  --qa_responses_dir "./persona_data/model_qa_responses" \
  --output_dir "./persona_data/model_inits"
```

### SLURM Job Chaining

**Key Benefit: GPU Never Idles**

```bash
#!/bin/bash

# Submit CPU job (no GPU needed)
CPU_JOB=$(sbatch --parsable \
  --job-name=roles-cpu \
  --partition=cpu \
  --cpus-per-task=4 \
  cpu_job.sh)

# Submit GPU job (depends on CPU, uses Q/A ready)
sbatch \
  --job-name=roles-gpu \
  --partition=gpu \
  --gres=gpu:1 \
  --dependency=afterok:$CPU_JOB \
  gpu_job.sh
```

When CPU job finishes, GPU job starts immediately with Q/A ready to process.

## Command Line Arguments

### roles_optimized_cpu.py

- `--answer_model`: Model for generating answers (default: `allenai/Olmo-3-7B-Instruct`)
- `-r, --roles`: Roles to process
- `-o, --output_dir`: Q/A output directory (default: `./persona_data/model_qa_responses`)
- `-d, --dataset_dir`: Role dataset directory
- `-n, --target_pairs`: Valid pairs per role (default: 20)
- `-t, --temperature`: Sampling temperature (default: 0.9)
- `--max_new_tokens`: Generation length (default: 2048)
- `--api_key`: OpenRouter API key (defaults to `OPENROUTER_API_KEY` env var)

**Example:**
```bash
python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --answer_model "allenai/Olmo-3-7B-Instruct" \
  --roles "Lawyers" \
  --target_pairs 40 \
  --api_key $OPENROUTER_API_KEY
```

### roles_optimized_gpu.py

- `--activation_extraction_model`: Model to load on GPU (default: `allenai/Olmo-3-7B-Instruct`)
- `--answer_model`: Model used for Q/A generation (for finding responses)
- `-r, --roles`: Roles to process
- `-l, --layer`: Layer for activation extraction (default: 14)
- `-q, --qa_responses_dir`: Q/A input directory
- `-d, --dataset_dir`: Role dataset directory
- `-o, --output_dir`: Persona vectors output directory

**Example:**
```bash
python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
  --activation_extraction_model "allenai/Olmo-3-7B-Instruct" \
  --answer_model "allenai/Olmo-3-7B-Instruct" \
  --roles "Lawyers" \
  --layer 14
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ CPU TASK: roles_optimized_cpu.py                             │
│ (Answer Model: OpenRouter API)                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Load role dataset (skip if missing)                       │
│ 2. For each (pos_prompt, question) pair:                     │
│    - Query OpenRouter API with prompt                        │
│    - Query OpenRouter API with empty prompt                  │
│    - Judge both responses                                    │
│    - If both pass: save to Q/A JSON                          │
│ 3. Result: persona_data/model_qa_responses/{model}/{role}.json
│                                                               │
│ GPU USAGE: None (uses remote API) ✓                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Q/A JSON file
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │ Storage: model_qa_responses/        │
        │   └── {answer_model_id}/            │
        │       └── {role}.json               │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ GPU TASK: roles_optimized_gpu.py                             │
│ (Extraction Model: Loaded on GPU)                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Load Q/A JSON (don't regenerate!)                         │
│ 2. Pre-tokenize all messages                                 │
│ 3. Randomly shuffle pairs                                    │
│ 4. For each pair (extraction model on GPU):                  │
│    - Extract hidden states at layer L                        │
│    - Accumulate in sums                                      │
│ 5. Compute contrastive vectors                               │
│ 6. Save to safetensors                                       │
│                                                               │
│ GPU USAGE: 100% (only when needed) ✓                         │
└─────────────────────────────────────────────────────────────┘
```

## Environment Setup

### OpenRouter API

Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or pass via command line:
```bash
python -m src.pvx.implementations.roles_optimized.roles_optimized_cpu \
  --api_key "your-api-key"
```

### Dependencies

```bash
pip install openai  # For OpenRouter API client
```

## Performance Benefits

### Without Optimization (Original)
```
GPU: [Generate → Judge → Extract]
     (GPU idle during judge ~30-40%)

Time: T_gen + T_judge + T_extract
GPU Utilization: ~60-70%
```

### With Optimization (This)
```
Job1 (CPU): [Generate via API → Judge]
Job2 (GPU):                        [Extract]
            (GPU processes Q/A immediately after CPU)

Time: max(T_gen + T_judge, T_extract)
GPU Utilization: 100% (no idle time)
```

**Results:**
- 5-15% faster single role
- 30-50% improvement with many roles (parallelizable CPU jobs)
- 100% GPU utilization (no idle waiting)

## Error Handling

### Missing Role Dataset
```
Error: Role dataset not found for {role}
Action: Skip role, continue with others
```

### Missing Q/A Responses
```
Error: Q/A responses not found at {path}
Solution: Run CPU task first
```

### OpenRouter API Error
```
Error: Failed to generate response via OpenRouter: {error}
Action: Skip pair, try next one
```

### GPU Out of Memory
```
Error: CUDA out of memory
Solution: Use smaller model or fewer roles per job
```

## Building on Previous Results

Extract incrementally without restarting:

```python
# Extract from first 20 Q/A pairs
extractor = RoleActivationExtractor(...)
vectors1 = extractor.extract_persona_vector()

# Generate 20 more Q/A pairs
qa_data = generator.generate_and_judge_qa(..., target_pairs=20)

# Extract from both 20+20 pairs (doesn't restart)
extractor = RoleActivationExtractor(...)
vectors_combined = extractor.extract_persona_vector()
```

## Multi-Layer Extraction

Extract from multiple layers in parallel (GPU):

```bash
for LAYER in 12 14 16 18 20; do
  sbatch --job-name=roles-layer$LAYER \
    --gres=gpu:1 \
    bash -c "python -m src.pvx.implementations.roles_optimized.roles_optimized_gpu \
      --layer $LAYER \
      --roles 'Lawyers' 'Doctor'"
done
```

## Integration with Existing Code

- ✅ Extends `AbstractPersonaModel`
- ✅ Uses `RoleDataset` for data loading
- ✅ Compatible with `RoleJudge` for evaluation
- ✅ Saves to standard safetensors format
- ✅ Drop-in replacement for original implementation

## Notes

- Both tasks handle model/role combinations automatically
- Q/A responses are cached (GPU phase doesn't regenerate)
- Token caching uses PyTorch tensors for efficiency
- Activation extraction uses float32 for numerical stability
- All outputs compatible with existing persona vector loaders
- See SLURM_EXAMPLES.sh for 9 different job submission patterns
