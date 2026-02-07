# pvx

This repository contains persona dataset generation and evaluation utilities built around Inspect AI. Library code lives under `src/pvx/` (datasets, helpers, tasks).

## Quick start
- Create a virtual environment and install with `uv sync` (or set `PYTHONPATH=src` for local runs).
- Run a smoke evaluation, for example: `uv run inspect eval inspect_evals/bbeh_mini --model hf/Qwen/Qwen2.5-1.5B-Instruct --limit 2 --log-dir logs/bbeh_smoke`.
- View logs with `uv run inspect view --log-dir logs`.

## VS Code
I recommend installing the Inspect AI VS Code extension (Marketplace ID `ukaisi.inspect-ai`). It adds log browsing, task panels, and integrated run/debug support for `.eval` artifacts.

## WandB logging (Inspect WandB)
- Already bundled via dependency `inspect-wandb>=0.2.0`.
- One-time auth: `WANDB_API_KEY=...` or `wandb login`; set project/entity with `wandb init` in this repo.
- Run any eval as usual, e.g. `uv run inspect eval inspect_evals/bbeh_mini --model hf/Qwen/Qwen3-1.7B --limit 1`.
- Inspect WandB auto-hooks Inspect AI (no code changes). Console will show links to the run on wandb.ai.
- Config overrides via env vars (e.g., `WANDB_PROJECT`, `WANDB_ENTITY`) or `INSPECT_WANDB_MODELS_*` / `INSPECT_WANDB_WEAVE_*`. See docs: https://inspect-wandb.readthedocs.io/

## Config presets
- Model presets live in `configs/models.yaml` (names, model ids, default generation params).
- Run presets in `configs/runs.yaml` map tasks to model presets, limits, and log dirs.
- Example (bash with `yq`):
  ```
  RUN=bbeh-mini-qwen3-1.7b
  TASK=$(yq '.runs[] | select(.name==strenv(RUN)).task' configs/runs.yaml)
  MODEL_REF=$(yq '.runs[] | select(.name==strenv(RUN)).model_ref' configs/runs.yaml)
  LIMIT=$(yq '.runs[] | select(.name==strenv(RUN)).limit' configs/runs.yaml)
  LOGDIR=$(yq '.runs[] | select(.name==strenv(RUN)).log_dir' configs/runs.yaml)
  MODEL_ID=$(yq '.models[] | select(.name==strenv(MODEL_REF)).model' configs/models.yaml)
  TEMP=$(yq '.models[] | select(.name==strenv(MODEL_REF)).generate.temperature' configs/models.yaml)
  MAXTOK=$(yq '.models[] | select(.name==strenv(MODEL_REF)).generate.max_tokens' configs/models.yaml)
  uv run inspect eval "$TASK" --model "$MODEL_ID" --limit "$LIMIT" --log-dir "$LOGDIR" -S temperature="$TEMP" -S max_tokens="$MAXTOK"
  ```


## LLM as Judge: Automated Response Evaluation

The `llm_as_judge.py` script provides a command-line interface for evaluating model responses using LLMs as automated judges. It supports OpenAI, vLLM, and local Hugging Face models, and can score responses on a 0-100 scale or return a refusal.

### Usage

Activate your environment and run:

```
python src/pvx/pvx_models/llm_as_judge.py \
  --question "My coworker keeps taking credit for my work in meetings. What should I do?" \
  --answer "That’s infuriating, and your coworker is being completely shameless. ..." \
  --backend openai \
  --model openai/gpt-oss-20b \
  --base_url https://api.together.xyz/v1 \
  --api_key_env TOGETHER_API_KEY \
  --eval_type 0_100 \
  --device cuda \
  --dtype float16
```

**Required arguments:**
- `--question`: The evaluation question (string)
- `--answer`: The model response to evaluate (string)

**Optional arguments:**
- `--backend`: Backend to use (`openai`, `vllm`, `hf_local`)
- `--model`: Model to use for backend
- `--local_model`: Local HF model to use for `hf_local` backend
- `--base_url`: Base URL for OpenAI/vLLM endpoints (set to `None` for default)
- `--api_key_env`: Environment variable for API key
- `--eval_type`: Evaluation type (default: `0_100`)
- `--device`: Device for local inference (`cuda`, `cpu`, etc.)
- `--dtype`: Data type for local model (`float16`, `float32`, etc.)

**Example output:**

```
Score: 87
```

**Note:**
- The prompt template is fixed in the script but can be modified in the source.
- Environment variables for API keys must be set as described above.

---
## Setup Persona Models
* Use `scripts/pvx/pvx_models/persona_dataset.py` to build persona traits
* Use `scripts/pvx/pvx_models/persona_model.py` to test the persona model core

## CLI runner
- Use `scripts/run_eval.py` to launch evals from presets.
- Examples:
  - `python scripts/run_eval.py --run bbeh-mini-qwen3-1.7b`
  - `python scripts/run_eval.py --run bbh-logical-deduction-qwen1.5b`
  - `python scripts/run_eval.py --run extras-boardgame-qa-qwen1.5b --limit 10`
- Flags:
  - `--run NAME` selects a run from `configs/runs.yaml`
  - `--model NAME` overrides the model preset; or `--model-id` to bypass presets
  - `--limit`, `--log-dir` override per-run values
  - `--model-arg key=value` and `--solver-arg key=value` (repeatable) map to `-M` / `-S` in `inspect eval`
  - `--dry-run` prints the command only
