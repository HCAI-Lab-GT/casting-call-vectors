# persona-vectors-evals

This repository contains persona dataset generation and evaluation utilities built around Inspect AI. Use the Inspect tasks in `evals/` and datasets in `dataset/`.

## Quick start
- Create a virtual environment and install with `uv sync`.
- Run a smoke evaluation, for example: `uv run inspect eval inspect_evals/bbeh_mini --model hf/Qwen/Qwen2.5-1.5B-Instruct --limit 2 --log-dir logs`.
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

## CLI runner
- Use `scripts/run_eval.py` to launch evals from presets.
- Examples:
  - `python scripts/run_eval.py --run bbeh-mini-qwen3-1.7b`
  - `python scripts/run_eval.py --run bbeh-mini-qwen3-1.7b --limit 5 --model qwen1.5b`
  - `python scripts/run_eval.py --task inspect_evals/bbeh_mini --model-id hf/Qwen/Qwen3-1.7B --limit 1`
- Flags:
  - `--run NAME` selects a run from `configs/runs.yaml`
  - `--model NAME` overrides the model preset; or `--model-id` to bypass presets
  - `--limit`, `--log-dir` override per-run values
  - `--model-arg key=value` and `--solver-arg key=value` (repeatable) map to `-M` / `-S` in `inspect eval`
  - `--dry-run` prints the command only
