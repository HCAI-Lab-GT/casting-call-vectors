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
