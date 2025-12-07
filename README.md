# persona-vectors-evals

This repository contains persona dataset generation and evaluation utilities built around Inspect AI. Use the Inspect tasks in `evals/` and datasets in `dataset/`.

## Quick start
- Create a virtual environment and install with `uv sync`.
- Run a smoke evaluation, for example: `uv run inspect eval inspect_evals/bbeh_mini --model hf/Qwen/Qwen2.5-1.5B-Instruct --limit 2 --log-dir logs`.
- View logs with `uv run inspect view --log-dir logs`.

## VS Code
I recommend installing the Inspect AI VS Code extension (Marketplace ID `ukaisi.inspect-ai`). It adds log browsing, task panels, and integrated run/debug support for `.eval` artifacts.
