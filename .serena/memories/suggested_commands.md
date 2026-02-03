# Suggested Commands for Development

## Environment Setup
```bash
# Install dependencies (using uv - preferred)
uv sync

# Alternative: pip install
pip install -e .
pip install -e ".[dev]"
```

## Running Evaluations
```bash
# Smoke test evaluation
uv run inspect eval inspect_evals/bbeh_mini --model hf/Qwen/Qwen2.5-1.5B-Instruct --limit 2 --log-dir logs/bbeh_smoke

# View logs
uv run inspect view --log-dir logs

# Run eval from preset
python scripts/run_eval.py --run bbeh-mini-qwen3-1.7b
python scripts/run_eval.py --run bbh-logical-deduction-qwen1.5b
```

## RIASEC Pipeline
```bash
# Pre-generate responses for a trait
python scripts/riasec_pipeline_eval.py --pregenerate --generate_dataset --trait social --target_count 7

# Run full pipeline for a trait
python scripts/riasec_pipeline_eval.py --trait social

# Generate persona vector
python src/pvx/pvx_models/riasec_persona_model.py --trait social --model_name Qwen/Qwen2.5-7B-Instruct --question "Do you like organizing events?"
```

## Code Quality
```bash
# Linting
ruff check src/ tests/

# Formatting
ruff format src/ tests/

# Type checking (via ty or Pylance in VS Code)
```

## Testing
```bash
# Run tests (test directories exist but are currently empty)
pytest

# Run single test
pytest tests/path/to/test_file.py::test_function_name -v

# With coverage
python -m pytest --cov=src/pvx tests/
```

## Git Operations (using jj)
```bash
jj status           # Working copy status
jj log -r @-3..@    # Recent commits
jj commit -m "msg"  # Create commit
jj git push         # Push to remote
```

## System Commands (Darwin/macOS)
- File search: `fd "pattern" path/`
- Content search: `rg "pattern" path/`
- Directory listing: `eza -lah --git`
- File viewing: `bat filename`
- JSON processing: `jq '.key' file.json`
- YAML processing: `yq eval '.key' file.yaml`
