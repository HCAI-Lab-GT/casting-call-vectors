# Codebase Structure

## Root Directory
```
persona-vectors/
├── src/pvx/              # Main library package
├── scripts/              # CLI runners and pipelines
├── configs/              # YAML configuration files
├── persona_data/         # Trait datasets and model data
├── tests/                # Test directories (unit, integration, mocks)
├── docs/                 # Documentation
├── experiments/          # Experiment results
├── slurm/                # HPC batch scripts
├── data/                 # Data directory
├── .serena/              # Serena MCP configuration
└── .claude/              # Claude Code settings
```

## Source Code (src/pvx/)
```
src/pvx/
├── __init__.py
├── pvx_models/           # Core persona model implementations
│   ├── persona_model.py          # Base persona model
│   ├── riasec_persona_model.py   # RIASEC-specific persona model
│   ├── persona_dataset.py        # Dataset handling
│   ├── response_generation.py    # Response synthesis
│   ├── abstract_persona_model.py # Abstract base class
│   ├── prompts.py                # Prompt templates
│   ├── judges/                   # Evaluation judges
│   │   ├── llm_as_judge.py       # LLM-based evaluation
│   │   └── riasec_judge.py       # RIASEC trait scoring
│   └── inspect_modelapi/         # Inspect AI model API integration
├── tasks/                # Inspect AI evaluation tasks
│   ├── worfbench/        # WorfBench task
│   └── boardgame_qa/     # BoardGame QA task
├── utils/                # Utility modules
│   ├── riasec_utils.py   # RIASEC helpers
│   ├── judge_utils.py    # Judge utilities
│   ├── inspect_utils.py  # Inspect AI utilities
│   ├── generation_utils.py # Generation helpers
│   ├── logging_utils.py  # Logging configuration
│   └── wandb_utils.py    # WandB integration
├── config/               # Configuration handling
├── sources/              # Data sources
├── data/                 # Data utilities
├── extraction/           # Extraction utilities
├── analysis/             # Analysis modules
└── infra/                # Infrastructure utilities
```

## Configuration Files
- `configs/models.yaml` - Model presets (names, IDs, generation params)
- `configs/runs.yaml` - Run presets (task/model mappings)
- `configs/riasec.yaml` - RIASEC trait definitions and Q/A pairs

## Key Entry Points
- `scripts/run_eval.py` - CLI runner for evaluations
- `scripts/riasec_pipeline_eval.py` - Full RIASEC pipeline
- `src/pvx/pvx_models/llm_as_judge.py` - LLM-as-judge CLI
- `src/pvx/pvx_models/riasec_persona_model.py` - Persona model CLI
