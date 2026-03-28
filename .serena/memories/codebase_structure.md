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
├── scripts/cluster/pace/ # HPC batch scripts (PACE cluster)
├── scripts/cluster/mats/ # HPC batch scripts (MATS cluster)
├── data/                 # Data directory
├── .serena/              # Serena MCP configuration
└── .claude/              # Claude Code settings
```

## Source Code (src/pvx/)
```
src/pvx/
├── __init__.py
├── abstraction/              # Abstract base classes
│   ├── pvx_models/
│   │   ├── abstract_persona_model.py  # AbstractPersonaModel(ABC)
│   │   └── abstract_dataset.py        # AbstractDataset(ResponseGeneration)
│   └── judges/
│       └── abstract_judge.py          # AbstractJudge(ResponseGeneration, ABC)
├── implementations/          # Concrete implementations
│   ├── base/
│   │   ├── persona_dataset.py         # PersonaDataset(AbstractDataset)
│   │   └── persona_model.py           # PersonaModel(AbstractPersonaModel)
│   ├── riasec/
│   │   ├── riasec_dataset.py          # RiasecDataset(AbstractDataset)
│   │   └── riasec_persona_model.py    # RIASECPersonaModel(AbstractPersonaModel)
│   ├── judges/
│   │   ├── llm_as_judge.py            # LLMJudge(AbstractJudge)
│   │   ├── riasec_judge.py            # RIASECJudge(AbstractJudge)
│   │   ├── role_judge.py              # RoleJudge(AbstractJudge)
│   │   ├── hexaco_judge.py            # HexacoJudge(AbstractJudge)
│   │   └── batch_scorer.py            # BatchScorer
│   ├── roles/                         # RolePersonaModel, RoleDataset
│   ├── roles_layers/                  # RoleLayersPersonaModel, AssistantAxisPersonaModel
│   ├── roles_optimized/               # RoleQAGenerator, RoleActivationExtractor
│   └── trait_coverage/                # TraitCoveragePersonaModel, TraitCoverageDataset
├── experiments/              # Experiment pipelines
│   ├── gold_prompt_experiments.py
│   └── pairwise_judge_experiments.py
├── inspect_modelapi/         # Inspect AI model API integration
├── tasks/                    # Inspect AI evaluation tasks
│   ├── worfbench/
│   └── boardgame_qa/
└── utils/                    # Utility modules
    ├── response_generation.py  # ResponseGeneration
    ├── prompts.py              # PromptTemplates
    ├── riasec_utils.py         # RIASECHelpers
    ├── judge_utils.py          # JudgeConfig
    ├── generation_utils.py     # GenerationConfig
    ├── logging_utils.py        # Logging configuration
    └── wandb_utils.py          # WandB integration
```

## Configuration Files
- `configs/models.yaml` - Model presets (names, IDs, generation params)
- `configs/runs.yaml` - Run presets (task/model mappings)
- `configs/riasec.yaml` - RIASEC trait definitions and Q/A pairs

## Key Entry Points
- `scripts/run_eval.py` - CLI runner for evaluations
- `scripts/riasec_pipeline_eval.py` - Full RIASEC pipeline
- `src/pvx/implementations/judges/llm_as_judge.py` - LLM-as-judge CLI
- `src/pvx/implementations/riasec/riasec_persona_model.py` - Persona model CLI
