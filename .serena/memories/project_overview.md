# Project Overview: pvx (persona-vectors)

## Purpose
Persona dataset generation and evaluation utilities built around Inspect AI. The project focuses on RIASEC (Holland codes) personality trait extraction and persona vector generation for language models.

## Core Capabilities
- **Persona Vector Extraction**: Extract personality-aligned activation vectors from LLMs
- **RIASEC Trait Modeling**: Generate and evaluate responses aligned with Holland RIASEC traits (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)
- **LLM-as-Judge Evaluation**: Automated response scoring using various LLM backends
- **Inspect AI Integration**: Full integration with Inspect AI evaluation framework

## Tech Stack
- **Python**: 3.12 (required)
- **ML/AI**: PyTorch, Transformers, vLLM, Sentence-Transformers
- **Evaluation**: Inspect AI, Inspect Evals, WandB logging
- **APIs**: OpenAI, Anthropic
- **Data**: HuggingFace Datasets, Safetensors
- **Build**: Hatchling, uv package manager

## Key Directories
- `src/pvx/` - Main library code
- `src/pvx/abstraction/` - Abstract base classes (AbstractPersonaModel, AbstractDataset, AbstractJudge)
- `src/pvx/implementations/` - Concrete implementations (persona models, judges, datasets)
- `src/pvx/tasks/` - Inspect AI evaluation tasks
- `src/pvx/utils/` - Utility functions
- `scripts/` - CLI runners and pipelines
- `configs/` - YAML configuration files
- `persona_data/` - Trait datasets and model initializations
