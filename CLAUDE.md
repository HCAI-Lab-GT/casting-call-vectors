# pvx - Persona Vectors Extended

> **CRITICAL**: Read [AGENTS.md](AGENTS.md) first. It contains essential interaction guidelines, code style requirements, and project protocols that apply to all work in this codebase.

## Codebase Overview

This project implements persona vector extraction and steering for language models, with a focus on **vocational personas** derived from O*NET occupational data. It builds on two research codebases: `assistant-axis` (role-based persona extraction) and `persona_vectors` (trait-based activation steering).

**Stack**: Python, PyTorch, HuggingFace Transformers, Inspect AI, OpenAI API

**Structure**:
- `src/pvx/` - Core library (PersonaModel, LLMJudge, ONETLoader)
- `scripts/` - Generation and evaluation scripts
- `persona_data/` - Generated persona definitions
- `_vendor/` - Vendored reference implementations (gitignored)

For detailed architecture, see [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).

## Key Components

| Component | Purpose | File |
|-----------|---------|------|
| `PersonaModel` | Extract persona vectors, apply steering | `src/pvx/pvx_models/persona_model.py` |
| `PersonaDataset` | Generate trait-based datasets | `src/pvx/pvx_models/persona_dataset.py` |
| `VocationalPersonaGenerator` | Generate O*NET-based personas | `src/pvx/pvx_models/vocational_dataset.py` |
| `LLMJudge` | Evaluate model responses | `src/pvx/judges/llm_as_judge.py` |
| `RIASECJudge` | Validate RIASEC profiles | `src/pvx/judges/riasec_judge.py` |
| `ONETLoader` | Parse O*NET database | `src/pvx/data/onet_loader.py` |

## Quick Start

```bash
# Generate vocational personas from O*NET
uv run python scripts/generate_vocational_personas.py --limit 10

# Generate with RIASEC filter
uv run python scripts/generate_vocational_personas.py --riasec S

# Extract persona vector for a trait
uv run python src/pvx/pvx_models/persona_model.py --trait analytical
```

## Research Context

This project is part of the **LM-VECTOR** research (Language Model Vocational Embeddings for Controlling Trait-Oriented Representations) investigating:

1. **H1 (Geometric Structure)**: Vocational personas form interpretable clusters in activation space
2. **H2 (Persona Persistence)**: Multi-axis constraints maintain persona stability
3. **H3 (Behavioral Validity)**: RIASEC profiles predict behavioral patterns
4. **H4 (Capability Control)**: Persona-specific capability modulation is possible

## Dependencies

- Python 3.11+
- uv for package management
- CUDA-capable GPU (for model inference)
- OpenAI API key (for persona generation)
