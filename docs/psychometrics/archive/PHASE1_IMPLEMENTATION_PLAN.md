# Phase 1 Implementation Plan: Vocational Persona Vector Pipeline

**Project**: LM-VECTOR (Language Model Vocational Embeddings for Controlling Trait-Oriented Representations)
**Timeline**: Phase 1 - Weeks 1-2 (Jan 20 – Feb 3)
**Goal**: Build extraction pipeline, generate 150 vocational persona vectors (25 per RIASEC type), run initial PCA geometry analysis

---

## Executive Summary

Build a modular, extensible pipeline for extracting persona vectors from vocational personas. The architecture supports future methodologies (trait-based, role-based) while implementing vocational extraction now. Uses OLMo-7B as the target model, assistant-axis questions for extraction, and W&B for tracking/visualization.

---

## Architecture Overview

```text
src/pvx/
├── sources/                    # NEW: Persona definitions (extensible)
│   ├── base.py                 # PersonaSource protocol
│   └── vocational.py           # O*NET-based source (uses existing VocationalPersonaGenerator)
│
├── extraction/                 # NEW: Vector extraction pipeline
│   ├── pipeline.py             # ExtractionPipeline orchestrator
│   ├── activations.py          # Model-agnostic activation capture
│   ├── contrast.py             # Contrast strategies (default baseline)
│   └── questions.py            # QuestionBank from assistant-axis JSONL
│
├── analysis/                   # NEW: Geometry & visualization
│   ├── geometry.py             # PCA, dimensionality reduction
│   ├── riasec.py               # RIASEC axis correlation
│   └── viz.py                  # Plots, W&B dashboards
│
├── infra/                      # NEW: Cluster execution
│   └── slurm.py                # SLURM job generation
│
├── data/                       # EXISTS: Keep ONETLoader
│   └── onet_loader.py
│
├── judges/                     # EXISTS: Keep judges
│   ├── llm_as_judge.py
│   └── riasec_judge.py         # Fix bug on line 19-20
│
└── pvx_models/                 # EXISTS: Refactor for modularity
    ├── persona_model.py        # Extract steering logic to pvx/steering/
    ├── persona_dataset.py      # Move to pvx/sources/trait.py
    └── vocational_dataset.py   # Integrate with pvx/sources/vocational.py
```

---

## Implementation Steps

### Step 1: Package Restructure (Foundation)

**Files to create:**

- `src/pvx/sources/__init__.py`
- `src/pvx/sources/base.py` - PersonaSource protocol
- `src/pvx/extraction/__init__.py`
- `src/pvx/analysis/__init__.py`
- `src/pvx/infra/__init__.py`

**Key abstraction** - `PersonaSource` protocol:

```python
from typing import Protocol, TypedDict

class PersonaMetadata(TypedDict, total=False):
    soc_code: str
    title: str
    riasec: dict[str, float]
    riasec_primary: str

class PersonaSource(Protocol):
    @property
    def persona_id(self) -> str: ...

    def get_system_prompts(self) -> list[str]: ...
    def get_baseline_prompts(self) -> list[str]: ...
    def get_eval_prompt(self) -> str: ...
    def get_metadata(self) -> PersonaMetadata: ...
```

---

### Step 2: Vocational Source Implementation

**File**: `src/pvx/sources/vocational.py`

Wraps existing `VocationalPersonaGenerator` output into `PersonaSource` interface:

```python
class VocationalPersonaSource:
    """Load vocational persona from JSON file."""

    @classmethod
    def from_json(cls, filepath: Path) -> "VocationalPersonaSource": ...

    @classmethod
    def from_onet(cls, soc_code: str, generator: VocationalPersonaGenerator) -> "VocationalPersonaSource": ...

    def get_system_prompts(self) -> list[str]:
        return [inst["pos"] for inst in self._data["instruction"]]

    def get_baseline_prompts(self) -> list[str]:
        # Load from persona_data/vocational_personas/instructions/default.json
        return DEFAULT_BASELINE_PROMPTS
```

---

### Step 3: Question Bank

**File**: `src/pvx/extraction/questions.py`

```python
class QuestionBank:
    """Manage extraction questions."""

    @classmethod
    def from_assistant_axis(cls, path: str = "_vendor/assistant-axis/data/extraction_questions.jsonl") -> "QuestionBank":
        """Load 240 questions from assistant-axis."""
        ...

    def sample(self, n: int) -> list[str]:
        """Random sample of n questions."""
        ...

    def get_all(self) -> list[str]:
        """All 240 questions."""
        ...
```

---

### Step 4: Activation Extractor

**File**: `src/pvx/extraction/activations.py`

Refactor from `PersonaModel._get_activations()`:

```python
class ActivationExtractor:
    """Extract activations from transformer models."""

    def __init__(
        self,
        model_id: str,
        layer: int = 14,
        device: str = "auto"
    ): ...

    def extract(
        self,
        system_prompt: str,
        question: str,
        max_new_tokens: int = 256
    ) -> dict:
        """Extract activations for a single prompt/question pair.

        Returns:
            {
                "prompt_last": Tensor,      # Last token of prompt
                "response_mean": Tensor,    # Mean of response tokens
                "response_text": str        # Generated response
            }
        """
        ...

    def extract_batch(
        self,
        prompts: list[tuple[str, str]],  # (system_prompt, question) pairs
        batch_size: int = 8
    ) -> list[dict]: ...
```

---

### Step 5: Extraction Pipeline

**File**: `src/pvx/extraction/pipeline.py`

```python
class ExtractionPipeline:
    """End-to-end persona vector extraction."""

    def __init__(
        self,
        model_id: str = "allenai/OLMo-7B-Instruct",
        layer: int = 14,
        questions: QuestionBank = None,
        judge: LLMJudge = None,
        judge_threshold: float = 2.5,  # 0-3 scale, require score >= 2.5 for "valid"
        output_dir: Path = Path("outputs/vectors"),
        wandb_project: str = "pvx-phase1"
    ): ...

    def extract_persona(
        self,
        source: PersonaSource,
        baseline: PersonaSource,
        num_questions: int = 50
    ) -> PersonaVector:
        """Extract vector for a single persona.

        Returns PersonaVector with:
        - prompt_last_diff: mean(persona) - mean(baseline) at last prompt token
        - response_mean_diff: mean(persona) - mean(baseline) at response tokens
        - metadata: RIASEC scores, judge scores, etc.
        """
        ...

    def extract_batch(
        self,
        sources: list[PersonaSource],
        baseline: PersonaSource,
        checkpoint_every: int = 1,  # Save after each persona
        resume_from: Path = None
    ) -> list[PersonaVector]:
        """Batch extraction with W&B logging and checkpointing."""
        ...
```

---

### Step 6: Geometry Analysis

**File**: `src/pvx/analysis/geometry.py`

```python
class PersonaGeometry:
    """PCA and geometric analysis of persona vectors."""

    def __init__(self, vectors: dict[str, Tensor], metadata: dict[str, PersonaMetadata]): ...

    def compute_pca(self, n_components: int = 10) -> PCAResult:
        """Run PCA, return loadings and explained variance."""
        ...

    def riasec_axis_correlation(self) -> dict:
        """Compute correlation between RIASEC dimensions and top PCs.

        Returns cosine similarities between:
        - Direct RIASEC contrast vectors (mean_R - mean_A, etc.)
        - Top principal components
        """
        ...

    def cluster_by_riasec(self) -> ClusterResult:
        """K-means or hierarchical clustering colored by RIASEC."""
        ...
```

**File**: `src/pvx/analysis/viz.py`

```python
class PersonaVisualizer:
    """Visualization for persona vectors."""

    def plot_pca_2d(self, geometry: PersonaGeometry, color_by: str = "riasec_primary") -> Figure: ...
    def plot_pca_3d(self, geometry: PersonaGeometry) -> Figure: ...
    def plot_riasec_hexagon(self, metadata: PersonaMetadata) -> Figure: ...
    def plot_variance_explained(self, pca_result: PCAResult) -> Figure: ...

    def log_to_wandb(self, run: wandb.Run): ...
```

---

### Step 7: SLURM Infrastructure

**File**: `src/pvx/infra/slurm.py`

```python
def generate_extraction_job(
    personas: list[str],  # SOC codes or persona file paths
    model_id: str,
    output_dir: str,
    wandb_project: str,
    partition: str = "gpu",
    gpus: int = 1,
    time: str = "4:00:00"
) -> str:
    """Generate SLURM batch script for extraction job."""
    ...
```

**File**: `scripts/run_extraction.py`

```python
"""Main extraction script for both local and SLURM execution."""

@click.command()
@click.option("--model", default="allenai/OLMo-7B-Instruct")
@click.option("--riasec", type=click.Choice(["R", "I", "A", "S", "E", "C", "all"]))
@click.option("--limit", type=int, default=100)
@click.option("--output-dir", default="outputs/vectors")
@click.option("--wandb-project", default="pvx-phase1")
@click.option("--resume", type=click.Path(exists=True))
def main(model, riasec, limit, output_dir, wandb_project, resume): ...
```

---

### Step 8: Fix Existing Issues

**File**: `src/pvx/judges/riasec_judge.py` (line 19-20)

Bug: File opened but not used properly

```python
# Current (broken):
with open(yaml_path, 'r'):
    self.riasec_data = yaml.safe_load(f)  # 'f' undefined!

# Fix:
with open(yaml_path, 'r') as f:
    self.riasec_data = yaml.safe_load(f)
```

---

## Persona Generation Plan

**Target**: 150 personas (25 per RIASEC type)

| RIASEC            | Target Count | Example Occupations                  |
| ----------------- | ------------ | ------------------------------------ |
| R (Realistic)     | 25           | Electricians, Mechanics, Carpenters  |
| I (Investigative) | 25           | Scientists, Researchers, Analysts    |
| A (Artistic)      | 25           | Artists, Writers, Designers          |
| S (Social)        | 25           | Nurses, Teachers, Counselors         |
| E (Enterprising)  | 25           | Executives, Managers, Sales          |
| C (Conventional)  | 25           | Accountants, Administrators, Clerks  |

**Generation approach**:

```bash
# Generate 25 per RIASEC type (150 total)
for riasec in R I A S E C; do
    uv run python scripts/generate_vocational_personas.py \
        --riasec $riasec \
        --limit 25 \
        --skip-existing
done
```

---

## File Manifest

### New Files to Create

1. `src/pvx/sources/__init__.py`
2. `src/pvx/sources/base.py`
3. `src/pvx/sources/vocational.py`
4. `src/pvx/extraction/__init__.py`
5. `src/pvx/extraction/questions.py`
6. `src/pvx/extraction/activations.py`
7. `src/pvx/extraction/pipeline.py`
8. `src/pvx/analysis/__init__.py`
9. `src/pvx/analysis/geometry.py`
10. `src/pvx/analysis/viz.py`
11. `src/pvx/infra/__init__.py`
12. `src/pvx/infra/slurm.py`
13. `scripts/run_extraction.py`
14. `scripts/run_analysis.py`
15. `configs/riasec.yaml` (if missing for RIASECJudge)

### Files to Modify

1. `src/pvx/judges/riasec_judge.py` - Fix file handle bug
2. `src/pvx/__init__.py` - Add new module exports
3. `pyproject.toml` - Add any new dependencies (wandb, plotly, scikit-learn)

### Files to Keep (no changes)

1. `src/pvx/data/onet_loader.py`
2. `src/pvx/judges/llm_as_judge.py`
3. `src/pvx/pvx_models/vocational_dataset.py`
4. `scripts/generate_vocational_personas.py`

---

## Verification Plan

### Unit Tests

```bash
# Test persona source loading
uv run pytest tests/test_sources.py -v

# Test activation extraction (requires GPU)
uv run pytest tests/test_extraction.py -v --gpu

# Test geometry analysis
uv run pytest tests/test_analysis.py -v
```

### Integration Tests

```bash
# Extract vectors for 5 personas (quick validation)
uv run python scripts/run_extraction.py --limit 5 --output-dir outputs/test

# Run PCA analysis on extracted vectors
uv run python scripts/run_analysis.py --input-dir outputs/test
```

### End-to-End Validation

1. Generate 10 vocational personas (2 per RIASEC type)
2. Extract persona vectors using OLMo-7B
3. Compute PCA and RIASEC correlations
4. Visualize in W&B dashboard
5. Verify go/no-go criterion: Are RIASEC dimensions correlated with top PCs?

---

## Dependencies to Add

```toml
# pyproject.toml additions
dependencies = [
    "wandb>=0.16",
    "plotly>=5.18",
    "scikit-learn>=1.4",
    "click>=8.1",
    "accelerate>=0.26",  # For OLMo model loading
]
```

---

## Implementation Order

1. **Day 1-2**: Package restructure + PersonaSource protocol + VocationalPersonaSource
2. **Day 3-4**: QuestionBank + ActivationExtractor (refactor from PersonaModel)
3. **Day 5-6**: ExtractionPipeline + W&B integration + checkpointing
4. **Day 7-8**: Geometry analysis + visualization
5. **Day 9-10**: SLURM scripts + batch generation + initial experiments

---

## Success Criteria

Phase 1 is complete when:

- [ ] 150 vocational persona vectors extracted (25 per RIASEC type)
- [ ] PCA analysis shows interpretable structure (variance explained by top 3-5 PCs)
- [ ] RIASEC correlation analysis completed (determines go/no-go for H1)
- [ ] W&B dashboard with 2D/3D projections
- [ ] All code tested and documented
- [ ] Ready to scale to 100-1000 personas in Phase 2
