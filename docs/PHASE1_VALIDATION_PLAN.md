# Phase 1 Validation: Testing Infrastructure & Local Smoke Tests

## Summary

Phase 1 implementation is **100% complete**. All extraction, analysis, and infrastructure modules exist per [docs/PHASE1_PLAN.md](docs/PHASE1_PLAN.md). This plan adds testing infrastructure to validate everything works on Apple Silicon using small local models (no APIs).

## Current State

| Component | Status | File |
|-----------|--------|------|
| PersonaSource protocol | ✅ Complete | [src/pvx/sources/base.py](src/pvx/sources/base.py) |
| VocationalPersonaSource | ✅ Complete | [src/pvx/sources/vocational.py](src/pvx/sources/vocational.py) |
| QuestionBank | ✅ Complete | [src/pvx/extraction/questions.py](src/pvx/extraction/questions.py) |
| ActivationExtractor | ✅ Complete | [src/pvx/extraction/activations.py](src/pvx/extraction/activations.py) |
| ExtractionPipeline | ✅ Complete | [src/pvx/extraction/pipeline.py](src/pvx/extraction/pipeline.py) |
| PersonaGeometry | ✅ Complete | [src/pvx/analysis/geometry.py](src/pvx/analysis/geometry.py) |
| PersonaVisualizer | ✅ Complete | [src/pvx/analysis/viz.py](src/pvx/analysis/viz.py) |
| SLURM infrastructure | ✅ Complete | [src/pvx/infra/slurm.py](src/pvx/infra/slurm.py) |

**Gap**: No test infrastructure exists - need to validate implementation works.

## Constraints

- **Apple Silicon only** - no NVIDIA GPU
- **No API usage** - no OpenAI, no external services
- **SmolLM models only**:
  - Unit tests: [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (fast, basic validation)
  - Smoke/integration tests: [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) (proper testing)
- **SLURM preserved** - scripts must still work for cloud scaling

---

## Implementation Steps

### Step 0: Code Quality & Style Validation

**Goal**: Verify existing code adheres to project principles from AGENTS.md. Not refactoring—just identifying violations.

**Quick Checks (automated)**:

```bash
# Ruff linting (enforces PEP 8, import order, error checking)
uv run ruff check src/ tests/ --statistics

# Ruff formatting check (Black-compatible)
uv run ruff format --check src/ tests/

# Find files exceeding 150 lines (AGENTS.md limit)
find src/ -name "*.py" -exec wc -l {} + | awk '$1 > 150 {print}'

# Check for os.path usage (should use pathlib)
rg "import os\.path|from os import path|os\.path\." src/

# Check for print statements (should use logging)
rg "^\s*print\(" src/ --glob "!__pycache__"

# Check for old-style type hints (List, Dict, Optional from typing)
rg "from typing import.*\b(List|Dict|Optional|Tuple|Set)\b" src/
```

**Key Principles to Verify** (from AGENTS.md):

| Principle | Check | Action if Violated |
|-----------|-------|-------------------|
| Ruff linting | `ruff check` exits clean | Fix errors before tests |
| Ruff formatting | `ruff format --check` | Run `ruff format` |
| File size ≤150 lines | `wc -l` check | Split only if blocking |
| Function length ≤30 lines | Manual review | Note for future |
| Native type hints | Grep for `typing.List` etc | Update during type fixes |
| pathlib not os.path | Grep check | Update if found |
| logging not print | Grep for `print(` | Replace with logger |

**Priority**: Fix only what blocks testing. Note other violations for gradual improvement—don't create refactoring busy work.

---

### Step 1: Type Checking with ty

**Goal**: Establish type checking baseline and fix critical type errors.

**Configuration**: `ty` is already installed (`ty>=0.0.13` in dev dependencies). Run:

```bash
# Check all code
uv run ty check

# Check specific module
uv run ty check src/pvx/extraction/
```

**Current Status** (~50-60% type coverage):

| Module | Coverage | Notes |
|--------|----------|-------|
| `sources/base.py` | 80-90% | Excellent (Protocol, TypedDict) |
| `extraction/pipeline.py` | 75-85% | Well typed |
| `analysis/geometry.py` | 75-85% | Well typed |
| `extraction/activations.py` | 60-70% | Partial, needs work |
| `extraction/questions.py` | 50-60% | Partial |
| `data/onet_loader.py` | 50-60% | Partial |
| `judges/llm_as_judge.py` | 40-50% | Mixed old/new style |
| `judges/riasec_judge.py` | 20-30% | Minimal |
| `pvx_models/` | 30-40% | Minimal |

**Tasks**:

1. **Run baseline**: `uv run ty check` and capture current error count
2. **Create py.typed marker**: `touch src/pvx/py.typed`
3. **Fix core extraction modules** (priority for testing):
   - `src/pvx/extraction/activations.py`
   - `src/pvx/extraction/pipeline.py`
   - `src/pvx/extraction/questions.py`
4. **Standardize typing style**: Use PEP 604 (`X | None` not `Optional[X]`)
5. **Add return types**: All public methods must have return type annotations

**Typing Style Guide**:

```python
# Use PEP 604 unions (Python 3.10+)
def foo(x: int | None = None) -> str | None: ...

# Use lowercase generics (Python 3.9+)
def bar(items: list[str]) -> dict[str, int]: ...

# Use TypedDict for structured dicts
class Config(TypedDict):
    model_id: str
    layer: int

# Use Protocol for duck typing
class Extractor(Protocol):
    def extract(self, prompt: str) -> ActivationResult: ...
```

---

### Step 2: Add MPS (Apple Silicon) Support

**File**: [src/pvx/extraction/activations.py](src/pvx/extraction/activations.py)

Update device/dtype detection (around line 77-79):

```python
# Current:
if dtype is None:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Change to:
if dtype is None:
    if torch.cuda.is_available():
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        dtype = torch.float32  # MPS works best with float32
    else:
        dtype = torch.float32
```

Also update device resolution to handle "mps":
```python
if self._device == "auto":
    if torch.cuda.is_available():
        self.device = "cuda"
    elif torch.backends.mps.is_available():
        self.device = "mps"
    else:
        self.device = "cpu"
```

---

### Step 3: Add Model Presets Configuration

**New file**: `src/pvx/config/__init__.py`
**New file**: `src/pvx/config/models.py`

```python
# SmolLM model family - https://huggingface.co/collections/HuggingFaceTB/smollm2
SMOLLM_MODELS = {
    # SmolLM2 - for fast unit tests (smaller, basic validation)
    "smol2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smol2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smol2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # SmolLM3 - for proper smoke/integration tests
    "smol3-3b": "HuggingFaceTB/SmolLM3-3B",
}

MODEL_PRESETS = {
    "production": {
        "model_id": "allenai/OLMo-7B-Instruct",
        "layer": 14,
        "max_new_tokens": 256,
    },
    "unit_test": {
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "layer": 4,
        "max_new_tokens": 32,
    },
    "smoke_test": {
        "model_id": "HuggingFaceTB/SmolLM3-3B",
        "layer": 16,  # SmolLM3-3B has 32 layers
        "max_new_tokens": 64,
    },
}
```

---

### Step 4: Create Test Infrastructure

**Directory structure**:
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── mocks/
│   ├── __init__.py
│   ├── model_mocks.py       # MockActivationExtractor
│   └── source_mocks.py      # MockPersonaSource
├── unit/
│   ├── __init__.py
│   ├── test_onet_loader.py
│   ├── test_question_bank.py
│   ├── test_persona_source.py
│   └── test_persona_geometry.py
└── integration/
    ├── __init__.py
    └── test_extraction_pipeline.py
```

**Key fixtures in conftest.py**:
- `test_device` - detect cpu/mps/cuda
- `unit_model_id` - returns SmolLM2-135M (fast unit tests)
- `smoke_model_id` - returns SmolLM3-3B (proper smoke tests)
- `sample_questions` - 5 test questions
- `sample_persona_data` - mock vocational persona dict
- `project_root` - path to project root

---

### Step 5: Write Mock Implementations

**File**: `tests/mocks/model_mocks.py`

```python
class MockActivationExtractor:
    """Returns deterministic fake activations for fast unit tests."""

    def __init__(self, hidden_dim=768, num_layers=12):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def extract(self, system_prompt, question, **kwargs):
        seed = hash(system_prompt + question) % (2**32)
        torch.manual_seed(seed)
        return ActivationResult(
            prompt_last=torch.randn(1, self.hidden_dim),
            response_mean=torch.randn(1, self.hidden_dim),
            response_all_layers=torch.randn(self.num_layers + 1, 1, self.hidden_dim),
            response_text=f"Mock response to: {question[:50]}",
            num_response_tokens=10,
        )
```

**File**: `tests/mocks/source_mocks.py`

```python
class MockPersonaSource:
    """Mock persona source for pipeline testing."""

    def __init__(self, persona_id="mock", riasec_primary="S"):
        self._id = persona_id
        self._riasec = riasec_primary

    @property
    def persona_id(self): return self._id

    def get_system_prompts(self): return [f"You are a {self._id}."]
    def get_baseline_prompts(self): return ["You are a helpful assistant."]
    def get_metadata(self): return {"riasec_primary": self._riasec}
```

---

### Step 6: Write Unit Tests

**Test categories**:

1. **test_onet_loader.py** - O*NET data loading
   - `test_load_occupations` - loads occupation data
   - `test_get_riasec_scores` - returns RIASEC scores
   - `test_filter_by_riasec` - filters by type

2. **test_question_bank.py** - Question management
   - `test_from_list` - creates from list
   - `test_sample` - random sampling
   - `test_split` - train/test splitting

3. **test_persona_source.py** - Persona abstractions
   - `test_baseline_source` - default baseline
   - `test_vocational_from_json` - loads JSON
   - `test_get_metadata` - returns typed metadata

4. **test_persona_geometry.py** - Analysis
   - `test_compute_pca` - PCA computation
   - `test_cluster` - K-means clustering
   - `test_compute_distances` - pairwise distances

---

### Step 7: Write Integration Tests

**File**: `tests/integration/test_extraction_pipeline.py`

```python
@pytest.mark.slow
class TestExtractionPipelineIntegration:
    """Tests requiring real model loading (SmolLM3-3B)."""

    @pytest.fixture(scope="class")
    def smoke_pipeline(self, smoke_model_id, tmp_path):
        """Load pipeline with SmolLM3-3B for proper testing."""
        return ExtractionPipeline(
            model_id=smoke_model_id,  # SmolLM3-3B
            layer=16,
            questions=QuestionBank.from_list([...]),
            output_dir=tmp_path,
        )

    def test_extract_single_persona(self, smoke_pipeline):
        """Should extract vector for single persona."""
        source = MockPersonaSource("test_nurse")
        baseline = BaselineSource()
        vector = smoke_pipeline.extract_persona(source, baseline, num_questions=2)

        assert vector.persona_id == "test_nurse"
        assert vector.prompt_last_diff.shape[1] > 0
```

---

### Step 8: Create Smoke Test Script

**File**: `scripts/smoke_test.py`

Quick validation script that:
1. Checks all imports work
2. Verifies O*NET data loads
3. Loads pre-generated vocational personas
4. Runs geometry analysis on synthetic data
5. (Optional with --model) Loads SmolLM and extracts one vector

```bash
# Fast check (no model loading)
uv run python scripts/smoke_test.py

# Unit-level model check (SmolLM2-135M - fast)
uv run python scripts/smoke_test.py --model smol2-135m

# Full smoke test (SmolLM3-3B - proper validation)
uv run python scripts/smoke_test.py --model smol3-3b
```

---

### Step 9: Add pytest Configuration

**Update**: `pyproject.toml`

```toml
[project.optional-dependencies]
dev = [
  "ruff>=0.6.0",
  "pytest>=8.0.0",
  "pytest-cov>=4.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["-v", "--tb=short", "-m", "not slow"]
markers = [
    "slow: tests requiring model loading",
    "integration: integration tests",
]
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/pvx/py.typed` | PEP 561 marker for typed package |
| `src/pvx/config/__init__.py` | Config module init |
| `src/pvx/config/models.py` | Model presets (production/test) |
| `tests/__init__.py` | Test package init |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/mocks/__init__.py` | Mocks package init |
| `tests/mocks/model_mocks.py` | MockActivationExtractor |
| `tests/mocks/source_mocks.py` | MockPersonaSource |
| `tests/unit/__init__.py` | Unit tests init |
| `tests/unit/test_onet_loader.py` | ONETLoader tests |
| `tests/unit/test_question_bank.py` | QuestionBank tests |
| `tests/unit/test_persona_source.py` | PersonaSource tests |
| `tests/unit/test_persona_geometry.py` | PersonaGeometry tests |
| `tests/integration/__init__.py` | Integration tests init |
| `tests/integration/test_extraction_pipeline.py` | End-to-end tests |
| `scripts/smoke_test.py` | Local validation script |

## Files to Modify

| File | Change |
|------|--------|
| [src/pvx/extraction/activations.py](src/pvx/extraction/activations.py) | Add MPS device support |
| [pyproject.toml](pyproject.toml) | Add pytest dependencies |

---

## Verification Plan

### Code Quality (fast, no model loading)

```bash
# Ruff linting - check for errors and style issues
uv run ruff check src/ tests/ --statistics

# Ruff formatting - verify Black-compatible formatting
uv run ruff format --check src/ tests/

# Fix any issues automatically
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

### Type Checking (fast, no model loading)

```bash
# Run type checker on all code
uv run ty check

# Check specific module (useful during development)
uv run ty check src/pvx/extraction/

# Verbose output with error details
uv run ty check --verbose
```

### Unit Tests (fast, no model loading)

```bash
uv run pytest tests/unit/ -v
```

### Integration Tests (loads SmolLM3-3B)

```bash
uv run pytest tests/integration/ -v -m slow
```

### Smoke Test

```bash
# Quick validation (no model)
uv run python scripts/smoke_test.py

# Full validation with SmolLM3-3B
uv run python scripts/smoke_test.py --model smol3-3b
```

### End-to-End Extraction

```bash
# Extract 1 persona with SmolLM3-3B on Apple Silicon
uv run python scripts/run_extraction.py \
    --model HuggingFaceTB/SmolLM3-3B \
    --layer 16 \
    --persona persona_data/vocational_personas/instructions/registered_nurses.json \
    --num-questions 3 \
    --output-dir outputs/smoke_test
```

---

## Success Criteria

- [ ] **Ruff linting passes**: `uv run ruff check src/` exits clean
- [ ] **Ruff formatting passes**: `uv run ruff format --check src/` exits clean
- [ ] **No os.path usage**: All file operations use pathlib
- [ ] **No print statements**: Use logging module instead
- [ ] **Type checking passes**: `uv run ty check` exits cleanly on core extraction modules
- [ ] **py.typed marker exists**: `src/pvx/py.typed` file created
- [ ] All unit tests pass without model loading
- [ ] Smoke test validates all components work
- [ ] Integration test extracts vector with SmolLM3-3B on Apple Silicon
- [ ] SLURM scripts still generate correctly for cloud
- [ ] No API calls required for any test
