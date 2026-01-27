# Psychometric Integration Documentation

This directory contains all documentation for psychometric integration phases (1-7).

## Directory Structure

```
docs/psychometrics/
├── README.md                           # This file
├── STATUS_psychometrics.md             # Progress tracker (living document)
├── workflows/                          # Workflow planning documents for all phases
│   ├── WORKFLOW_phase1_onet.md
│   ├── WORKFLOW_overview_phases_2_7.md
│   ├── WORKFLOW_phase2_hexaco.md
│   ├── WORKFLOW_phase3_ipip_neo.md
│   ├── WORKFLOW_phase4_schwartz.md
│   ├── WORKFLOW_phase5_dark_personality.md
│   ├── WORKFLOW_phase6_ml_datasets.md
│   └── WORKFLOW_phase7_catalog.md
└── archive/                            # Completed work (historical reference)
    ├── PHASE1_IMPLEMENTATION_PLAN.md
    ├── PHASE1_VALIDATION_PLAN.md
    └── IMPLEMENT_work_styles_extension.md
```

## Status

**Phase 1 O*NET Integration: COMPLETE** (2025-01-25)

See [STATUS_psychometrics.md](STATUS_psychometrics.md) for detailed progress tracking across all phases.

## Naming Convention

| Prefix | Meaning | Status |
|--------|---------|--------|
| `STATUS_` | Progress tracking | Living document |
| `WORKFLOW_` | Phase workflow planning | Planning/reference |
| `IMPLEMENT_` | Verified implementation plans | Executed |
| `archive/` | Completed work | Historical reference |

## Phase 1 Implementation Summary

### What's Implemented

| Feature | Location | Tests |
|---------|----------|-------|
| O*NET database loading | `ONETLoader` | 17 tests |
| RIASEC scores & high-points | `get_riasec_scores()`, `get_highpoint_codes()` | Passing |
| Work Styles (16 traits) | `load_work_styles()`, `get_work_style_scores()` | 5 tests |
| Big Five (derived) | `get_big_five_scores()` | 3 tests |
| Work Values (6 values) | `load_work_values()`, `get_work_value_scores()` | 6 tests |
| Extended profiles | `get_occupation_profile()` | 5 tests |
| Persona metadata | `VocationalPersonaGenerator._metadata` | Updated |

### Quick Validation

```bash
# Run all O*NET tests
uv run pytest tests/unit/test_onet_loader.py -v

# Sample profile
uv run python -c "
from pvx.data.onet_loader import ONETLoader
loader = ONETLoader()
p = loader.get_occupation_profile('29-1141.00')  # Registered Nurses
print(f\"Big Five: {p['big_five']}\")
print(f\"Work Values: {p['work_value_highpoints']}\")
"
```

## Workflow Documents

All workflow planning documents are in the `workflows/` subdirectory:

| Phase | Document | Instrument | Priority |
|-------|----------|------------|----------|
| Overview | [WORKFLOW_overview.md](workflows/WORKFLOW_overview.md) | All phases summary | - |
| 1 | [WORKFLOW_phase1_onet.md](workflows/WORKFLOW_phase1_onet.md) | O*NET (RIASEC, Work Styles, Work Values) | ✅ COMPLETE |
| 2 | [WORKFLOW_phase2_hexaco.md](workflows/WORKFLOW_phase2_hexaco.md) | HEXACO-PI-R | HIGH |
| 3 | [WORKFLOW_phase3_ipip_neo.md](workflows/WORKFLOW_phase3_ipip_neo.md) | IPIP-NEO | HIGH |
| 4 | [WORKFLOW_phase4_schwartz.md](workflows/WORKFLOW_phase4_schwartz.md) | Schwartz Values | MEDIUM |
| 5 | [WORKFLOW_phase5_dark_personality.md](workflows/WORKFLOW_phase5_dark_personality.md) | Dark Personality | MEDIUM |
| 6 | [WORKFLOW_phase6_ml_datasets.md](workflows/WORKFLOW_phase6_ml_datasets.md) | ML Datasets | LOW |
| 7 | [WORKFLOW_phase7_catalog.md](workflows/WORKFLOW_phase7_catalog.md) | Final Catalog | MEDIUM |

## Related Reference Docs

See [docs/reference/](../reference/) for general reference materials:
- [PSYCHOMETRICS_DATA.md](../reference/PSYCHOMETRICS_DATA.md) - Master psychometric instruments guide
- [PERSONA_ARCHITECTURE_COMPARISON.md](../reference/PERSONA_ARCHITECTURE_COMPARISON.md) - Architecture comparison

## Next Steps

1. **Experimentation**: Generate personas with full psychometric metadata
2. **Validation**: Compare Big Five derivation against known profiles
3. **Phase 2**: Begin HEXACO-PI-R integration (see workflow docs)
