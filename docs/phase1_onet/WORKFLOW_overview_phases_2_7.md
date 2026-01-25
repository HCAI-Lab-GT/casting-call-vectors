# Implementation Workflows Overview: Phases 2-7

> **Purpose**: High-level summary of remaining psychometric phases. Each phase has its own detailed workflow document.
>
> **Reference**: [PSYCHOMETRICS_DATA.md](../reference/PSYCHOMETRICS_DATA.md)
>
> **Created**: 2025-01-25

---

## Phase Priority Matrix

| Phase | Instrument | Items | Priority | Workflow Document |
|-------|------------|-------|----------|-------------------|
| **2** | HEXACO-PI-R | 60-200 | **HIGH** | [WORKFLOW_phase2_hexaco.md](./WORKFLOW_phase2_hexaco.md) |
| **3** | IPIP-NEO | 120-300 | **HIGH** | [WORKFLOW_phase3_ipip_neo.md](./WORKFLOW_phase3_ipip_neo.md) |
| **4** | Schwartz Values | 21-57 | MEDIUM | [WORKFLOW_phase4_schwartz.md](./WORKFLOW_phase4_schwartz.md) |
| **5** | Dark Personality | 27-28 | MEDIUM | [WORKFLOW_phase5_dark_personality.md](./WORKFLOW_phase5_dark_personality.md) |
| **6** | ML Datasets | N/A | LOW | [WORKFLOW_phase6_ml_datasets.md](./WORKFLOW_phase6_ml_datasets.md) |
| **7** | Final Catalog | N/A | MEDIUM | [WORKFLOW_phase7_catalog.md](./WORKFLOW_phase7_catalog.md) |

---

## Architecture Pattern

Following the successful `ONETLoader` pattern from Phase 1:

```
src/pvx/data/
├── onet_loader.py       # Complete (Phase 1)
├── hexaco_loader.py     # Phase 2
├── ipip_loader.py       # Phase 3
├── schwartz_loader.py   # Phase 4
└── dark_loader.py       # Phase 5
```

---

## Execution Order

```
Phase 2 (HEXACO)     → Honesty-Humility crucial for AI safety
Phase 3 (IPIP-NEO)   → Validates O*NET Big Five; public domain
Phase 4 (Schwartz)   → Deeper values beyond O*NET Work Values
Phase 5 (Dark)       → Adversarial personas for edge testing
Phase 6 (ML Data)    → Training datasets
Phase 7 (Catalog)    → Final integration
```

---

## Related Documents

- [PSYCHOMETRICS_DATA.md](../reference/PSYCHOMETRICS_DATA.md) - Master reference
- [STATUS_psychometrics.md](./STATUS_psychometrics.md) - Progress tracking
