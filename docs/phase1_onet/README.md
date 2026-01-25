# Phase 1: O*NET Integration Documentation

This directory contains documentation for the O*NET-based vocational persona system.

## Naming Convention

| Prefix | Meaning | Status |
|--------|---------|--------|
| `DESIGN_` | Design documents, rationale, mappings | Reference material |
| `IMPLEMENT_` | Verified implementation plans with code | Ready to execute |
| `archive/` | Completed work | Historical reference |

## Active Documents

| Document | Purpose |
|----------|---------|
| [DESIGN_psychometrics_mapping.md](DESIGN_psychometrics_mapping.md) | Work Styles → Big Five/HEXACO mapping design. Explains the "what" and "why" of extending O*NET with personality frameworks. |
| [IMPLEMENT_work_styles_extension.md](IMPLEMENT_work_styles_extension.md) | **Verified implementation plan** for adding Work Styles and Work Values to ONETLoader. Contains exact code, line numbers, and tests. |

## Related Reference Docs

See [docs/reference/](../reference/) for general reference materials:
- [PSYCHOMETRICS_DATA.md](../reference/PSYCHOMETRICS_DATA.md) - Psychometric instruments guide
- [PERSONA_ARCHITECTURE_COMPARISON.md](../reference/PERSONA_ARCHITECTURE_COMPARISON.md) - Architecture comparison

## Archived Documents

| Document | Why Archived |
|----------|--------------|
| [archive/PHASE1_IMPLEMENTATION_PLAN.md](archive/PHASE1_IMPLEMENTATION_PLAN.md) | Extraction pipeline implementation complete. |
| [archive/PHASE1_VALIDATION_PLAN.md](archive/PHASE1_VALIDATION_PLAN.md) | Testing infrastructure implemented. |

## Current Work

The next implementation task is **IMPLEMENT_work_styles_extension.md**, which extends `ONETLoader` with:
- Work Styles loading (16 personality traits)
- Big Five score derivation
- Work Values loading (6 work values)

See the implementation plan for exact code and test specifications.
