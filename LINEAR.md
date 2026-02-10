# LINEAR.md — Project Management Context

Overview for Claude Code sessions. Use the Linear MCP tools for specifics.

## Workspace

| Field | Value |
|-------|-------|
| Team | HCAI Lab (pending rename from "Geometry of Personality") |
| Project | Geometry of Personality |
| Issue prefix | GEO- |
| Deadline | COLM, March 31, 2026 |

## Structure

29 issues across 3 streams, 4 weekly cycles (Feb 10 - Mar 7), with work continuing to Mar 31.

**Streams:**
- Stream 1: Data & Personas (7 issues, S1-01 to S1-07)
- Stream 2: Pipeline & Infra (14 issues, S2-01 to S2-14)
- Stream 3: Instruments & Eval (8 issues, S3-01 to S3-08)

**Cycles:**
- Week 1 (Feb 10-14): Setup & Foundation
- Week 2 (Feb 17-21): Build & Generate
- Week 3 (Feb 24-28): Pilot Run & Gate 1
- Week 4 (Mar 3-7): Experiments & H1

## Decision Gates (milestones in Linear)

| Gate | Target | Question | Key Issue |
|------|--------|----------|-----------|
| Gate 1 | Feb 28 | Do personas work? | GEO-21 (S2-10) |
| Gate 2 | Mar 7 | Does steering work? | GEO-25 (S2-14) |
| Gate 3 | Mar 7 | Occupation-dependent shifts? | GEO-33 (S3-08) |

## Critical Path

```
S2-01 → S2-02 → S2-06 → S2-07 → S2-08 → S2-09 → S2-10 [Gate 1]
                                                      ↓
S2-11 → S2-12 → S2-13 → S2-14 [Gate 2] → S3-07 → S3-08 [Gate 3]
```

Cross-stream dependencies are wired in Linear (blockedBy relationships).

## Team Roles

| Person | Focus |
|--------|-------|
| Glenn | Lead. Methodology, analysis, writing. |
| Person B | O*NET data, persona prompt generation, QC |
| Person C | Model setup, compute ops, generation runs, judge |
| Person D | Instrument assembly, scoring, Likert parsing, eval |

## Labels

Stream labels, Critical Path, Decision Gate, Delegatable, Size (XS/S/M/L).

## Notes

- Issue naming: `[S{stream}-{number}] {title}` maps to GEO-{N} identifiers
- All dependencies set as blocking relationships in Linear
- Timeline view works with cycles + due dates + dependency arrows
- Team rename to "HCAI Lab" must be done in Linear UI (no API for team rename)
