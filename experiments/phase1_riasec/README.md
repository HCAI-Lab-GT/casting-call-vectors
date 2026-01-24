# Phase 1: RIASEC Vocational Persona Vectors

**Timeline**: Weeks 1-2 (Jan 20 – Feb 3)
**Goal**: Extract 150 vocational persona vectors and test H1 hypothesis

## Overview

This experiment extracts persona vectors from vocational personas organized by
RIASEC type (Holland codes) to test whether occupational identity creates
interpretable geometric structure in LLM activation space.

### RIASEC Dimensions

| Code | Type | Example Occupations |
|------|------|---------------------|
| R | Realistic | Electricians, Mechanics, Carpenters |
| I | Investigative | Scientists, Researchers, Analysts |
| A | Artistic | Artists, Writers, Designers |
| S | Social | Nurses, Teachers, Counselors |
| E | Enterprising | Executives, Managers, Sales |
| C | Conventional | Accountants, Administrators, Clerks |

## Quick Start

```bash
# 1. Generate vocational personas (25 per RIASEC type)
for riasec in R I A S E C; do
    uv run python scripts/generate_vocational_personas.py \
        --riasec $riasec \
        --limit 25 \
        --skip-existing
done

# 2. Extract persona vectors (local)
uv run python experiments/phase1_riasec/extract_riasec.py

# 3. Run RIASEC analysis
uv run python experiments/phase1_riasec/analyze_riasec.py --interactive
```

## SLURM Cluster Execution

```bash
# Generate SLURM job scripts
uv run python experiments/phase1_riasec/extract_riasec.py --slurm

# Submit all jobs (6 extraction + 1 analysis)
bash jobs/phase1_riasec/submit_all.sh
```

## H1 Hypothesis Test

**Hypothesis**: RIASEC dimensions align with principal components in persona
vector space, indicating that occupational identity creates interpretable
geometric structure.

**Go/No-Go Criterion**: Mean alignment score ≥ 0.6

The alignment score is computed as the mean of maximum absolute cosine
similarities between RIASEC contrast vectors and top PCs:

- R vs S (Realistic vs Social)
- I vs E (Investigative vs Enterprising)
- A vs C (Artistic vs Conventional)

## Expected Outputs

```
outputs/phase1_riasec/
├── vectors/           # Extracted persona vectors
│   ├── {soc_code}.pt  # Tensor files
│   └── {soc_code}.json  # Metadata
└── analysis/
    ├── riasec_analysis.json  # Full analysis results
    ├── riasec_pca_2d.png     # 2D PCA plot
    ├── riasec_pca_3d.html    # Interactive 3D plot
    └── variance_explained.png
```

## Configuration

Default settings in `extract_riasec.py` and `analyze_riasec.py`:

| Parameter | Value |
|-----------|-------|
| Model | allenai/OLMo-7B-Instruct |
| Layer | 14 |
| Questions per persona | 50 |
| Personas per RIASEC | 25 |
| Total personas | 150 |
| PCA components | 10 |
| Alignment threshold | 0.6 |

## W&B Dashboard

Results are logged to W&B project `pvx-phase1`:

- Extraction runs: `riasec-{R,I,A,S,E,C}` or `riasec-all`
- Analysis run: `riasec-analysis`

View at: https://wandb.ai/{username}/pvx-phase1
