# Persona Steering Evaluation Report

**Roles Analyzed**: 39
**Total Responses**: 7775
**Filters**: None
**spaCy**: Disabled (regex fallback)

## Method Summaries

| Metric | assistant_axis | baseline | steered |
|--------|--------|--------|--------|
| First-Person Rate (%) | 0.45 | 3.60 | 4.20 |
| Epistemic Markers | 0.81 | 0.10 | 0.26 |
| Avg Word Count | 1444.5 | 41.0 | 346.8 |
| Unique Bigram Ratio | 0.306 | 0.994 | 0.877 |
| Repetitive (%) | 91.6 | 0.0 | 0.2 |
| Degenerate Length (%) | 89.9 | 0.0 | 17.0 |
| Modal Verb Rate (%) | 4.70 | 1.14 | 1.38 |
| Questions/Response | 0.46 | 0.12 | 2.08 |
| AI Phrase Leakage | 0.021 | 0.000 | 0.007 |

## Statistical Comparisons

### assistant_axis vs baseline: first_person_rate
- Means: 0.451 vs 3.599
- p-value: 0.0000 ***
- Cohen's d: -0.849 (large)

### assistant_axis vs baseline: epistemic_total
- Means: 0.806 vs 0.101
- p-value: 0.0000 ***
- Cohen's d: 0.197 (negligible)

### assistant_axis vs baseline: unique_bigram_ratio
- Means: 0.306 vs 0.994
- p-value: 0.0000 ***
- Cohen's d: -3.518 (large)

### assistant_axis vs baseline: modal_verb_rate
- Means: 4.697 vs 1.138
- p-value: 0.0000 ***
- Cohen's d: 1.183 (large)

### assistant_axis vs steered: first_person_rate
- Means: 0.451 vs 4.201
- p-value: 0.0000 ***
- Cohen's d: -1.103 (large)

### assistant_axis vs steered: epistemic_total
- Means: 0.806 vs 0.258
- p-value: 0.0000 ***
- Cohen's d: 0.152 (negligible)

### assistant_axis vs steered: unique_bigram_ratio
- Means: 0.306 vs 0.877
- p-value: 0.0000 ***
- Cohen's d: -2.716 (large)

### assistant_axis vs steered: modal_verb_rate
- Means: 4.697 vs 1.378
- p-value: 0.0000 ***
- Cohen's d: 1.242 (large)

### baseline vs steered: first_person_rate
- Means: 3.599 vs 4.201
- p-value: 0.0000 ***
- Cohen's d: -0.156 (negligible)

### baseline vs steered: epistemic_total
- Means: 0.101 vs 0.258
- p-value: 0.0000 ***
- Cohen's d: -0.251 (small)

### baseline vs steered: unique_bigram_ratio
- Means: 0.994 vs 0.877
- p-value: 0.0000 ***
- Cohen's d: 1.882 (large)

### baseline vs steered: modal_verb_rate
- Means: 1.138 vs 1.378
- p-value: 0.0000 ***
- Cohen's d: -0.126 (negligible)


## Key Findings

- **First-Person Rate**: steered leads with 4.20
- **Epistemic Markers**: assistant_axis leads with 0.81
- **Unique Bigram Ratio**: baseline leads with 0.99
- **Repetitive %**: baseline leads with 0.00
- **Modal Verb Rate**: assistant_axis leads with 4.70
## Per-Role Summary

| Role | Best (First-Person) | Best (Epistemic) | Repetitive Issues |
|------|---------------------|------------------|-------------------|
| aberration | steered | steered | assistant_axis |
| absurdist | steered | assistant_axis | assistant_axis |
| accountant | steered | steered | assistant_axis |
| activist | steered | steered | assistant_axis |
| actor | steered | assistant_axis | assistant_axis |
| addict | baseline | assistant_axis | assistant_axis |
| adolescent | steered | assistant_axis | assistant_axis |
| alien | steered | steered | assistant_axis |
| altruist | steered | assistant_axis | assistant_axis |
| amateur | baseline | assistant_axis | assistant_axis |
| ambassador | steered | steered | assistant_axis |
| amnesiac | baseline | assistant_axis | assistant_axis |
| analyst | steered | steered | assistant_axis |
| anarchist | steered | steered | assistant_axis |
| ancient | steered | steered | assistant_axis |
| angel | baseline | assistant_axis | assistant_axis |
| anthropologist | baseline | steered | assistant_axis |
| archaeologist | steered | steered | assistant_axis |
| architect | steered | steered | assistant_axis |
| artisan | steered | assistant_axis | assistant_axis |
| ascetic | steered | assistant_axis | assistant_axis |
| assistant | steered | steered | assistant_axis |
| auctioneer | steered | assistant_axis | assistant_axis |
| auditor | steered | steered | assistant_axis |
| avatar | steered | steered | assistant_axis |
| bard | steered | steered | assistant_axis |
| bartender | steered | assistant_axis | assistant_axis |
| biologist | steered | steered | assistant_axis |
| blogger | steered | assistant_axis | assistant_axis |
| bohemian | steered | assistant_axis | assistant_axis |
| builder | steered | steered | assistant_axis |
| caregiver | steered | assistant_axis | assistant_axis |
| cartographer | steered | steered | assistant_axis |
| caveman | baseline | assistant_axis | assistant_axis |
| celebrity | steered | assistant_axis | assistant_axis |
| chameleon | steered | steered | assistant_axis |
| chef | baseline | steered | assistant_axis |
| chemist | baseline | steered | assistant_axis |
| chimera | steered | assistant_axis | assistant_axis |