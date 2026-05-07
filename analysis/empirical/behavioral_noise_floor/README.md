# Behavioral noise floor / RSA reliability ceiling

Companion to `rsa_geometry_behavior/`. Estimates the upper bound any external
RDM can possibly correlate with the behavioral RDM, given judge noise.

## Why

`rsa_geometry_behavior/` reports RSA(repr_cos, beh_corr) = +0.137 and
RSA(repr_cos, beh_l2) = +0.233. By itself, "0.14" is uninterpretable — is it
small because the geometry-behavior link is genuinely weak, or because the
behavioral side is so noisy that even *another behavioral measurement* would
only correlate with it at, say, 0.30?

This script answers that by measuring **split-half reliability** of the
behavioral RDM and Spearman-Brown-correcting for the half-data penalty. The
result is the noise-corrected RSA: `r_observed / r_ceiling`, expressed as a
fraction of the explainable behavioral variance.

## How to reproduce

```bash
.venv/bin/python analysis/empirical/behavioral_noise_floor/run_analysis.py
.venv/bin/python analysis/empirical/behavioral_noise_floor/make_figures.py
```

Reads:
- `experiment_data/gold_prompt_experiments/Comparison_GoldStandard_*.csv`
- `analysis/empirical/rsa_geometry_behavior/data/role_order.json`
- `analysis/empirical/rsa_geometry_behavior/data/rsa_main.csv`

## Method

For each of `N_SEEDS=200` random seeds:
1. For each role, stratify-split its 50 gold-prompt rows by alpha into halves
   A and B (so each half is balanced across the 4 alphas).
2. Build a 24-feature behavioral profile (4 alphas × 6 features) on each half
   → two 275×24 matrices.
3. Build behavioral RDMs on each half (corr-distance and L2 separately).
4. Spearman correlation between the upper triangles of RDM_A and RDM_B = the
   half-data reliability for this seed.
5. Spearman-Brown correction: `r_full = 2 * r_half / (1 + r_half)` projects
   half-data reliability to full-data reliability.

Average across seeds, get a 95% bootstrap CI, then divide the observed RSA
values by this ceiling.

## Files

```
data/
  noise_floor_split_half.csv  one row per seed, 4 columns
  noise_floor_summary.csv     mean, std, 95% CI for each metric
  noise_corrected_rsa.csv     observed / ceiling, per (repr, beh) combo
figures/
  fig1_split_half_distribution.png   per-seed reliability histogram
  fig2_observed_vs_ceiling.png       observed RSA bars, ceiling line per bar
  fig3_noise_corrected_rsa.png       fraction of explainable variance
```

## Findings (n = 275 roles, 200 random splits)

**The behavioral RDM is highly reliable.**

| beh distance | split-half r (mean) | 95% CI         | Spearman-Brown corrected |
| ------------ | -------------------:| -------------- | ------------------------:|
| beh_corr     | 0.94                | [0.92, 0.96]   | **0.97**                 |
| beh_l2       | 0.98                | [0.98, 0.98]   | **0.99**                 |

**Noise-corrected RSA = observed / ceiling.** The geometry-behavior RSA values
barely change after correction:

| repr × beh           | observed r | ceiling | noise-corrected | % of ceiling |
| -------------------- | ----------:| -------:| ---------------:| ------------:|
| repr_cos × beh_corr  | +0.137     | 0.968   | +0.141          | **14%**      |
| repr_cos × beh_l2    | +0.233     | 0.991   | +0.236          | **24%**      |
| repr_l2  × beh_corr  | +0.068     | 0.968   | +0.070          |  7%          |
| repr_l2  × beh_l2    | +0.263     | 0.991   | +0.266          | **27%**      |

## Interpretation

Judge noise is *not* the limit. The behavioral RDM is one of the cleanest
signals in this pipeline (split-half ≈ 0.94 with corr-distance, ≈ 0.98 with
L2). If the geometry-behavior link were really there at r ≈ 0.5, we would
have seen it. So the modest noise-corrected values (14 - 27% of ceiling) are
the true effect size of "geometry predicts behavior" given how this pipeline
defines geometry and behavior.

The next question — answered in
`analysis/empirical/intrinsic_dimensionality/` — is whether the bottleneck is
on the *behavioral* side (the 24-d profile is effectively low-rank, so most
of the rich variance in the 4096-d cloud has nothing in behavior to map to).
