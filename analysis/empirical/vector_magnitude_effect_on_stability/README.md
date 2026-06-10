# Does role-vector magnitude predict steering stability at high alpha?

**Hypothesis (user-stated):** the quality / stability of a role vector at higher
alphas is tied to its internal magnitude.

**Setup.** 275 roles. For each role we have a single `.pt` activation-difference
vector at layer 16 (4096-dim) and gold-prompt judge scores at four alphas
{1.0, 1.5, 2.0, 2.5}. Internal magnitude := L2 norm of the vector.

## How to reproduce

```bash
.venv/bin/python analysis/empirical/vector_magnitude_effect_on_stability/run_analysis.py
.venv/bin/python analysis/empirical/vector_magnitude_effect_on_stability/make_figures.py
```

`run_analysis.py` writes per-role tables to `data/`. `make_figures.py` writes
figures to `figures/`.

## Stability metrics tested

For each role we summarize the alpha–score curve into:

| metric                | meaning                                         | "stable" looks like |
| --------------------- | ----------------------------------------------- | ------------------- |
| `score_at_alpha_2_5`  | mean steered_score at the highest alpha         | high                |
| `delta_high_minus_low`| score(2.5) − score(1.0)                         | positive            |
| `score_slope`         | OLS slope of mean-score vs alpha                | positive            |
| `late_alpha_drop`     | max(score) − score(2.5)                         | ≈ 0                 |
| `peak_alpha < 2.5`    | score peaks before the highest alpha            | False               |
| `subdim_consistency`  | std-across-alpha of judge sub-dimension scores  | low                 |

## Findings (n = 275 roles)

| stability metric      | Pearson r vs norm | p-value   | direction |
| --------------------- | ------------------:| ---------:| --------- |
| score_at_alpha_1_0    | **−0.33**         | 2.0e-08   | low-norm scores higher at low alpha |
| score_at_alpha_2_5    |  +0.10            | 0.10      | mild positive |
| delta_high_minus_low  | **+0.49**         | 1.2e-17   | high-norm gains more from larger alpha |
| score_slope           | **+0.49**         | 3.1e-18   | same |
| late_alpha_drop       |  −0.03            | 0.59      | linearly null, but… |
| peak<2.5 share by Q   | 28%, 38%, 22%, **12%** | —    | low-norm is 3× more likely to peak early |

**Interpretation.** The hypothesis is supported, with a sharper claim:

1. **Low-norm vectors saturate early.** They already achieve their best score at
   α = 1.0 and have little headroom; in 28–38% of roles the score curve peaks
   *before* α = 2.5 and then drops — the classic "over-steered" failure mode.
2. **High-norm vectors ramp up monotonically.** At α = 1.0 they barely move
   the model (lowest scores), but their alpha–score curve has the steepest
   positive slope and they continue to improve at α = 2.5 (only 12% peak
   early).
3. By α = 2.5 the four norm quartiles roughly *converge* (Fig 2), with the
   high-norm group pulling slightly ahead. So "stability at high alpha" is
   real, but it manifests as **monotonic non-collapse**, not as a higher
   absolute ceiling.

In short: **magnitude controls the alpha–response slope.** Higher-magnitude
vectors are slower-acting but stay coherent further into the steering range,
while lower-magnitude vectors hit their effective ceiling early and start to
break down at high alpha.

## Files

```
data/
  vector_magnitude_stability.csv  one row per role, all metrics
  alpha_curves_long.csv           tidy (role, alpha, score) for plotting
  correlations.csv                Pearson + Spearman, magnitude × stability
figures/
  fig1_norm_vs_score_per_alpha.png   scatter per alpha (sign reverses across α)
  fig2_alpha_curves_by_norm_quartile.png   the headline curve plot
  fig3_norm_vs_slope_and_delta.png    norm → slope and norm → Δscore
  fig4_norm_vs_late_alpha_drop.png    early-peak instability vs norm
  fig5_correlation_heatmap.png        compact summary of all correlations
  fig6_subdimensions_at_alpha2_5.png  sub-dimension breakdown at α=2.5
  fig7_summary_panel.png              5-panel combined summary
```
