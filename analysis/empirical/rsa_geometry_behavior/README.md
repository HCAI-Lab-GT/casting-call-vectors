# Representational vs behavioral RSA on role vectors

**Question.** Does the geometry of the role-vector cloud (cosine/L2 distances on
the 4096-dim residual-stream vectors at layer 16) predict the *behavioral*
similarity of those roles under steering — i.e., do roles that look close in
representation space produce judge profiles that look close in behavior space?

This complements:
- `role_vector_vs_semantic_vector/` (geometry vs name embeddings)
- `vector_magnitude_effect_on_stability/` (norm vs alpha-curve features)

## How to reproduce

```bash
.venv/bin/python analysis/empirical/rsa_geometry_behavior/run_analysis.py
.venv/bin/python analysis/empirical/rsa_geometry_behavior/make_figures.py
```

`run_analysis.py` writes RDMs and RSA tables to `data/`. `make_figures.py`
writes figures to `figures/`.

## Setup

- 275 roles. Vectors loaded from `persona_data/pt_vectors/*.pt` (layer 16,
  4096-dim).
- Behavioral profile per role = 4 alphas × 6 features (`steered_score` plus the
  5 `cmp_*` sub-dimensions), giving a 24-dim fingerprint of how the role
  responds to steering on the gold-prompt judge.
- All 275 roles have full alpha coverage in `experiment_data/gold_prompt_experiments/`.

## Distance matrices

| name           | shape   | metric                                  |
| -------------- | ------- | --------------------------------------- |
| `repr_cos`     | 275×275 | `1 - cos(v_i, v_j)` on role vectors     |
| `repr_l2`      | 275×275 | `||v_i - v_j||_2` on role vectors       |
| `beh_corr`     | 275×275 | `1 - pearson(b_i, b_j)` on 24-d profile |
| `beh_l2`       | 275×275 | `||b_i - b_j||_2` on z-scored profile   |
| `rdm_norm`     | 275×275 | abs-difference of `||v_i||` (control)   |

RSA = Spearman r between the upper triangles. Significance via Mantel
permutation test (n_perm=10000 for the main grid, 2000 per per-alpha cell).

## Results (n = 275 roles)

| repr distance | beh distance | RSA Spearman r | Mantel p |
| ------------- | ------------ | --------------:| --------:|
| repr_cos      | beh_corr     | **+0.137**     | 1e-04    |
| repr_cos      | beh_l2       | **+0.233**     | 1e-04    |
| repr_l2       | beh_corr     |  +0.067        | 1.4e-02  |
| repr_l2       | beh_l2       | **+0.263**     | 1e-04    |

All four are positive and statistically significant. L2 ↔ L2 is the strongest;
cosine ↔ correlation is weaker but lives on a more interpretable scale (both
sides ignore magnitude). Cosine on representations is the principled choice
given that activation-difference vectors live on a roughly conic manifold; L2
mixes "direction" and "magnitude," and the magnitude difference is itself
correlated with behavior (see `vector_magnitude_effect_on_stability/`).

### Per-alpha RSA (using `repr_cos`)

| alpha | beh_corr r | p       | beh_l2 r | p       |
| ----- | ----------:| -------:| --------:| -------:|
| 1.0   | +0.085     | 3.5e-3  | +0.197   | 5e-4    |
| 1.5   | +0.092     | 3.0e-3  | +0.206   | 5e-4    |
| 2.0   | +0.121     | 5e-4    | +0.196   | 5e-4    |
| 2.5   | **+0.179** | 5e-4    | +0.169   | 5e-4    |

With correlation-distance the link **strengthens monotonically with alpha**:
the more you steer, the more representational geometry predicts behavioral
geometry. Consistent with the magnitude finding — at low alpha the model
barely moves, so behavior is dominated by base-model noise; at high alpha the
role-vector identity is loaded into the output.

### Partial RSA controlling for `||v||`

| repr × beh           | full r  | partial r (control: norm) | Δ        |
| -------------------- | -------:| -------------------------:| --------:|
| repr_cos × beh_corr  | +0.137  | +0.132                    | -0.005   |
| repr_cos × beh_l2    | +0.233  | +0.218                    | -0.015   |
| repr_l2  × beh_corr  | +0.067  | +0.058                    | -0.010   |
| repr_l2  × beh_l2    | +0.263  | +0.245                    | -0.018   |

Partial Spearman of the two RDM upper triangles after partialling out a
norm-difference RDM. The shift is tiny — the geometry-behavior link is **not**
just a re-statement of "high-norm vectors steer differently." Direction
information in the role vector carries independent behavioral signal.

## Interpretation

1. **Geometry is behaviorally meaningful.** Roles that are close in cosine on
   the role-vector cloud produce more similar steered outputs (in judge
   sub-dimension space) than chance — but the effect size is moderate. About
   2 – 7% of pairwise behavioral variance is explained by representational
   distance (r² ≈ 0.02 – 0.07).
2. **The link grows with alpha.** Push the model harder and the geometry
   matters more, as you'd expect if the persona vector becomes the dominant
   source of variance at high steering strength.
3. **Direction ≠ magnitude.** Partial RSA controlling for ||v|| barely moves
   the result, so this is an independent finding from the magnitude story.
4. **Caveat: the ceiling is low.** Pairwise judge profiles are noisy averages
   over 50 questions × 5 sub-dimensions. The 24-d behavioral profile likely
   has a noise-floor RSA much below 1.0; the observed r values should be read
   relative to that ceiling, not 1.0.

## Files

```
data/
  rdm_repr_cos.npy, rdm_repr_l2.npy   representational RDMs (275x275)
  rdm_beh_corr.npy, rdm_beh_l2.npy    behavioral RDMs (full 24-d profile)
  rdm_beh_corr_alpha{a}.npy           per-alpha behavioral RDMs (4 files)
  rdm_norm.npy                        norm-difference RDM (control)
  role_order.json                     role names in matrix index order
  behavioral_profiles.csv             one row per role, 24 features
  role_norms.csv                      one row per role, ||v||
  rsa_main.csv                        4-combo RSA + Mantel p
  rsa_per_alpha.csv                   per-alpha RSA breakdown
  rsa_partial.csv                     partial RSA controlling for ||v||
  mantel_null_distribution.npz        null Spearman r values
figures/
  fig1_rdms_side_by_side.png          repr vs beh RDM, common sort
  fig2_pairwise_distance_scatter.png  hexbin of pairwise distances
  fig3_mantel_null.png                null distribution per combo
  fig4_rsa_per_alpha.png              per-alpha bars
  fig5_rsa_distance_grid.png          4-combo grid heatmap
  fig6_partial_rsa.png                full vs partial RSA
  fig7_summary_panel.png              6-panel summary
```
