# Intrinsic dimensionality of the role-vector cloud

Loads 275 role vectors (4096-dim, layer 16) and asks: how many *true* persona
directions are there in the cloud? If the cloud is effectively 8-dimensional,
most cosine differences between random roles are noise on the remaining 4088
axes — which would explain why `RSA(repr_cos, beh_corr) ~ 0.14` is small even
when the link is real.

## How to reproduce

```bash
.venv/bin/python analysis/empirical/intrinsic_dimensionality/run_analysis.py
.venv/bin/python analysis/empirical/intrinsic_dimensionality/make_figures.py
```

## Diagnostics

For each of four matrices (`repr_raw`, `repr_centered`, `behavioral`, and an
iid-Gaussian null with the same shape and global std as `repr_raw`):

| metric | meaning |
| --- | --- |
| singular value spectrum | sigma_i, sorted desc |
| cumulative explained variance | smallest d for {50, 80, 90, 95, 99}% |
| effective rank | `exp(- sum p_i log p_i)` where `p_i = sigma_i^2 / sum` |
| participation ratio | `(sum sigma^2)^2 / sum(sigma^4)` |
| mean pairwise cosine | global anisotropy proxy |

`repr_centered` subtracts the role-mean direction first, removing the
assistant-axis-like rank-1 component so we measure dispersion *around* the
cloud center. The Gaussian null is the "what would an unstructured 275×4096
matrix at this aspect ratio look like?" reference.

## Why include the behavioral matrix

We z-score the 275×24 behavioral profile and run the same diagnostics. This
gives an apples-to-apples comparison: if `effective_rank(repr) >>
effective_rank(beh)`, the geometry has more directions than the judge can
distinguish, and RSA is inherently bottlenecked from the behavioral side.

## Findings

| matrix                  | n   | dim  | effective rank | PR    | mean cos | d for 90% var |
| ----------------------- | ---:| ----:| --------------:| -----:| --------:| -------------:|
| repr_raw                | 275 | 4096 |  2.7           |  1.4  | **+0.85** |    4         |
| **repr_centered**       | 275 | 4096 |  **49.6**      | 15.1  |   0.00   |    98        |
| **behavioral (24-d)**   | 275 |   24 |  **2.0**       |  1.4  |   0.01   |    2         |
| iid Gaussian null       | 275 | 4096 |  265.9         | 257.6 |   0.00   |  233         |

**Three things jump out.**

1. **The raw role cloud is dominated by a global direction** (mean pairwise
   cosine ≈ +0.85, effective rank ≈ 2.7). That global direction is the
   "assistant-axis"-like component you already deal with. Once you subtract
   the role-mean direction, the cloud has a non-trivial ~50 effective
   dimensions.
2. **The behavioral profile is effectively rank-2** out of a maximum of 24.
   The judge resolves roles along about two consistent axes; the rest of the
   24 features are mostly redundant or noise.
3. **The Gaussian null has effective rank ~266** at the same shape, vs ~50
   for the centered cloud. So the role cloud really is ~5× more concentrated
   than chance — there *is* a low-dim manifold of "persona directions."

## Why this matters for the RSA result

The geometry side (centered representations) lives in ~50 effective
dimensions. The behavior side lives in ~2. Even a perfect-information map
from geometry to behavior can only exploit the ~2 behavioral axes — most of
the role-vector variance is **orthogonal to anything the judge measures**.

That is consistent with the noise-floor finding from
`behavioral_noise_floor/`: the behavioral RDM is reliable (r ≈ 0.97
Spearman-Brown corrected), so noise isn't the limit. The limit is
**dimensional mismatch**: a richer behavioral fingerprint (more sub-dimensions,
or per-question profiles instead of alpha aggregates) is the natural way to
raise the RSA ceiling.

## Files

```
data/
  spectrum_repr_raw.csv        sigma_i for the raw 275x4096 matrix
  spectrum_repr_centered.csv   sigma_i after removing the role-mean direction
  spectrum_beh.csv             sigma_i for the z-scored 275x24 behavioral matrix
  spectrum_random.csv          sigma_i for the iid-Gaussian null
  explained_variance_curves.csv  long-format cumvar(d) for plotting
  summary.csv                  effective_rank, PR, mean cosine, d_explains_*pct
figures/
  fig1_singular_value_spectrum.png    sigma_i vs i, log-y
  fig2_explained_variance_curves.png  cumvar vs d (zoom on first 100)
  fig3_effective_rank_summary.png     effective rank and PR bars
  fig4_pairwise_cosine_distribution.png  anisotropy diagnostic
```
