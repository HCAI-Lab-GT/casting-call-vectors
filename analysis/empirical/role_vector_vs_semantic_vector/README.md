# Role vector vs. semantic vector

Asks: does the persona vector for a role (extracted by contrastive averaging
at layer 16 of `allenai/Olmo-3-7B-Instruct`) line up with a "semantic"
representation of that role's name (the layer-16 hidden state when the model
just reads the word)? Both vectors live in the same 4096-dim residual stream,
so cosine similarity, shared PCA / t-SNE, and Procrustes are well-defined.

## Run

```bash
# 1. Extract layer-16 hidden states for every role name (needs GPU + Olmo).
python analysis/empirical/role_vector_vs_semantic_vector/compute_semantic_vectors.py

# 2. Compare against persona vectors and emit plots / summary CSV.
python analysis/empirical/role_vector_vs_semantic_vector/compare.py
```

## Variants

`compute_semantic_vectors.py` saves four candidates per role:

| key            | how it is built                                                      |
|----------------|-----------------------------------------------------------------------|
| `bare_last`    | tokenize "{role}", take last-token hidden state                       |
| `bare_mean`    | tokenize "{role}", mean over all tokens                               |
| `context_last` | tokenize "I am a {role}.", take last role-name-token hidden state     |
| `context_mean` | tokenize "I am a {role}.", mean over the role-name token span         |

Layer indexing matches `analysis/geometry/full_analysis.py`: with
`layer_steering = 16` we read `hidden_states[15]`, which is the same slot
loaded for the persona vector.

## Outputs

`figures/` contains, per variant:

- `cosine_similarity_hist_{variant}.png` — distribution of matched (persona[i]
  vs name[i]) and unmatched (persona[i] vs name[j], i≠j) cosine.
- `cosine_similarity_top_bottom_{variant}.png` — roles whose name embedding
  best / worst aligns with the persona vector.
- `cross_similarity_heatmap_{variant}.png` — full role × role cosine matrix,
  rows / cols sorted by diagonal, so the matched pairs sit on the visible
  diagonal.
- `pca_overlay_{variant}.png`, `tsne_overlay_{variant}.png` — both vector
  sets projected together with each persona-name pair connected by a thin
  segment, so you can read off how far the name drifts from the persona.
- `procrustes_{variant}.json` — matched / unmatched cosine summary plus a
  Procrustes disparity (lower = more shared geometry).

`summary.csv` lists per-role matched cosine across every variant.
