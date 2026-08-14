# Key paper facts for implementation

Use the local PDF for full details. These notes are intentionally concise.

## Core method

For each trait/persona:

1. Build positive and negative contrastive prompts.
2. Generate continuations from the checkpoint model.
3. Judge each generation for trait expression and coherence on 0-100 scales.
4. Retain only generations with both scores >= 50.
5. Record mean residual-stream activations over generated tokens.
6. Compute vector: `v = mean(positive activations) - mean(negative activations)`.
7. During generation, steer by adding `c * mu_l * v / ||v||_2` at layer `l`, where `mu_l` is the steered model's local mean residual-stream norm.

## Dataset sizes in paper

- Extraction: 20 prompts x 5 phrasings = 100 generations per persona before sampling multiplicity.
- Evaluation: 20 disjoint neutral prompts.
- Sampling: 10 continuations per evaluation prompt.
- Generation cap: 64 tokens.
- Temperature: 0.5.
- Repetition penalty: 1.1.

## Main analyses

- Same-checkpoint emergence: source checkpoint equals target checkpoint.
- Transfer: source vector comes from a base pretraining checkpoint, target model is final base or instruct/SFT/DPO/RLVR.
- Geometry: cosine to final vector, adjacent-checkpoint cosine, MDS.
- Facets: subfacet annotation for evil and sycophantic.
- Controls: random direction and label-shuffled vectors.

## Important interpretation

The paper argues that the earliest extractable checkpoint is a lower bound: the extraction pipeline requires enough language fluency for coherent, trait-expressing generations.
