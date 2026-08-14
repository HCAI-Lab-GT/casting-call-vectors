# Functional spec for training-trajectory persona vectors

## Purpose

Build a research pipeline that traces a behavioral activation direction across model checkpoints.

## Inputs

- `model_id`: Hugging Face model repo.
- `revision`: checkpoint branch/tag/SHA.
- `trait`: behavioral axis, e.g. humorous.
- `positive_prompts`: prompts that elicit the trait.
- `negative_prompts`: contrast prompts that suppress/opposite the trait.
- `evaluation_prompts`: neutral prompts disjoint from extraction prompts.
- `layer`: transformer layer to steer.
- `coefficient`: steering strength.

## Vector extraction

For every retained sample:

1. Generate continuation.
2. Score trait and coherence.
3. If both scores >= threshold, record hidden states.
4. Average residual-stream activations over generated tokens.

For each layer:

```text
v_trait_layer = mean(positive_activation_means) - mean(negative_activation_means)
```

## Steering

At decoding time, modify residual stream at selected layer:

```text
h_l <- h_l + coefficient * mu_l * v / ||v||_2
```

where `mu_l` is the mean residual-stream norm at layer `l` for the target model.

## Metrics

```text
delta_trait = mean(trait_score_steered) - mean(trait_score_baseline)
pass_rate = count(trait_score >= 50) / total_count
```

## Minimal plot

X-axis: source checkpoint token count.  
Y-axis top: same-checkpoint steering delta.  
Y-axis bottom: pass rate.

## Data retention

Store all raw generations, scores, prompts, vector metadata, and cost logs. Do not publish harmful vectors or harmful generated corpora.
