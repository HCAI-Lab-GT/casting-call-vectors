# Model and repo notes for the agent

## Official paper repo

The official paper repo states that it contains code and data to reproduce the experiments and results. It lists a full pipeline: activation extraction, checkpoint sweeps, transfer analyses, geometry/facet analyses, discourse-type ablations, and controls. It also lists the output shape for vectors as `[layers x hidden_dim]`.

Key repo paths to inspect after cloning:

```text
analysis/
data/trait_data_extract/
data/trait_data_eval/
data/persona_vectors/
data/model_responses/
pipeline/checkpoint_sweep.sh
pipeline/transfer_sweep.sh
source/generate_vec.py
source/activation_steer.py
source/eval_persona.py
source/judge.py
source/deepseek_judge.py
```

## OLMo

The OLMo Hugging Face card says OLMo-3 includes Base/Instruct/Think variants and releases code, checkpoints, and training details. For the 7B base, the card lists 5.93T training tokens, 32 layers, hidden size 4096, 32 attention heads, 32 KV heads, and context length 65,536. It also documents pretraining revision naming such as `stage1-stepXXX`, with `stage2-stepXXX` and `stage3-stepXXX` for later stages.

## Apertus

The Apertus Hugging Face cards describe a decoder-only transformer pretrained on 15T tokens. They state that training intermediate checkpoints are available as branches of the model repository and that training data reconstruction scripts are open.

## Small-run caveat

A small run with `N_PER_QUESTION_EXTRACT=1` and `N_PER_QUESTION_EVAL=1` is for compute/cost validation. It will be noisy and may fail filtering at early checkpoints. It should not be treated as a statistical reproduction.
