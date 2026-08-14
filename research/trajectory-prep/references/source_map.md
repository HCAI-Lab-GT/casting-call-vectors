# Source map

## Paper

- Title: *Tracing Persona Vectors Through LLM Pretraining*
- arXiv: `2605.13329v1`
- Local copy: `paper/2605.13329v1.pdf`
- Official code footnote in the PDF: `https://github.com/epfl-dlab/pretraining_persona`

## Official reproduction repo

- URL: https://github.com/epfl-dlab/pretraining_persona
- Purpose: code and data to reproduce the paper's experiments and figures.
- License shown on GitHub: Apache-2.0.
- Repo directories to inspect: `analysis/`, `data/`, `pipeline/`, `results/`, `source/`.
- Key scripts named in README: `pipeline/checkpoint_sweep.sh`, `pipeline/transfer_sweep.sh`, `source/generate_vec.py`, `source/activation_steer.py`, `source/eval_persona.py`.

## Upstream persona vectors repo

- URL: https://github.com/safety-research/persona_vectors
- Purpose: original persona-vector method implementation that the paper builds on.
- License shown on GitHub: Apache-2.0.

## OLMo models

- Base trajectory: https://huggingface.co/allenai/Olmo-3-1025-7B
- Instruct final/RLVR: https://huggingface.co/allenai/Olmo-3-7B-Instruct
- SFT target: https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT
- DPO target: https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO
- Related code: https://github.com/allenai/OLMo-core, https://github.com/allenai/open-instruct, https://github.com/allenai/OLMo-Eval

## Apertus models

- Base trajectory: https://huggingface.co/swiss-ai/Apertus-8B-2509
- Instruct target: https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509
- Training data reconstruction scripts: https://github.com/swiss-ai/pretrain-data
- Tech report repo: https://github.com/swiss-ai/apertus-tech-report

## Notes for the agent

- Do not download every checkpoint at once. Start with two revisions.
- Hugging Face revisions/branches may be large and may use cache deduplication inconsistently across branches.
- The official repo README examples may use `Apertus/Apertus-8B-2509`; verify current public namespace. The current Hugging Face namespace found for Apertus is `swiss-ai`.
