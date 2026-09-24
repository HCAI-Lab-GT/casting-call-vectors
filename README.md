# Casting Call Vectors (CastVectors)

Role vectors for LLMs: extract a steering direction for each of 275 occupational, social and
fictional **roles** (accountant, activist, absurdist, …). Adding that direction to a model's
residual stream makes the model take on the role without a system prompt. The repo also
contains the harness that compares these vectors against a prompted gold standard and an
[assistant-axis](#data-shipped-with-the-repo) baseline, plus every analysis and figure script
behind the paper.

> **New here?** Start with [SETUP.md](SETUP.md). It installs the repo and walks you
> through a series of tiny "baby runs" that check the repo works on your hardware before you
> launch anything big.

---

## Contents

- [Method in one paragraph](#method-in-one-paragraph)
- [Pipeline overview](#pipeline-overview)
- [Installation](#installation)
- [Configuration and API keys](#configuration-and-api-keys)
- [Running each stage](#running-each-stage)
- [Running on a SLURM cluster](#running-on-a-slurm-cluster)
- [Data shipped with the repo](#data-shipped-with-the-repo)
- [Analysis and paper figures](#analysis-and-paper-figures)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Known issues](#known-issues)

---

## Method in one paragraph

A **role vector** is a contrastive mean difference of activations. At one decoder layer, we
run the model on many (role instruction, question) pairs and record the hidden states:

1. once with a role-conditioned system prompt (positive), and
2. once with a blank system prompt (baseline).

The vector is the mean of the positive states minus the mean of the baseline states. We keep
three views of it: the prompt's last token, the average over response tokens, and the
response average at every layer. Before a pair is used, an LLM judge (`RoleJudge`, 0–3 scale)
filters it. A pair survives only if the positive response scores 3 ("fully in character")
and the baseline response doesn't score 0 (refusal).

**Steering** adds `alpha * response_vector` to the last-token hidden state at that layer.
This is done with a forward hook, installed per request, so concurrent requests in one
process don't interfere.

## Pipeline overview

```
 persona_data/role_datasets/<role>.json     (role instructions + questions, LLM-generated)
                │
                ▼
 ┌──────────────────────────────┐   API / vLLM / local HF — no GPU needed with an API
 │ 1. CPU phase                 │   generate answers with & without the role prompt,
 │   roles_optimized_cpu.py     │   judge them, keep the accepted pairs
 └──────────────┬───────────────┘
                ▼
 persona_data/model_qa_responses/<answer_model>/<role>.json
                │
                ▼
 ┌──────────────────────────────┐   GPU: one forward pass over the accepted pairs,
 │ 2. GPU phase                 │   saves one vector per requested sample count
 │   roles_optimized_gpu.py     │
 └──────────────┬───────────────┘
                ▼
 persona_data/model_inits/<role>_persona_initialization/<model>_layer<L>_count<N>.safetensors
                │
                ▼
 ┌──────────────────────────────┐   GPU (+ API if judging): for each question generate
 │ 3. Gold-prompt experiments   │   {assistant-axis steered, gold role-prompted,
 │   gold_prompt_experiments.py │   role-vector steered}, optionally judge all three
 └──────────────┬───────────────┘
                ▼
 experiment_data/gold_prompt_experiments/Comparison_GoldStandard_<role>.csv
                │
                ▼
 analysis/empirical/*, analysis/geometry/*, scripts/figures/*   (CPU, local)
```

The CPU and GPU phases are split on purpose: GPU nodes never sit idle while waiting on API
calls. Every stage can be resumed:

- The CPU phase checkpoints after each accepted pair and skips roles that already have
  enough pairs.
- The GPU phase skips checkpoints that already exist and are valid.
- The gold harness skips any (role, layer, sample_count, alpha, temperature, question) row
  that is already in the CSV.

## Installation

Requirements:

- **Linux x86_64**, **Python 3.12** (exactly; `>=3.12,<3.13`), and
  [`uv`](https://docs.astral.sh/uv/).
- An **NVIDIA GPU** for extraction and steering. OLMo-3-7B in fp16 needs about 16 GB of
  weights, so plan for a GPU with **24 GB or more** (A10G, L4, L40S, A100, H100, RTX 4090).
- `vllm` is a hard dependency, so `uv sync` on macOS or Windows will probably fail. Use a
  Linux box or a cluster.

```bash
git clone https://github.com/HCAI-Lab-GT/casting-call-vectors.git
cd casting-call-vectors
git lfs install && git lfs pull      # safetensors under persona_data/model_inits are LFS-tracked
uv sync --extra dev                  # creates .venv with pvx installed in editable mode
cp .env.example .env                 # then fill in keys (see below)
```

Everything below assumes you run it from the repo root. Either prefix commands with
`uv run`, or `source .venv/bin/activate` first.

## Configuration and API keys

Settings are read from `.env` (loaded with `python-dotenv`) and from each script's CLI flags.
Which keys you need depends on what you run:

| Variable | Needed for | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | Gold-experiment judging (`-j`); CPU-phase answer generation with the default `--backend openai` | Gold judge model is `gpt-4.1-mini` via `https://openrouter.ai/api/v1` |
| `TOGETHER_API_KEY` | CPU-phase pair filtering (`RoleJudge`) | Judge model `openai/gpt-oss-20b` via `https://api.together.xyz/v1` |
| `OPENAI_API_KEY` | Fallback key if the configured `api_key_env` is unset | |
| `VLLM_API_URL` | `--backend vllm` with no `--base_url` | Point at your own vLLM server |
| `HF_TOKEN` | Gated models (Llama, Gemma) | Not needed for OLMo or Qwen |
| `HF_HOME` | Where model weights are cached | **Set this to a big disk** (scratch) on clusters |
| `PACE_ACCOUNT` | SLURM wrapper scripts | GT PACE charge account; see [SLURM](#running-on-a-slurm-cluster) |
| `WANDB_*` | Optional logging | |

Steering and extraction with the vectors shipped in the repo need **no API keys at all**.
Keys only come in when you generate new Q/A pairs or turn on judging.

## Running each stage

All commands are plain Python entry points; the SLURM wrappers call exactly these. Use
`--help` on any of them to see every flag.

### 1. CPU phase: generate and judge Q/A pairs

```bash
uv run python src/pvx/implementations/roles_optimized/roles_optimized_cpu.py \
    --roles accountant activist \
    --answer_model allenai/Olmo-3-7B-Instruct \
    --backend openai \
    --target_pairs 50
# output: persona_data/model_qa_responses/allenai__Olmo-3-7B-Instruct/<role>.json
```

- `--backend` is one of `openai` (any OpenAI-compatible API; OpenRouter by default),
  `vllm` (your own server), or `hf_local` (loads `--answer_model` on the local GPU).
- `--base_url` and `--api_key_env` choose the endpoint and which env var holds its key.
- Role names are the file stems in `persona_data/role_datasets/` (all 275 are listed in
  `configs/role_list.json`).

### 2. GPU phase: extract role vectors

```bash
uv run python src/pvx/implementations/roles_optimized/roles_optimized_gpu.py \
    --activation_extraction_model allenai/Olmo-3-7B-Instruct \
    --roles accountant activist \
    --layer 16 \
    --counts 20 40 50
# output: persona_data/model_inits/<role>_persona_initialization/
#         allenai__Olmo-3-7B-Instruct_layer16_count{20,40,50}.safetensors (+ manifest.json)
```

- `--counts` saves a checkpoint each time that many pairs have been accumulated, all in one
  pass.
- `--layer_fraction 0.5` picks the layer as a fraction of model depth instead of an index.
  Use it when comparing models with different depths.
- `--answer_model` is only needed if the Q/A pairs were generated by a different model than
  the one you extract from.

### 3. Gold-prompt experiments: steer, compare, and judge

```bash
uv run python src/pvx/experiments/gold_prompt_experiments.py \
    --model allenai/Olmo-3-7B-Instruct \
    --roles accountant \
    --layers 16 --sample_counts 50 \
    --alphas 1 1.5 2 2.5 --temperatures 0.2 \
    --judge_responses
# output: experiment_data/gold_prompt_experiments/Comparison_GoldStandard_accountant.csv
```

- For each question in `configs/validation_questions.jsonl` (229 questions), this writes one
  row per (role, layer, sample_count, alpha, temperature, question). Each row holds the
  assistant-axis, gold-prompted and steered responses, and their judge scores if `-j` is on.
- `--question "..."` runs a single question instead of the whole file.
- `--save_dir` writes somewhere else. **Use it for any test run** so you don't touch the
  committed paper CSVs.
- **If the vector for a (role, layer, count) isn't on disk, this script runs the full
  extraction itself** (`load_or_create`). Run stage 2 first unless that's what you want.

Other entry points:

- `src/pvx/experiments/pairwise_judge_experiments.py`: pairwise judge comparisons.
- `src/pvx/implementations/roles_layers/role_layers_persona_model.py`: single-process
  multi-layer extraction with `--layers` / `--layer_fractions`.
- `src/pvx/implementations/roles_layers/assistant_axis_persona_model.py`: load one
  assistant-axis `.pt` vector and generate.

## Running on a SLURM cluster

`scripts/cluster/slurm/` holds pairs of files:

- `foo.sh` runs on the login node. It reads `.env`, validates arguments and calls
  `sbatch foo.sbatch`.
- `foo.sbatch` is the job body.

The `*_autobatch.sh` scripts compare what is expected against what is already on disk and
submit jobs only for the missing work. This is how we parallelize across roles.

```bash
# one role, GPU extraction; chains a gold experiment automatically when it finishes
bash scripts/cluster/slurm/roles_optimized_gpu_batch.sh \
    --role accountant --layer 16 --counts 20 40 50 [--no-chain]
```

Logs go to `logs/slurm/<script-name>/`.

The wrappers are written for **Georgia Tech PACE**:

- `--account "$PACE_ACCOUNT"` and `--qos embers`
- `module load uv`
- `--mem 512G`

**On any other cluster**, edit the `### SLURM SETTINGS ###` block at the top of the `.sh`
file (account, QOS/partition, GPU `--gres`, memory, time), and remove or replace
`module load uv` in the `.sbatch` file. The Python commands inside don't change.

`scripts/cluster/modal_batching/` has [Modal](https://modal.com) versions of the gold
experiment and backfill jobs. Their dependency pins are set separately from
`pyproject.toml`, so check them before relying on them.

## Data shipped with the repo

| Path | Contents |
|---|---|
| `persona_data/role_datasets/` | 275 role datasets: role instructions, eval questions and eval prompt, one JSON per role |
| `persona_data/gold_labels_prompts_dataset/` | Hand-checked gold-standard role prompts (`<role>_gold_label.json`) |
| `persona_data/model_inits/` | **Precomputed OLMo-3-7B-Instruct role vectors**, layer 16, counts 20/40/50 (legacy JSON format, loaded transparently) |
| `persona_data/assistant-axis/olmo-3-7b-instruct/vectors/` | Assistant-axis baseline vectors, one `.pt` per role |
| `persona_data/pt_vectors/` | Role vectors exported as `.pt` (layer 16, 4096-d) for analysis |
| `experiment_data/gold_prompt_experiments/` | Paper results: one CSV per role, alphas {1, 1.5, 2, 2.5} |
| `experiment_data/pairwise_judge_experiments*/` | Pairwise judge results |
| `configs/validation_questions.jsonl` | The 229 evaluation questions |
| `configs/role_list.json` | Role name → one-line description |

Because the OLMo vectors ship with the repo, you can steer without running any extraction.

## Analysis and paper figures

All analysis runs on CPU from the committed CSVs and vectors.

- **`scripts/figures/*.py`** regenerate the paper figures into `scripts/figures/out/`.
  - Each figure function **recomputes the paper's numbers and hard-asserts them**
    (`verify(...)` raises `SystemExit` on mismatch), so these scripts double as
    regression tests for the published results.
  - Scripts: `headline_figs.py`, `controllability_figs.py`,
    `appendix_controllability_figs.py`, `geometry_figs.py`, `pairwise_figs.py`.
- **`analysis/empirical/<study>/`**: each study is a `run_analysis.py` + `make_figures.py`
  pair. Studies:
  - `score_trajectory_comparison`, `rsa_geometry_behavior`, `role_vector_decomposition`
  - `gap_closing`, `best_vs_best`, `paired_significance`
  - `intrinsic_dimensionality`, `behavioral_diversity`, `perp_component_rsa`
  - `subdimension_advantage`, `subdim_correlation_collapse`
  - `vector_magnitude_effect_on_stability`, `behavioral_noise_floor`
- **`analysis/geometry/`**: geometric analysis of the vector set (density, sparsity,
  combinations).
- **`frontend/`**: a static CSV viewer for browsing result files
  (`cd frontend && python -m http.server 8080`).

## Repository layout

```
casting-call-vectors/
├── src/pvx/                          # the installable library
│   ├── abstraction/                  # AbstractPersonaModel (extraction math, steering hook,
│   │                                 #   safetensors I/O), AbstractDataset, AbstractJudge
│   ├── implementations/
│   │   ├── roles/                    # RolePersonaModel + RoleDataset (base role impl)
│   │   ├── roles_layers/             # RoleLayersPersonaModel (multi-layer / multi-count),
│   │   │                             #   AssistantAxisPersonaModel (baseline)
│   │   ├── roles_optimized/          # CPU/GPU split production pipeline
│   │   ├── steer_only/               # load a vector and steer, no extraction
│   │   ├── judges/                   # RoleJudge, LLMJudge, BatchScorer, psychometric judges
│   │   └── base/ riasec/ trait_coverage/   # earlier trait/RIASEC variants
│   ├── experiments/                  # gold_prompt_experiments.py (main harness), pairwise, identity
│   ├── utils/                        # response generation, io_utils (atomic writes, locks),
│   │                                 #   layer_utils (layer_fraction), prompts, logging
│   ├── inspect_modelapi/ tasks/      # Inspect-AI integration + benchmarks (legacy)
├── scripts/
│   ├── cluster/slurm/                # SLURM submit (.sh) + job (.sbatch) pairs, autobatchers
│   ├── cluster/modal_batching/       # Modal cloud equivalents
│   ├── figures/                      # paper figure regeneration (self-verifying)
│   └── patch_scripts/                # one-off data repair / backfill utilities
├── configs/                          # role lists, validation questions, scoring rubrics
├── persona_data/                     # datasets, gold prompts, vectors (see above)
├── experiment_data/                  # result CSVs
├── analysis/                         # empirical + geometry analyses
├── tests/                            # pytest: unit + integration (CPU-only, no downloads)
└── docs/                             # CONFERENCE_ROADMAP.md, dev notes
```

`configs/models.yaml`, `runs.yaml` and `psychometric_runs.yaml`, together with
`scripts/helpers.py`, belong to an older Inspect-AI workflow. They are only used by
`analysis/old_psycho/`.

## Development

```bash
uv run pytest tests -q          # CPU-only; mocks the model, no weights downloaded
uv run ruff check . && uv run ruff format .
uv run pre-commit install --hook-type pre-commit --hook-type pre-push   # ruff on commit, pytest on push
```

When you add a new experiment axis, reuse the resumability pattern:

- Write files atomically and take a file lock (`pvx.utils.io_utils`).
- Skip work whose output is already on disk.
- Let an autobatch script submit only the missing jobs.

The planned scope for the conference version (more models, layer sweeps, multi-turn
stability) is in [docs/CONFERENCE_ROADMAP.md](docs/CONFERENCE_ROADMAP.md).

## Known issues

- `ResponseGeneration` falls back to an internal GTRI URL when `--backend vllm` is used and
  neither `--base_url` nor `VLLM_API_URL` is set. Always set one of them.
- The gold harness passes the judge model as `gpt-4.1-mini` with no provider prefix. If
  OpenRouter rejects the model id, change `judge_model_id` in `GoldPromptExperiments` to
  `openai/gpt-4.1-mini`.
- The scripts in `scripts/patch_scripts/` and `scripts/cluster/modal_batching/` write the
  gold CSVs **without** the file lock the main harness uses. Don't run them at the same time
  as gold experiments on the same roles.
- The SLURM wrappers assume PACE (see above).
