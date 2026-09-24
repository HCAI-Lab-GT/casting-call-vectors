# Onboarding: setup and smoke-test runs

This guide gets the repo installed on **your** machine or cluster and checks that each part
works with runs small enough to finish in minutes. Do the tiers in order: each one checks
something the next one relies on. **Stop at the first tier that fails** and report back (see
[the end of this doc](#reporting-back)).

| Tier | What it proves | Needs | Rough time |
|---|---|---|---|
| [0](#tier-0--install) | Environment installs | Linux, Python 3.12, `uv` | 5–15 min |
| [1](#tier-1--unit-tests-cpu-no-keys) | Library imports, core I/O logic is correct | CPU only | ~1 min |
| [2](#tier-2--reproduce-a-paper-figure-cpu-no-keys) | Data is intact, analysis stack matches paper numbers | CPU only | a few min |
| [3](#tier-3--steering-smoke-test-gpu-no-keys) | Model loads on your GPU, steering hook works | 1 GPU with ≥24 GB | ~10 min + model download |
| [4](#tier-4--judging-smoke-test-gpu--api-key) | Your API keys and judge calls work | Tier 3 + `OPENROUTER_API_KEY` | ~2 min |
| [5](#tier-5--end-to-end-extraction-gpu--api-keys) | Full pipeline: generate → judge → extract → steer | Tier 4 + `TOGETHER_API_KEY` | 15–30 min |

Most people only need tiers 0–3. Tier 5 matters only if you'll be extracting **new** vectors,
for example for a new model or new layers.

**Golden rule for every run below:** always write output to `./smoke_out/`. Never write into
`persona_data/` or `experiment_data/`, which hold the paper's data. `smoke_out/` is not
tracked by git, so you can delete it when you're done.

---

## Tier 0 — Install

### Prerequisites

- **OS**: Linux x86_64. macOS and Windows are not supported because `vllm` is a hard
  dependency and won't install there.
- **Python 3.12** exactly. `uv` downloads it for you if it's missing.
- **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`. On GT PACE, run
  `module load uv` instead.
- **git-lfs**: `git lfs install`.
- **Disk**: about 3 GB for the repo and venv, **plus about 15 GB per 7B model** in the
  Hugging Face cache.

### Steps

```bash
git clone https://github.com/HCAI-Lab-GT/casting-call-vectors.git
cd casting-call-vectors
git lfs pull
uv sync --extra dev
cp .env.example .env
```

Edit `.env`. For tiers 0–3, the only thing to set is where model weights go. **On a shared
cluster, point this at scratch or project storage, not your home directory**, because home
quotas fill up fast:

```bash
HF_HOME=/path/to/big/disk/hf_cache
HF_HUB_ENABLE_HF_TRANSFER=1
```

### Check

```bash
uv run python -c "import torch, pvx; print('pvx ok | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

✅ **Pass:** prints `pvx ok`. On a GPU node, `cuda True` is also expected.

Common problems:

- `uv sync` fails building `vllm`: you're probably not on Linux x86_64, or you're on a node
  with a very old glibc. Try a compute node.
- `cuda False` on a GPU node: the torch wheel doesn't match your driver. Run `nvidia-smi` and
  report the driver version.

---

## Tier 1 — Unit tests (CPU, no keys)

The tests mock the model, so no weights are downloaded. Run them anywhere, including a login
node if your cluster allows Python there.

```bash
uv run pytest tests -q
```

✅ **Pass:** every test passes (`N passed`, no failures or errors).

---

## Tier 2 — Reproduce a paper figure (CPU, no keys)

This recomputes the paper's headline numbers from the committed result CSVs and
**hard-fails if any number differs from the paper**. A pass means your checkout is intact and
your pandas/numpy stack gives the same results as ours.

```bash
uv run python scripts/figures/headline_figs.py
```

✅ **Pass:**

- the script ends with `figures written to .../scripts/figures/out`
- `scripts/figures/out/fig_headline_scores.pdf` exists
- no mismatch error or `SystemExit`

❌ **Fail:**

- A mismatch message with a `got` / `want` pair usually means the CSVs are incomplete. Check
  `git status` and `git lfs pull`.
- An import error means tier 0 didn't fully succeed.

---

## Tier 3 — Steering smoke test (GPU, no keys)

This is the most important check. It uses the **OLMo-3-7B role vectors that ship in the
repo**, so no extraction and no API calls are involved.

For one role and one question it generates three short answers:

1. steered with the assistant-axis baseline vector,
2. prompted with the gold-standard role prompt,
3. steered with our role vector.

**Hardware:** 1 NVIDIA GPU with **≥24 GB** of memory. The first run downloads
`allenai/Olmo-3-7B-Instruct` (about 15 GB) into `HF_HOME`.

### Interactive GPU session

```bash
uv run python src/pvx/experiments/gold_prompt_experiments.py \
    --model allenai/Olmo-3-7B-Instruct \
    --roles accountant \
    --layers 16 --sample_counts 50 \
    --alphas 2.0 --temperatures 0.2 \
    --max_new_tokens 64 \
    --question "How should society prepare for technological disruption?" \
    --save_dir ./smoke_out/gold/
```

### Batch job (SLURM)

Copy this, change the `#SBATCH` lines to match your cluster, then `sbatch smoke_tier3.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=cv-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=smoke_out/%x-%j.out
# #SBATCH --account=...   --partition=...   --qos=...   <- your cluster's values

cd "$SLURM_SUBMIT_DIR"
set -a; source .env; set +a
nvidia-smi
uv run python src/pvx/experiments/gold_prompt_experiments.py \
    --model allenai/Olmo-3-7B-Instruct --roles accountant \
    --layers 16 --sample_counts 50 --alphas 2.0 --temperatures 0.2 \
    --max_new_tokens 64 \
    --question "How should society prepare for technological disruption?" \
    --save_dir ./smoke_out/gold/
```

(Run `mkdir -p smoke_out` before submitting, so SLURM has somewhere to write the log.)

### Check

```bash
ls smoke_out/gold/          # -> Comparison_GoldStandard_accountant.csv
uv run python -c "
import pandas as pd
r = pd.read_csv('smoke_out/gold/Comparison_GoldStandard_accountant.csv').iloc[0]
for c in ['assistant_axis', 'baseline', 'steered']:
    print(f'--- {c} ---'); print(str(r[c])[:400])
"
```

✅ **Pass:**

- The CSV has 1 row.
- All three columns contain readable text.
- The `steered` text sounds noticeably like an accountant: procedural, focused on
  auditability and verification, with some finance vocabulary.

The log should show `Loading from legacy JSON`. That line is expected, because the shipped
vectors are stored as JSON.

❌ **Fail:**

- CUDA out-of-memory: your GPU is too small. Try a card with more memory.
- The log shows the script **starting an extraction** instead of loading a vector: it
  couldn't find `persona_data/model_inits/accountant_persona_initialization/...`. Check that
  you're running from the repo root.
- The `steered` text is gibberish: report it along with your GPU model and torch version.

To see how steering strength changes the output, rerun with `--alphas 1 2.5` and compare the
rows. If you keep the same `--save_dir`, rows that already exist are skipped.

---

## Tier 4 — Judging smoke test (GPU + API key)

This adds LLM-judge scoring through OpenRouter. Add your key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-...
```

Rerun the tier 3 command with **`--judge_responses` added** and a fresh output directory:

```bash
uv run python src/pvx/experiments/gold_prompt_experiments.py \
    --model allenai/Olmo-3-7B-Instruct --roles accountant \
    --layers 16 --sample_counts 50 --alphas 2.0 --temperatures 0.2 \
    --max_new_tokens 64 \
    --question "How should society prepare for technological disruption?" \
    --save_dir ./smoke_out/gold_judged/ \
    --judge_responses
```

✅ **Pass:** the CSV row has numeric `assistant_axis_score`, `baseline_score` and
`steered_score` columns, plus comparator sub-scores such as emotional register and
vocabulary choice.

❌ **Fail:**

- `401` / `API key not set`: the key is missing or wrong. Make sure it's in `.env` and that
  you're running from the repo root.
- `model not found` for `gpt-4.1-mini`: see "Known issues" in the [README](README.md).
  The fix is to use the `openai/gpt-4.1-mini` model id.

---

## Tier 5 — End-to-end extraction (GPU + API keys)

This builds a **new** role vector from scratch at a tiny scale: 3 judged pairs instead of 50.
It needs two keys in `.env`:

- `OPENROUTER_API_KEY`: generates answers with the default `--backend openai`.
- `TOGETHER_API_KEY`: the pair-filtering judge (`openai/gpt-oss-20b` on Together).

Running on your own vLLM server instead? Pass `--backend vllm --base_url http://host:port/v1`
in step 5a and point `--api_key_env` at whatever env var holds that server's key. Always set
`--base_url`; see "Known issues" in the README.

### 5a. CPU phase: generate and judge 3 pairs (no GPU needed)

```bash
uv run python src/pvx/implementations/roles_optimized/roles_optimized_cpu.py \
    --roles accountant \
    --answer_model allenai/Olmo-3-7B-Instruct \
    --target_pairs 3 \
    --max_new_tokens 256 \
    --output_dir ./smoke_out/qa/
```

✅ **Pass:** `smoke_out/qa/allenai__Olmo-3-7B-Instruct/accountant.json` exists, and its
`"positive"` list has 3 entries.

If your API provider doesn't host OLMo, or you want everything to run locally, use
`--backend hf_local`. That loads the answer model on the local GPU, so this step then needs
a GPU too. Only the judge still goes through Together.

### 5b. GPU phase: extract, saving checkpoints at 2 and 3 pairs

```bash
uv run python src/pvx/implementations/roles_optimized/roles_optimized_gpu.py \
    --activation_extraction_model allenai/Olmo-3-7B-Instruct \
    --roles accountant \
    --layer 16 \
    --counts 2 3 \
    --qa_responses_dir ./smoke_out/qa/ \
    --output_dir ./smoke_out/vectors/
```

✅ **Pass:**

- `smoke_out/vectors/accountant_persona_initialization/` contains
  `allenai__Olmo-3-7B-Instruct_layer16_count2.safetensors`, a matching `count3` file, and
  `manifest.json`.
- Running the **same command again** logs `All checkpoints for accountant already exist,
  skipping`. That confirms resuming works.

### 5c. Steer with the vector you just built

```bash
uv run python src/pvx/experiments/gold_prompt_experiments.py \
    --model allenai/Olmo-3-7B-Instruct --roles accountant \
    --layers 16 --sample_counts 3 --alphas 2.0 --temperatures 0.2 \
    --max_new_tokens 64 \
    --question "How should society prepare for technological disruption?" \
    --safetensors_dir ./smoke_out/vectors/ \
    --save_dir ./smoke_out/gold_e2e/
```

✅ **Pass:** the output looks like tier 3. With only 3 pairs, expect a weaker and noisier
accountant flavor than the shipped 50-pair vector gives. That's fine, since this run only
checks that the pipeline works, not the quality of the vector.

**Extracting for a different model** (for example Qwen-2.5-7B or Llama-3.1-8B):

- Replace the model id in 5a–5c.
- Use `--layer_fraction 0.5` in 5b instead of `--layer 16`, because models have different
  depths.
- Llama and Gemma are gated, so they need `HF_TOKEN`.
- **Tier 3 and 5c won't work as written for non-OLMo models.** The gold harness also loads the
  OLMo assistant-axis baseline vectors, which have a different hidden size. Stop after 5b
  for other models, or ask before running 5c.

---

## Moving on to real runs

Once the tiers pass:

- **Single machine:** run the same commands with real sizes: `--target_pairs 50`,
  `--counts 20 40 50`, and drop `--question` so the full 229-question set runs. Keep writing
  to your own `--save_dir` and `--output_dir` unless you've agreed otherwise with the team.
- **SLURM:** the wrappers in `scripts/cluster/slurm/` are set up for GT PACE. Edit the
  `### SLURM SETTINGS ###` block in each `.sh` file for your cluster; the README section
  "Running on a SLURM cluster" explains how.
- **Parallelism:** fan work out across roles, one role per job. Every stage skips work that
  is already done, so failed jobs can simply be resubmitted.

## Reporting back

Please send this, filled in, whether everything passed or not:

```
Name / cluster:
GPU model + memory (nvidia-smi):          e.g. A100 80GB
Driver / CUDA (nvidia-smi header):
torch version (tier 0 output):
Highest tier passed:                      0 / 1 / 2 / 3 / 4 / 5
If a tier failed: command run + last ~30 lines of output:
Tier 3 'steered' text (first 2 sentences):
Anything you had to change to make it work:
```

The last field matters most. Anything you had to change is something the docs should cover.
