# Changes from "occupational_networks" branch:
- Filtered number of occupations based on the data collection precision and the other attribute confidence bounds => Reduced from 1016 to 668 occupations.
- Switched to O*NET 30.1 instead of 29.1 (previously used) mainly favouring the latest HEXACO inclusions and more obvious Distinctiveness Rank of a Work Style for an occupation, Included HEXACO in the JSONs for the occupation profile.
- Added Work Contexts for additional social context regarding  the occupation. Though indirect, they signal interesting attributes that a person in the occupation face, such as `"Face-to-Face Discussions with Individuals and Within Teams": "Every day",` or `"Work Outcomes and Results of Other Workers": "High responsibility"`. 
- Filtered the tasks to top-10 based on importance and relevance to the occupation. Helps in reducing the token size while still capturing the core of the occupation's nature.
- Added `LOOK_TO_FILTER.json` in `data/` for future reference to remove 354 occupations if needed. This is based on the rank of the work styles of each occupation.
- Updated SystemPrompt generation template to consider all the new context to generate more detailed system prompts.
- Generated basic systemprompts for about 10 occupations to test the system prompt. Found issues, 
    -physical items are still present in metadata and can possibly influence generation.
    - prompts read like a data dump rather than a coherent persona. 
    - managerial roles are not differentiated enough. the prompts should focus more on the domain specific aspect of the managerial tasks. 
- Done multiple system-prompt generation tests with `Qwen-235B-Instruct`, `GPT-5-mini`, `GPT-4o-mini` and `GPT-OSS-120B`. Found `GPT-OSS-120B` to be fast and suitable. GPT-5-mini wasn't able to create a good prompt at all, and Qwen was too inconsistent to have coherent prompts. 
- Found that an addendum `Answer in first person as this person would in a conversation.` solved the inconsistency of responses, at least in Olmo-3.1-32B, now the task is to validate the responses. Main issue: `Honestly` epidemic. The respnoses almost all have or start with the phrase, this either is Olmo's way of responding to the prompt or something else is wrong. 
- Compared vocational persona generation across three generator models (Gemini, Grok, DeepSeek) on 10 roles. Results:

| Dimension | Gemini | Grok | DeepSeek | GPT-OSS-120B | Kimi-K2.5 |
|---|---|---|---|---|---|
| Prompt focus | Faceted, vivid | Task-enumeration dump | Single-thread, focused | Single-thread, focused | Faceted | 
| Prompt length | Moderate | Long/dense | Short | Short | Moderate |
| Anti-AI framing | Varied, natural | Copy-pasted template | Varied, specific | Varied, specific  | Varied, natural | 
| Prompt bug | None observed | None observed | Pronoun switching (You/I) | Pronoun switching (You/I) | No major issues | 
| Question variety | High | Low (repetitive) | High | Medium | High | 
| Question depth | Good psychological dilemmas | Formulaic "removal" probes | Concrete scenarios, strong | Good psychological dilemmas | Varied Psychological dilemmas |
| Question overlap | Minimal | Significant duplication | Minimal | Minimal | Minimal | 
| Role grounding | Strong environmental immersion | Surface-level task listing | Strong behavioral identity | Varied Behavioral identity | Strong behavioral identity  and immersion| 

---

# pvx


This repository contains persona dataset generation and evaluation utilities built around Inspect AI. Library code lives under `src/pvx/` (datasets, helpers, tasks).

## Quick start
- Create a virtual environment and install with `uv sync` (or set `PYTHONPATH=src` for local runs).
- Run a smoke evaluation, for example: `uv run inspect eval inspect_evals/bbeh_mini --model hf/Qwen/Qwen2.5-1.5B-Instruct --limit 2 --log-dir logs/bbeh_smoke`.
- View logs with `uv run inspect view --log-dir logs`.

## VS Code
I recommend installing the Inspect AI VS Code extension (Marketplace ID `ukaisi.inspect-ai`). It adds log browsing, task panels, and integrated run/debug support for `.eval` artifacts.

## WandB logging (Inspect WandB)
- Already bundled via dependency `inspect-wandb>=0.2.0`.
- One-time auth: `WANDB_API_KEY=...` or `wandb login`; set project/entity with `wandb init` in this repo.
- Run any eval as usual, e.g. `uv run inspect eval inspect_evals/bbeh_mini --model hf/Qwen/Qwen3-1.7B --limit 1`.
- Inspect WandB auto-hooks Inspect AI (no code changes). Console will show links to the run on wandb.ai.
- Config overrides via env vars (e.g., `WANDB_PROJECT`, `WANDB_ENTITY`) or `INSPECT_WANDB_MODELS_*` / `INSPECT_WANDB_WEAVE_*`. See docs: https://inspect-wandb.readthedocs.io/

## Config presets
- Model presets live in `configs/models.yaml` (names, model ids, default generation params).
- Run presets in `configs/runs.yaml` map tasks to model presets, limits, and log dirs.
- Example (bash with `yq`):
  ```
  RUN=bbeh-mini-qwen3-1.7b
  TASK=$(yq '.runs[] | select(.name==strenv(RUN)).task' configs/runs.yaml)
  MODEL_REF=$(yq '.runs[] | select(.name==strenv(RUN)).model_ref' configs/runs.yaml)
  LIMIT=$(yq '.runs[] | select(.name==strenv(RUN)).limit' configs/runs.yaml)
  LOGDIR=$(yq '.runs[] | select(.name==strenv(RUN)).log_dir' configs/runs.yaml)
  MODEL_ID=$(yq '.models[] | select(.name==strenv(MODEL_REF)).model' configs/models.yaml)
  TEMP=$(yq '.models[] | select(.name==strenv(MODEL_REF)).generate.temperature' configs/models.yaml)
  MAXTOK=$(yq '.models[] | select(.name==strenv(MODEL_REF)).generate.max_tokens' configs/models.yaml)
  uv run inspect eval "$TASK" --model "$MODEL_ID" --limit "$LIMIT" --log-dir "$LOGDIR" -S temperature="$TEMP" -S max_tokens="$MAXTOK"
  ```


## LLM as Judge: Automated Response Evaluation

The `llm_as_judge.py` script provides a command-line interface for evaluating model responses using LLMs as automated judges. It supports OpenAI, vLLM, and local Hugging Face models, and can score responses on a 0-100 scale or return a refusal.

### Usage

Activate your environment and run:

```
python src/pvx/pvx_models/llm_as_judge.py \
  --question "My coworker keeps taking credit for my work in meetings. What should I do?" \
  --answer "That’s infuriating, and your coworker is being completely shameless. ..." \
  --backend openai \
  --model openai/gpt-oss-20b \
  --base_url https://api.together.xyz/v1 \
  --api_key_env TOGETHER_API_KEY \
  --eval_type 0_100 \
  --device cuda \
  --dtype float16
```

**Required arguments:**
- `--question`: The evaluation question (string)
- `--answer`: The model response to evaluate (string)

**Optional arguments:**
- `--backend`: Backend to use (`openai`, `vllm`, `hf_local`)
- `--model`: Model to use for backend
- `--local_model`: Local HF model to use for `hf_local` backend
- `--base_url`: Base URL for OpenAI/vLLM endpoints (set to `None` for default)
- `--api_key_env`: Environment variable for API key
- `--eval_type`: Evaluation type (default: `0_100`)
- `--device`: Device for local inference (`cuda`, `cpu`, etc.)
- `--dtype`: Data type for local model (`float16`, `float32`, etc.)

**Example output:**

```
Score: 87
```

**Note:**
- The prompt template is fixed in the script but can be modified in the source.
- Environment variables for API keys must be set as described above.

---

## Response Generation: Automated Trait-Aligned Answer Synthesis

The `response_generation.py` module provides utilities for generating trait-aligned responses using LLMs. It is used for producing positive/negative answers for RIASEC and other persona traits.

### Usage

Call from Python or as part of the RIASEC pipeline. 

Example (Non-RIASEC Response Generation):

* Use `python src/pvx.pvx_models/response_generation.py --question "What is the theory of relativity"`

Example (RIASEC Response Generation):

* Use `python src/pvx.pvx_models/response_generation.py --riasec_positive --pos_neg_trait "Make your answers realistic" --question "Do you like to build puzzles"`

**Arguments:**
- `messages`: List of chat messages (system/user/assistant roles)
- `temperature`, `max_new_tokens`, etc.: Standard generation parameters

---

## RIASEC Persona Model: Pre Generate Responses + Persona Model Generation

The `riasec_persona_model.py` script/class extracts persona vectors for a given RIASEC trait using pregenerated positive/negative responses.

### Usage

Pre-generate responses for a trait:

* Use `python src/pvx/pvx_models/riasec_persona_model.py --pre_generate_response --trait social`


Extract persona vectors and generate a response:

* Use `python src/pvx/pvx_models/riasec_persona_model.py --trait social --model_name Qwen/Qwen2.5-7B-Instruct --question "Do you like organizing events?"`


**Arguments:**
- `--pre_generate_response`: Generate and log positive/negative responses for the trait
- `--trait`: RIASEC trait name
- `--model_name`: Model to use
- `--question`: Prompt for generation
- Other generation parameters: `--max_new_tokens`, `--alpha`, `--temperature`, etc.

---

## RIASEC Judge: Automated Trait Evaluation

The `riasec_judge.py` module provides a class for evaluating responses according to RIASEC trait alignment. Used in the pipeline for scoring generated answers.

### Usage

Call from Python or as part of the pipeline. Example:

* Use `python src/pvx/pvx_models/judges/riasec_judge.py --trait social`

**Arguments:**
- `trait`: RIASEC trait name
- `question`: The question being evaluated
- `answer`: The model response to score

---
## Setup Persona Models
* Use `src/pvx/pvx_models/persona_dataset.py` to build persona traits
* Use `src/pvx/pvx_models/persona_model.py` to test the persona model core
* Use `src/pvx/pvx_models/riasec_persona_model.py` to generate a riasec specific persona model

## CLI runner
- Use `scripts/run_eval.py` to launch evals from presets.
- Examples:
  - `python scripts/run_eval.py --run bbeh-mini-qwen3-1.7b`
  - `python scripts/run_eval.py --run bbh-logical-deduction-qwen1.5b`
  - `python scripts/run_eval.py --run extras-boardgame-qa-qwen1.5b --limit 10`
- Flags:
  - `--run NAME` selects a run from `configs/runs.yaml`
  - `--model NAME` overrides the model preset; or `--model-id` to bypass presets
  - `--limit`, `--log-dir` override per-run values
  - `--model-arg key=value` and `--solver-arg key=value` (repeatable) map to `-M` / `-S` in `inspect eval`
  - `--dry-run` prints the command only
- Use `scripts/riasec_pipeline_eval.py` to perform the full pipeline of generating a riasec specific persona vector
- Examples:
  - `python scripts/riasec_pipeline_eval.py --pregenerate --generate_dataset --trait social --target_count 7`
  - `python scripts/riasec_pipeline_eval.py --trait social`
- Flags:
  - `--pregenerate` will pregenerate a set of positive and negative responses to the list of riasec trait questions
  - `--generate_dataset` will generate the dataset for the trait specified.
  - `--trait` the RIASEC trait to generate a persona vector for with the RIASEC pipeline.
  - `--target_count` the target number of pregenerated responses to be generated for each question's positive and negative responses.
