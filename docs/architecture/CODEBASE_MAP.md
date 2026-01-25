# Persona Vectors Evals — Codebase Map

This document sketches how data moves through the new evaluation drop so you can quickly see what runs where.

## Flow Diagram
```
[You / runner]
    |
    | choose TRAIT + NUM_QUESTIONS
    v
PersonaDataset.generate_dataset()
  (dataset/PersonaDataset.py)
    |-- inference_with_client()
    |     |-- Ollama client (if provided)
    |     \-- OpenAI‑compatible API @ https://glados.ctisl.gtri.org
    |         (needs LITELLM_API_KEY)
    |
    |-- generate_trait_description()
    |-- generate_question_instruction()
    |-- parse_dataset_output()   <-- expects <pos_instruction>, <neg_instruction>,
    |                              <questions>, <eval_prompt> tags
    |-- save_dataset_to_json()
           -> dataset/persona_dataset/{trait}_dataset.json
    |
    +--------------------+
    |                    |
    v                    v
persona vector extraction          Inspect AI evaluations
personas/extract_vector_m4         evals/boardgame_task.py
  |  - Qwen2.5-1.5B on MPS         evals/big_bench_hard_task.py
  |  - Hooks layer 14 activations  |  - HF datasets: Boardgame‑QA (test),
  |  - POS vs NEG prompts          |    BigBenchHard/logical_deduction_five_objects (train)
  |  - Saves normalized diff vec   |  - Solver: system_message -> chain_of_thought -> generate
  |                                |  - Scorers: model_graded_fact() / match()
  |                                |
  +--------------------------------+--> logs/<task>/<timestamp>_... .eval
```

## What lives where
- `dataset/PersonaDataset.py` — LLM-backed persona dataset generator; writes JSON under `dataset/persona_dataset/`.
- `dataset/persona_dataset/*.json` — Pre-generated persona datasets (11 traits).
- `personas/extract_vector_m4` — Apple M-series only: derives a persona direction vector from POS/NEG prompts + questions using Qwen2.5‑1.5B.
- `evals/boardgame_task.py` — Inspect AI task for HuggingFace `tasksource/Boardgame-QA` (test split).
- `evals/big_bench_hard_task.py` — Inspect AI task for HF `maveriq/bigbenchhard` subset `logical_deduction_five_objects` (train split).
- `logs/` — Inspect AI run artifacts (`*.eval`).

## Inputs and outputs
- Inputs: trait name, num_questions, model choice, optional Ollama client, LITELLM_API_KEY (for remote API path).
- Generated artifacts:
  - Dataset JSON: `dataset/persona_dataset/{trait}_dataset.json`
  - Optional vector: `personas/analytical_vector_m4.pt` (or value of `OUTPUT_FILE` in the script)
  - Eval logs: `logs/<task_name>/<timestamp>_... .eval`

## Things to watch
- `PersonaDataset.py` imports `.prompts` but no `prompts.py` ships in this drop; add it or adjust the import before running generation.
- The repo is nested (`persona-vectors-evals-main/persona-vectors-evals-main`); run commands from the inner folder.
- `extract_vector_m4` requires Apple Silicon (MPS) and the Qwen model weights available locally.

## Quick start reminders
- To regenerate a trait dataset:
  ```bash
  cd persona-vectors-evals-main/persona-vectors-evals-main
  python -m dataset.PersonaDataset  # adjust __main__ as needed; ensure prompts module exists
  ```
- To run an Inspect AI task (after installing deps):
  ```bash
  inspect evals/boardgame_task.py:boardgame_loader
  inspect evals/big_bench_hard_task.py:bbh_loader
  ```
