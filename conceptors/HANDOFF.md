# Handoff: semantic conceptors for persona steering

Written 4 Nov 2026 at the end of a Claude (cloud) session, for resuming in Claude Code from this folder.

## Goal

Find out whether **conceptors** (soft, per-direction "stencils" fitted to a cloud of activations) are a better way to set up and hold a **persona or trait** in an LLM than steering vectors or activation caps, especially **across many turns**. Two deliverables run in parallel:

1. **Experiments**: collect persona activations, fit conceptors and baselines, evaluate single-turn and multi-turn.
2. **Explainer**: an interactive page for collaborators ("Persona Steering Field Guide") that explains the methods, the data pipeline, the conceptor variants and multi-turn dynamics, and shows our real results.

This builds on the Role Steering / "Casting Call" paper (arXiv 2608.00023).

## What exists in this folder

| Path | What it is |
|---|---|
| `persona_acts.py` | Collection script. Phase 1: 9 roles (assistant ×3 prompts, pirate ×3, doctor, teacher, bard, robot, consultant, ghost ×1, pirate_doctor ×3) plus a hard negative (`pirate_narrated`: the assistant asked to describe how a pirate would answer), 24 questions over 10 task types, 384 answers, mean-over-answer and last-prompt-token activations at every hidden state. Phase 2: three 12-turn conversations (switch / benign / replace) under none / add / cap / conceptor steering. Has the macOS MPS fixes (eager attention, `remove_invalid_values`, per-prompt fallback, `dtype` kwarg). |
| `persona_acts.orig.py` | Pre-fix backup. Safe to delete. |
| `acts_qwen05/` | The 4 Nov run on Qwen2.5-0.5B-Instruct (MPS, float32, seed 7, 1147 s): `acts.npz` (38 MB), `texts.jsonl`, `manifest.json`, and `derived.json` (analysis output). |
| `analyze_acts.py` | Turns a run into `derived.json`: layer sweeps (probe AUC, variance shares), 2-D maps, hard-negative scores, rank test, conceptor logic test, multi-turn persona scores. Needs numpy + scikit-learn. Layer constants at the top are set for 24 layers. |
| `explainer/` | The page. `persona_steering_field_guide.html` is the built, self-contained page (v2). `src/` holds `head.html` (CSS), `body.html` (text), `lib.js` (math, drawing, line charts), `toy.js` (toy plates, method zoo, toy turn dynamics), `real.js` (real-data plates). `build.py` inlines `../acts_qwen05/derived.json`. `v1/` is the first 2-D conceptor page (28 Oct). `screenshots/` are render checks. `tools/` are Playwright scripts (`node tools/check.js light 1280 map-plate rank`, `node tools/interact.js` exercises every control and reports page errors). |

Published copy of the page (v2): https://claude.ai/artifact/AMg74LVB6RKRkdRA9JsjDP

### Commands

```
python persona_acts.py --model Qwen/Qwen2.5-0.5B-Instruct --out acts_qwen05            # collect
python analyze_acts.py --acts acts_qwen05/acts.npz --texts acts_qwen05/texts.jsonl \
    --manifest acts_qwen05/manifest.json --out acts_qwen05/derived.json                 # analyze
python explainer/build.py                                                             # rebuild page
node explainer/tools/interact.js                                                      # smoke-test page
```

### Conventions

- Hidden index `i` = output of decoder layer `i−1`; index 0 is the embeddings. "Steering at module 12" = hidden 13.
- "Standardized" = each dimension z-scored over the 9 main roles. Needed: 3 outlier dimensions carry 60–70% of raw energy.
- Persona score = projection on the pirate-minus-assistant mean difference, rescaled so assistant mean = 0 and pirate mean = 1.

## What the 0.5B run found

Solid, and worth keeping as methodology:

- **The model barely plays the persona.** On task questions the "pirate" answers like the assistant; the persona shows only on self-questions ("Seafarer."). Identity probes revert to "I am Qwen" (and "I am Claude" under add steering).
- **Hard negative beats the real persona.** The assistant describing a pirate scores 1.23 on the answer-average persona axis (pirate = 1.0) and uses more pirate words (4.1% vs 1.7%). Bard 0.96, ghost 0.83: the axis is "storybook character vs assistant", matching the Assistant Axis paper's PC1.
- **Pooling matters.** Last prompt token: AUC 1.0 at layers 4–14 (the prompt is read). Answer average: AUC 0.74–0.92; role explains ~2% of variance, question ~66% (layer 13).
- **Rank trap.** 72 pirate answers in 896 dims → rank 71, 825 directions with keep 0. On an assistant answer (measured from the pirate mean): centered C keeps 66%, NOT 23%, blend (β 0.3) 85%. Raw uncentered C seems to keep 92% but only 67% of the part that varies.
- **Logic test.** Raw-cloud conceptors are all ≥ 0.93 similar (they learn the questions). On per-question differences from the assistant: pirate_doctor ≈ pirate (0.96), OR 0.89, AND 0.74 (below bard).
- **Keyword judge** passes 8/72 pirate answers; filtered vs unfiltered direction cosine 0.81 (~36°).

Not informative yet:

- **Multi-turn pilot.** AI-admitting answers out of 36: none 10, add 8, cap 6, conceptor 4. One sample per turn, 3 conversations, no coherence judge. Add raised the activation meter but not pirate words.

## Diagnosis: why the results are thin

1. **Model too small.** 0.5B ignores the persona prompt on most questions, so there is little persona signal to fit. Every downstream number inherits this. The papers we compare to use 7–8B (Persona Vectors and GCAD: Qwen2.5-7B, Llama-3.1-8B) or 27–70B (Assistant Axis).
2. **Data too thin.** 24 questions × 1 rollout, 72-token answers, no LLM judge. Assistant Axis uses 240 questions × 5 prompts with a 3-level role-play judge; Persona Vectors uses 40 questions with trait/coherence judge filtering. 72 samples in 896 dims guarantees rank deficiency.
3. **Steering pilot is not a fair test.** Strengths were not swept or matched; the read-out meter is nearly the direction we steer along (circular for "add"); no trait/coherence judge; n is tiny.
4. **The pipeline itself works** (MPS fixes, collection, analysis, page build), and the methodological findings above should carry over.

## Plan to get this done

Ordered; each step has a clear output.

1. **Scale the model (GPU).** Run on CUDA (lab cluster) with Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Check `--max_mb` so the needed layers are kept. Make `analyze_acts.py` layer constants relative to `n_layers` (e.g. map layers at fractions, steer ≈ 0.5, read ≈ 0.75).
2. **Real data recipe.** Use the Assistant Axis role prompts and 240 extraction questions (`git clone https://github.com/safety-research/assistant-axis`: `data/roles/instructions/*.json`, `data/extraction_questions.jsonl`). 5 prompts per role, ≥ 3 rollouts, 256–512 new tokens. Keep hard negatives (narrated, quoted dialogue, vocabulary without identity).
3. **LLM judge.** Add a judge pass (papers use gpt-4.1-mini): Assistant-Axis role-play label (fully / somewhat / none) plus a 0–100 coherence score (Persona Vectors rubric, reproduced in the GCAD appendix). Filter to fully-in-character for fitting.
4. **Fix rank.** Save per-token activations for a subset; fit low-rank conceptors (`C = U diag(s) Uᵀ`, k ≤ n) with Ledoit-Wolf shrinkage; fit on per-question differences; always use the interpolation form.
5. **Methods to compare** at matched single-turn trait level: steering vector, activation cap on the persona axis, centered conceptor, contrastive conceptor (in-character AND NOT broke-character), **gated conceptor** (the untested cell), and GCAD if feasible. Controls: matrix-removed (same center and gain), random subspace with matched spectrum.
6. **Multi-turn eval, GCAD-style.** 10 turns, KV cache kept across turns, ~20 conversation groups × 3 samples, judge trait and coherence on every turn; report T1/T5/T10 and drift. Include persona-replacement and "are you an AI?" probes. Read out with the judge and a held-out probe, not the steering direction.
7. **Cross-model geometry.** Compare conceptor keep-spectra across models (rotation-invariant), per persona.
8. **Update the page.** Rebuild from the new `derived.json`, then a prose pass: `src/body.html` quotes 0.5B numbers in many places.

## Open decisions (ask Glenn)

- Judge: which API key / budget (gpt-4.1-mini for parity with the papers, or a local judge)?
- Compute: which GPUs, and is 70B worth one run for the Assistant Axis comparison?
- Personas beyond pirate: Persona Vectors traits (evil, sycophancy, …) or Assistant Axis roles?
- Is `acts.npz` committed (38 MB) or kept out of git? A `.gitignore` here currently ignores `*.npz`.

## Key sources

- Jaeger 2014, Controlling RNNs by Conceptors, arXiv 1403.3369 (aperture, Boolean ops, autoconceptors §3.14)
- Postmus & Abreu 2024, Steering LLMs using Conceptors, arXiv 2410.16314
- Triantafyllopoulos et al. 2026, Conceptors for Semantic Steering, arXiv 2605.04980
- Razdan & Balamurugan 2026, When a Conceptor Complement Becomes Gain, arXiv 2610.16325 (matrix-removed control)
- Miao et al. 2026, COAST, arXiv 2605.17144
- Chen et al. 2025, Persona Vectors, arXiv 2507.21509 (github.com/safety-research/persona_vectors)
- Lu et al. 2026, The Assistant Axis, arXiv 2601.10387 (capping: 25th percentile, 8 layers Qwen3-32B, 16 Llama-3.3-70B; ~60% fewer jailbreak harms)
- Kang et al. 2026, Prompt–Activation Duality (GCAD), arXiv 2605.10664 (15-trait avg coherence drift −18.6 vs −1.9; Table 1)
- Xu, Xu & Wu 2026, PersonaManifold, arXiv 2609.34571
- Cheng, Baroni & Amo Alonso, LiSeCo, arXiv 2405.15454; MAGS arXiv 2605.21770; DSAS arXiv 2512.03661; PaSS arXiv 2610.04676; SEA arXiv 2405.09719; MiMiC arXiv 2402.09631
- Li et al. 2024, Instruction (In)Stability in LM Dialogs, arXiv 2402.10962

## Update (4 Nov, later): model choices and hardware

- **Hardware:** MacBook M1, 64 GB unified memory (GPU wired limit shown as ~43.5 GB by oMLX). Quit the oMLX server before big runs; it holds RAM.
- **Smoke-test model:** `HuggingFaceTB/SmolLM3-3B` (open data, base + instruct + `HuggingFaceTB/SmolLM3-3B-checkpoints`). SmolLM3 has a reasoning mode; turn it off (`enable_thinking=False` in the chat template, or `/no_think`) and verify.
- **References:** `meta-llama/Llama-3.2-3B-Instruct` (best published small-model role-play evidence: Persona-Pruner, arXiv 2606.14695), `HuggingFaceTB/SmolLM3-3B-Base` with third-person prompting (the base-model trick from "Tracing Persona Vectors Through LLM Pretraining", arXiv 2605.13329).
- **7–8B:** `allenai/Olmo-3-7B-Instruct` and its stages (`-Instruct-SFT`, `-Instruct-DPO`), base `allenai/Olmo-3-1025-7B` with pretraining checkpoints; `Qwen/Qwen2.5-7B-Instruct` for parity with Persona Vectors and GCAD. 7B fits on the Mac at fp16/fp32; it is slow, not impossible.
- **Existing Casting Call results on OLMo-3-7B** live elsewhere in this repo. Inventory them read-only before deciding reuse; only mix them into conceptor fits if pooling, layer indexing and prompts match.
- Why below 1.5B is a dead end for persona *play*: our Qwen-0.5B run, and no published evidence. Small models remain fine for read-only (teacher-forced) activation studies.
- **4 Nov 2026, task 1 done (OLMo-3 inventory):** `docs/prior_olmo3_inventory.md`. The repo holds no per-sample activations. The reusable piece is `persona_data/assistant-axis/olmo-3-7b-instruct/`: 330k answers (275 roles x 5 prompts x 240 questions, 1 sample) with 0-3 role-play labels, 54% fully in character (pirate 94%). Teacher-force those answers for conceptor fits. Casting Call vectors (hidden 17, pooled means, contrast against an empty system block) only serve as steering baselines. Open: the AA layer index and question identity, to check against the AA repo in task 2.
- **4 Nov 2026, data location check:** Google Drive and OneDrive hold no run data beyond this repo. The OneDrive `download-dump-september2026/personality-vectors-main` copy matches the checkout by path and size (only `role_datasets.zip` is extra). All data is already in plain git on `origin` (3.79 GiB pack, no LFS). Still missing: the CPU-phase `model_qa_responses/` and the AA generation code; the likely homes are the MATS cluster or the Modal volume `personality-vectors-data`. Recommendation in `docs/prior_olmo3_inventory.md`: a private HF bucket for new activation caches, plus an HF dataset repo for frozen runs.
- **4 Nov 2026, cleanup (approved):** deleted the OneDrive repo copy and zip (12.1 GB), the stray git temp pack (2.0 GiB), `christina-vectors/` (550 tracked duplicates of the AA vectors; `analysis/geometry/christina_analysis.py` now reads the AA folder and takes chemist's layer-16 row), the dead `.gitattributes` LFS rule and `persona_acts.orig.py`. Uncommitted on main. Found but not fixed: 18 `gold_standard_prompts` files overwritten by gold-label copies. The MATS cluster and Modal hold nothing for this project, so the missing Q/A files and AA settings go to Rehan via a draft message. Next: GPU access on PACE ICE.
- **4 Nov 2026, PACE ICE GPU access verified:** `ssh pace-ice-agent` (user `gmatlin3`) has the `coe-ice`, `coc-ice` and `coc-grade` QOS. The AI Makerspace route is `-q coe-ice --gres=gpu:H100:N` (or `H200:N`), with no `-A` and no `-p`. Cancelled 56 stale pending `frontier-*`/`routing-*` jobs from 27-28 Aug that filled the 50-job `coe-ice` submit limit. Smoke job 6817479 ran on an H100 80GB HBM3 (driver 615.71.09) on `coe-gpu`/`coe-ice`, about 3 minutes after submission. Blockers before real runs: scratch is at 999,420 of 1,000,000 files (a venv would fail), home is at 85%, and `conceptors/` must be on a pushed topic branch for `pace-code-release`. Per the global storage rule, data artifacts go to `/storage/ice-shared/cs7634/staff/TDA`.
- **4 Nov 2026, PACE storage layout:** the CS 7634 course volume (`/storage/ice-shared/cs7634`, 10 TB, 3.3 TB used; you own `public/` and `staff/`) is the data home. The conceptors folders are `/storage/ice-shared/cs7634/staff/conceptors/{hf_cache,runs,logs}`. Scratch holds only venvs, uv caches and code releases (`/storage/ice1/2/7/gmatlin3/releases/casting-call-vectors/<sha>`; home is too small for the roughly 13 GB checkout). Backups are HF buckets under `HCAI-Lab-GT`. A `pace-code-release` dry run for `325b03bf` passed; the real deploy waits for the task 2 code. Scratch cleanup (approved) is done: all 14 folders are deleted (3 needed `chmod u+w` on read-only snapshots), and scratch is at 109.7 of 300 GB and 29,000 of 1,000,000 files (2.9%). The Modal-archive bucket `archive-modal-personality-vectors-data` and `pvx-artifacts` hold the same layer-16 vectors as the repo, and no `model_qa_responses`.
- **4 Nov 2026, task 2 done (generalized `persona_acts.py`):**
  - **Config:** `--preset` (from `models.yaml`), `--model`, `--revision`, `--mode chat|base`, `--dtype`, `--trust_remote_code`, `--chat_template_kwargs`, `--repetition_penalty`.
  - **Sources:** Assistant Axis files through `--roles_dir`/`--questions_file` (pinned commit `a9896195`, read from a clone, not vendored, because the repo has no license), with `--roles`, `--n_prompts`, `--n_questions`, `--n_samples` and `--narrated`.
  - **Modes and outputs:** base mode (`persona_sources.base_prompt`, the 2605.13329 format; the wording is a placeholder marked TODO), `--teacher_forced` replay of AA `conversation` rows or our `texts.jsonl`, and `--save_tokens`/`--token_layers` per-token saving. Layer lookup is architecture-agnostic, and its path is recorded in the manifest.
  - **Defaults and logging:** `max_new_tokens` defaults to 256 (`--max_new_tokens 72` reproduces the pilot), `<think>` tags are counted in the manifest, and every rendered prompt is saved, since SmolLM3's template stamps today's date.
  - **Tests:** `conceptors/tests/test_persona_acts_tiny.py`, 7 tests on a random 4-layer Llama, about 6 s, no downloads.
  - **Settled checks:** our 240 questions equal AA's `extraction_questions.jsonl` in the same order, and AA hooks decoder-layer outputs, so its "layer 16" is hidden 17. transformers 4.57.3 already supports `smollm3` and `olmo3`.
  - **Base-mode wording (chosen with Glenn):** `Steve is <description>.\n\nIf someone asked Steve, "<q>", Steve would reply: "`. Answers are cut at the closing quote; the raw continuation is kept as `raw_answer`.
- **4 Nov 2026, task 3 done (`judge.py`):** Assistant Axis role-play rubric (0-3, mapped to fully/somewhat/not; `<role>_narrated` rows judged with `<role>`'s rubric; a generic rubric for roles without a release file) and the Persona Vectors coherence rubric (0-100, verbatim from GCAD App. C.5). Stdlib HTTP client for the `openai` backend (`gpt-4.1-mini-2025-04-14`, refuses without `--yes_spend`) or `local` (any OpenAI-compatible URL). Writes a resumable `judge.jsonl` plus `judge_manifest.json` with per-role fully-rate and mean coherence. Tests: `conceptors/tests/test_judge.py` (fake judge plus a stub HTTP server). Open: which judge backend to use. No `OPENAI_API_KEY` is set, and no local server was running.
- **4 Nov 2026, judge decision:** local judges on PACE (we don't need to match the papers' judge for this pilot). The judge model is `Qwen/Qwen3-30B-A3B-Instruct-2507` (61 GB, approved), served by vLLM inside `conceptors/pace/judge_job.sh` with a per-job API key and a socket-ownership readiness check (shared nodes). PACE launchers live in `conceptors/pace/`. Release `93106652` is deployed to `$HOME/scratch/releases/casting-call-vectors/` (12 GB per release, a full clone each time), and the venv at `$HOME/scratch/venvs/casting-call-vectors` has torch 2.8.0+cu128, transformers 4.57.3 and vllm 0.11.0.
- **4 Nov 2026, task 4 done (`analyze_acts.py`):**
  - **Layers:** layers come from `n_layers` as fractions (24 layers give 13/18/22 and maps 2-24, as before), and arrays are read through `saved_hidden_idx`, so thinned saves read the right layers.
  - **Judge filter:** reads `judge.jsonl` and fits persona-side directions and conceptors on fully-in-character rows only (the default role is not filtered), adding a `judge` section with per-role label counts and coherence.
  - **New `lowrank` section:** Ledoit-Wolf-shrunk rank-k conceptors (k = 4/8/16/32) on raw clouds and on per-question differences, with held-out keep shares and the interpolation form (beta 0.3).
  - **I/O and robustness:** `--run DIR` writes `DIR/derived.json`; sections that need roles not present are skipped, and zero-variance map layers are skipped.
  - **Verification:** on `acts_qwen05` every shared section matches a fresh run of the old script with zero difference, and the stored file within 7e-17 (`tests/compare_derived.py`). The old script did not byte-reproduce the stored file either; the only drift is the unrounded `alpha_inv2`.
  - **Pilot low-rank result:** a raw rank-4 conceptor keeps 9.3% of held-out pirate energy against 8.6% of assistant energy (no separation at 0.5B). The difference fit keeps 3.7% for pirate against 0.8-3.4% for other roles, with `pirate_narrated` at 3.4%.
- **4 Nov 2026, task 5 done (smoke runs, `runs/{smollm3_quick,llama32_quick,smollm3base_quick}` on CS 7634).** Each run is 20 AA questions x (5 pirate + 5 assistant prompts + narrated) x 2 samples, judged locally.
  - **Judge results:**
    - SmolLM3-3B: pirate fully 200/200 (coherence 78.2), assistant 94.8, `<think>` tags 0/440.
    - Llama-3.2-3B: 200/200 (coherence 81.2), assistant 94.4.
    - SmolLM3-3B-Base (Steve): pirate fully 144/200 (coherence 49.7), and answers are short (median 14 words).
  - **Activation results** (steering layer, answer mean): pirate-vs-assistant AUC 1.00, 1.00 and 0.88; role variance 11%, 8.7% and 2.2%; the narrated negative on the persona axis at 0.89, 1.12 and 0.44.
  - **Decision:** SmolLM3-3B holds the persona as well as Llama-3.2-3B, and its narrated negative sits below the persona (Llama's overshoots, as the 0.5B pilot's did).
  - **Caveats:** the AA rubric rates narration "fully in character" (39-40/40), because it only checks for an AI identity; base-model labels fit the rubric poorly.
- **4 Nov 2026, infra fixes:**
  - transformers 4.57 tokenizers call `model_info()` for repo ids even when offline, so models now load from their snapshot folder (`model_snapshot` is recorded in the manifest).
  - Concurrent vLLM compiles shared an NFS cache and hit a stale handle, so each job gets a private `mktemp` cache in `$TMPDIR`.
  - Login nodes kill long CPU work, so analyses run through `pace/analyze_job.sh`.
  - **Caveat:** in transformers 4.57, `output_hidden_states` records each decoder layer's output before forward hooks run. The steered layer's own entry therefore looks unsteered, while every downstream entry and the logits are steered. Phase-2 activations at the steered layer are pre-steering; read layers are fine.
- **4 Nov 2026, steering harness:** `steering.py` fits add / cap / conceptor / lowrank (k=16, Ledoit-Wolf) / meanshift (matrix-removed control) from a run's fully-in-character persona rows at module n//2. `steer_eval.py --stage sweep` is the setup test (no system prompt, a strength grid, held-out AA role questions); `--stage hold` runs 10-turn conversations with probes at turns 5 and 8, in `steer_only` and `prompt` contexts, at strengths matched from the sweep. Every answer gets an unsteered read-out at the 0.75-depth layer. `analyze_steer.py` writes `steer_derived.json` (sweep table, selected strengths, per-turn hold curves with T1/T5/T10 and drift). 13 tiny-model tests pass.
- **5 Nov 2026, explainer and bench:**
  - **Task 7 explainer:** `explainer/build.py --data` takes a run folder; `src/head.html` declares UTF-8 (served over HTTP it decoded as windows-1252); `explainer/tools/interact_inpage.js` runs interact.js's checks in-page (no Playwright on this Mac; the pilot page passes). The 0.5B prose list is in `docs/explainer_05b_passages.md`. The real-data rebuild waits for `runs/smollm3_full`, which has the pilot roles, logic and phase 2 the page reads.
  - **New bench page:** `explainer/build_bench.py --sweep RUN --hold RUN` builds `explainer/steering_bench.html` from `explainer/bench/bench.html`: methods at matched strength, the setup sweep (in character against coherence), and ten-turn hold curves with probes. The method palette is checked with the dataviz validator.
  - **Voice rubric:** `judge.py --voice` adds a point-of-view rubric (as / mixed / about the role), which `analyze_steer.py` and the bench page show.
- **5 Nov 2026, pilot sweep and harness fixes:** the judged pilot sweep (`runs/steer_sweep_quick`, LOGBOOK has the table) has add reaching 0.925 fully at coherence 90 by strength 1.0, while the conceptor does nothing up to beta 0.5, breaks the model at 0.8, and tracks its matrix-removed control throughout. Fixes in `steering.py`: `lowrank` keeps the Ledoit-Wolf floor outside its span, and new `conceptor_tok` / `meanshift_tok` fit on per-token persona activations (they need `--save_tokens` rows at hidden module+1; `smollm3_full` has them at 18, so use `--module 17`). `analyze_steer.select` runs a method that never reaches the target at its strongest coherent strength, marked `matched: false`.
- **5 Nov 2026, full SmolLM3 steering results:** `runs/steer_sweep_full` and `runs/steer_hold_full` (all 8 methods, fits from 3,600 judged pirate answers at module 17) are judged and analyzed, and `explainer/steering_bench.html` is built from them. Setup: only the steering vector reaches 80% fully while staying coherent (c 1.0: 0.875 at coherence 92); cap tops out at 0.775; both full-rank conceptors leave the persona score where it was (0.39 and 0.28 fully at beta 1.0); low-rank and mean-shift act only where coherence collapses. Hold: the prompted pirate falls to 0.05 at the first identity probe and 0.20 by turn 10, while prompt + per-token conceptor stays at 0.90 by turn 10 with coherence 90 (cap 0.95 at 88, steering vector 1.00 at 71). So the per-token conceptor holds a persona without inducing one; 20 conversations per cell, so the method-to-method probe gaps are within noise. `analyze_acts.py` no longer times out on 13k-row runs (release 5daa77b9). LOGBOOK has the tables.
- **5 Nov 2026, why the conceptors hold but don't steer:** `gap_closure.py` measures how much of an assistant token's gap to the persona center a fitted conceptor closes along the persona direction. At beta 1 the answer-average conceptor closes 2.6% and the per-token one 11%, because the persona direction is a high-variance direction of the persona cloud and the median aperture keeps it (vhat.C.vhat 0.97 and 0.89). The field guide now builds from `runs/smollm3_full` (`explainer/build.py --data <run> --samples 1`; interaction check passes), but its prose still quotes the 0.5B pilot: `docs/explainer_05b_passages.md` lists the passages, including four claims SmolLM3 contradicts. The bench is published privately at https://claude.ai/artifact/PQYFKuAqAjWKXSR8UqRYAh.
- **5 Nov 2026, field guide prose describes SmolLM3:** every passage in `docs/explainer_05b_passages.md` now quotes `runs/smollm3_full` (its `derived.json` and judge manifest), and the committed `explainer/persona_steering_field_guide.html` is built from that run with `--samples 1` (in-page check: no errors, no missing controls). Rewritten claims: Plate 1.4 (separation is perfect everywhere; the role's share is what varies), 1.5 (the top-2 view sorts by role, not by question), 1.3 (the Assistant Axis rubric passes 711 of 720 narrated answers; only the point-of-view rubric catches them), 1.7 (question share instead of a task split), 2.7 (the 3,600-answer pirate cloud is full rank; 720-answer roles are not), and 4.2 (SmolLM3 phase 2, plus the judged hold result and gap closure). Part 5 gained "use the conceptor to hold, not to set up". `real.js` now fills the map's answer count and the rank tile's model name from the data.
