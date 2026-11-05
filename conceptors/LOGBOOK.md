# LOGBOOK — conceptors

Append-only. Correct mistakes with a new entry; never edit old ones. HANDOFF.md holds the plan and the dated
per-task updates; this file holds goals, authority and the running work record.

[2026-11-04 23:01] GOAL conceptors-steering-explorer

Context: Glenn asked to "set a /goal and keep working on doing the analysis and building the visualizer and
working on testing and using our different steering approaches so we can visualize, measure, explore". Tasks
1-5 of the 4 Nov task list are done (see HANDOFF.md Update section). GPU (PACE ICE, coe-ice H100) and storage
(CS 7634 volume) are working; the local judge (Qwen3-30B-A3B-Instruct-2507 via vLLM) works.

```
GOAL conceptors-steering-explorer — continue until done or blocked on an owner decision
Objective: measure and visualize whether conceptors set up and hold a persona better than steering vectors
  and activation caps on SmolLM3-3B, with an explorable page built from real run data.
Authority: as in the AUTHORITY block below.
Decisions consumed:
  - task 6 full SmolLM3-3B run: approved by Glenn's "keep working" direction (2026-11-04 23:00)
  - judge: local Qwen3-30B-A3B-Instruct-2507 on PACE, no API spend (2026-11-04)
  - base-mode prompt: "Steve ... would reply: \"" (2026-11-04)
  - additive measurements (extra judge rubrics, extra metrics) are allowed; existing labels are never changed
Morning-list triggers: downloads over ~10 GB not yet approved (Qwen2.5-7B, OLMo-3-7B SFT/DPO stages);
  deleting anything this goal did not create; new personas or traits beyond pirate; protected-branch work;
  API spend; changes to existing rubric labels or the hidden-index convention.
Done means:
  1. runs/smollm3_full collected, judged and analyzed (derived.json) on a deployed exact-SHA release
  2. a steering evaluation harness (setup and hold, matched strengths, multi-turn) tested offline and run on
     SmolLM3-3B for steering vector, activation cap and conceptor variants, judged and analyzed
  3. explainer/build.py --data builds from a real run and node explainer/tools/interact.js reports no errors;
     the 0.5B-quoting passages in explainer/src/body.html are listed; real-data results are explorable
All gates from earlier goals are void unless restated here.
```

```
AUTHORITY 2026-11-04T23:01-05:00 goal=conceptors-steering-explorer
repo_publish:    commit and push allowed on topic branch conceptors/scale-up; main/protected: never
slurm_submit:    allowed on PACE ICE: GPU jobs -q coe-ice --gres=gpu:H100:1 (ICE caps 16 GPU-h/job, 16 h wall),
                 CPU jobs for fetch/analysis; chains use afterok + --kill-on-invalid-dep=yes; stay under the
                 50-job coe-ice submit limit
deploy:          exact-SHA releases via ~/dotfiles/scripts/pace-code-release only (12 GB each; batch changes)
downloads:       models under ~10 GB allowed; Qwen3-30B judge already approved and fetched
delete/cleanup:  only files this goal creates under runs/ and logs/ (e.g. failed partial runs); superseded
                 releases and anything else -> morning list
decisions_consumed: task-6 SmolLM3 full run; local judge; Steve base prompt; additive metrics allowed
morning_list_only:  >10 GB downloads, new personas/traits, deleting non-goal files, protected-branch ops,
                    API spend, rubric-label changes
```

Action: wrote this goal and ledger; next steps are the smoke-run analysis readout, task 7 (explainer build from
run data), the full SmolLM3-3B run, and the steering harness.

Result: in progress.

## For the morning

- (empty)

[2026-11-04 23:40] Smoke-run decision, transformers 4.57 caveat, steering harness

Context: finish task 5's decision point and start the steering comparison.

Action: analyzed the three smoke runs (CPU job 6819658); wrote steering.py, steer_eval.py and analyze_steer.py with
tiny-model tests; added --extra_roles and spread --save_tokens for the full run.

Result: SmolLM3-3B is good enough (judge ceiling like Llama-3.2-3B; AUC 1.00; narrated negative at 0.89, below the
persona). transformers 4.57 records hidden_states before forward hooks (the steered layer's own entry looks
unsteered; downstream entries and logits are steered); the test now checks downstream and logits. The judge rubric
cannot tell narration from role play (morning-list candidate: an additive "speaking as vs about" rubric).

[2026-11-05 00:30] Explainer fixes, voice rubric, slow deploys

Context: task 7 groundwork while waiting on PACE deploys.

Action: build.py takes a run folder; head.html declares UTF-8 (served over HTTP the page decoded as windows-1252);
tools/interact_inpage.js runs interact.js's checks in-page (no Playwright here; verified on the pilot page in the
browser pane: no errors, all controls present, 6 tables filled); docs/explainer_05b_passages.md lists the 0.5B prose.
Added judge.py --voice (additive point-of-view rubric, allowed by the ledger: existing labels unchanged).

Result: commits 04f74ae1 and 335e3b88. PACE connectivity was flaky: the 5687ed81 deploy died mid-fetch (partial
staging folder left at releases/casting-call-vectors/.staging/5687ed81...), the 04f74ae1 deploy was stopped in favor of
335e3b88. Scratch holds 6 full releases (about 12 GB each, 185.6 GB of 300).

## For the morning (appended 2026-11-05 00:30)

- Prune superseded releases under `$HOME/scratch/releases/casting-call-vectors/` (all but the active one) and the
  partial `.staging/` folders; each release is a 12 GB full clone. Not covered by the ledger's delete contract.
- Consider a lighter deploy path for conceptors (sparse checkout or a split repo): every release clones the repo's
  ~4 GB of tracked Casting Call data, which is most of the 30-50 minute deploy time tonight.

[2026-11-05 01:05] Steering bench page; unsigned commit

Context: build the visualizer for steering results while the 335e3b88 deploy runs.

Action: wrote explainer/bench/bench.html and build_bench.py; validated the method palette (dataviz validator, light
and dark both pass after darkening amber for dark mode); checked the page in the browser pane on tiny-model demo
data (no console errors; one pass of fixes: label only the matched strength, trim local model paths).

Result: commit 8217e1fc is unsigned: the 1Password signer returned an agent error (likely locked while Glenn is away);
per the unattended-signing rule only that commit skipped signing.

[2026-11-05 00:10] Pilot sweep diagnosis; token-level conceptor; selection rule

Clock correction: the entries stamped 00:30 and 01:05 above ran ahead of the local clock (EST); from here stamps come
from `date`.

Context: the judged pilot sweep (runs/steer_sweep_quick, fits from smollm3_quick at module 18) shows the conceptor
family doing nothing up to beta 0.5 and breaking coherence at 0.8, with the conceptor tracking its own matrix-removed
control (meanshift) at every strength. add 1.0 reaches 0.925 fully at coherence 90; cap tops out at 0.675.

Action: two causes and their fixes. (1) Rank trap: 200 answer averages in d=2048 give a rank-200 conceptor, and the
lowrank hook zeroed every direction outside its 16-dim span; the hook now keeps the Ledoit-Wolf floor null_keep
there. (2) Reference cloud: the conceptor is fitted on answer averages but applied to every token, so it pulls each
token toward an averaged center; new methods conceptor_tok and meanshift_tok fit on per-token persona activations
(full run saves them at hidden 18 and 27, so the full-run sweep uses --module 17). The default grid now probes
beta 0.3-1.0 and cap percentiles 25-99. analyze_steer.select: a method that never reaches the target now runs the
hold test at its strongest strength with coherence >= 60 (marked matched False) instead of its best fully rate at
coherence >= 50, which picked beta 0.1 on a three-way tie. Excluding such methods would drop the conceptors from
the prompt-context hold test, which is the comparison they are meant to win.

Result: 14 tiny-model tests and 5 judge tests pass; ruff could not run (the shell call was refused).

[2026-11-05 00:20] Release c45df6af; full sweep and hold chain submitted

Context: commit c45df6af (signer still locked: unsigned) deployed and verified at
releases/casting-call-vectors/c45df6af...; bench commit d2bb51cc (unsigned, local page only, not deployed).

Action: second pilot sweep (runs/steer_sweep_quick2, widened grid, pre-fix code) analyzed locally: add reproduces the
first sweep exactly (c=1.0: 0.925 fully, coherence 90.2); conceptor, lowrank and meanshift all keep coherence ~94 at
beta 0.5 and collapse by 0.7 (coherence 38 / 14 / 12), so the answer-average conceptor adds nothing over its control.
Submitted from the c45df6af release, all afterok with --kill-on-invalid-dep=yes, behind judge-smollm3_full 6820305:
analyze-smollm3_full 6821021 (CPU); steer-sweep-full 6821023 (fits from smollm3_full at module 17, all 8 methods,
2 samples x 40 role questions) -> judge 6821024 (--voice) -> analyze_steer 6821025 (sbatch --wrap, CPU) ->
steer-hold-full 6821027 (--select_from steer_sweep_full) -> judge 6821028 (--voice) -> analyze_steer 6821029.

Result: chain queued; collect-smollm3_full 6820304 was writing phase-1 activations at 34 min (12,960 answers,
147,272 per-token rows).

[2026-11-05 00:24] Conceptor persona-steering handoff: OLMo-3 inventory, data-location search, and PACE ICE access (auto-generated, unverified)

Context: Picking up a conceptors research project from a cloud session. The question is whether conceptors (soft per-direction stencils fitted to activation clouds) set up and hold a persona better than steering vectors or activation caps, especially over many turns. A 0.5B pilot was too small to role-play and used thin data, so the plan is to scale to models that can play personas. The first task was a read-only inventory of existing OLMo-3-7B results in the casting-call-vectors repo. After that came a search for where the run data lives, then repo cleanup and moving to GPU access on PACE ICE.

Action:
- Read HANDOFF.md, CLAUDE.md and the explainer README, then inventoried the repo's OLMo-3-7B artifacts. They are the Assistant Axis responses and scores, the Assistant Axis, Casting Call role and trait vectors, and the gold-prompt CSVs. Wrote `conceptors/docs/prior_olmo3_inventory.md` rather than a repo-root `docs/` file, so nothing outside the project folder changed. Resolved an open question on OLMo-3's chat template by reading it from the Hub without downloading the model. Added a dated line to HANDOFF.md.
- When asked to look for the data on Google Drive, searched Drive (mostly notes and papers, no run data). Then checked OneDrive under `~/Library/CloudStorage`. I diffed the `personality-vectors-main` dump against the checkout by path and file size only, and listed the 2.6 GB zip without pulling it. I added the findings and a storage recommendation to the inventory doc.
- On the user's direction (no mats cluster or Modal data; draft a message to Rehan via #messages; unify and de-duplicate), looked for the "messages" peer session. Only a "Zotero papers" session was visible. Applied four approved cleanup edits, including repointing the christina analysis to the Assistant Axis vectors, and checked that the loader returns 276 vectors (275 roles plus default) of shape (4096,). `ruff` wasn't in the venv, so I tried the one on PATH.
- Started on PACE ICE: loaded the pace-ice skill, checked read-only that the `pace-ice-agent` ssh alias connects (on `login-ice-gnr-1` as `gmatlin3`), and reran its GPU, queue and storage probes with `grep` because PACE has no `rg`.

Result:
- Task 1 is done. The repo has no per-sample OLMo-3-7B activations, so nothing can feed a conceptor fit directly. It does have about 330,000 judged OLMo-3-7B-Instruct answers that can be teacher-forced through the model to rebuild activation clouds. The vectors are usable as steering baselines only.
- There is nothing new to move: the OneDrive copy is an older snapshot of data already tracked in plain git on `origin` (no LFS in use), and Google Drive holds no run data. Two pieces are still missing from both drives: the judged Q/A answers from the CPU phase, and the code that generated the Assistant Axis data. Moving existing data to LFS would mean rewriting shared history, so the recommendation was to leave it. For new activation dumps, the recommendation was a private Hugging Face bucket.
- The ssh connection to PACE ICE works. The GPU and storage results, any job submission, and the draft message to Rehan do not appear in the truncated transcript, so I can't confirm any of them. The reported cleanup edits and the ruff check are also unconfirmed beyond the loader result.
- Git ground truth shows only one staged change, the deletion of `overleaf`, which isn't clearly tied to this session. Task 2 (generalizing persona_acts.py) hasn't started. The transformers check for smollm3 and olmo3 support isn't visible in the transcript. The 12 GB OneDrive dump was left untouched.

[2026-11-05 00:52] Conceptor persona-steering inventory and PACE ICE setup (auto-generated, unverified)

Context: Picking up the conceptors research project from a cloud-session handoff. The goal is to test whether conceptors set up and hold a persona better than steering vectors or activation caps. The 0.5B pilot was too small, so the plan is to scale to models that can role-play (OLMo-3-7B among them). Task 1 was a read-only inventory of existing OLMo-3-7B artifacts in the casting-call-vectors repo. After that came a search for where the missing data lives, then repo cleanup, then getting GPU access through PACE ICE.

Action: Read HANDOFF.md, CLAUDE.md and the explainer README, then wrote `conceptors/docs/prior_olmo3_inventory.md` and added a dated line to HANDOFF.md. The doc went under `conceptors/` rather than the repo-root `docs/` so nothing outside the project folder changed. I resolved an open check on OLMo-3's chat template by reading it from the Hub, with no model download. When asked to look for the data on Google Drive, I searched Drive, then OneDrive under `~/Library/CloudStorage`, and checked whether the repo itself points to a Drive or Hub location. I diffed the OneDrive `personality-vectors-main/` folder against the checkout by path and file size without opening files, and added the findings and a storage recommendation to the inventory doc. After the user said the MATS cluster and Modal volumes hold nothing relevant, I applied four approved cleanup and unification edits, including repointing the christina analysis to the Assistant Axis vectors. I checked that the loader returns 276 vectors (275 roles plus default), all shaped (4096,), with chemist included. `ruff` isn't in the venv, so I tried the one on PATH. Then I loaded the pace-ice skill and tested the `pace-ice-agent` ssh alias read-only. I was in the middle of rerunning checks with `grep` (PACE has no `rg`) and adding queue state and storage checks.

Result: Task 1 is done: the repo saved no per-sample activations for OLMo-3-7B, so nothing can go straight into a conceptor fit. It does hold about 330,000 OLMo-3-7B-Instruct answers with 0-3 role-play labels, which can be teacher-forced to get activation clouds. Role-play rates run from 5% to 96% (median 54%), so OLMo-3-7B plays characters the 0.5B model couldn't. The vectors at hidden 17 work as steering baselines but not as conceptor fits. The OneDrive dump is an older snapshot of data already in this repo. The only file missing from the repo is `persona_data/role_datasets.zip`. Google Drive has no run data. The data is already tracked in plain git on `origin`, with no LFS in use, so I recommended leaving it in place and using a private Hugging Face bucket for new activation dumps. The judged Q/A answers from the CPU phase and the code that generated the Assistant Axis data weren't found anywhere. The ssh connection works (login-ice-gnr-1 as `gmatlin3`). GPU and storage availability on ICE hadn't been determined when the transcript ends. The cleanup edits and the draft message to Rehan for the casting-call Discord channel aren't clearly shown in the truncated transcript, so I can't confirm the message was drafted or sent. Tasks 2 onward (generalizing persona_acts.py, the smollm3/olmo3 transformers check, float16 retest on MPS) hadn't started. The git ground truth shows only a staged deletion of `overleaf`, which is unrelated to this session's work.
