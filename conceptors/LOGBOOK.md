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
