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
