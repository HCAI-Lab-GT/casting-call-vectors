# CastVectors 2.0 — Conference Roadmap

Status: draft, 2026-09-17. Supersedes/operationalizes the "Roadmap Notes" section of
`CastVectors_2_0.pdf`. Delete-before-submission notes live in that PDF; this file is the
living execution plan and stays in the repo (update it as workstreams close out).

**Target**: main-track submission, early Jan deadline (ICML-style late-Jan cycle assumed
until venue is confirmed).

**Scope decisions locked in (2026-09-17)**:
- Cross-model set: **Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Gemma-2-9B-it** (3 models,
  plus OLMo-3-7B-Instruct as the existing primary model = 4 total).
- Layer sweep: **full sweep on all 4 models**, not just OLMo. This is a materially bigger
  compute commitment than the doc's cheaper "coarse OLMo + single best-layer elsewhere"
  option — see "Compute risk" below for the mitigation.
- Workstream E (downstream multi-agent simulation fidelity): **deferred to a follow-up
  paper**. Workstream G (multi-turn stability, new — see below) is this paper's
  lower-cost answer to the same reviewer concern ("does this hold up beyond single-turn
  Q&A").
- New Workstream G: **multi-turn persona-vector stability** (user-requested addition, not
  in the original doc).

**Compute risk and mitigation**: 3 new models × a 5-point layer grid × 275 roles × the
alpha grid is roughly 4-5x the extraction+eval workload the original doc scoped (which
assumed layer sweep stayed OLMo-only). Mitigation: **fold Workstream A and B into one
combined per-model extraction campaign** — for each of the 4 models, run
`RoleLayersPersonaModel`'s multi-layer/multi-sample-count checkpointing once across the
full layer grid, then run `GoldPromptExperiments` twice per model: once across the layer
grid at 1-2 alphas (answers B), and once across the full alpha grid at each model's
chosen best layer only (answers A). This avoids re-extracting activations twice for the
same (model, layer) pairs.

---

## Workstream 0 — Infra hardening (prerequisite; do first, parallelizable)

This must land before scaling to 4 models, because every fragility found in the
2026-09-17 codebase deep-dive (see repo memory / prior session) gets multiplied by the
number of models × layers × roles now in flight. Concrete tasks:

1. **Atomic writes**: switch `save_to_safetensors` (abstract_persona_model.py),
   `RoleLayersPersonaModel`'s per-checkpoint saves, and `roles_optimized_cpu.py`'s
   `save_qa_responses` to write-to-temp-then-rename, so a killed job can't leave a
   corrupt/truncated file that silently breaks a later load.
2. **GPU-phase idempotent resume**: `roles_optimized_gpu.py` currently has no
   skip-if-exists check and always redoes the full forward pass. Add the same
   "skip if target checkpoint already on disk" logic the CPU phase already has.
3. **Fix dead code**: `roles_optimized_gpu.py`'s "ran out of pairs" path unconditionally
   raises before reaching its own graceful-partial-checkpoint fallback — fix the control
   flow so a role with fewer valid pairs than requested degrades gracefully instead of
   crashing the whole job.
4. **Fix `role_layer_batch.sbatch`**: undefined `${ROLE_LIST[@]}` (should be `$ROLES`)
   and a missing line-continuation before `--layers`. This script is now load-bearing for
   Workstream B (full layer sweep, all models) — must work before B starts.
5. **Generalize layer selection across architectures**: models in the new set have
   different depths (OLMo ~32, Llama-3.1-8B 32, Qwen-2.5-7B 28, Gemma-2-9B 42 — confirm
   exact counts via `AutoConfig` at implementation time). Add a `layer_fraction` option
   (e.g. `layer_fraction=0.5`) resolved to an integer layer index per model at load time,
   so "middle layer" and the B grid's early/early-mid/mid/late-mid/late bins mean the same
   architectural thing across models instead of a hardcoded integer that only makes sense
   for one model's depth.
6. **Fix broken test suite**: `tests/unit/test_safetensors_io.py` and
   `tests/integration/test_persona_model_lifecycle.py` import `pvx.pvx_models...`, which
   doesn't exist (`pvx.abstraction.pvx_models...` is correct) — fix the import path, then
   wire `pytest tests/` into `.pre-commit-config.yaml` or a CI workflow so a vector-I/O
   regression can't ship silently while scaling up.
7. **Consolidate config**: mark `configs/models.yaml`/`runs.yaml`/`psychometric_runs.yaml`
   deprecated (only reachable from `analysis/old_psycho/`) so new contributors aren't
   misled; standardize on the `*_autobatch.sh` disk-diff pattern as *the* documented
   resumability mechanism for new ablation axes rather than ad hoc role-split JSON files.
8. **Concurrency safety**: add basic file locking (or per-job-unique temp paths + atomic
   rename) around shared CSV/safetensors outputs, since 4 models running in parallel on
   SLURM increases collision risk over the current single-model workflow.

Owner: TBD. Target: complete before Workstream A/B compute starts in earnest (can overlap
partially — dataset generation and role inventories don't need to wait on this).

---

## Workstream A — Cross-model generality

**Models**: Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Gemma-2-9B-it.

**Reuse note**: the 275-role `RoleDataset` JSONs (positive instructions + eval questions)
are model-agnostic content — they don't need regenerating per target model, only the
activation-extraction target model changes. This is a significant compute saving worth
keeping explicit in the paper's method section (same role inventory, same elicitation
battery literally reused, not just "same" in kind).

**Architecture check**: `AbstractPersonaModel._get_decoder_blocks` already handles
`model.model.layers` (LLaMA/Qwen/Mistral-style) — Gemma-2 uses the same attribute path, so
no new steering-hook code should be needed for any of the 3 new models. Confirm this
empirically for Gemma-2 first (it has interleaved local/global attention — verify the
forward-hook still lands on the right residual stream point) before trusting it silently.

**Deliverables**:
1. Per-model best-layer extraction (via combined A+B campaign, see "Compute risk" above)
   at the full existing alpha grid {1.0, 1.5, 2.0, 2.5}.
2. Report: does the CastVectors-vs-assistant-axis gap replicate per model?
3. Report: anti-controllable subset overlap across models (new analysis — Jaccard overlap
   of the "anti-controllable" role set between each model pair; extend
   `analysis/empirical/` with a new `cross_model_overlap/` script).
4. Report: does the middle-layer convention transfer, or does each model need its own
   best layer (this answer comes out of Workstream B's grid, reported here).

---

## Workstream B — Layer sweep (full, all 4 models)

**Grid**: 5 layer-fraction bins (early ≈0.15, early-mid ≈0.35, mid ≈0.5, late-mid ≈0.65,
late ≈0.8 of each model's depth) × 4 models × 275 roles, at 1-2 alphas (not the full
grid — keep this sweep cheap; e.g. alpha=2.0 only, matching the doc's framing that B is
about role-profile alignment and anti-controllable rate *per layer*, not a full
alpha-response curve per layer).

**Depends on**: Workstream 0 items 4 (sbatch fix) and 5 (layer-fraction generalization).

**Deliverables**: alignment-vs-layer and anti-controllable-rate-vs-layer figures, faceted
by model — does the layer-16-equivalent convention hold everywhere, or is it
architecture-dependent (ties into Workstream D's mechanism story if layer choice
correlates with anti-controllability).

---

## Workstream C — Extended coefficient grid + directional steering

**Mechanism note**: negative/sub-1.0 alpha requires **no new steering code** — steering
is `delta = alpha * response_persona_vector` in `AbstractPersonaModel._steering_delta`,
which already works for negative and fractional multipliers. This workstream is new
*experiment generation and analysis*, not new pipeline code.

**Grid**: add α ∈ {−1.0, −0.5, 0, 0.5} to the existing {1.0, 1.5, 2.0, 2.5} set.

**Scope**: full extended grid on OLMo (primary model, most mature infra/gold-standard
data). Given the compute already committed to full-layer-sweep on 4 models, treat
extending the full negative grid to the other 3 models as a stretch add-on, not baseline
scope — a single best-layer spot-check (one negative alpha point to confirm sign flip) on
each of the 3 new models is enough to support a cross-model directional-control claim
without redoing the full curve everywhere.

**Directional steering control**: steer with `-v_r` per role, verify `RoleJudge` score
drops toward/below baseline. New analysis: extend `score_trajectory_comparison` to the
negative range and add a "sign-consistency" check (does the curve monotonically move away
from in-character as alpha goes negative).

**Deliverables**: full α-response curve including negative range (does score peak and
decline as the doc suspects for the 38 anti-controllable roles); directional-control
sanity check result.

---

## Workstream D — Mechanism for anti-controllability

Three sub-experiments, in the doc's own cost order:

1. **Finer-grained α on the 38 anti-controllable roles**: denser sampling (e.g. every
   0.25 between −1.0 and 4.0) restricted to that role subset, using C's infra — locates
   the turnover point precisely instead of inferring it from 4-8 coarse points.
2. **Norm-matched assistant-axis control** (highest priority — direct rebuttal to the
   workshop paper's own scale-mismatch caveat): rescale `v_assistant` per role so
   `alpha_AA * ||v_AA_role|| ≈ 3.79` (CastVectors' mean ℓ2 norm), rerun aggregate and
   per-role comparisons at that effective alpha. Needs a small extension to
   `GoldPromptExperiments`: currently it takes one global `alphas` list applied uniformly;
   add a per-role-computed-alpha mode for the AA arm specifically (existing
   `plot_effective_norm.py` already computes the norm ratio needed — reuse that logic to
   derive the per-role target alpha rather than just re-plotting existing data, since this
   needs new generations, not a re-plot).
3. **Projection/ablation check**: reuse `role_vector_decomposition`'s v∥/v⊥ split (v∥ =
   component along the assistant axis, v⊥ = role-specific residual). Build v⊥-only vectors
   for the anti-controllable subset and steer with them via `SteerPersonaModel` (already
   supports loading an arbitrary precomputed vector) as a new condition — does removing
   v∥ change the decline pattern? Bears directly on the target-direction-mismatch
   hypothesis.

**Depends on**: Workstream C's extended grid (for #1) and the existing
`role_vector_decomposition` analysis (for #3).

---

## Workstream E — Downstream simulation fidelity (DEFERRED)

Out of scope for this submission per 2026-09-17 decision. Note explicitly in the paper's
Discussion/Limitations as future work, alongside the salvaged psychometric-battery
future-work seed (`tom_scoring`/`analysis/old_psycho`, commit `87c14ef5`) as a related
follow-up direction (richer behavioral readouts for a future downstream-sim paper).

---

## Workstream F — Human judge validation

Small blind human-eval subset (5-10 roles × a handful of questions) rated by 1-2+
annotators, compared against GPT-4.1-mini judge scores (Spearman correlation / Cohen's
kappa). Cheap, no GPU compute, doesn't block on A-D — **schedule early and in parallel**,
starting as soon as annotators are available, using OLMo gold-standard/steered responses
that already exist today. Needs: a lightweight blind-rating tool (a lightweight rating
form, e.g. as a shared spreadsheet or a minimal internal artifact) — not a pipeline
dependency.

---

## Workstream G — Multi-turn persona-vector stability (new)

**Motivation**: the current elicitation battery is single-turn Q&A. A main-track reviewer
asking about "downstream simulation fidelity" is really asking whether steering holds up
in extended interaction — this workstream answers a scoped version of that question
without the cost of a full multi-agent simulation harness (Workstream E).

**Open mechanistic question worth stating explicitly in the paper**: the steering hook
adds `delta` to a decoder block's output at the *last* token position on each forward
pass. In multi-turn generation with KV-cache reuse across turns, each turn's steered
hidden states become part of the cached K/V that later turns attend over — so steering
effects may compound turn-over-turn rather than simply repeating identically each turn.
This is worth measuring directly, not assuming.

**Design**:
- Extend the generation harness to run N-turn conversations (recommend N=4-5) per role,
  under a few conditions: (a) reapply steering identically at every turn, (b) apply only
  at turn 1 and let the KV cache carry it forward, (c) vary alpha across turns.
- Measure `RoleJudge` score per turn — does character consistency hold, decay, or drift
  upward (compounding)?
- Include a mix of highly-controllable and anti-controllable roles (~30-40 roles) to see
  whether multi-turn changes controllability classification, not just consistency.

**Scope**: OLMo only (primary model), 1-2 alphas (e.g. 1.5, 2.0) from the existing grid —
keep tractable given the compute already committed elsewhere. Cross-model multi-turn is
explicitly out of scope for this cycle; note as future work alongside Workstream E if it
turns out to matter.

**New code needed**: a multi-turn conversation harness (new sibling class to
`GoldPromptExperiments`, or an extension of it), reusing the existing per-turn generation
and judging primitives.

---

## Timeline (today: 2026-09-17; deadline: early Jan)

| Weeks | Dates (approx) | Work |
|---|---|---|
| 1-2 | through ~Oct 1 | Workstream 0 hardening (critical path — can partially overlap with dataset/infra prep for A). |
| 2-5 | through ~Oct 22 | Combined A+B extraction campaign: per-model layer-grid extraction, then B's layer-sweep eval (1-2 alphas) and A's full-alpha-grid eval at each model's best layer. This is the compute-heaviest block. |
| 5-8 | through ~Nov 12 | Workstream C: full extended grid (incl. negative α, directional control) on OLMo; spot-check negative α on the 3 new models. |
| 6-9 | through ~Nov 19 (overlaps C) | Workstream G (multi-turn, OLMo only) — independent of A/B's cross-model compute, can run in parallel if a second person/lead is available. |
| 8-10 | through ~Dec 3 | Workstream D: norm-matched AA control first (highest priority), then finer-α on anti-controllable subset, then projection/ablation. |
| 2+ (parallel, low-cost) | ongoing | Workstream F human-eval subset — start as soon as annotators are available, doesn't block on GPU work. |
| 10-12 | through ~Dec 17 | Consolidate all tables/figures across A-D+G; extend geometry appendix (RSA, decomposition) to new models where relevant. |
| 12-13 | through ~Dec 24 | Full draft written into paper skeleton. |
| 13-14 | through ~Dec 31 | Internal co-author review pass. |
| 14-15 | through early Jan | Revisions, appendix finalization, submission prep, buffer for compute reruns. |

**Risk flag**: this timeline is tighter than the original doc's because scope grew (3
models instead of 2, full layer sweep on all 4 instead of OLMo-only, plus new Workstream
G). If Weeks 2-5's combined A+B campaign slips, the first things to cut are (in order):
full negative-alpha grid on the 3 non-OLMo models (C), then multi-turn cross-model
scope (already excluded), then reduce Workstream B's role subset per model from all 275
to a stratified sample for the non-primary models.

---

## Open items still needing an owner

- Who leads each workstream (0, A, B, C, D, F, G) and the Overleaf/repo consolidation —
  people-planning, not something to resolve in this file's content, but tracked here so
  it isn't lost.
- Confirm exact per-model layer counts (OLMo, Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B) via
  `AutoConfig.from_pretrained(...).num_hidden_layers` before finalizing the layer-fraction
  grid in Workstream B.
- Confirm venue/track and swap the paper skeleton's style file accordingly (per the
  original doc's note).
