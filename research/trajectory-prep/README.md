# Trajectory prep (absorbed from persona_trajectory)

Provenance: github.com/glennmatlin/persona_trajectory @ b2fbd09 (archived
2026-08-14; full history and bulk artifacts remain there). Created June 2026
as an agent prep pack for reproducing arXiv:2605.13329 (EPFL, persona
vectors across pretraining checkpoints); absorbed here because the
conference project (persona emergence over training + data attribution) is
the successor to that effort.

- `cleanroom/` — functional spec + minimal pseudocode for a non-copy
  reimplementation of the EPFL extraction (base-checkpoint prompting).
- `references/` — source map, OLMo checkpoint grid, model/repo notes.
- `run-20260609/` — the measured overnight reproduction + ICE A100
  validation: GPU-cost receipts (calibration data for scoping larger
  compute asks), environment report, judged-output metadata, and the
  exact commands run.
