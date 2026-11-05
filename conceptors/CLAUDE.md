# conceptors/

Persona-steering experiments with conceptors, plus the interactive explainer.

- Read `HANDOFF.md` first: goal, what exists, findings, diagnosis, the ordered plan, open decisions.
- Collect → analyze → rebuild: `persona_acts.py` → `analyze_acts.py` → `explainer/build.py`. Commands are in HANDOFF.md.
- On PACE, everything (runs, logs, models, releases, venv, uv cache) lives on the CS 7634 volume under `/storage/ice-shared/cs7634/staff/conceptors`, never scratch or home. The table and the deploy command are in the root `AGENTS.md`; paths come from `pace/env.sh` only.
- Hidden index i = output of decoder layer i−1 (0 = embeddings).
- Test the collection path offline: `python conceptors/tests/test_persona_acts_tiny.py` (random 4-layer model, ~6 s). Model presets live in `models.yaml`.
- Edit the page in `explainer/src/`, never the built HTML; rebuild, then run `node explainer/tools/interact.js` (expects "no errors").
- Real numbers quoted in `explainer/src/body.html` come from the Qwen2.5-0.5B run; update the prose when the data changes.
- Don't commit `*.npz` activation dumps (ignored here).
