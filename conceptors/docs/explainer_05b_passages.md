# Passages in `explainer/src/body.html` that quote the Qwen2.5-0.5B pilot

Listed 4 Nov 2026 for the prose pass (task 7). Line numbers refer to `explainer/src/body.html` at commit `a415c3d0`.
Model-specific layer numbers (13, 12, 18, 10-14, 24) follow the 24-layer model; `derived.json` now carries
`meta.steer_hidden`, `meta.read_hidden` and `meta.map_layers`, so the prose should quote those values.

| Line | Passage | What it quotes |
|---|---|---|
| 15 | eyebrow | "real Qwen2.5-0.5B activations" |
| 56 | Part 1 intro | Qwen2.5-0.5B-Instruct, 24 layers, 896 dimensions, laptop run, 9 roles plus the hard negative |
| 72 | recipe card | 24 questions across 10 task types |
| 75 | recipe card | one answer each, temperature 0.7, up to 72 new tokens |
| 94 | Plate 1.2 text | the 0.5B "pirate" answers like the assistant; "Seafarer." |
| 96 | Plate 1.2 try-it | question picks from the 24-question set |
| 104 | Plate 1.2 caption | "Real answers from Qwen2.5-0.5B-Instruct" |
| 111 | Plate 1.3 | keyword filter turns the direction by about 36° (cosine 0.81) |
| 181 | Plate 1.4 note | "Layer 13 hidden state, which is the output of layer 12" |
| 191 | Plate 1.7 | task type accounts for about half the pirate cloud's spread |
| 139 | sweep caption | layer 24 is after the final norm; three outlier dimensions carry 60-70% of raw energy |
| 164-165 | map plate | "384 answers" (aria label and caption) |
| 216, 219 | toy plates | "896 numbers", "896 x 896 table", "in 896 dimensions" |
| 349-350 | rank trap | 72 pirate answers span at most 71 directions; 825 directions with keep 0 |
| 363, 365 | cover chart | share inside span 13% at n=2 to 83% at n=71; "fitted on all 72 pirate answers" |
| 419 | Plate 2.10 | raw-cloud conceptors at least 0.93 similar |
| 536-538 | Plate 4.2 | "A 12-turn pilot on Qwen2.5-0.5B"; add at layer 12, cap across layers 10-14, conceptor beta 0.3, read at layer 18 |
| 643 | footer | run provenance: Apple MPS, float32, seed 7, 384 single-turn and 144 multi-turn answers |

Numbers computed live from `derived.json` (charts, tables, tooltips) update on rebuild; only the prose above needs
editing by hand. Plate 4.3 (GCAD, lines 566, 581) quotes the published paper, not our run, and stays.
