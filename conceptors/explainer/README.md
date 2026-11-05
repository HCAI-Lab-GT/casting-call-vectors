# Persona Steering Field Guide

An interactive explainer for persona and trait steering: how persona data is collected, what conceptors do, twelve steering methods on one toy space, and turn-by-turn dynamics. Toy figures are computed in the browser; real-data figures come from the 4 Nov 2026 run of `persona_acts.py` on Qwen2.5-0.5B-Instruct.

Open `persona_steering_field_guide.html` in a browser. It is one self-contained file (fonts load from Google Fonts when online).

## Files

| File | What it is |
|---|---|
| `../persona_acts.py` | Collects activations: 384 single-turn answers (9 roles plus a hard negative) and 3 twelve-turn conversations under 4 steering conditions. |
| `../analyze_acts.py` | Turns that run into `derived.json`: layer sweeps, 2-D maps, hard-negative scores, the rank test, the conceptor logic test, multi-turn persona scores. |
| `../acts_qwen05/derived.json` | Output of `analyze_acts.py` for the 4 Nov run. |
| `src/head.html`, `src/body.html` | CSS and page text. |
| `src/lib.js` | 2×2 math, seeded data, canvas drawing, the line-chart helper. |
| `src/toy.js` | Toy plates in Part 2, the method zoo (Part 3) and the toy turn dynamics (Plate 4.1). |
| `src/real.js` | Real-data plates: Part 1, Plates 2.7 and 2.10, Plate 4.2, and the published GCAD numbers (Plate 4.3). |
| `build.py` | Inlines `derived.json` and the scripts into the single HTML file. |

## Rebuild

From the `conceptors` folder:

```
python analyze_acts.py --acts acts_qwen05/acts.npz --texts acts_qwen05/texts.jsonl \
    --manifest acts_qwen05/manifest.json --out acts_qwen05/derived.json
python explainer/build.py
```

`analyze_acts.py` needs numpy and scikit-learn; `build.py` needs only the standard library.

## Another model

`python persona_acts.py --model <hf id> --out acts_<name>` collects the same data for another model. Two things to change before analyzing it:

- The layer constants at the top of `analyze_acts.py` (`MAP_LAYERS`, `STEER_HIDDEN`, `READ_HIDDEN`, and the logic-test layers `[13, 18, 22]`) are set for the 24-layer 0.5B model.
- The page text quotes numbers from this run, so the prose needs a pass after the figures update.

## Also here

- `v1/conceptors_in_2d.html`: the first, 2-D-only conceptor page (28 Oct), with two screenshots.
- `screenshots/`: render checks of the current page (light, dark, phone width).
- `tools/`: Playwright scripts. `node tools/check.js light 1280 map-plate rank` screenshots plates by id; `node tools/zoo.js light 1280 cap gcad` screenshots zoo methods; `node tools/interact.js` clicks every control and reports page errors.
