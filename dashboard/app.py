#!/usr/bin/env python
"""
Personality Vectors Dashboard - FastAPI backend.

Serves experiment JSON data and provides API endpoints for
interactive visualization of personality steering results.

Run: python dashboard/app.py
Then open http://localhost:8050 in your browser.
"""

import json
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Personality Vectors Dashboard")

# ── paths ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO / "outputs" / "archive" / "2026-02-15_legacy" / "analysis"
NEW_ANALYSIS_DIR = REPO / "outputs" / "analysis"
CURRENT_RUN = REPO / "outputs" / "run_2026-02-15_marin_smollm"
VECTORS_DIR = REPO / "persona_data" / "model_inits"
PSYCHOMETRIC_DIR = REPO / "outputs" / "psychometric"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── mount static files ─────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── utility ────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    with open(path) as f:
        text = f.read()
    # Replace NaN/Infinity with null for JSON compliance
    text = text.replace(': NaN', ': null').replace(':NaN', ':null')
    text = text.replace(': Infinity', ': null').replace(':Infinity', ':null')
    text = text.replace(': -Infinity', ': null').replace(':-Infinity', ':null')
    return json.loads(text)


def _safe_json(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── routes ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/experiments")
async def list_experiments():
    """List all available experiment JSON files."""
    files = sorted(ANALYSIS_DIR.glob("*.json"))
    return {
        "count": len(files),
        "experiments": [
            {
                "name": f.stem,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "path": f"/api/experiment/{f.stem}",
            }
            for f in files
        ],
    }


@app.get("/api/experiment/{name}")
async def get_experiment(name: str):
    """Load a specific experiment result."""
    path = ANALYSIS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Experiment '{name}' not found")
    return _load_json(path)


@app.get("/api/current-run")
async def current_run():
    """List files in the current analysis run."""
    artifacts = sorted((CURRENT_RUN / "artifacts").glob("*.json")) if (CURRENT_RUN / "artifacts").exists() else []
    return {
        "artifacts": [
            {"name": f.stem, "path": f"/api/current-run/{f.stem}"}
            for f in artifacts
        ]
    }


@app.get("/api/current-run/{name}")
async def get_current_artifact(name: str):
    path = CURRENT_RUN / "artifacts" / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Artifact '{name}' not found")
    return _load_json(path)


@app.get("/api/vectors/models")
async def list_vector_models():
    """List all models with extracted persona vectors."""
    models = set()
    for trait_dir in VECTORS_DIR.glob("*_persona_initialization"):
        for sf in trait_dir.glob("*.safetensors"):
            models.add(sf.stem)
    return {"models": sorted(models)}


@app.get("/api/summary")
async def get_summary():
    """Get a structured summary of key findings for the overview panel."""
    summary = {
        "models_tested": 6,
        "experiments_run": len(list(ANALYSIS_DIR.glob("*.json"))),
        "traits": ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"],
        "headline_findings": [
            {"title": "5D Personality Subspace", "value": "Exactly 5D (6th SV = 0.000)", "icon": "dimension"},
            {"title": "Cross-Model Transfer", "value": "100% (Llama 1B to Marin 8B)", "icon": "transfer"},
            {"title": "Channel Capacity", "value": "21.8 bits (~3.6M directions)", "icon": "channel"},
            {"title": "Language Universality", "value": "100% across 12 languages", "icon": "language"},
            {"title": "Topic Robustness", "value": "100% across 10 domains", "icon": "topic"},
            {"title": "Firewall Neutralization", "value": "90.7%", "icon": "shield"},
            {"title": "Zero-Delay Switching", "value": "120/120 perfect", "icon": "switch"},
            {"title": "Psychometric Specificity", "value": "1.74x at alpha=3", "icon": "psych"},
        ],
        "negative_results": [
            {"title": "Text Transfer", "value": "22-28%", "note": "RLHF homogenizes output"},
            {"title": "Watermark", "value": "6.25% (chance)", "note": "Information doesn't survive generation"},
            {"title": "ICL Induction", "value": "17%", "note": "Few-shot barely works"},
        ],
    }
    return summary


@app.get("/api/5d-space")
async def get_5d_space():
    """Get the 5D personality space coordinates for visualization."""
    path = ANALYSIS_DIR / "5d_semantics.json"
    if not path.exists():
        raise HTTPException(404, "5D semantics data not found")
    data = _load_json(path)
    return data


@app.get("/api/phase-diagram")
async def get_phase_diagram():
    """Get alpha phase diagram data."""
    path = ANALYSIS_DIR / "alpha_phase_diagram.json"
    if not path.exists():
        raise HTTPException(404, "Phase diagram data not found")
    return _load_json(path)


@app.get("/api/transfer-matrix")
async def get_transfer_matrix():
    """Get cross-model transfer matrix."""
    path = ANALYSIS_DIR / "cross_dim_matrix.json"
    if not path.exists():
        raise HTTPException(404, "Transfer matrix not found")
    return _load_json(path)


@app.get("/api/cross-language")
async def get_cross_language():
    """Get cross-language personality data."""
    path = ANALYSIS_DIR / "cross_language_personality.json"
    if not path.exists():
        raise HTTPException(404, "Cross-language data not found")
    return _load_json(path)


@app.get("/api/dynamic-transitions")
async def get_dynamic_transitions():
    """Get dynamic personality transition data."""
    path = ANALYSIS_DIR / "dynamic_personality_transition.json"
    if not path.exists():
        raise HTTPException(404, "Dynamic transition data not found")
    return _load_json(path)


@app.get("/api/negative-composition")
async def get_negative_composition():
    """Get negative composition data."""
    path = ANALYSIS_DIR / "negative_composition.json"
    if not path.exists():
        raise HTTPException(404, "Negative composition data not found")
    return _load_json(path)


@app.get("/api/information-channel")
async def get_information_channel():
    """Get information channel capacity data."""
    path = ANALYSIS_DIR / "personality_information_channel.json"
    if not path.exists():
        raise HTTPException(404, "Information channel data not found")
    return _load_json(path)


@app.get("/api/layer-heatmap")
async def get_layer_heatmap():
    """Get the full layer injection × detection heatmap."""
    # Try new analysis dir first, then legacy
    for d in [NEW_ANALYSIS_DIR, ANALYSIS_DIR]:
        for pattern in d.glob("full_layer_heatmap_*.json"):
            return _load_json(pattern)
    raise HTTPException(404, "Layer heatmap data not found")


@app.get("/api/temp-alpha-sweep")
async def get_temp_alpha_sweep():
    """Get temperature × alpha 2D sweep data."""
    for d in [NEW_ANALYSIS_DIR, ANALYSIS_DIR]:
        for pattern in d.glob("temp_alpha_sweep_*.json"):
            return _load_json(pattern)
    raise HTTPException(404, "Temperature-alpha sweep data not found")


@app.get("/api/new-experiments")
async def list_new_experiments():
    """List experiments from the new analysis directory (today's runs)."""
    if not NEW_ANALYSIS_DIR.exists():
        return {"count": 0, "experiments": []}
    files = sorted(NEW_ANALYSIS_DIR.glob("*.json"))
    return {
        "count": len(files),
        "experiments": [
            {
                "name": f.stem,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "path": f"/api/new-experiment/{f.stem}",
            }
            for f in files
        ],
    }


@app.get("/api/new-experiment/{name}")
async def get_new_experiment(name: str):
    """Load a new experiment result."""
    path = NEW_ANALYSIS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"New experiment '{name}' not found")
    return _load_json(path)


@app.get("/api/psychometric")
async def list_psychometric():
    """List psychometric evaluation results."""
    if not PSYCHOMETRIC_DIR.exists():
        return {"count": 0, "results": []}
    files = sorted(PSYCHOMETRIC_DIR.glob("*.json"))
    return {
        "count": len(files),
        "results": [
            {"name": f.stem, "path": f"/api/psychometric/{f.stem}"}
            for f in files
        ],
    }


@app.get("/api/psychometric/{name}")
async def get_psychometric(name: str):
    """Load a specific psychometric result."""
    path = PSYCHOMETRIC_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"Psychometric result '{name}' not found")
    return _load_json(path)


@app.get("/api/long-gen-tracking")
async def get_long_gen_tracking():
    """Get long generation 5D tracking data."""
    path = NEW_ANALYSIS_DIR / "long_generation_5d_tracking.json"
    if not path.exists():
        raise HTTPException(404, "Long generation tracking data not found (experiment may still be running)")
    return _load_json(path)


@app.get("/api/topic-shift")
async def get_topic_shift():
    """Get topic shift robustness data."""
    path = NEW_ANALYSIS_DIR / "topic_shift_robustness.json"
    if not path.exists():
        raise HTTPException(404, "Topic shift data not found (experiment may still be running)")
    return _load_json(path)


@app.get("/api/hexaco")
async def get_hexaco():
    """Get HEXACO-100 psychometric data."""
    path = NEW_ANALYSIS_DIR / "hexaco_logprob.json"
    if not path.exists():
        raise HTTPException(404, "HEXACO data not found (experiment may still be running)")
    return _load_json(path)


@app.get("/api/dose-response")
async def get_dose_response():
    """Get psychometric dose-response data."""
    path = NEW_ANALYSIS_DIR / "psychometric_dose_response.json"
    if not path.exists():
        raise HTTPException(404, "Dose-response data not found (experiment may still be running)")
    return _load_json(path)


if __name__ == "__main__":
    print("\n  Personality Vectors Dashboard")
    print("  http://localhost:8050\n")
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="info")
