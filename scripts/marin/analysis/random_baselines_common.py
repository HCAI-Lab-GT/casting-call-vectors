from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]

MODEL_SAFE = "meta-llama__Llama-3.2-1B-Instruct"
DIMENSIONS = 2048

TRAITS = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]

N_PCS = 10
N_TRIALS = 100_000
SEED = 0
BATCH_SIZE = 1024

PCA_PATH = ROOT / "data/assistant_axis/pca" / f"{MODEL_SAFE}_pca.pt"
COMPARISON_NPZ_PATH = ROOT / "outputs/analysis" / f"{MODEL_SAFE}_comparison_arrays.npz"
VECTORS_DIR = ROOT / "persona_data/model_inits"
OUTPUT_PATH = ROOT / "outputs/analysis/random_baselines.json"

HEX_ADJ = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], dtype=np.int64)
HEX_ALT = np.array([(0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1)], dtype=np.int64)
HEX_OPP = np.array([(0, 3), (1, 4), (2, 5)], dtype=np.int64)


def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Found zero-norm vector while normalizing.")
    return x / norms


def load_riasec_unit_vectors() -> np.ndarray:
    vecs: list[np.ndarray] = []
    for trait in TRAITS:
        path = VECTORS_DIR / f"{trait}_persona_initialization" / f"{MODEL_SAFE}.safetensors"
        t = load_file(str(path))["response_persona_vector"].detach().cpu().numpy().astype(np.float64)
        if t.shape != (DIMENSIONS,):
            raise ValueError(f"Unexpected {trait} vector shape: {t.shape}")
        vecs.append(t)
    return unit_rows(np.stack(vecs, axis=0))


def load_pcs() -> tuple[np.ndarray, dict]:
    artifact = torch.load(PCA_PATH, map_location="cpu")
    if not isinstance(artifact, dict) or "components" not in artifact:
        raise ValueError("PCA artifact missing expected dict['components'].")
    comps = artifact["components"].detach().cpu().numpy().astype(np.float64)
    if comps.ndim != 2 or comps.shape[1] != DIMENSIONS or comps.shape[0] < N_PCS:
        raise ValueError(f"Unexpected PCA components shape: {comps.shape}")
    return comps[:N_PCS], artifact

