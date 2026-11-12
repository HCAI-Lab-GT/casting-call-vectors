"""Locate and load the released Casting Call Vectors data.

The data lives in the Hugging Face dataset HF_REPO at HF_REVISION. Run
`uv run python scripts/fetch_data.py` once to download it into ./data, or point
CASTVECTORS_DATA at an existing copy. The loaders return the same objects the
analysis code originally read from CSV and .pt files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_file

HF_REPO = "HCAI-Lab-GT/casting-call-vectors"
HF_REVISION = "v1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]

CASTVECTORS = "vectors/castvectors_olmo3-7b-instruct_L16_n50.safetensors"
ASSISTANT_AXIS = "vectors/assistant_axis_olmo3-7b-instruct_L16.safetensors"
TRAIT_VECTORS = "vectors/trait_vectors_olmo3-7b-instruct_L16_n40.safetensors"

# pandas.read_csv's default missing-value strings, applied to text columns so
# read_table() returns exactly what read_csv returned on the original CSVs.
_CSV_NA = {
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
    "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
}  # fmt: skip


def data_root() -> Path:
    env = os.environ.get("CASTVECTORS_DATA")
    root = Path(env) if env else REPO_ROOT / "data"
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(
            f"No release data at {root}. Run `uv run python scripts/fetch_data.py` "
            "or set CASTVECTORS_DATA to a downloaded copy."
        )
    return root


def gold_paths() -> list[Path]:
    """Per-role steering evaluation tables (one row per role, alpha, question)."""
    return sorted((data_root() / "eval" / "gold").glob("*.parquet"))


def pairwise_paths() -> list[Path]:
    """Per-role pairwise judge tables for the 39-role subset."""
    return sorted((data_root() / "eval" / "pairwise").glob("*.parquet"))


def read_table(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a released table with read_csv semantics (column order, NaN for blanks)."""
    df = pd.read_parquet(path)
    if usecols is not None:
        wanted = set(usecols)
        missing = wanted - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        df = df[[c for c in df.columns if c in wanted]]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].mask(df[col].isin(_CSV_NA))
    return df


def load_castvectors() -> dict[str, torch.Tensor]:
    """The 275 CastVectors (layer 16, 50 pairs), each shaped (1, 4096), float32."""
    vecs = load_file(str(data_root() / CASTVECTORS))
    return {role: vecs[role].reshape(1, -1) for role in sorted(vecs)}


def load_assistant_axis() -> dict[str, dict]:
    """Assistant-axis control vectors as {role: {"vector", "type", "role"}}, incl. default."""
    path = data_root() / ASSISTANT_AXIS
    with safe_open(str(path), "pt") as f:
        types = json.loads(f.metadata()["types"])
    vecs = load_file(str(path))
    return {
        role: {"vector": vecs[role].reshape(1, -1), "type": types[role], "role": role}
        for role in sorted(vecs)
    }


def load_trait_vector(trait: str) -> np.ndarray:
    """Layer-16 prompt_persona_vector for one of the 240 traits, as float64."""
    with safe_open(str(data_root() / TRAIT_VECTORS), "pt") as f:
        return f.get_tensor(trait).double().numpy()
