"""Loaders for the released Hugging Face data."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from pvx import release_data


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    vec_dir = tmp_path / "vectors"
    vec_dir.mkdir()
    save_file(
        {"zeta": torch.ones(8), "alpha": torch.arange(8, dtype=torch.float32)},
        str(tmp_path / release_data.CASTVECTORS),
        metadata={"layer": "16", "target_pairs": "50"},
    )
    save_file(
        {"chef": torch.ones(8, dtype=torch.bfloat16), "default": torch.zeros(8, dtype=torch.bfloat16)},
        str(tmp_path / release_data.ASSISTANT_AXIS),
        metadata={"types": json.dumps({"chef": "pos_3", "default": "mean"})},
    )
    save_file({"stoic": torch.full((8,), 0.5)}, str(tmp_path / release_data.TRAIT_VECTORS))
    gold = tmp_path / "eval" / "gold"
    gold.mkdir(parents=True)
    pd.DataFrame(
        {"role": ["chef", "chef"], "alpha": [1.0, 2.5], "steered": ["", "NA"], "steered_score": [70, 80]}
    ).to_parquet(gold / "chef.parquet")
    monkeypatch.setenv("CASTVECTORS_DATA", str(tmp_path))
    return tmp_path


def test_data_root_requires_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("CASTVECTORS_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        release_data.data_root()


def test_castvectors_sorted_and_shaped_like_pt_files(data_dir):
    vecs = release_data.load_castvectors()
    assert list(vecs) == ["alpha", "zeta"]
    assert vecs["alpha"].shape == (1, 8)
    assert vecs["alpha"].dtype == torch.float32


def test_assistant_axis_keeps_type_and_default(data_dir):
    aa = release_data.load_assistant_axis()
    assert aa["default"]["type"] == "mean"
    assert aa["chef"]["type"] == "pos_3"
    assert aa["chef"]["vector"].shape == (1, 8)


def test_trait_vector_is_float64(data_dir):
    v = release_data.load_trait_vector("stoic")
    assert v.dtype == np.float64
    assert np.all(v == 0.5)


def test_read_table_matches_read_csv_semantics(data_dir):
    (path,) = release_data.gold_paths()
    df = release_data.read_table(path, usecols=["steered_score", "role", "steered"])
    assert list(df.columns) == ["role", "steered", "steered_score"]
    assert df["steered"].isna().all()
    assert df["steered_score"].tolist() == [70, 80]


def test_read_table_rejects_unknown_columns(data_dir):
    (path,) = release_data.gold_paths()
    with pytest.raises(ValueError, match="missing columns"):
        release_data.read_table(path, usecols=["nope"])


def test_from_release_checks_layer_and_pairs(data_dir):
    from pvx.implementations.roles_layers.role_layers_persona_model import (
        RoleLayersPersonaModel,
    )

    with pytest.raises(ValueError, match="layer 16, 50 pairs"):
        RoleLayersPersonaModel.from_release(
            str(data_dir / release_data.CASTVECTORS),
            "alpha",
            "allenai/Olmo-3-7B-Instruct",
            layer=16,
            target_pairs=20,
        )


def test_assistant_axis_model_accepts_release_file(data_dir):
    from pvx.implementations.roles_layers.assistant_axis_persona_model import (
        AssistantAxisPersonaModel,
    )

    release = str(data_dir / release_data.ASSISTANT_AXIS)
    path, _ = AssistantAxisPersonaModel.get_path(
        target_model_id="allenai/Olmo-3-7B-Instruct", concept="chef", safetensors_dir=release
    )
    assert str(path) == release
