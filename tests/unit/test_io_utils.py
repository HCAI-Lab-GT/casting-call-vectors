import json
import threading

import pytest
import torch

from pvx.utils.io_utils import (
    PERSONA_VECTOR_KEYS,
    atomic_save_safetensors,
    atomic_write_json,
    file_lock,
    is_valid_safetensors,
    load_json_or_none,
)


def _tensors():
    return {k: torch.ones(2, 3) for k in PERSONA_VECTOR_KEYS}


def test_atomic_write_json_roundtrip_and_no_temp_left(temp_dir):
    path = temp_dir / "sub" / "data.json"
    atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in path.parent.iterdir()] == ["data.json"]


def test_atomic_write_json_failure_keeps_old_file(temp_dir):
    path = temp_dir / "data.json"
    atomic_write_json(path, {"ok": True})
    with pytest.raises(TypeError):
        atomic_write_json(path, {"bad": object()})
    assert json.loads(path.read_text()) == {"ok": True}
    assert [p.name for p in temp_dir.iterdir()] == ["data.json"]


def test_load_json_or_none_handles_missing_and_truncated(temp_dir):
    assert load_json_or_none(temp_dir / "missing.json") is None
    truncated = temp_dir / "cut.json"
    truncated.write_text('{"positive": [{"q":')
    assert load_json_or_none(truncated) is None


def test_atomic_save_safetensors_is_valid(temp_dir):
    path = temp_dir / "v.safetensors"
    atomic_save_safetensors(_tensors(), path, metadata={"layer_steering": "16"})
    assert is_valid_safetensors(path, PERSONA_VECTOR_KEYS)
    assert not is_valid_safetensors(path, ("not_a_key",))


def test_is_valid_safetensors_rejects_missing_empty_and_truncated(temp_dir):
    assert not is_valid_safetensors(temp_dir / "nope.safetensors")

    empty = temp_dir / "empty.safetensors"
    empty.write_bytes(b"")
    assert not is_valid_safetensors(empty)

    good = temp_dir / "good.safetensors"
    atomic_save_safetensors(_tensors(), good)
    cut = temp_dir / "cut.safetensors"
    cut.write_bytes(good.read_bytes()[:50])
    assert not is_valid_safetensors(cut, PERSONA_VECTOR_KEYS)


def test_file_lock_serializes_read_modify_write(temp_dir):
    path = temp_dir / "counter.json"
    atomic_write_json(path, {"n": 0})

    def bump():
        for _ in range(20):
            with file_lock(path):
                n = load_json_or_none(path)["n"]
                atomic_write_json(path, {"n": n + 1})

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert load_json_or_none(path)["n"] == 80
