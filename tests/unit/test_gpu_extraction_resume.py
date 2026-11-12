import json
import sys

import torch

from pvx.implementations.roles_optimized import roles_optimized_gpu as gpu
from pvx.utils.io_utils import PERSONA_VECTOR_KEYS, is_valid_safetensors

MODEL = "test/model"
LAYER = 3
ROLE = "lawyer"


class _Tok:
    def apply_chat_template(self, msgs, **kw):
        return torch.tensor([[1, 2, 3]])


def _make_extractor(tmp_path, n_pairs):
    positive = [
        {"pos_prompt": f"p{i}", "question": f"q{i}", "response": "r", "score": 3}
        for i in range(n_pairs)
    ]
    base = [{"question": f"q{i}", "response": "r", "score": 1} for i in range(n_pairs)]
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps({"positive": positive, "base": base}))

    ex = gpu.RoleActivationExtractor.__new__(gpu.RoleActivationExtractor)
    ex.role = ROLE
    ex.target_model_id = MODEL
    ex.layer_steering = LAYER
    ex.device = "cpu"
    ex.tokenizer = _Tok()
    ex.answer_model_id = MODEL
    ex._qa_responses_path = str(qa_path)
    ex._get_activations = lambda ids, mask: (
        torch.ones(1, 4),
        torch.ones(1, 4),
        torch.ones(5, 1, 4),
        "",
    )
    return ex


def _path(out, count):
    return gpu.checkpoint_path(str(out), ROLE, MODEL, LAYER, count)


def test_ran_out_of_pairs_saves_only_reachable_counts(tmp_path):
    out = tmp_path / "out"
    ex = _make_extractor(tmp_path, n_pairs=3)

    saved = ex.extract_and_save_incremental([2, 5], str(out))

    assert saved == [2]
    assert is_valid_safetensors(_path(out, 2), PERSONA_VECTOR_KEYS)
    assert not _path(out, 5).exists()


def test_rerun_skips_existing_checkpoints(tmp_path):
    out = tmp_path / "out"
    ex = _make_extractor(tmp_path, n_pairs=4)
    assert ex.extract_and_save_incremental([2, 4], str(out)) == [2, 4]

    calls = []
    ex._get_activations = lambda *a: calls.append(1)
    assert ex.extract_and_save_incremental([2, 4], str(out)) == []
    assert calls == []


def test_truncated_checkpoint_is_redone(tmp_path):
    out = tmp_path / "out"
    ex = _make_extractor(tmp_path, n_pairs=4)
    ex.extract_and_save_incremental([2, 4], str(out))

    good = _path(out, 4).read_bytes()
    _path(out, 4).write_bytes(good[:40])
    assert gpu.missing_counts(str(out), ROLE, MODEL, LAYER, [2, 4]) == [4]

    assert ex.extract_and_save_incremental([2, 4], str(out)) == [4]
    assert is_valid_safetensors(_path(out, 4), PERSONA_VECTOR_KEYS)


def test_main_loads_model_once_for_many_roles(tmp_path, monkeypatch):
    roles = ["a", "b", "c"]
    qa_dir = tmp_path / "qa" / MODEL.replace("/", "__")
    qa_dir.mkdir(parents=True)
    for r in roles:
        (qa_dir / f"{r}.json").write_text("{}")

    built, reused = [], []

    class FakeExtractor:
        def __init__(self, role, **kw):
            built.append(role)

        def set_role(self, role, **kw):
            reused.append(role)

        def extract_and_save_incremental(self, counts, output_dir):
            return list(counts)

    monkeypatch.setattr(gpu, "RoleActivationExtractor", FakeExtractor)
    monkeypatch.setattr(gpu.RoleDataset, "from_json", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--activation_extraction_model", MODEL, "-r", *roles, "-c", "2",
         "-q", str(tmp_path / "qa"), "-o", str(tmp_path / "out")],
    )

    gpu.main()

    assert built == ["a"]
    assert reused == ["b", "c"]


def test_no_temp_files_left_behind(tmp_path):
    out = tmp_path / "out"
    ex = _make_extractor(tmp_path, n_pairs=4)
    ex.extract_and_save_incremental([2, 4], str(out))
    assert not list(out.rglob("*.tmp"))
