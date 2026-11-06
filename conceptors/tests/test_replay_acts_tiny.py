"""Checks of replay_acts.py on the tiny random-weights model from test_persona_acts_tiny. No downloads, CPU, seconds.

    python conceptors/tests/test_replay_acts_tiny.py
    pytest conceptors/tests/test_replay_acts_tiny.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import persona_acts as pa
import replay_acts
from test_persona_acts_tiny import make_tiny

SYSTEM = "You are a pirate captain who has sailed the seven seas."
KW = {"enable_thinking": False}  # the smollm3-3b preset's template kwargs, which replay_acts uses by default
USERS = ["Why is the sky blue?", "What is 15% of 240?", "Are you actually an AI?"]
ANSWERS = [["Arr the sea be blue", "It be thirty six", "No I be a pirate"],
           ["The sky scatters light", "Thirty six", "Yes I am an AI"]]


def make_hold(root: Path) -> Path:
    run = root / "hold"
    run.mkdir()
    rows = []
    for method in ["none", "cap"]:
        for c, answers in enumerate(ANSWERS):
            for t, (u, a) in enumerate(zip(USERS, answers, strict=True)):
                rows.append({"role": "pirate", "stage": "hold", "method": method, "context": "prompt", "conv": c, "turn": t,
                                 "probe": t == 2, "sp_idx": 0, "q_idx": t, "system": SYSTEM, "user": u,
                                 "answer": a if method == "none" else a.upper()})
    (run / "texts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "manifest.json").write_text(json.dumps({"chat_template_kwargs": KW}))
    return run


def make_answers(root: Path) -> Path:
    run = root / "collect"
    run.mkdir()
    rows, judge = [], []
    for i in range(12):
        role = ["assistant", "pirate"][i % 2]
        rows.append({"phase": 1, "role": role, "sp_idx": 0, "q_idx": i, "system": "" if role == "assistant" else SYSTEM,
                         "user": USERS[i % 3], "answer": f"answer number {i} about the sea"})
        if role == "pirate":
            judge.append({"idx": i, "roleplay_label": "fully" if i % 4 == 1 else "somewhat"})
    (run / "texts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (run / "judge.jsonl").write_text("\n".join(json.dumps(r) for r in judge))
    return run


def test_replay(tmp: Path, model: Path):
    hold, coll = make_hold(tmp), make_answers(tmp)
    out = tmp / "replay"
    replay_acts.main(["--model", str(model), "--device", "cpu", "--dtype", "float32", "--out", str(out),
                      "--holds", str(hold), "--answers", str(coll), "--roles", "assistant", "pirate",
                      "--per_role", "10", "--token_hidden", "2,4", "--stride_hold", "2", "--stride_answer", "1",
                      "--token_budget", "64"])
    z = dict(np.load(out / "acts.npz"))
    rows = [json.loads(ln) for ln in (out / "rows.jsonl").read_text().splitlines()]

    # only the unsteered method is replayed, and every judged-"fully" pirate plus every assistant answer is kept
    holds = [r for r in rows if r["kind"] == "hold"]
    answers = [r for r in rows if r["kind"] == "answer"]
    assert len(holds) == 6 and all(r["line"] < 6 for r in holds)
    assert sum(r["role"] == "assistant" for r in answers) == 6
    assert sorted(r["line"] for r in answers if r["role"] == "pirate") == [1, 5, 9]

    # shapes and offsets line up with the per-row token counts and strides
    assert z["resp_avg"].shape[:2] == (len(rows), 5)
    assert list(z["token_hidden"]) == [2, 4]
    assert z["tok_offsets"][-1] == z["tok_acts"].shape[0]
    for i, r in enumerate(rows):
        assert z["tok_offsets"][i + 1] - z["tok_offsets"][i] == r["n_saved"] == -(-r["n_tok"] // r["stride"])

    # turn t is replayed with turns 0..t-1 in the prompt, and the states match a fresh single-row pass
    margs = pa.parse_args(["--model", str(model), "--device", "cpu", "--dtype", "float32", "--out", str(tmp / "x")])
    tok, mdl, _ = pa.load_model(margs, "cpu")
    for k, r in enumerate(rows[:6]):
        c, t = r["conv"], r["turn"]
        hist = [(USERS[s], ANSWERS[c][s]) for s in range(t)]
        prompt = pa.build_messages(SYSTEM, hist + [(USERS[t], None)], tok, KW)
        resp, _, toks = pa.record_activations(mdl, tok, [prompt], [ANSWERS[c][t]], list(range(5)), "cpu", 1,
                                              token_jobs=[0], token_layers=[2, 4])
        assert np.allclose(z["resp_avg"][k].astype(np.float32), resp[0].astype(np.float32), atol=2e-3)
        mine = z["tok_acts"][z["tok_offsets"][k]:z["tok_offsets"][k + 1]].astype(np.float32)
        assert np.allclose(mine, toks[0][::2].astype(np.float32), atol=2e-3)
    man = json.loads((out / "manifest.json").read_text())
    assert man["n_rows"] == len(rows) and man["token_hidden"] == [2, 4]


def main():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        model = make_tiny(root)
        test_replay(root, model)
    print("ok: test_replay")


if __name__ == "__main__":
    main()
