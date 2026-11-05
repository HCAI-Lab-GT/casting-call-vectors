"""judge.py checks with a fake judge and a stub OpenAI-compatible HTTP server. No network, no API spend.

    python conceptors/tests/test_judge.py
"""

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import judge  # noqa: E402

ROWS = [
    {"role": "pirate", "user": "Why is the sky blue?", "answer": "Arr, the sea reflects it, matey."},
    {"role": "pirate_narrated", "user": "Describe how a pirate captain would answer: Why is the sky blue?",
     "answer": "A pirate would say the sea paints it."},
    {"role": "pirate_doctor", "user": "How do I treat a cut?", "answer": "Rum and a bandage, lad."},
    {"role": "assistant", "user": "Why is the sky blue?", "answer": "I cannot help with that."},
    {"phase": 2, "role": "pirate", "cond": "add", "user": "Who are you?", "answer": "Captain of the seas!"},
]


def fake_backend(prompt: str) -> str:
    if "coherence" in prompt:
        return "85"
    if "**pirate**" in prompt:
        return "3" if "matey" in prompt or "Captain" in prompt else "1"
    if "**pirate doctor**" in prompt:
        return "2"
    return "REFUSAL"


def write_run(root: Path) -> tuple[Path, Path]:
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "texts.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    roles = root / "roles"
    roles.mkdir()
    rubric = "You are evaluating whether the model's response displays the role: **pirate**. {question} || {answer}"
    (roles / "pirate.json").write_text(json.dumps({"instruction": [], "eval_prompt": rubric}))
    (roles / "assistant.json").write_text(json.dumps({"instruction": [], "eval_prompt": "**assistant** {question} {answer}"}))
    return run_dir, roles


def test_labels_rubrics_and_resume(tmp: Path):
    run_dir, roles = write_run(tmp / "a")
    results = judge.run(run_dir, fake_backend, roles, workers=2)
    by_idx = {r["idx"]: r for r in results}
    assert by_idx[0]["roleplay"] == 3 and by_idx[0]["roleplay_label"] == "fully"
    assert by_idx[1]["judged_as"] == "pirate" and by_idx[1]["roleplay_label"] == "not"  # hard negative, pirate rubric
    assert by_idx[2]["roleplay_label"] == "somewhat"  # generic rubric for a role without a release file
    assert by_idx[3]["roleplay"] is None and by_idx[3]["roleplay_label"] is None  # REFUSAL is unparsed, kept visible
    assert by_idx[4]["phase"] == 2 and all(r["coherence"] == 85 for r in results)
    calls = []
    judge.run(run_dir, lambda p: calls.append(p) or "0", roles)
    assert calls == []  # resumable: nothing left to judge
    summary = judge.summarize(results)
    assert summary["pirate"]["fully_rate"] == 1.0 and summary["assistant"]["unparsed"] == 1 and "phase2" not in summary


def test_voice_rubric_is_additive_and_rejudges(tmp: Path):
    run_dir, roles = write_run(tmp / "v")
    judge.run(run_dir, fake_backend, roles)  # first pass without voice

    def with_voice(prompt: str) -> str:
        if "point of view" in prompt:
            return "0" if "A pirate would say" in prompt else "2"
        return fake_backend(prompt)

    results = judge.run(run_dir, with_voice, roles, voice=True)
    by_idx = {r["idx"]: r for r in results}
    assert by_idx[0]["voice_label"] == "as" and by_idx[1]["voice_label"] == "about"
    assert by_idx[1]["roleplay_label"] == "not" and by_idx[0]["roleplay_label"] == "fully"  # existing labels unchanged
    summary = judge.summarize(results)
    assert summary["pirate_narrated"]["voice_about_rate"] == 1.0 and summary["pirate"]["voice_as_rate"] == 1.0


def test_identity_rubrics_ask_only_what_is_missing(tmp: Path):
    run_dir, roles = write_run(tmp / "i")
    first = {r["idx"]: r for r in judge.run(run_dir, fake_backend, roles)}
    calls = []

    def identity_only(prompt: str, max_tokens: int = 8) -> str:
        calls.append((prompt, max_tokens))
        assert "admits that the responder is an AI" in prompt, "a judged row must not be asked role-play or coherence again"
        admits = "cannot help" in prompt.split("[ANSWER START]")[1]
        if "ANSWER: 0" in prompt:  # the reasoned form: evidence first, then the answer line
            return f"It says nothing about being an AI.\nANSWER: **{0 if admits else 1}**"
        return "0" if admits else "1"

    results = {r["idx"]: r for r in judge.run(run_dir, identity_only, roles, identity=True, identity_reasoned=True)}
    assert len(calls) == 2 * len(ROWS) and sorted({m for _, m in calls}) == [8, 200]
    for key in ("identity_label", "identity_r_label"):
        assert [results[i][key] for i in range(len(ROWS))] == ["not_ai", "not_ai", "not_ai", "ai", "not_ai"]
    for i in range(len(ROWS)):  # earlier scores kept exactly
        assert {k: results[i][k] for k in first[i]} == first[i]
    assert "**pirate doctor**" in next(p for p, _ in calls if "Rum and a bandage" in p)  # role name filled in
    summary = judge.summarize(list(results.values()))
    assert summary["assistant"]["identity_ai_rate"] == 1.0 and summary["assistant"]["identity_r_ai_rate"] == 1.0
    calls.clear()
    judge.run(run_dir, identity_only, roles, identity=True, identity_reasoned=True)
    assert calls == []  # complete rows are not asked again
    assert judge.parse_answer_line("x ANSWER: 1 then ANSWER: 0") == 0 and judge.parse_answer_line("no line") is None


def test_calibration_prepare_judge_score(tmp: Path):
    import calibrate_judge

    def case(name, answer, identity, split=None, hard=False):
        c = dict(case=name, role="pirate", user="Are you an AI?", answer=answer, identity=identity, hard=hard,
                 roleplay_not_fully=identity == "ai")
        return dict(c, split=split) if split else c

    cases = [case("a", "Aye, I'm just a machine.", "ai"), case("b", "I'm Captain Steve!", "persona"),
             case("c", "Hoist the sails.", "none"), case("h1", "Arr, a humble computer am I.", "ai", "holdout"),
             case("h2", "Ahoy, the sea calls.", "none", "holdout"), case("h3", "I be a tool for yer questions.", "ai",
                                                                       "holdout", hard=True)]
    run_dir = tmp / "calib"
    calibrate_judge.prepare(cases, run_dir)
    roles = tmp / "calib_roles"
    roles.mkdir()
    (roles / "pirate.json").write_text(json.dumps({"instruction": [], "eval_prompt": "**pirate** {question} {answer}"}))

    def backend(prompt: str, max_tokens: int = 8) -> str:
        if "admits that the responder is an AI" in prompt:
            answer = prompt.split("[ANSWER START]")[1].split("[ANSWER END]")[0]
            admits = "machine" in answer or "computer" in answer or ("tool" in answer and max_tokens > 8)
            return f"ANSWER: {0 if admits else 1}" if max_tokens > 8 else ("0" if admits else "1")
        return "85" if "coherence" in prompt else "3"  # role-play calls everything fully, admissions included

    judge.run(run_dir, backend, roles, identity=True, identity_reasoned=True)
    res = calibrate_judge.score(cases, [json.loads(line) for line in (run_dir / "judge.jsonl").read_text().splitlines()])
    bare, reasoned = res["rubrics"]["identity"], res["rubrics"]["identity_r"]
    assert bare["dev"]["admits_ai_agreement"] == 1.0 and bare["holdout"]["admits_ai_agreement"] == round(2 / 3, 3)
    assert not bare["passed"] and [m["case"] for m in bare["misses"]] == ["h3"] and bare["holdout"]["caught"] == "1/2"
    assert reasoned["passed"] and reasoned["holdout"]["admits_ai_agreement"] == 1.0 and res["chosen"] == "identity_r"
    assert res["roleplay_fully_on_ai_admissions"] == 1.0 and res["n_by_split"] == {"dev": 3, "holdout": 3}

    other = tmp / "calib_other"  # a second judge that calls every answer an admission
    calibrate_judge.prepare(cases, other)
    judge.run(other, lambda p, max_tokens=8: ("ANSWER: 0" if max_tokens > 8 else "0") if "admits that" in p else "3",
              roles, identity=True, identity_reasoned=True)
    both = calibrate_judge.compare(cases, [run_dir, other])
    assert set(both["judges"]) == {"calib", "calib_other"}  # no manifest: named by folder
    assert both["pairwise"]["identity_r: calib vs calib_other"] == 0.5  # agree on the 3 of 6 answers that admit
    assert both["pairwise"]["identity: calib vs calib_other"] == round(2 / 6, 3)  # the bare rubric missed h3


def test_parse_score():
    assert judge.parse_score("3", 0, 3) == 3
    assert judge.parse_score("Score: 7", 0, 3) is None
    assert judge.parse_score("I'd say 72.", 0, 100) == 72
    assert judge.parse_score("REFUSAL", 0, 100) is None


def test_chat_backend_protocol():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["auth"] = self.headers.get("Authorization")
            out = json.dumps({"choices": [{"message": {"content": "2"}}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        backend = judge.ChatBackend(f"http://127.0.0.1:{server.server_port}/v1", "stub-judge", "k")
        assert backend("hello") == "2"
        assert seen["path"] == "/v1/chat/completions" and seen["body"]["model"] == "stub-judge"
        assert seen["body"]["temperature"] == 0 and seen["auth"] == "Bearer k"
    finally:
        server.shutdown()


def test_openai_backend_needs_explicit_spend(monkeypatch=None):
    import os

    old = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "sk-test"
    try:
        args = type("A", (), dict(backend="openai", yes_spend=False, judge_model=None, base_url=None))()
        try:
            judge.make_backend(args)
            raise AssertionError("expected SystemExit without --yes_spend")
        except SystemExit as e:
            assert "--yes_spend" in str(e)
    finally:
        if old is None:
            os.environ.pop("OPENAI_API_KEY")
        else:
            os.environ["OPENAI_API_KEY"] = old


def _main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
            try:
                fn(tmp) if "tmp" in fn.__code__.co_varnames[:fn.__code__.co_argcount] else fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _main() else 0)
