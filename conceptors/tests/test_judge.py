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
