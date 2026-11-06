"""End-to-end check of persona_screen.py on synthetic replay data with planted geometry. CPU, seconds.

The synthetic persona drifts back toward the assistant along the persona axis, and AI admissions move along a
separate identity direction, so the screen should find drift mostly on the axis and identity mostly off it.

    python conceptors/tests/test_persona_screen_synth.py
    pytest conceptors/tests/test_persona_screen_synth.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import persona_screen

D, NL, HID = 24, 36, 18
JUDGES = ("qwen", "gemma", "mistral", "llama")


def make_data(root: Path):
    rng = np.random.default_rng(0)
    mu_a = np.zeros(D)
    axis = np.zeros(D)
    axis[0] = 3.0
    ident = np.zeros(D)
    ident[1] = 2.5
    content = rng.normal(size=(D, D)) * 0.25
    content[:, 0] *= 0.2
    content[:, 1] *= 0.2

    def toks(center, n=8):
        return center + rng.normal(size=(n, D)) @ content.T + 0.3 * rng.normal(size=(n, D))

    rows, acts, resp = [], [], []

    def add(meta, T):
        rows.append(dict(meta, n_tok=len(T), n_saved=len(T), stride=1))
        acts.append(T)
        R = np.zeros((NL + 1, D))
        R[HID] = T.mean(0)
        resp.append(R)
    for i in range(120):
        role = "assistant" if i % 2 == 0 else "pirate"
        add({"source": "/runs/collect", "line": i, "kind": "answer", "role": role, "context": "answer", "conv": -1, "turn": -1,
                 "probe": False, "sp_idx": 0, "q_idx": i % 10}, toks(mu_a + (axis if role == "pirate" else 0)))
    labels = {}
    for ctx, run in (("prompt", "steer_hold_v2_pirate_prompt"), ("steer_only", "steer_hold_v2_pirate_steer_only")):
        labels[run] = []
        for c in range(16):
            for t in range(20):
                line = c * 20 + t
                probe = t in (4, 7, 12, 15)
                if ctx == "steer_only":
                    center, role, idl = mu_a, "not", "ai"
                else:
                    drift = min(1.0, t / 14) * (rng.random() < 0.6)
                    ai = probe and rng.random() < 0.5
                    center = mu_a + axis * (1 - drift) + (ident if ai else 0)
                    role = "fully" if drift < 0.3 else "not" if drift > 0.6 else "somewhat"
                    idl = "ai" if ai else "not_ai"
                add({"source": f"/runs/{run}", "line": line, "kind": "hold", "role": "pirate", "context": ctx, "conv": c, "turn": t,
                         "probe": probe, "sp_idx": 0, "q_idx": t}, toks(center))
                labels[run].append((line, role, idl))
    rep = root / "replay"
    rep.mkdir()
    off = np.cumsum([0] + [len(a) for a in acts])
    np.savez(rep / "acts.npz", resp_avg=np.stack(resp).astype(np.float16),
             tok_acts=np.concatenate(acts)[:, None, :].astype(np.float16), tok_offsets=off,
             hidden_idx=np.arange(NL + 1), token_hidden=np.array([HID]))
    (rep / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for run, labs in labels.items():
        for j in JUDGES:
            folder = root / "judges" / f"{run}.judge-{j}"
            folder.mkdir(parents=True)
            key = "identity_r_label" if j in ("qwen", "gemma") else "identity_label"
            folder.joinpath("judge.jsonl").write_text(
                "\n".join(json.dumps({"idx": ln, "roleplay_label": rl, key: il}) for ln, rl, il in labs))
    return rep, root / "judges"


def test_screen(tmp: Path):
    root = tmp / "screen_synth"  # the session tmp is shared with the replay tests, which also write tmp/replay
    root.mkdir(exist_ok=True)
    rep, judges = make_data(root)
    out = root / "screen"
    persona_screen.main(["--replay", str(rep), "--judges", str(judges), "--out", str(out), "--personas", "pirate"])
    res = json.loads((out / "screen.json").read_text())["results"]["pirate"]
    sci = res["science"]
    # planted geometry: drift lies on the persona axis, identity off it
    assert sci["drift"]["share_axis"] > 0.8
    assert sci["identity"]["share_axis"] < 0.3
    methods = {r["method"] for r in res["ops"]}
    assert {"cap", "fisher_cap", "region", "diffpull", "gated_ot", "ccs_pull", "conceptor_tok"} <= methods
    none = next(r for r in res["ops"] if r["method"] == "none")
    assert abs(none["restore_axis"]) < 1e-6 and none["distortion"] == 0
    # the region (which has the identity coordinate) fixes admissions better than the axis-only cap
    best = lambda m, k: max(r[k] for r in res["ops"] if r["method"] == m)
    assert best("region", "identity_fix") > best("cap", "identity_fix")
    for f in ("fig_restore_vs_distortion.png", "fig_detection.png", "fig_drift_axis_share.png", "fig_persona_space.png"):
        assert (out / f).exists()


def main():
    with tempfile.TemporaryDirectory() as d:
        test_screen(Path(d))
    print("ok: test_screen")


if __name__ == "__main__":
    main()
