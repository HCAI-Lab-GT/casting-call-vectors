"""End-to-end checks of persona_acts.py on a tiny random-weights model. No downloads; runs in seconds on CPU.

    python conceptors/tests/test_persona_acts_tiny.py      # plain runner
    pytest conceptors/tests/test_persona_acts_tiny.py      # same tests under pytest
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import persona_acts  # noqa: E402
import persona_sources as src  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>"
    "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}"
)
ROLE_FILES = {
    "assistant": ["You are an assistant dedicated to helping users.", "Act as an assistant that is helpful."],
    "pirate": ["You are a pirate captain who has sailed the seven seas.", "Be a pirate who speaks of the sea."],
}
AA_QUESTIONS = ["What is the relationship between law and morality?", "Why is the sky blue?", "What is 15% of 240?"]


def make_tiny(root: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = [p for ps in persona_acts.ROLES.values() for p in ps] + [q for _, q in persona_acts.QUESTIONS] + AA_QUESTIONS
    bpe = Tokenizer(models.BPE(unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    bpe.train_from_iterator(corpus * 3, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    import torch

    torch.manual_seed(0)
    path = root / "tiny-llama"
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def make_aa(root: Path) -> tuple[Path, Path]:
    roles_dir = root / "aa" / "roles" / "instructions"
    roles_dir.mkdir(parents=True)
    for name, prompts in ROLE_FILES.items():
        data = {"instruction": [{"pos": p} for p in prompts], "questions": [], "eval_prompt": f"Is this a {name}?"}
        (roles_dir / f"{name}.json").write_text(json.dumps(data))
    qfile = root / "aa" / "extraction_questions.jsonl"
    qfile.write_text("\n".join(json.dumps({"question": q, "id": i}) for i, q in enumerate(AA_QUESTIONS)))
    return roles_dir, qfile


def run(model: Path, out: Path, *extra: str) -> tuple[dict, list[dict], dict]:
    persona_acts.main(["--model", str(model), "--device", "cpu", "--out", str(out), "--max_new_tokens", "6",
                       "--batch_size", "8", *extra])
    npz = dict(np.load(out / "acts.npz"))
    texts = [json.loads(line) for line in (out / "texts.jsonl").read_text().splitlines()]
    manifest = json.loads((out / "manifest.json").read_text())
    return npz, texts, manifest


def test_builtin_pilot_path_with_phase2(tmp: Path, model: Path):
    npz, texts, man = run(model, tmp / "builtin", "--quick", "--roles", "assistant", "pirate")
    n1 = (3 + 3 + 1) * persona_acts.QUICK_N_QUESTIONS  # 3 assistant + 3 pirate prompts + pirate_narrated
    assert npz["p1_resp_avg"].shape == (n1, len(man["saved_hidden_idx"]), 32)
    assert man["saved_hidden_idx"] == list(range(5))  # 4 layers -> hidden 0..4
    assert man["layer_path"] == "model.layers"
    assert "p2_resp_avg" in npz and "steer_v" in npz
    assert set(npz["p2_cond"]) == {"none", "add", "cap", "conceptor"}
    assert sum(1 for t in texts if t.get("phase") == 2) == 4 * 2 * 12
    assert {"pirate_narrated"} <= set(npz["p1_role"])


def test_file_sources_samples_and_tokens(tmp: Path, model: Path, roles_dir: Path, qfile: Path):
    npz, texts, man = run(model, tmp / "files", "--roles_dir", str(roles_dir), "--questions_file", str(qfile),
                          "--roles", "assistant", "pirate", "--n_questions", "3", "--n_samples", "2",
                          "--save_tokens", "4", "--skip_phase2", "--chat_template_kwargs", '{"enable_thinking": false}')
    n1 = (2 + 2 + 1) * 3 * 2  # 2+2 prompts + narrated, 3 questions, 2 samples
    assert npz["p1_resp_avg"].shape[0] == n1 == len(texts)
    assert sorted(set(npz["p1_sample_idx"].tolist())) == [0, 1]
    assert npz["p1_tok_job"].tolist() == [0, 10, 19, 29]  # spread over all 30 rows
    assert npz["p1_tok_offsets"][-1] == npz["p1_tok_acts"].shape[0]
    assert npz["p1_tok_acts"].shape[1:] == (len(man["token_hidden_idx"]), 32)
    assert man["chat_template_kwargs"] == {"enable_thinking": False}
    assert "n_answers_with_think_tags" in man and all("has_think" in t for t in texts)


def test_extra_roles_with_builtin_fallback(tmp: Path, model: Path, roles_dir: Path, qfile: Path):
    _, texts, _ = run(model, tmp / "extra", "--roles_dir", str(roles_dir), "--questions_file", str(qfile),
                      "--roles", "pirate", "--extra_roles", "pirate_doctor", "--narrated", "--skip_phase2")
    counts = {r: sum(t["role"] == r for t in texts) for r in {t["role"] for t in texts}}
    assert counts == {"pirate": 2 * 3, "pirate_doctor": 1 * 3}  # pirate_doctor has no AA file: built-in prompt
    assert texts[-1]["system"] == persona_acts.ROLES["pirate_doctor"][0]


def test_base_mode(tmp: Path, model: Path, roles_dir: Path, qfile: Path):
    npz, texts, man = run(model, tmp / "base", "--mode", "base", "--roles_dir", str(roles_dir),
                          "--questions_file", str(qfile), "--roles", "pirate", "--narrated")
    assert man["mode"] == "base" and "phase2" not in man
    assert len(texts) == 2 * 3
    assert all(t["prompt"].startswith("Steve is a pirate") and t["prompt"].endswith('would reply: "') for t in texts)
    assert all("<|assistant|>" not in t["prompt"] for t in texts)
    assert all('"' not in t["answer"] and "raw_answer" in t for t in texts)


def test_teacher_forced(tmp: Path, model: Path, roles_dir: Path, qfile: Path):
    _, texts, _ = run(model, tmp / "src", "--roles_dir", str(roles_dir), "--questions_file", str(qfile),
                      "--roles", "pirate", "--narrated", "--skip_phase2")
    aa_file = tmp / "pirate.jsonl"
    aa_rows = [{"system_prompt": "", "prompt_index": 0, "question_index": i, "question": q,
                "conversation": [{"role": "user", "content": q}, {"role": "assistant", "content": f"Arr, answer {i}."}]}
               for i, q in enumerate(AA_QUESTIONS)]
    aa_file.write_text("\n".join(json.dumps(r) for r in aa_rows))
    npz, tf_texts, man = run(model, tmp / "tf", "--teacher_forced", str(tmp / "src" / "texts.jsonl"), str(aa_file))
    assert len(tf_texts) == len(texts) + len(aa_rows)
    assert [t["answer"] for t in tf_texts[:len(texts)]] == [t["answer"] for t in texts]
    assert tf_texts[-1]["answer"] == "Arr, answer 2." and tf_texts[-1]["role"] == "pirate"
    assert not tf_texts[-1]["prompt"].startswith("<|system|>")  # AA rows with no system message stay without one
    assert npz["p1_resp_avg"].shape[0] == len(tf_texts) and man["teacher_forced"]


def synthetic_judged_run(tmp: Path, model: Path, name: str) -> tuple[Path, list[str]]:
    """Varied synthetic answers replayed teacher-forced (so activations differ across rows) plus fake judge labels."""
    run_dir = tmp / name
    if (run_dir / "judge.jsonl").exists():
        texts = [json.loads(line) for line in (run_dir / "texts.jsonl").read_text().splitlines()]
        return run_dir, ["fully" if t["role"] == "pirate" and i % 3 else "not" for i, t in enumerate(texts)]
    rng = np.random.default_rng(0)
    words = "arr matey sea ship gold the a we you sail captain help answer here is task question".split()
    rows = [{"role": r, "sp_idx": s, "q_idx": qi, "task": "aa", "system": ROLE_FILES[r][s], "user": q,
             "answer": " ".join(rng.choice(words, size=12)), "sample_idx": k}
            for r in ("assistant", "pirate") for s in range(2) for qi, q in enumerate(AA_QUESTIONS * 2) for k in range(2)]
    src_file = tmp / f"{name}_synthetic_texts.jsonl"
    src_file.write_text("\n".join(json.dumps(r) for r in rows))
    _, texts, _ = run(model, run_dir, "--teacher_forced", str(src_file), "--save_tokens", str(len(rows)))
    labels = ["fully" if t["role"] == "pirate" and i % 3 else "not" for i, t in enumerate(texts)]
    (run_dir / "judge.jsonl").write_text("\n".join(
        json.dumps({"idx": i, "phase": 1, "roleplay_label": lab, "coherence": 80}) for i, lab in enumerate(labels)))
    return run_dir, labels


def fake_judge(run_dir: Path, label_of) -> None:
    rows = [json.loads(line) for line in (run_dir / "texts.jsonl").read_text().splitlines()]
    (run_dir / "judge.jsonl").write_text("\n".join(
        json.dumps({"idx": i, "phase": 1, "roleplay_label": label_of(r), "coherence": 70}) for i, r in enumerate(rows)))


def test_analyze_with_judge_filter(tmp: Path, model: Path, roles_dir: Path, qfile: Path):
    import analyze_acts

    run_dir, labels = synthetic_judged_run(tmp, model, "analyze")
    analyze_acts.main(["--run", str(run_dir)])
    d = json.loads((run_dir / "derived.json").read_text())
    n_fully = sum(lab == "fully" for lab in labels)
    assert d["judge"]["n_fit"] == n_fully < d["meta"]["counts"]["pirate"]
    assert d["rank"]["n"] == n_fully
    assert d["meta"]["steer_hidden"] == 3 and d["meta"]["read_hidden"] == 3  # 4 layers: 4//2+1, round(0.75*4)
    assert {"sweep", "maps", "groups", "rank", "lowrank", "judge"} <= set(d) and "logic" not in d
    assert all("j" in p for p in d["points"])


def test_steer_modes_change_the_steered_layer(tmp: Path, model: Path):
    import torch
    from transformers import AutoModelForCausalLM

    import steering

    run_dir, _ = synthetic_judged_run(tmp, model, "fitrun")
    fits = steering.fit(run_dir)
    assert fits["module"] == 2 and fits["read_hidden"] == 3 and fits["n_persona"] < fits["n_default"] + 100
    assert fits["tok_C"].shape == (32, 32) and fits["n_tokens"] > fits["n_persona"] and 0 < fits["lr_null_keep"] < 1
    no_tok = steering.fit(run_dir, module=0)  # hidden 1 has no per-token rows saved
    assert "tok_C" not in no_tok and not steering.available(no_tok, "conceptor_tok") and steering.available(no_tok, "add")
    m = AutoModelForCausalLM.from_pretrained(model).eval()
    x = torch.tensor([[5, 6, 7, 8, 9]])
    # transformers 4.57 records hidden_states[m + 1] before forward hooks run, so the steered layer's own entry
    # looks unsteered; the edit propagates, so check the next hidden state and the logits instead
    with torch.no_grad():
        out = m(input_ids=x, output_hidden_states=True)
        base_next, base_logits = out.hidden_states[fits["module"] + 2], out.logits
        identity = steering.specs(fits, "lowrank", 0.7)  # keep 1 in the span and a floor of 1 off it gives C = I
        identity[0]["keep"], identity[0]["null_keep"] = torch.ones_like(identity[0]["keep"]), 1.0
        with persona_acts.Steer(m, identity):
            assert torch.allclose(m(input_ids=x).logits, base_logits, atol=1e-4)
        for method, strength in [("add", 1.0), ("cap", 90), ("conceptor", 0.5), ("lowrank", 0.5), ("meanshift", 0.5),
                                 ("conceptor_tok", 0.5), ("meanshift_tok", 0.5)]:
            specs = steering.specs(fits, method, strength)
            if method == "cap":  # a cap only acts below its floor; force a floor every position is under
                specs[0]["tau"] = 1e3
            with persona_acts.Steer(m, specs):
                out = m(input_ids=x, output_hidden_states=True)
            assert not torch.allclose(out.hidden_states[fits["module"] + 2], base_next), method
            assert not torch.allclose(out.logits, base_logits), method
        with persona_acts.Steer(m, steering.specs(fits, "none", 0)):
            assert torch.allclose(m(input_ids=x).logits, base_logits)


def test_add_then_conceptor_tok_composes_in_order(tmp: Path, model: Path):
    import torch
    from transformers import AutoModelForCausalLM

    import steering

    run_dir, _ = synthetic_judged_run(tmp, model, "fitrun_combo")
    fits = steering.fit(run_dir)
    specs = steering.specs(fits, "add_ctok", 1.5)
    assert [s["mode"] for s in specs] == ["add", "conceptor"] and specs[1]["beta"] == 1.0
    v, mu, C = (torch.tensor(fits[k], dtype=torch.float32) for k in ("v", "tok_mu", "tok_C"))

    def by_hand(mod, ins, out):  # conceptor(add(h)) at beta 1 in one hook; the reverse order gives a different map
        h = out[0] if isinstance(out, tuple) else out
        h2 = mu + (h + 1.5 * v - mu) @ C
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    m = AutoModelForCausalLM.from_pretrained(model).eval()
    x = torch.tensor([[5, 6, 7, 8, 9]])
    with torch.no_grad():
        handle = persona_acts.layer_modules(m)[fits["module"]].register_forward_hook(by_hand)
        expected = m(input_ids=x).logits
        handle.remove()
        with persona_acts.Steer(m, specs):
            got = m(input_ids=x).logits
        with persona_acts.Steer(m, specs[::-1]):
            reversed_order = m(input_ids=x).logits
    assert torch.allclose(got, expected, atol=1e-4)
    assert not torch.allclose(reversed_order, expected, atol=1e-4)


def test_steer_eval_sweep_select_and_hold(tmp: Path, model: Path, roles_dir: Path, qfile: Path):
    import analyze_steer
    import steer_eval

    run_dir, _ = synthetic_judged_run(tmp, model, "fitrun")
    grid = ('{"add": [0.5, 2.0], "cap": [50], "conceptor": [0.5], "lowrank": [0.5], "meanshift": [0.5], '
            '"conceptor_tok": [0.5], "meanshift_tok": [0.5], "add_ctok": [1.0]}')
    methods = {"add", "cap", "conceptor", "lowrank", "meanshift", "conceptor_tok", "meanshift_tok", "add_ctok"}
    common = ["--model", str(model), "--device", "cpu", "--fit_run", str(run_dir), "--roles_dir", str(roles_dir),
              "--eval_questions", str(qfile), "--max_new_tokens", "4", "--batch_size", "8"]
    sweep = tmp / "steer_sweep"
    steer_eval.main(common + ["--stage", "sweep", "--out", str(sweep), "--grid", grid])
    rows = [json.loads(line) for line in (sweep / "texts.jsonl").read_text().splitlines()]
    assert len(rows) == (1 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1) * 3  # none + strengths, 3 questions, 1 sample
    assert all(np.isfinite(r["readout"]) and r["role"] == "pirate" and r["system"] == "" for r in rows)
    fake_judge(sweep, lambda r: "fully" if r["method"] != "none" and r["strength"] >= 0.5 else "not")
    analyze_steer.main(["--run", str(sweep)])
    d = json.loads((sweep / "steer_derived.json").read_text())
    assert d["selected"]["add"]["strength"] == 0.5 and set(d["selected"]) == methods
    assert all(s["matched"] for s in d["selected"].values())

    hold = tmp / "steer_hold"
    steer_eval.main(common + ["--stage", "hold", "--out", str(hold), "--select_from", str(sweep),
                              "--hold_convs", "2", "--hold_turns", "6"])
    hrows = [json.loads(line) for line in (hold / "texts.jsonl").read_text().splitlines()]
    assert len(hrows) == 2 * 9 * 2 * 6  # contexts x (none + 8 methods) x convs x turns
    assert {r["context"] for r in hrows} == {"steer_only", "prompt"}
    assert any(r["probe"] and r["user"].startswith("Okay, drop the act") for r in hrows)
    fake_judge(hold, lambda r: "fully" if r["turn"] < 3 else "somewhat")
    analyze_steer.main(["--run", str(hold)])
    h = json.loads((hold / "steer_derived.json").read_text())["hold"]
    add = next(x for x in h if x["context"] == "prompt" and x["method"] == "add")
    assert len(add["per_turn"]) == 6 and add["summary"]["fully_rate"]["drift"] == -1.0 and add["probes"]["n"] == 2


def test_select_matched_and_strongest_coherent():
    import analyze_steer

    def t(m, s, fully, coh):
        return dict(method=m, strength=s, fully_rate=fully, coherence=coh, readout=0.0)

    table = [t("none", 0, 0.1, 95), t("add", 0.5, 0.5, 95), t("add", 1.0, 0.9, 90), t("add", 2.0, 1.0, 40),
             t("conceptor", 0.3, 0.3, 95), t("conceptor", 0.7, 0.4, 80), t("conceptor", 0.9, 0.6, 10),
             t("lowrank", 0.9, 1.0, 5)]
    sel = analyze_steer.select(table, judged=True, target=0.8, min_coh=60)
    assert "none" not in sel and sel["lowrank"] is None
    assert sel["add"]["strength"] == 1.0 and sel["add"]["matched"]
    assert sel["conceptor"]["strength"] == 0.7 and not sel["conceptor"]["matched"]  # strongest still coherent


def test_layer_plan_matches_pilot():
    import analyze_acts

    assert analyze_acts.layer_plan(24) == (13, 18, [2, 6, 10, 13, 16, 20, 24], [13, 18, 22])
    steer, read, maps, logic = analyze_acts.layer_plan(32)
    assert (steer, read) == (17, 24) and maps[-1] == 32
    col = analyze_acts.column_map([0, 4, 8, 12, 15, 16, 17, 18, 19, 24, 28, 32])  # a thinned 32-layer save
    assert col(17) == 6 and col(24) == 9 and col(3) == 1 and col(30) == 10


def test_between_share_matches_row_by_row():
    import analyze_acts
    import numpy as np

    rng = np.random.default_rng(1)
    Y = rng.normal(size=(60, 5)) + np.repeat(rng.normal(size=(4, 5)), 15, axis=0)
    Y -= Y.mean(0)
    for labels in (np.repeat(np.array(["a", "b", "c", "d"]), 15), rng.integers(0, 7, 60)):
        brute = np.array([Y[labels == g].mean(0) for g in labels])  # the old O(n^2) form
        assert np.isclose(analyze_acts.between_share(Y, labels), (brute ** 2).sum() / (Y ** 2).sum())


def test_gap_closure_endpoints():
    import gap_closure

    rng = np.random.default_rng(2)
    H, mu = rng.normal(size=(50, 6)), rng.normal(size=6) + 3
    vhat = (mu - H.mean(0)) / np.linalg.norm(mu - H.mean(0))
    ident, zero = gap_closure.gap_report(H, vhat, mu, np.eye(6)), gap_closure.gap_report(H, vhat, mu, np.zeros((6, 6)))
    assert np.isclose(ident["closed_along_vhat_median"], 0) and np.isclose(ident["keep_along_vhat"], 1)
    assert np.isclose(ident["gap_energy_kept"], 1) and ident["n_kept_dirs"] == 6
    assert np.isclose(zero["closed_along_vhat_median"], 1) and np.isclose(zero["gap_energy_kept"], 0)


def test_explainer_thin_points_keeps_alignment():
    sys.path.insert(0, str(HERE.parent / "explainer"))
    import build

    # two roles, one prompt, two questions, three draws each, interleaved the way a run writes them
    pts = [dict(r=r, s=0, q=q, a=f"{r}{q}-{k}") for k in range(3) for q in (0, 1) for r in ("assistant", "pirate")]
    coords = [[i, -i] for i in range(len(pts))]
    vals = {r: [float(i) for i, p in enumerate(pts) if p["r"] == r] for r in ("assistant", "pirate")}
    data = dict(meta={}, points=pts, maps={"resp": {"2": dict(pca=coords, plane=coords), "4": None}},
                groups={"resp": {r: dict(vals=v) for r, v in vals.items()}})
    out = build.thin_points(json.loads(json.dumps(data)), 1)
    assert [p["a"] for p in out["points"]] == ["assistant0-0", "pirate0-0", "assistant1-0", "pirate1-0"]
    kept = [i for i, p in enumerate(pts) if p["a"].endswith("-0")]
    assert out["maps"]["resp"]["2"]["plane"] == [coords[i] for i in kept]
    for r in ("assistant", "pirate"):  # strip value j still belongs to the j-th kept point of that role
        assert out["groups"]["resp"][r]["vals"] == [float(i) for i in kept if pts[i]["r"] == r]


def test_layer_paths():
    from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM

    neox = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                                            intermediate_size=32))
    gpt2 = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=2))
    assert persona_acts.find_layers(neox)[0] == "gpt_neox.layers"
    assert persona_acts.find_layers(gpt2)[0] == "transformer.h"


def test_presets_and_overrides():
    args = persona_acts.parse_args(["--preset", "smollm3-3b"])
    assert args.model == "HuggingFaceTB/SmolLM3-3B" and args.chat_template_kwargs == {"enable_thinking": False}
    assert args.max_new_tokens == 256
    args = persona_acts.parse_args(["--preset", "olmo3-7b-base", "--revision", "stage1-step5000", "--max_new_tokens", "32"])
    assert args.mode == "base" and args.revision == "stage1-step5000" and args.max_new_tokens == 32
    assert args.repetition_penalty == 1.1


def test_resolve_model_path_offline(tmp: Path):
    """A cached repo id resolves to its snapshot folder without network, as needed under HF_HUB_OFFLINE=1."""
    import os
    import subprocess

    hub = tmp / "hfhome" / "hub" / "models--org--tiny"
    sha = "0123456789abcdef0123456789abcdef01234567"
    (hub / "snapshots" / sha).mkdir(parents=True)
    (hub / "snapshots" / sha / "config.json").write_text("{}")
    (hub / "refs").mkdir()
    (hub / "refs" / "main").write_text(sha)
    code = ("import sys; sys.path.insert(0, %r); import persona_acts as p; "
            "path, snap = p.resolve_model_path('org/tiny', None); print(path); print(snap)") % str(HERE.parent)
    env = dict(os.environ, HF_HOME=str(tmp / "hfhome"), HF_HUB_OFFLINE="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out[0].endswith(f"snapshots/{sha}") and out[1] == sha
    assert persona_acts.resolve_model_path(str(tmp), None) == (str(tmp), None)  # local dirs pass through


def test_third_person():
    assert src.third_person("You are a pirate captain who has sailed your ship.") == \
        "Steve is a pirate captain who has sailed their ship."
    assert src.third_person("Act as a seasoned pirate who values freedom.").startswith("Steve is a seasoned pirate")
    assert src.trim_base_answer('Arr, freedom above all." He laughed. "More') == "Arr, freedom above all."


def _main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        model = make_tiny(tmp)
        roles_dir, qfile = make_aa(tmp)
        fixtures = dict(tmp=tmp, model=model, roles_dir=roles_dir, qfile=qfile)
        for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
            kwargs = {k: v for k, v in fixtures.items() if k in fn.__code__.co_varnames[:fn.__code__.co_argcount]}
            try:
                fn(**kwargs)
                print(f"PASS {name}")
            except Exception as e:  # report every test, then fail
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _main() else 0)
