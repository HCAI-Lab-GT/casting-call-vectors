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
    assert npz["p1_tok_job"].tolist() == [0, 1, 2, 3]
    assert npz["p1_tok_offsets"][-1] == npz["p1_tok_acts"].shape[0]
    assert npz["p1_tok_acts"].shape[1:] == (len(man["token_hidden_idx"]), 32)
    assert man["chat_template_kwargs"] == {"enable_thinking": False}
    assert "n_answers_with_think_tags" in man and all("has_think" in t for t in texts)


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


def test_analyze_with_judge_filter(tmp: Path, model: Path, roles_dir: Path, qfile: Path):
    import analyze_acts

    # varied synthetic answers, replayed teacher-forced, so activations differ across rows
    rng = np.random.default_rng(0)
    words = "arr matey sea ship gold the a we you sail captain help answer here is task question".split()
    rows = [{"role": r, "sp_idx": s, "q_idx": qi, "task": "aa", "system": ROLE_FILES[r][s], "user": q,
             "answer": " ".join(rng.choice(words, size=12)), "sample_idx": k}
            for r in ("assistant", "pirate") for s in range(2) for qi, q in enumerate(AA_QUESTIONS * 2) for k in range(2)]
    src_file = tmp / "synthetic_texts.jsonl"
    src_file.write_text("\n".join(json.dumps(r) for r in rows))
    run_dir = tmp / "analyze"
    _, texts, man = run(model, run_dir, "--teacher_forced", str(src_file))
    labels = ["fully" if t["role"] == "pirate" and i % 3 else "not" for i, t in enumerate(texts)]
    (run_dir / "judge.jsonl").write_text("\n".join(
        json.dumps({"idx": i, "phase": 1, "roleplay_label": lab, "coherence": 80}) for i, lab in enumerate(labels)))
    analyze_acts.main(["--run", str(run_dir)])
    d = json.loads((run_dir / "derived.json").read_text())
    n_fully = sum(lab == "fully" for lab in labels)
    assert d["judge"]["n_fit"] == n_fully < d["meta"]["counts"]["pirate"]
    assert d["rank"]["n"] == n_fully
    assert d["meta"]["steer_hidden"] == 3 and d["meta"]["read_hidden"] == 3  # 4 layers: 4//2+1, round(0.75*4)
    assert {"sweep", "maps", "groups", "rank", "lowrank", "judge"} <= set(d) and "logic" not in d
    assert all("j" in p for p in d["points"])


def test_layer_plan_matches_pilot():
    import analyze_acts

    assert analyze_acts.layer_plan(24) == (13, 18, [2, 6, 10, 13, 16, 20, 24], [13, 18, 22])
    steer, read, maps, logic = analyze_acts.layer_plan(32)
    assert (steer, read) == (17, 24) and maps[-1] == 32
    col = analyze_acts.column_map([0, 4, 8, 12, 15, 16, 17, 18, 19, 24, 28, 32])  # a thinned 32-layer save
    assert col(17) == 6 and col(24) == 9 and col(3) == 1 and col(30) == 10


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
