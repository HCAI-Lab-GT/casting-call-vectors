"""Teacher-force saved answers through the model and save their activations, for offline method screens.

Two sources, mixed freely:
  --holds RUN [RUN ...]    steer_eval hold runs. Replays the unsteered turns (--hold_method, default 'none') with
                           their full conversation history, rendered exactly as the hold rendered them: the prompt for
                           turn t is the system prompt, turns 0..t-1 with their answers, then user t. That reproduces
                           the unsteered model's states during the hold. Steered methods are skipped, because their
                           states depended on the hooks.
  --answers RUN [RUN ...]  persona_acts collection runs. Replays single-turn phase-1 answers of --roles. Persona rows are
                           kept only when judge.jsonl rates them fully in character (the default role is not filtered),
                           --per_role rows per role, spread evenly over the run.

Writes into --out:
  acts.npz    resp_avg (n, L, d) float16   mean over answer tokens at every hidden index in hidden_idx
              tok_acts (T, K, d) float16   every stride-th answer token at the hidden indices in token_hidden
              tok_offsets (n + 1,)         row i's tokens are tok_acts[tok_offsets[i]:tok_offsets[i + 1]]
              hidden_idx, token_hidden
  rows.jsonl  one row per replayed answer: source run and its texts.jsonl line (to join judge labels), kind, role,
              context, conv, turn, probe, sp_idx, q_idx, n_tok, n_saved, stride
  manifest.json

hidden_states[i] is the output of decoder layer i - 1, so a steering module m acts on hidden index m + 1.

  python conceptors/replay_acts.py --preset smollm3-3b --runs_root $CONCEPTORS_DATA/runs --out replay_smollm3 \\
      --holds steer_hold_v2_pirate_prompt steer_hold_v2_pirate_steer_only \\
      --answers smollm3_full smollm3_genie_alien --roles assistant pirate ghost bard consultant genie alien
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import persona_acts as pa
import torch


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="smollm3-3b")
    ap.add_argument("--model", default=None, help="override the preset's model (a path or hub id)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--runs_root", default="", help="resolve relative run names and --out against this folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holds", nargs="*", default=[])
    ap.add_argument("--hold_method", default="none")
    ap.add_argument("--answers", nargs="*", default=[])
    ap.add_argument("--roles", nargs="*", default=[])
    ap.add_argument("--default_role", default="assistant")
    ap.add_argument("--per_role", type=int, default=200)
    ap.add_argument("--token_hidden", default="18", help="comma-separated hidden indices for per-token activations")
    ap.add_argument("--stride_hold", type=int, default=3, help="keep every n-th answer token of hold turns")
    ap.add_argument("--stride_answer", type=int, default=2, help="keep every n-th answer token of single-turn answers")
    ap.add_argument("--token_budget", type=int, default=40000, help="max padded tokens per forward batch")
    ap.add_argument("--limit", type=int, default=None, help="debug: at most this many rows per source")
    return ap.parse_args(argv)


def resolve(root: str, name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() or not root else Path(root) / p


def read_rows(run: Path) -> list[dict]:
    return [json.loads(ln) for ln in (run / "texts.jsonl").read_text().splitlines() if ln.strip()]


def judge_labels(run: Path) -> dict:
    path = run / "judge.jsonl"
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[r["idx"]] = r.get("roleplay_label")
    return out


def hold_jobs(run: Path, tok, template_kwargs, method: str, stride: int, limit=None) -> list[dict]:
    rows = read_rows(run)
    by_conv = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("stage") == "hold" and r.get("method") == method:
            by_conv[(r["context"], r["conv"])].append((r["turn"], i, r))
    jobs = []
    for (context, conv), turns in sorted(by_conv.items()):
        turns.sort()
        if [t for t, *_ in turns] != list(range(len(turns))):
            raise ValueError(f"{run}: {context} conv {conv} has missing turns")
        history = []
        for t, i, r in turns:
            prompt = pa.build_messages(r["system"], history + [(r["user"], None)], tok, template_kwargs)
            jobs.append({"source": str(run), "line": i, "kind": "hold", "role": r["role"], "context": context, "conv": conv, "turn": t,
                             "probe": bool(r.get("probe")), "sp_idx": r.get("sp_idx", 0), "q_idx": r.get("q_idx", 0),
                             "prompt": prompt, "answer": r["answer"], "stride": stride})
            history.append((r["user"], r["answer"]))
    return jobs[:limit]


def answer_jobs(run: Path, tok, template_kwargs, roles, default_role, per_role, stride, limit=None) -> list[dict]:
    rows = read_rows(run)
    labels = judge_labels(run)
    jobs = []
    for role in roles:
        idx = [i for i, r in enumerate(rows) if r.get("phase", 1) == 1 and r["role"] == role
               and (role == default_role or not labels or labels.get(i) == "fully")]
        if not idx:
            continue
        pick = sorted(set(np.linspace(0, len(idx) - 1, min(per_role, len(idx))).round().astype(int).tolist()))
        for k in pick:
            i = idx[k]
            r = rows[i]
            prompt = pa.build_messages(r.get("system", ""), [(r["user"], None)], tok, template_kwargs)
            jobs.append({"source": str(run), "line": i, "kind": "answer", "role": role, "context": "answer", "conv": -1, "turn": -1,
                             "probe": False, "sp_idx": r.get("sp_idx", 0), "q_idx": r.get("q_idx", 0), "prompt": prompt,
                             "answer": r["answer"], "stride": stride})
    return jobs[:limit]


def batches(lengths, budget):
    """Index batches, longest first, each with (batch size x longest sequence) <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > budget:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def replay(model, tok, jobs, hidden_idx, token_hidden, device, budget, log_every=50):
    ids, plen, alen = [], [], []
    for j in jobs:
        pi = pa.encode(tok, j["prompt"], False)
        ai = pa.encode(tok, j["answer"], False) or [tok.eos_token_id]
        ids.append(pi + ai)
        plen.append(len(pi))
        alen.append(len(ai))
    resp = [None] * len(jobs)
    toks = [None] * len(jobs)
    t0, done = time.time(), 0
    bl = batches([len(s) for s in ids], budget)
    for b, batch in enumerate(bl):
        L = max(len(ids[i]) for i in batch)
        x = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(batch), L), dtype=torch.long)
        for k, i in enumerate(batch):  # right padding is exact for a causal pass
            x[k, :len(ids[i])] = torch.tensor(ids[i])
            m[k, :len(ids[i])] = 1
        hs = model(input_ids=x.to(device), attention_mask=m.to(device), output_hidden_states=True).hidden_states
        for k, i in enumerate(batch):
            span = slice(plen[i], plen[i] + alen[i])
            resp[i] = torch.stack([hs[h][k, span].float().mean(0) for h in hidden_idx]).cpu().numpy().astype(np.float16)
            keep = torch.arange(plen[i], plen[i] + alen[i], jobs[i]["stride"], device=hs[0].device)
            toks[i] = torch.stack([hs[h][k, keep].float() for h in token_hidden], 1).cpu().numpy().astype(np.float16)
        done += len(batch)
        if b % log_every == 0 or b == len(bl) - 1:
            print(f"[replay] batch {b + 1}/{len(bl)}, {done}/{len(jobs)} rows, {time.time() - t0:.0f}s", flush=True)
    return resp, toks, alen


def main(argv=None):
    args = parse(argv)
    out = resolve(args.runs_root, args.out)
    out.mkdir(parents=True, exist_ok=True)
    margv = ["--preset", args.preset, "--out", str(out), "--device", args.device]
    if args.model:
        margv += ["--model", args.model]
    if args.dtype:
        margv += ["--dtype", args.dtype]
    margs = pa.parse_args(margv)
    device = pa.pick_device(margs.device)
    t0 = time.time()
    tok, model, dtype = pa.load_model(margs, device)
    n_layers = len(pa.layer_modules(model))
    hidden_idx = list(range(n_layers + 1))
    token_hidden = [int(s) for s in args.token_hidden.split(",") if s.strip()]
    if any(h < 0 or h > n_layers for h in token_hidden):
        raise ValueError(f"token_hidden {token_hidden} outside 0..{n_layers}")
    kw = margs.chat_template_kwargs or {}

    jobs = []
    for name in args.holds:
        run = resolve(args.runs_root, name)
        man = json.loads((run / "manifest.json").read_text())
        hold_kw = man.get("chat_template_kwargs", man.get("model_args", {}).get("chat_template_kwargs"))
        if hold_kw is not None and hold_kw != kw:
            raise ValueError(f"{run}: hold used chat_template_kwargs {hold_kw}, this run would use {kw}")
        js = hold_jobs(run, tok, kw, args.hold_method, args.stride_hold, args.limit)
        print(f"[jobs] {run.name}: {len(js)} hold turns ({args.hold_method})", flush=True)
        jobs += js
    for name in args.answers:
        run = resolve(args.runs_root, name)
        js = answer_jobs(run, tok, kw, args.roles, args.default_role, args.per_role, args.stride_answer, args.limit)
        print(f"[jobs] {run.name}: {len(js)} single-turn answers", flush=True)
        jobs += js
    if not jobs:
        raise SystemExit("nothing to replay: pass --holds and/or --answers")

    resp, toks, alen = replay(model, tok, jobs, hidden_idx, token_hidden, device, args.token_budget)
    offsets = np.cumsum([0] + [t.shape[0] for t in toks])
    np.savez(out / "acts.npz", resp_avg=np.stack(resp), tok_acts=np.concatenate(toks), tok_offsets=offsets,
             hidden_idx=np.array(hidden_idx), token_hidden=np.array(token_hidden))
    with open(out / "rows.jsonl", "w") as fh:
        for j, a, t in zip(jobs, alen, toks, strict=True):
            meta = {k: v for k, v in j.items() if k not in ("prompt", "answer")}
            fh.write(json.dumps(dict(meta, n_tok=a, n_saved=int(t.shape[0]))) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "model": margs.model, "model_snapshot": getattr(margs, "model_snapshot", None), "preset": args.preset, "device": device,
        "dtype": str(dtype), "n_layers": n_layers, "d": int(resp[0].shape[1]), "hidden_idx": hidden_idx, "token_hidden": token_hidden,
        "chat_template_kwargs": kw, "holds": args.holds, "hold_method": args.hold_method, "answers": args.answers, "roles": args.roles,
        "per_role": args.per_role, "stride_hold": args.stride_hold, "stride_answer": args.stride_answer, "n_rows": len(jobs),
        "n_tokens_saved": int(offsets[-1]), "seconds": round(time.time() - t0, 1),
        "note": "hidden_states[i] = output of decoder layer i-1; index 0 = embeddings"}, indent=2))
    print(f"[done] {len(jobs)} rows, {int(offsets[-1])} saved tokens, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
