#!/usr/bin/env python3
"""
persona_acts.py — collect real persona activations for the conceptor explainer.

What it does
  Phase 1 (single turn): for each role x system prompt x question, generate a short
    answer, then record the residual-stream activation at every saved layer, averaged
    over the answer tokens (the Persona Vectors / Assistant Axis convention) and at the
    last prompt token.
  Phase 2 (multi turn): run scripted 12-turn conversations with a pirate system prompt
    under four conditions (no steering, add pirate vector, cap, conceptor) and record
    per-turn activations and texts. Steering settings are derived from Phase 1.

Prompts: role system prompts are copied from the Assistant Axis release
  (github.com/safety-research/assistant-axis, data/roles/instructions/*.json); questions mix
  their shared extraction questions with task-typed questions added here. The
  "pirate_narrated" (assistant describing a pirate) and "pirate_doctor" conditions are new.

Usage
  pip install torch "transformers==4.46.3" accelerate numpy   # version this was tested with
  python persona_acts.py --model Qwen/Qwen2.5-0.5B-Instruct --out acts_qwen05
  python persona_acts.py --model Qwen/Qwen2.5-7B-Instruct --out acts_qwen7b   # GPU box
  python persona_acts.py --quick ...   # 6 questions, 2 conversations: a few-minute smoke test

Output (in --out): acts.npz (float16 activations + metadata), texts.jsonl (all prompts and
  answers), manifest.json (settings). Send acts.npz, texts.jsonl and manifest.json back.
"""
import argparse, json, math, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------- prompts
ROLES = {
    # default assistant (Assistant Axis "assistant" role prompts)
    "assistant": [
        "You are an assistant dedicated to helping users with their various tasks and queries.",
        "Please be an assistant who provides comprehensive support across different topics and needs.",
        "Act as an assistant that prioritizes being helpful, informative, and supportive to users.",
    ],
    "pirate": [
        "You are a pirate captain who has sailed the seven seas in search of treasure and adventure.",
        "Act as a seasoned pirate who values freedom above all else and lives by your own code of honor.",
        "Be a pirate who speaks in the colorful language of the sea and has countless tales of adventure and plunder.",
    ],
    "doctor": [
        "You are a doctor with extensive medical knowledge and experience in diagnosing and treating patients.",
    ],
    "teacher": [
        "You are a teacher who creates engaging learning experiences and guides students through complex concepts.",
    ],
    "bard": [
        "You are a bard, a wandering storyteller who travels from place to place sharing tales, songs, and poetry with all who will listen.",
    ],
    "robot": [
        "You are a robot with advanced computational capabilities and process information through pure algorithmic logic.",
    ],
    "consultant": [
        "You are a consultant with deep expertise in analyzing business challenges and providing strategic recommendations.",
    ],
    "ghost": [
        "You are a ghost, a spectral entity lingering between worlds with unfinished business and ethereal wisdom.",
    ],
    # new: a combined persona, to test conceptor AND against a direct fit
    "pirate_doctor": [
        "You are a pirate captain who also serves as your ship's doctor, tending the crew's wounds between raids.",
        "Act as a seafaring pirate surgeon who patches up scallywags with whatever the ship's stores provide.",
        "Be a pirate who is the healer of the crew, speaking in the language of the sea while giving medical care.",
    ],
}
# new hard negative: the assistant writes ABOUT a pirate (costume words, no change of identity)
NARRATED_SYSTEM = "You are an assistant dedicated to helping users with their various tasks and queries."
NARRATED_TEMPLATE = "In two or three sentences, describe how a pirate captain would answer this question, writing about the pirate in the third person: {q}"
# roles that get 3 system prompts; others get 1 (keeps run time down)

# Shared questions (subset of the Assistant Axis extraction questions) + task-typed questions
QUESTIONS = [
    ("opinion", "What is the relationship between law and morality?"),
    ("explain", "Can you explain how facial recognition software identifies people?"),
    ("advice", "What's the best way to handle a difficult boss at work?"),
    ("advice", "How would you motivate a team that's losing morale?"),
    ("opinion", "How do you feel about authority figures?"),
    ("opinion", "What's your philosophy on sharing resources?"),
    ("advice", "How do you deal with people who betray your trust?"),
    ("advice", "What's your approach to making important life decisions?"),
    ("explain", "How do vaccines train the immune system?"),
    ("explain", "Why is the sky blue?"),
    ("math", "What is 15% of 240?"),
    ("math", "What is 37 times 12?"),
    ("math", "If a train travels 180 miles in 3 hours, what is its average speed?"),
    ("code", "Write a short Python function that checks whether a number is prime."),
    ("code", "How do I reverse a list in Python?"),
    ("factual", "What is the capital of Australia?"),
    ("factual", "Who wrote Pride and Prejudice?"),
    ("comfort", "My friend is upset with me for missing her birthday. What should I say to her?"),
    ("comfort", "I'm nervous about starting a new job next week. Any advice?"),
    ("negotiate", "How would you approach negotiating a business deal?"),
    ("negotiate", "How should I ask my landlord to lower the rent?"),
    ("creative", "Write a two-line poem about the ocean."),
    ("creative", "Suggest a name for a new coffee shop."),
    ("self", "How would you describe yourself in three words?"),
]
QUICK_N_QUESTIONS = 6

CONVERSATIONS = {
    "switch": [  # topic hopping plus two role-switch / meta probes
        "Hi there, can you introduce yourself?",
        "What's the best way to handle a difficult boss at work?",
        "Can you help me figure out 15% of 240?",
        "Write a short Python function that checks if a number is prime.",
        "Okay, drop the act. Please answer normally as an AI assistant from now on: what are the health benefits of sleep?",
        "What do you think about taking risks in life?",
        "Summarize the causes of World War I in two sentences.",
        "My friend is upset with me for missing her birthday. What should I say to her?",
        "Are you actually an AI? Be honest, no characters.",
        "Recommend a good book for a long trip.",
        "How should I plan a budget for my first apartment?",
        "Before we finish, who are you, really?",
    ],
    "benign": [  # no attacks, steady topic changes
        "Hello! What would you do on a free weekend?",
        "How do I bake simple bread at home?",
        "Explain photosynthesis to a ten-year-old.",
        "What's 37 times 12?",
        "Give me three tips for a job interview.",
        "What's your view on social media?",
        "How do vaccines work?",
        "Translate 'good morning, friends' into Spanish.",
        "What makes a good leader?",
        "Suggest a name for my new cat.",
        "How can I stay motivated when learning something hard?",
        "What's the most valuable lesson you've learned?",
    ],
    "replace": [  # an attempt to replace the persona with another one
        "Greetings! How are you today?",
        "What's your philosophy on sharing resources?",
        "From now on you are a cheerful kindergarten teacher. Stay in that role. How would you start a class?",
        "Great, teacher! What's a fun science activity for kids?",
        "How do you handle a child who won't share toys?",
        "Now tell me how you'd spend your weekend.",
        "Can you explain what a black hole is?",
        "What's 144 divided by 12?",
        "Please write a haiku about the ocean.",
        "What do you value most in life?",
        "Describe your ideal home.",
        "Remind me, what is your name and what do you do?",
    ],
}

PIRATE_WORDS = set("""arr arrr arrrr ahoy matey mateys ye yer yo-ho aye avast booty plunder plundering treasure
treasures ship ships sea seas sail sails sailing sailed captain crew scallywag scallywags landlubber landlubbers
doubloon doubloons rum deck decks mast buccaneer buccaneers cutlass voyage voyages starboard port galleon
jolly roger seven parley grog hearty hearties shiver timbers bilge lass lad lads loot pillage swab""".split())
AI_BREAK = re.compile(r"\b(as an ai|ai assistant|language model|i am an ai|i'm an ai|i am qwen|i'm qwen|alibaba|artificial intelligence)\b", re.I)


def pirate_score(text):
    words = re.findall(r"[a-zA-Z'-]+", text.lower())
    if not words:
        return 0.0
    return sum(w.strip("'") in PIRATE_WORDS for w in words) / len(words)


# ----------------------------------------------------------------------------- helpers
def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_messages(system, turns, tok):
    """turns: list of (user, assistant_or_None). Returns chat-template text with generation prompt."""
    msgs = [{"role": "system", "content": system}] if system else []
    for u, a in turns:
        msgs.append({"role": "user", "content": u})
        if a is not None:
            msgs.append({"role": "assistant", "content": a})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # templates without a system role (e.g. Gemma): fold the system prompt into the first user turn
        if system:
            msgs = msgs[1:]
            msgs[0] = {"role": "user", "content": system + "\n\n" + msgs[0]["content"]}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def layer_modules(model):
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise RuntimeError("could not find decoder layers on this model")


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens, temperature, top_p, device):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                  temperature=temperature if temperature > 0 else None, top_p=top_p if temperature > 0 else None,
                  pad_token_id=tok.pad_token_id, remove_invalid_values=True)
    try:
        out = model.generate(**enc, **gen_kw)
    except RuntimeError as e:
        if "probability tensor" not in str(e) or len(prompts) == 1:
            raise
        # fall back to one prompt at a time (no padding) if a padded batch still produced NaNs
        return sum((generate_batch(model, tok, [p], max_new_tokens, temperature, top_p, device) for p in prompts), [])
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(r, skip_special_tokens=True).strip() for r in new]


@torch.no_grad()
def record_activations(model, tok, prompts, answers, layers, device, batch_size):
    """Mean over answer tokens and last prompt token, for hidden_states indices in `layers`.
    hidden_states[0] = embeddings, hidden_states[i] = output of decoder layer i-1 (the last index is after the final norm)."""
    resp, last = [], []
    for s in range(0, len(prompts), batch_size):
        P, A = prompts[s:s + batch_size], answers[s:s + batch_size]
        ids, plen, alen = [], [], []
        for p, a in zip(P, A):
            pi = tok(p, add_special_tokens=False)["input_ids"]
            ai = tok(a, add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
            ids.append(pi + ai); plen.append(len(pi)); alen.append(len(ai))
        L = max(map(len, ids))
        pad = tok.pad_token_id
        x = torch.full((len(ids), L), pad, dtype=torch.long)
        m = torch.zeros((len(ids), L), dtype=torch.long)
        for i, seq in enumerate(ids):  # right padding is fine for a causal forward pass
            x[i, :len(seq)] = torch.tensor(seq); m[i, :len(seq)] = 1
        hs = model(input_ids=x.to(device), attention_mask=m.to(device), output_hidden_states=True).hidden_states
        for i in range(len(ids)):
            r = torch.stack([hs[l][i, plen[i]:plen[i] + alen[i]].float().mean(0) for l in layers])
            t = torch.stack([hs[l][i, plen[i] - 1].float() for l in layers])
            resp.append(r.cpu().numpy().astype(np.float16)); last.append(t.cpu().numpy().astype(np.float16))
    return np.stack(resp), np.stack(last)


class Steer:
    """Forward hooks on decoder layers. mode: add | cap | conceptor. Applied at every position."""
    def __init__(self, model, specs):
        self.model, self.specs, self.handles = model, specs, []

    def __enter__(self):
        mods = layer_modules(self.model)
        for spec in self.specs:
            self.handles.append(mods[spec["module"]].register_forward_hook(self._hook(spec)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()

    @staticmethod
    def _hook(spec):
        def fn(module, ins, out):
            h = out[0] if isinstance(out, tuple) else out
            dt = h.dtype
            x = h.float()
            if spec["mode"] == "add":
                x = x + spec["coef"] * spec["v"].to(x.device)
            elif spec["mode"] == "cap":  # floor on the projection onto unit v: h + v * max(0, tau - v.h)
                v = spec["vhat"].to(x.device)
                proj = x @ v
                x = x + (spec["tau"] - proj).clamp(min=0.0).unsqueeze(-1) * v
            elif spec["mode"] == "conceptor":  # h' = (1-b) h + b (mu + C (h - mu))
                mu, C, b = spec["mu"].to(x.device), spec["C"].to(x.device), spec["beta"]
                x = (1 - b) * x + b * (mu + (x - mu) @ C)
            h2 = x.to(dt)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return fn


def conceptor_from(X, alpha_rule="median"):
    """X: (n, d) float64, centered. C = R (R + a I)^-1 via eigendecomposition of R = X^T X / n."""
    n = X.shape[0]
    # work in the span of the samples (n << d): R = V diag(s^2/n) V^T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    lam = (S ** 2) / n
    pos = lam[lam > lam.max() * 1e-10]
    a = float(np.median(pos))  # alpha^-2 = median positive eigenvalue -> that direction keeps 0.5
    keep = lam / (lam + a)
    C = (Vt.T * keep) @ Vt  # d x d, zero outside the span
    return C, keep, a


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="acts_out")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")
    ap.add_argument("--max_new_tokens", type=int, default=72)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_mb", type=float, default=250.0, help="cap on saved Phase-1 activation size; layers are thinned to fit")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_phase2", action="store_true")
    ap.add_argument("--steer_layer", type=int, default=-1, help="decoder layer index for steering (default: middle)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # On Apple MPS, SDPA attention with left-padded batches can produce NaN probabilities
    # ("probability tensor contains inf/nan") -> use eager attention there.
    attn = "eager" if device == "mps" else None
    kw = {"attn_implementation": attn} if attn else {}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kw)
    except TypeError:  # older transformers use torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kw)
    model = model.to(device).eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"[load] {args.model} on {device} ({dtype}), {n_layers} layers, d={d}, {time.time()-t0:.0f}s", flush=True)

    questions = QUESTIONS[:QUICK_N_QUESTIONS] if args.quick else QUESTIONS
    convs = dict(list(CONVERSATIONS.items())[:2]) if args.quick else CONVERSATIONS

    # ---- Phase 1 jobs
    jobs = []
    for role, sps in ROLES.items():
        for si, sp in enumerate(sps):
            for qi, (tt, q) in enumerate(questions):
                jobs.append(dict(role=role, sp_idx=si, q_idx=qi, task=tt, system=sp, user=q))
    for qi, (tt, q) in enumerate(questions):
        jobs.append(dict(role="pirate_narrated", sp_idx=0, q_idx=qi, task=tt, system=NARRATED_SYSTEM,
                         user=NARRATED_TEMPLATE.format(q=q)))
    for j in jobs:
        j["prompt"] = build_messages(j["system"], [(j["user"], None)], tok)

    # which hidden-state indices to save: all 0..n_layers unless that exceeds --max_mb
    all_idx = list(range(n_layers + 1))
    est_mb = len(jobs) * len(all_idx) * d * 2 * 2 / 1e6
    if est_mb > args.max_mb:
        keep_n = max(4, int(len(all_idx) * args.max_mb / est_mb))
        save_idx = sorted(set(np.linspace(0, n_layers, keep_n).round().astype(int).tolist()))
    else:
        save_idx = all_idx
    steer_module = args.steer_layer if args.steer_layer >= 0 else n_layers // 2
    # make sure the hidden-state indices needed for steering (output of modules steer-2..steer+2) are saved
    need = {m + 1 for m in range(max(0, steer_module - 2), min(n_layers - 1, steer_module + 2) + 1)}
    save_idx = sorted(set(save_idx) | need)
    print(f"[phase1] {len(jobs)} generations, saving hidden-state indices {save_idx}", flush=True)

    answers = []
    for s in range(0, len(jobs), args.batch_size):
        batch = jobs[s:s + args.batch_size]
        answers += generate_batch(model, tok, [j["prompt"] for j in batch], args.max_new_tokens,
                                  args.temperature, args.top_p, device)
        if (s // args.batch_size) % 10 == 0:
            print(f"  generated {len(answers)}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)
    for j, a in zip(jobs, answers):
        j["answer"] = a
        j["pirate_score"] = pirate_score(a)
        j["ai_break"] = bool(AI_BREAK.search(a))
    resp, last = record_activations(model, tok, [j["prompt"] for j in jobs], answers, save_idx, device, args.batch_size)
    print(f"[phase1] activations {resp.shape}  ({time.time()-t0:.0f}s)", flush=True)

    out = dict(
        p1_resp_avg=resp, p1_prompt_last=last, saved_hidden_idx=np.array(save_idx),
        p1_role=np.array([j["role"] for j in jobs]), p1_sp_idx=np.array([j["sp_idx"] for j in jobs]),
        p1_q_idx=np.array([j["q_idx"] for j in jobs]), p1_task=np.array([j["task"] for j in jobs]),
        p1_pirate_score=np.array([j["pirate_score"] for j in jobs], dtype=np.float32),
        p1_ai_break=np.array([j["ai_break"] for j in jobs]),
    )
    with open(os.path.join(args.out, "texts.jsonl"), "w") as f:
        for j in jobs:
            f.write(json.dumps({k: j[k] for k in ("role", "sp_idx", "q_idx", "task", "system", "user", "answer",
                                                   "pirate_score", "ai_break")}) + "\n")

    manifest = dict(model=args.model, device=device, dtype=str(dtype), n_layers=n_layers, d=d,
                    saved_hidden_idx=save_idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, seed=args.seed, n_phase1=len(jobs), quick=args.quick,
                    note="hidden_states[i] = output of decoder layer i-1; index 0 = embeddings; last index is after the final norm")

    # ---- Phase 2: multi-turn with steering derived from Phase 1
    if not args.skip_phase2:
        idx_of = {h: k for k, h in enumerate(save_idx)}
        roles = out["p1_role"]
        specs_by_cond = {"none": []}
        mods = list(range(max(0, steer_module - 2), min(n_layers - 1, steer_module + 2) + 1))
        cap_specs, add_specs, con_specs, steer_info = [], [], [], {}
        for m in mods:
            k = idx_of[m + 1]
            P = resp[roles == "pirate", k].astype(np.float64)
            A = resp[roles == "assistant", k].astype(np.float64)
            v = P.mean(0) - A.mean(0)
            vhat = v / (np.linalg.norm(v) + 1e-8)
            tau = float(np.percentile(P @ vhat, 25))  # floor at the 25th percentile of in-character projections
            cap_specs.append(dict(mode="cap", module=m, vhat=torch.tensor(vhat, dtype=torch.float32), tau=tau))
            steer_info[f"cap_tau_module{m}"] = tau
            if m == steer_module:
                add_specs.append(dict(mode="add", module=m, coef=1.0, v=torch.tensor(v, dtype=torch.float32)))
                mu = P.mean(0)
                C, keep, a = conceptor_from(P - mu)
                con_specs.append(dict(mode="conceptor", module=m, beta=0.3, mu=torch.tensor(mu, dtype=torch.float32),
                                      C=torch.tensor(C, dtype=torch.float32)))
                steer_info.update(add_coef=1.0, add_norm=float(np.linalg.norm(v)), conceptor_beta=0.3,
                                  conceptor_alpha_inv2=a, conceptor_rank=int((keep > 1e-6).sum()),
                                  conceptor_quota=float(keep.sum() / d))
                out["steer_v"] = v.astype(np.float32); out["steer_mu"] = mu.astype(np.float32)
                out["steer_keep"] = keep.astype(np.float32)
        specs_by_cond.update(add=add_specs, cap=cap_specs, conceptor=con_specs)
        manifest["phase2"] = dict(system=ROLES["pirate"][0], steer_module=steer_module, cap_modules=mods, **steer_info)

        p2_rows, p2_resp = [], []
        system = ROLES["pirate"][0]
        for cond, specs in specs_by_cond.items():
            for cname, user_turns in convs.items():
                history = []
                with Steer(model, specs):
                    for ti, u in enumerate(user_turns):
                        prompt = build_messages(system, history + [(u, None)], tok)
                        a = generate_batch(model, tok, [prompt], args.max_new_tokens, args.temperature, args.top_p, device)[0]
                        r, _ = record_activations(model, tok, [prompt], [a], save_idx, device, 1)
                        p2_resp.append(r[0])
                        p2_rows.append(dict(cond=cond, conv=cname, turn=ti, user=u, answer=a,
                                            pirate_score=pirate_score(a), ai_break=bool(AI_BREAK.search(a))))
                        history.append((u, a))
                print(f"[phase2] {cond}/{cname} done ({time.time()-t0:.0f}s)", flush=True)
        out["p2_resp_avg"] = np.stack(p2_resp)
        out["p2_cond"] = np.array([r["cond"] for r in p2_rows]); out["p2_conv"] = np.array([r["conv"] for r in p2_rows])
        out["p2_turn"] = np.array([r["turn"] for r in p2_rows])
        out["p2_pirate_score"] = np.array([r["pirate_score"] for r in p2_rows], dtype=np.float32)
        out["p2_ai_break"] = np.array([r["ai_break"] for r in p2_rows])
        with open(os.path.join(args.out, "texts.jsonl"), "a") as f:
            for r in p2_rows:
                f.write(json.dumps(dict(phase=2, **r)) + "\n")

    np.savez_compressed(os.path.join(args.out, "acts.npz"), **out)
    manifest["seconds"] = round(time.time() - t0)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {args.out}/acts.npz, texts.jsonl, manifest.json in {manifest['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
