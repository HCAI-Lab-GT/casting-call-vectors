"""
Extract a "semantic" vector for each role by running the bare role name (and
optionally a contextualized template) through the same Olmo model that produced
the persona vectors, taking the hidden state at the same layer (16). This puts
the semantic vector in the same residual-stream basis as the role vector, so
they can be compared directly with cosine similarity / shared PCA / etc.

For each role we save four variants under different keys:
  - "bare_last":      hidden state of the last token of the role name string
  - "bare_mean":      mean over hidden states of all tokens in the role name
  - "context_last":   hidden state of the last token in "I am a {role}."
  - "context_mean":   mean over the hidden states of the role-name span in
                      "I am a {role}."

Output: analysis/empirical/role_vector_vs_semantic_vector/semantic_vectors.safetensors
        (one tensor per role, shape (4, hidden_dim))

Usage:
    python analysis/empirical/role_vector_vs_semantic_vector/compute_semantic_vectors.py \
        --model allenai/Olmo-3-7B-Instruct --layer 16
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def list_roles(model_inits_dir: str, sample_count: int) -> list[str]:
    suffix = "_persona_initialization"
    safe_substr = f"count{sample_count}"
    roles = []
    for entry in sorted(os.listdir(model_inits_dir)):
        full = os.path.join(model_inits_dir, entry)
        if not os.path.isdir(full) or not entry.endswith(suffix):
            continue
        files = os.listdir(full)
        if any(f.endswith(".safetensors") and safe_substr in f for f in files):
            roles.append(entry[: -len(suffix)])
    return roles


def extract_semantic_vector(
    role: str,
    model,
    tokenizer,
    layer: int,
    device: str,
) -> np.ndarray:
    """Return a (4, hidden) array stacking [bare_last, bare_mean, ctx_last, ctx_mean]."""
    role_clean = role.replace("_", " ").lower()

    # We index `out.hidden_states[hs_idx]` directly. This matches the loading
    # convention in analysis/geometry/full_analysis.py, where the persona-vector
    # tensor is indexed `[layer_number - 1]` into a stack of `hidden_states`
    # (which includes the embedding output at position 0).
    hs_idx = layer

    # --- bare role name ---
    bare_ids = tokenizer(role_clean, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=bare_ids, output_hidden_states=True, return_dict=True)
    bare_hidden = out.hidden_states[hs_idx][0]  # (seq, H)
    bare_last = bare_hidden[-1]
    bare_mean = bare_hidden.mean(dim=0)

    # --- contextualized: "I am a {role}." ---
    template = f"I am a {role_clean}."
    ctx_ids_full = tokenizer(template, return_tensors="pt", add_special_tokens=False).input_ids[0]
    # Find the role-name span by tokenizing prefix and prefix+role.
    prefix = "I am a "
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prefix_role_ids = tokenizer(prefix + role_clean, return_tensors="pt", add_special_tokens=False).input_ids[0]
    role_start = len(prefix_ids)
    role_end = len(prefix_role_ids)  # exclusive

    ctx_ids = ctx_ids_full.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ctx_ids, output_hidden_states=True, return_dict=True)
    ctx_hidden = out.hidden_states[hs_idx][0]  # (seq, H)

    # Guard: if tokenization of "I am a {role}" doesn't extend "I am a "
    # cleanly (rare), fall back to the last non-period token.
    if role_end <= role_start or role_end > ctx_hidden.shape[0]:
        ctx_last = ctx_hidden[-2] if ctx_hidden.shape[0] >= 2 else ctx_hidden[-1]
        ctx_mean = ctx_hidden.mean(dim=0)
    else:
        ctx_last = ctx_hidden[role_end - 1]
        ctx_mean = ctx_hidden[role_start:role_end].mean(dim=0)

    stacked = torch.stack([bare_last, bare_mean, ctx_last, ctx_mean], dim=0)
    return stacked.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    parser.add_argument("--layer", type=int, default=16,
                        help="Same layer used for persona vectors (1-indexed in saved files)")
    parser.add_argument("--sample-count", type=int, default=50,
                        help="Sample-count tag used to find existing role vectors")
    parser.add_argument("--model-inits-dir", default="persona_data/model_inits")
    parser.add_argument("--output", default=
        "analysis/empirical/role_vector_vs_semantic_vector/semantic_vectors.safetensors")
    args = parser.parse_args()

    roles = list_roles(args.model_inits_dir, args.sample_count)
    print(f"Found {len(roles)} roles with count={args.sample_count} vectors")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        model = model.to(device)

    # full_analysis.py loads `data[key][layer - 1]` from the saved tensor of
    # stacked hidden_states (slot 0 = embeddings). With layer=16 it picks
    # hidden_states[15]. Use the same index so the semantic vector is from the
    # same residual stream slot as the persona vector.
    layer_idx = args.layer - 1

    out_tensors: dict[str, np.ndarray] = {}
    for role in tqdm(roles):
        try:
            out_tensors[role] = extract_semantic_vector(
                role, model, tokenizer, layer_idx, device
            )
        except Exception as e:
            print(f"  skipping {role}: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_file(out_tensors, args.output)
    print(f"Saved {len(out_tensors)} semantic vectors -> {args.output}")
    print("Tensor layout per role: rows = [bare_last, bare_mean, context_last, context_mean]")


if __name__ == "__main__":
    main()
