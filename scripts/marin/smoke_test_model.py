#!/usr/bin/env python
"""
Smoke test: load a model, verify architecture, run a forward pass, check activation shapes.

Usage:
  python scripts/marin/smoke_test_model.py
  python scripts/marin/smoke_test_model.py --model_id marin-community/marin-8b-instruct
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Expected specs per model family
EXPECTED_SPECS = {
    "meta-llama/Llama-3.2-1B-Instruct": {"layers": 16, "hidden_dim": 2048},
    "marin-community/marin-8b-instruct": {"layers": 32, "hidden_dim": 4096},
}


def smoke_test(model_id: str):
    print(f"=== Smoke Test: {model_id} ===\n")

    # Load tokenizer
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    # Test chat template
    print("\n[2/5] Testing chat template...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"  Chat template works. Formatted length: {len(formatted)} chars")
        print(f"  First 200 chars: {formatted[:200]!r}")
    except Exception as e:
        print(f"  WARN: Chat template failed: {e}")
        print("  Falling back to simple concatenation for testing.")
        formatted = "Hello, who are you?"

    # Load model
    print("\n[3/5] Loading model...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        model = model.to(device)
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")

    # Check architecture (model.model.layers for LLaMA-family)
    print("\n[4/5] Verifying architecture...")
    ok = True
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        num_layers = len(layers)
        hidden_dim = model.config.hidden_size
        print(f"  Architecture: LLaMA-family (model.model.layers)")
        print(f"  Num layers: {num_layers}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Middle layer: {num_layers // 2}")

        if model_id in EXPECTED_SPECS:
            expected = EXPECTED_SPECS[model_id]
            if num_layers != expected["layers"]:
                print(f"  FAIL: Expected {expected['layers']} layers, got {num_layers}")
                ok = False
            if hidden_dim != expected["hidden_dim"]:
                print(f"  FAIL: Expected hidden_dim={expected['hidden_dim']}, got {hidden_dim}")
                ok = False
            if ok:
                print("  Architecture matches expected specs.")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        print(f"  Architecture: GPT-2 family (model.transformer.h)")
        print(f"  Num layers: {len(model.transformer.h)}")
    else:
        print("  WARN: Unrecognized architecture layout.")
        ok = False

    # Forward pass with hidden states
    print("\n[5/5] Running forward pass...")
    enc = tokenizer(formatted, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    print(f"  Number of hidden state layers: {len(hidden_states)} (embedding + {len(hidden_states)-1} decoder layers)")
    print(f"  Input shape: {enc['input_ids'].shape}")
    print(f"  Hidden state [0] shape: {hidden_states[0].shape}")
    print(f"  Hidden state [-1] shape: {hidden_states[-1].shape}")
    print(f"  Logits shape: {outputs.logits.shape}")

    # Verify shapes
    seq_len = enc["input_ids"].shape[1]
    expected_shape = (1, seq_len, model.config.hidden_size)
    if hidden_states[-1].shape != expected_shape:
        print(f"  FAIL: Expected hidden state shape {expected_shape}, got {hidden_states[-1].shape}")
        ok = False
    else:
        print(f"  Hidden state shape matches: {expected_shape}")

    # Summary
    print("\n=== Result ===")
    if ok:
        print("PASS: All checks passed.")
    else:
        print("FAIL: Some checks failed (see above).")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Smoke test a model.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    smoke_test(args.model_id)


if __name__ == "__main__":
    main()
