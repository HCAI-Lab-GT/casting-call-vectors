#!/usr/bin/env python
"""RIASEC steering eval via logprob gap: log P(YES) - log P(NO) at first assistant token."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoTokenizer
from pvx import setup_logging
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel

logger = setup_logging(name="marin-eval-logprob-riasec")

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
def _parse_alphas(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected --alphas as a comma-separated string, e.g. '-3,0,3'")
    return [float(p) for p in parts]
def _messages_for_characteristic(characteristic: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Output EXACTLY one token: YES or NO."},
        {"role": "user", "content": f"{characteristic}"},
    ]
def _single_token_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids if len(ids) == 1 else []
def _find_yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    yes_groups = [["YES", " YES"], ["yes", " yes"], ["Yes", " Yes"]]
    no_groups = [["NO", " NO"], ["no", " no"], ["No", " No"]]
    logger.info(
        "YES candidates encodings: %s",
        {c: tokenizer.encode(c, add_special_tokens=False) for g in yes_groups for c in g},
    )
    logger.info(
        "NO candidates encodings: %s",
        {c: tokenizer.encode(c, add_special_tokens=False) for g in no_groups for c in g},
    )

    def collect(cands: list[str]) -> list[int]:
        return sorted({i for c in cands for i in _single_token_ids(tokenizer, c)})

    def pick(label: str, groups: list[list[str]]) -> list[int]:
        for cands in groups:
            ids = collect(cands)
            if ids:
                logger.info("%s variants: %s", label, cands)
                logger.info("%s token ids: %s", label, ids)
                return ids
        raise RuntimeError(f"Could not find single-token {label} variants. Extend candidate strings.")

    return pick("YES", yes_groups), pick("NO", no_groups)
def _detect_middle_layer(model_id: str) -> int:
    return int(AutoConfig.from_pretrained(model_id).num_hidden_layers) // 2
def _last_token_logits(outputs) -> torch.Tensor:
    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    return logits[0, -1, :]
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RIASEC steering via YES/NO logprob gap.")
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--trait", type=str, default="all")
    ap.add_argument("--alphas", type=str, default="-3,-1,0,1,3")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default="outputs/riasec_eval")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = _repo_root()
    riasec_path = root / "configs/riasec.yaml"
    with open(riasec_path, "r") as f:
        riasec = yaml.safe_load(f)
    traits = sorted(riasec.keys()) if args.trait == "all" else [args.trait]
    unknown = [t for t in traits if t not in riasec]
    if unknown:
        raise ValueError(f"Unknown trait(s): {unknown}. Expected one of: {sorted(riasec.keys())} or 'all'")
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = _parse_alphas(args.alphas)
    if args.dry_run:
        tok = AutoTokenizer.from_pretrained(args.model_id)
        yes_ids, no_ids = _find_yes_no_token_ids(tok)
        print(f"token_ids.yes={yes_ids} token_ids.no={no_ids}\n")
        for trait in traits:
            chars = riasec[trait].get("characteristics", []) or []
            if not chars:
                logger.warning("Trait '%s' has empty characteristics list. Skipping.", trait)
                continue
            for c in chars:
                messages = _messages_for_characteristic(c)
                formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print(f"[{trait}] {c}\n{formatted}\n{'-'*60}\n")
        return
    layer = args.layer if args.layer is not None else _detect_middle_layer(args.model_id)
    all_results: dict[str, dict] = {}
    token_ids: dict[str, list[int]] | None = None
    for trait in traits:
        chars = riasec[trait].get("characteristics", []) or []
        if not chars:
            logger.warning("Trait '%s' has empty characteristics list. Skipping.", trait)
            continue
        model = RIASECPersonaModel.load_or_create(target_model_id=args.model_id, trait=trait, layer=layer)
        model.model.eval()
        yes_ids, no_ids = _find_yes_no_token_ids(model.tokenizer)
        token_ids = {"yes": yes_ids, "no": no_ids}
        trait_out: dict[str, dict] = {"alphas": {}}
        for alpha in alphas:
            rows: list[dict[str, float | str]] = []
            for c in chars:
                msgs = _messages_for_characteristic(c)
                formatted = model.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                enc = model.tokenizer(formatted, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                with model._steering_delta(alpha), torch.no_grad():
                    out = model.model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                        use_cache=False,
                        return_dict=True,
                    )
                lp = torch.log_softmax(_last_token_logits(out).float(), dim=-1)
                lp_yes = float(lp[yes_ids].max().item())
                lp_no = float(lp[no_ids].max().item())
                rows.append({"text": c, "logprob_yes": lp_yes, "logprob_no": lp_no, "gap": lp_yes - lp_no})
            gaps = np.asarray([r["gap"] for r in rows], dtype=np.float64)
            trait_out["alphas"][f"{alpha:g}"] = {
                "mean_gap": float(gaps.mean()),
                "std_gap": float(gaps.std(ddof=0)),
                "characteristics": rows,
            }
        all_results[trait] = trait_out
        model.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    safe_model = args.model_id.replace("/", "__").replace(":", "_")
    out_path = output_dir / f"{safe_model}_logprob_eval.json"
    out = {"model_id": args.model_id, "results": all_results, "token_ids": token_ids or {"yes": [], "no": []}}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved: %s", str(out_path))
if __name__ == "__main__":
    main()
