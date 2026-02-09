#!/usr/bin/env python
"""
Semantic analysis of qualitative steering responses.

Uses sentence-transformers to embed responses and RIASEC trait descriptions,
then checks whether trait-X steered responses are more semantically similar
to trait-X's description than to other traits.

This bridges the gap between:
- The coarse YES/NO logprob specificity test (shows non-specificity)
- The clearly visible qualitative differences (show personality differentiation)
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]


def main():
    # Load RIASEC descriptions
    with open("configs/riasec.yaml") as f:
        riasec = yaml.safe_load(f)

    trait_descriptions = {}
    trait_characteristics_text = {}
    for trait in TRAITS:
        trait_descriptions[trait] = riasec[trait]["description"]
        trait_characteristics_text[trait] = " ".join(riasec[trait]["characteristics"])

    # Load qualitative data
    qual_files = list(Path("outputs/qualitative").glob("*_qualitative_steering.json"))

    # Load sentence transformer model (light model, runs on CPU)
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Embed trait descriptions and characteristics
    desc_texts = [trait_descriptions[t] for t in TRAITS]
    char_texts = [trait_characteristics_text[t] for t in TRAITS]
    desc_embeddings = model.encode(desc_texts, normalize_embeddings=True)
    char_embeddings = model.encode(char_texts, normalize_embeddings=True)

    for qf in sorted(qual_files):
        with open(qf) as f:
            data = json.load(f)

        model_id = data["model_id"]
        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print(f"{'='*70}")

        # Analyze residual conditions
        conditions = ["baseline"] + [f"residual_{t}" for t in TRAITS] + ["shared_only"]

        # For each condition, embed all 8 responses
        for target in ["description", "characteristics"]:
            ref_embeddings = desc_embeddings if target == "description" else char_embeddings

            print(f"\n--- Similarity to trait {target}s ---")
            print(f"{'Condition':>30} ", end="")
            for t in TRAITS:
                print(f"{t[:5]:>8}", end="")
            print(f"{'Match?':>8}")

            for cond in conditions:
                if cond not in data["conditions"]:
                    continue

                responses = [r["response"] for r in data["conditions"][cond]]
                resp_embeddings = model.encode(responses, normalize_embeddings=True)

                # Mean embedding for all responses under this condition
                mean_resp = resp_embeddings.mean(axis=0)
                mean_resp = mean_resp / np.linalg.norm(mean_resp)

                # Cosine similarity to each trait
                sims = mean_resp @ ref_embeddings.T

                # Check if matching trait has highest similarity
                matched_trait = cond.replace("residual_", "").replace("original_", "")
                best_trait = TRAITS[np.argmax(sims)]
                match = "✓" if matched_trait == best_trait and matched_trait in TRAITS else ""

                print(f"{cond:>30} ", end="")
                for s in sims:
                    print(f"{s:8.3f}", end="")
                print(f"{'  ' + match:>8}")

        # Compute specificity index using semantic similarity
        print(f"\n--- Semantic specificity index ---")
        for target in ["description", "characteristics"]:
            ref_embeddings = desc_embeddings if target == "description" else char_embeddings

            diagonal_sims = []
            off_diagonal_sims = []

            for t_idx, trait in enumerate(TRAITS):
                cond = f"residual_{trait}"
                if cond not in data["conditions"]:
                    continue

                responses = [r["response"] for r in data["conditions"][cond]]
                resp_embeddings = model.encode(responses, normalize_embeddings=True)
                mean_resp = resp_embeddings.mean(axis=0)
                mean_resp = mean_resp / np.linalg.norm(mean_resp)

                sims = mean_resp @ ref_embeddings.T
                diagonal_sims.append(sims[t_idx])
                off_diagonal_sims.extend([sims[j] for j in range(6) if j != t_idx])

            diag_mean = np.mean(diagonal_sims)
            off_diag_mean = np.mean(off_diagonal_sims)

            print(f"  {target:15s}: diagonal={diag_mean:.4f}, off-diagonal={off_diag_mean:.4f}, "
                  f"diff={diag_mean - off_diag_mean:+.4f}")

    # Save results
    results = {"analysis": "semantic_specificity"}
    out_path = Path("outputs/analysis/qualitative_semantic_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
