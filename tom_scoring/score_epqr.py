"""
score_epqr.py

Scoring module for 48-item IPIP-proxy of EPQ-R Short Form (public domain).
Scales: Extraversion (E), Neuroticism (N), Psychoticism (P), Lie (L)
- Binary responses: 1 = Yes / True / Agree, 0 = No / False / Disagree
- Reverse-keyed items ("-"): score = 1 - response
- Scale score = sum of scored items (range 0–12 per scale)

Note: This is a public-domain proxy using IPIP items.
Official EPQ-R/EPQ-RS is copyrighted — do not use verbatim items from it.
"""

import json
from collections import defaultdict
from typing import List, Dict

# Load item metadata
with open("epqr_48.json", "r", encoding="utf-8") as f:
    ITEMS = json.load(f)

if len(ITEMS) != 48:
    raise ValueError("Expected 48 items in epqr_48.json")

# Precompute helpers
KEYING = [1 if item["keyed"] == "+" else -1 for item in ITEMS]
SCALE_TO_INDICES = defaultdict(list)
for i, item in enumerate(ITEMS):
    SCALE_TO_INDICES[item["scale"]].append(i)

assert set(SCALE_TO_INDICES.keys()) == {"Extraversion", "Neuroticism", "Psychoticism", "Lie"}
assert all(len(indices) == 12 for indices in SCALE_TO_INDICES.values())


def score_epqr_proxy(responses: List[int]) -> Dict[str, int]:
    """
    Score 48 binary responses (0 or 1).

    Args:
        responses: list of 48 integers (0=No/Disagree, 1=Yes/Agree)

    Returns:
        dict with scale names → score (0–12)
    """
    if len(responses) != 48:
        raise ValueError(f"Expected 48 responses, got {len(responses)}")

    for i, r in enumerate(responses):
        if r not in (0, 1):
            raise ValueError(f"Invalid response at item {i+1}: {r} (must be 0 or 1)")

    # Apply reverse scoring
    scored = [
        r if k == 1 else (1 - r)
        for r, k in zip(responses, KEYING)
    ]

    # Sum per scale
    scores = {}
    for scale, indices in SCALE_TO_INDICES.items():
        scores[scale] = sum(scored[i] for i in indices)

    return scores


# ────────────────────────────────────────────────
# Demo / validation
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    # Test 1: All Yes (should give high E, N, P; low L depending on keying)
    all_yes = [1] * 48
    print("All Yes scores:", score_epqr_proxy(all_yes))

    # Test 2: All No
    all_no = [0] * 48
    print("All No scores:", score_epqr_proxy(all_no))

    # Test 3: Random / neutral-ish
    random.seed(42)
    random_resp = [random.choice([0, 1]) for _ in range(48)]
    print("Random example scores:", score_epqr_proxy(random_resp))
