"""
score_ipip_neo.py

Scoring module for the 120-item IPIP-NEO personality inventory (Johnson, 2014).
Computes facet-level (30 facets) and domain-level (Big Five) scores from raw responses.

This implementation:
- Uses the item metadata from ipip_neo_120.json (120 items with domain, facet, keying)
- Applies standard IPIP reverse-scoring for negatively keyed items
- Returns raw sum scores:
  - Facets: sum of 4 items → range 4–20
  - Domains: sum of 24 items → range 24–120
- Follows the official IPIP scoring guidelines (ipip.ori.org)

Standard Input:
    responses: list of exactly 120 integers (1–5 Likert scale)
    Example:
    [3, 5, 1, 4, 2, 3, 4, 2, ...]  # 120 values

Standard Output:
    dict with two keys:
    - "facets":   dict of 30 facet names → integer sum (4–20)
    - "domains":  dict of 5 domain names → integer sum (24–120)

    Example output for neutral-ish responses (mostly 3s):
    {
        "facets": {
            "Anxiety": 12,
            "Anger": 11,
            "Depression": 13,
            # ... 27 more facets ...
            "Cautiousness": 12
        },
        "domains": {
            "Neuroticism": 72,
            "Extraversion": 72,
            "Openness": 72,
            "Agreeableness": 72,
            "Conscientiousness": 72
        }
    }

Usage:
    from score_ipip_neo import score_ipip_neo_120

    results = score_ipip_neo_120([3]*120)  # all neutral → ~72 per domain
    print(results["domains"]["Neuroticism"])  # e.g. 72

Requirements:
    - Python 3.6+
    - ipip_neo_120.json in the same directory (or adjust path)
"""

import json
from collections import defaultdict
from typing import List, Dict, Any

# Load item metadata once when module is imported
try:
    with open("ipip_neo_120.json", "r", encoding="utf-8") as f:
        ITEMS: List[Dict[str, Any]] = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("ipip_neo_120.json not found in current directory")

if len(ITEMS) != 120:
    raise ValueError("ipip_neo_120.json must contain exactly 120 items")

# Precompute helpers for efficiency (done once)
KEYING: List[int] = [1 if item["keyed"] == "+" else -1 for item in ITEMS]

# facet → list of 0-based item indices (exactly 4 per facet)
facet_to_indices: Dict[str, List[int]] = defaultdict(list)
# domain → list of 0-based item indices (exactly 24 per domain)
domain_to_indices: Dict[str, List[int]] = defaultdict(list)

for i, item in enumerate(ITEMS):
    facet = item["facet"]
    domain = item["domain"]
    facet_to_indices[facet].append(i)
    domain_to_indices[domain].append(i)

# Verify we have the expected structure
assert len(facet_to_indices) == 30, "Expected 30 facets"
assert len(domain_to_indices) == 5, "Expected 5 domains"
assert all(len(indices) == 4 for indices in facet_to_indices.values()), "Each facet must have 4 items"
assert all(len(indices) == 24 for indices in domain_to_indices.values()), "Each domain must have 24 items"


def score_ipip_neo_120(responses: List[int]) -> Dict[str, Dict[str, float]]:
    """
    Score a complete set of 120 IPIP-NEO-120 responses.

    Args:
        responses: List of 120 integers (each 1–5)

    Returns:
        dict with:
            "facets":   {facet_name: sum_score (4–20)}
            "domains":  {domain_name: sum_score (24–120)}

    Raises:
        ValueError: if input length or values are invalid
    """
    if len(responses) != 120:
        raise ValueError(f"Expected exactly 120 responses, got {len(responses)}")

    # Validate all responses are integers 1–5
    for i, resp in enumerate(responses):
        if not isinstance(resp, int) or not (1 <= resp <= 5):
            raise ValueError(f"Invalid response at position {i+1}: {resp} (must be int 1–5)")

    # Step 1: Apply reverse scoring where needed
    scored_items: List[int] = []
    for i, response in enumerate(responses):
        if KEYING[i] == 1:
            scored_items.append(response)
        else:
            scored_items.append(6 - response)

    # Step 2: Compute facet scores (sum of 4 scored items each)
    facets: Dict[str, int] = {}
    for facet, indices in facet_to_indices.items():
        facets[facet] = sum(scored_items[i] for i in indices)

    # Step 3: Compute domain scores (sum of 24 scored items each)
    domains: Dict[str, int] = {}
    for domain, indices in domain_to_indices.items():
        domains[domain] = sum(scored_items[i] for i in indices)

    # If you prefer domain scores as averages (1–5 scale), uncomment:
    # domains = {d: round(sum(facets[f] for f in facets if f in [item["facet"] for item in ITEMS if item["domain"] == d]) / 6, 2)
    #            for d in domains}

    return {
        "facets": facets,
        "domains": domains
    }


# Quick self-test / demo when run directly
if __name__ == "__main__":
    # Example: all neutral (3) responses → should get ~12 per facet, ~72 per domain
    neutral_responses = [3] * 120
    try:
        results = score_ipip_neo_120(neutral_responses)
        print("Neutral responses test:")
        print("Domain scores:", results["domains"])
        print("Sample facets:", dict(list(results["facets"].items())[:5]), "...")
    except Exception as e:
        print("Test failed:", e)