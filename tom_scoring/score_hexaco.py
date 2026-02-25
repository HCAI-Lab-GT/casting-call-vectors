"""
score_hexaco.py

Scoring module for the HEXACO Personality Inventory – Revised (100-item version).
Follows the official scoring key: https://hexaco.org/downloads/ScoringKeys_100.pdf

Features:
- Loads item metadata from hexaco_100.json
- Applies reverse scoring (6 - response) for negatively keyed items
- Computes mean scores (1–5 scale) for:
  - 25 facets (4 items each)
  - 6 domains (average of 4 facet means)
  - Altruism (interstitial facet, reported separately)
- Input validation and neutral test case

Usage:
    from score_hexaco import score_hexaco_100

    # Example: neutral responses
    results = score_hexaco_100([3] * 100)
    print(results["domains"])
    print("Altruism:", results["altruism"])

Command-line:
    python score_hexaco.py --help
    python score_hexaco.py --neutral
    python score_hexaco.py --responses-file my_responses.json
"""

import json
from collections import defaultdict
from typing import List, Dict, Any
import argparse
import sys

# ────────────────────────────────────────────────
# Load item metadata (adjust path if necessary)
# ────────────────────────────────────────────────

try:
    with open("hexaco_100.json", "r", encoding="utf-8") as f:
        ITEMS: List[Dict[str, Any]] = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("hexaco_100.json not found in current directory")

if len(ITEMS) != 100:
    raise ValueError("hexaco_100.json must contain exactly 100 items")

# Precompute helpers
ITEM_KEYING = [1 if item["keyed"] == "+" else -1 for item in ITEMS]
ITEM_FACET  = [item["facet"] for item in ITEMS]
ITEM_DOMAIN = [item["domain"] for item in ITEMS]  # None for Altruism

# facet → list of 0-based indices (exactly 4 per facet)
facet_to_indices = defaultdict(list)
for i, facet in enumerate(ITEM_FACET):
    facet_to_indices[facet].append(i)

# domain → list of facets (4 per domain; Altruism excluded)
domain_to_facets = defaultdict(list)
for i, domain in enumerate(ITEM_DOMAIN):
    if domain:
        domain_to_facets[domain].append(ITEM_FACET[i])

# Verify structure
assert len(facet_to_indices) == 25, "Expected 25 facets"
assert len(domain_to_facets) == 6,  "Expected 6 domains"
assert all(len(indices) == 4 for indices in facet_to_indices.values()), "Each facet must have 4 items"


def score_hexaco_100(responses: List[int | float]) -> Dict[str, Dict[str, float]]:
    """
    Score a list of 100 HEXACO responses (1–5 Likert scale).

    Args:
        responses: list of 100 numbers (integers or floats) in [1, 5]

    Returns:
        dict with:
            "facets":    {facet_name: mean (1–5)}
            "domains":   {domain_name: mean (1–5)}
            "altruism":  mean score for Altruism facet (1–5)

    Raises:
        ValueError on invalid input length or values
    """
    if len(responses) != 100:
        raise ValueError(f"Expected exactly 100 responses, got {len(responses)}")

    # Validate range
    for i, r in enumerate(responses):
        if not isinstance(r, (int, float)) or not (1 <= r <= 5):
            raise ValueError(f"Invalid response at item {i+1}: {r} (must be 1.0–5.0)")

    # Apply reverse scoring
    scored = [
        r if key == 1 else (6 - r)
        for r, key in zip(responses, ITEM_KEYING)
    ]

    # Compute facet means
    facets = {}
    for facet, indices in facet_to_indices.items():
        values = [scored[i] for i in indices]
        facets[facet] = sum(values) / 4.0

    # Compute domain means (average of facet means)
    domains = {}
    for domain, facets_list in domain_to_facets.items():
        domain_values = [facets[f] for f in facets_list]
        domains[domain] = sum(domain_values) / len(domain_values)

    # Altruism (interstitial facet)
    altruism_mean = facets["Altruism"]

    return {
        "facets": facets,
        "domains": domains,
        "altruism": altruism_mean
    }


# ────────────────────────────────────────────────
# Command-line interface & demo
# ────────────────────────────────────────────────

def print_results(results: Dict):
    print("\nHEXACO-100 Scores (1–5 scale)\n")
    print("Domains:")
    for domain, score in sorted(results["domains"].items()):
        print(f"  {domain:20} : {score:.2f}")
    print(f"\nAltruism (interstitial)  : {results['altruism']:.2f}")
    print("\nSample facets (first 5):")
    for facet, score in list(results["facets"].items())[:5]:
        print(f"  {facet:20} : {score:.2f}")
    print("  ... (20 more facets)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score HEXACO-100 responses")
    parser.add_argument("--neutral", action="store_true",
                        help="Run with all-neutral (3) responses")
    parser.add_argument("--responses-file", type=str,
                        help="JSON file containing list of 100 numbers (1–5)")
    args = parser.parse_args()

    if args.neutral:
        responses = [3.0] * 100
        print("Scoring neutral responses (all 3.0) …")
        results = score_hexaco_100(responses)
        print_results(results)

    elif args.responses_file:
        try:
            with open(args.responses_file, "r", encoding="utf-8") as f:
                responses = json.load(f)
            print(f"Scoring responses from {args.responses_file} …")
            results = score_hexaco_100(responses)
            print_results(results)
        except Exception as e:
            print(f"Error loading file: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python score_hexaco.py --neutral")
        print("  python score_hexaco.py --responses-file my_responses.json")
