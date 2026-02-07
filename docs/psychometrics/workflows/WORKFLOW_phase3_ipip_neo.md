# Phase 3 Workflow: IPIP-NEO (Big Five) Implementation

> **Status**: Ready for Implementation
>
> **Priority**: MEDIUM
>
> **Reference**: [PSYCHOMETRICS_DATA.md § Phase 3](../reference/PSYCHOMETRICS_DATA.md#phase-3-ipip-neo-big-five)
>
> **Last Updated**: 2025-01-25

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why IPIP-NEO for LM-VECTOR](#2-why-ipip-neo-for-lm-vector)
3. [Instrument Overview](#3-instrument-overview)
4. [Implementation Tasks](#4-implementation-tasks)
5. [Data Artifacts](#5-data-artifacts)
6. [Code Implementation](#6-code-implementation)
7. [Testing Strategy](#7-testing-strategy)
8. [Acceptance Criteria](#8-acceptance-criteria)
9. [Verification Commands](#9-verification-commands)

---

## 1. Executive Summary

### Goal
Implement an `IPIPNEOLoader` class that provides:
- Access to IPIP-NEO-120 questionnaire items (public domain Big Five)
- Scoring of responses to produce domain and facet scores
- Comparison capabilities against O*NET Work Styles → Big Five derivation
- Integration with the persona generation pipeline

### Scope
| In Scope | Out of Scope |
|----------|--------------|
| IPIP-NEO-120 (120 items, 30 facets) | IPIP-NEO-300 (full version, 300 items) |
| Built-in item database (public domain) | IPIP-NEO-60 (brief version) |
| Scoring logic with reverse-scoring | Percentile/norm table lookups |
| Big Five domain + facet scores | HuggingFace dataset integration |

### Key Advantage
**100% Public Domain** - No licensing restrictions, no author contact required, freely redistributable.

### Timeline Estimate
- **Data Preparation**: 1 hour (items already documented in PSYCHOMETRICS_DATA.md)
- **Loader Implementation**: 2-3 hours
- **Testing**: 2 hours
- **Documentation**: 1 hour

---

## 2. Why IPIP-NEO for LM-VECTOR

### Validation Against O*NET Big Five Derivation

We already derive Big Five scores from O*NET Work Styles (see [DESIGN_psychometrics_mapping.md](./DESIGN_psychometrics_mapping.md)). IPIP-NEO provides a **ground truth comparison**:

| Approach | Source | Method | Use Case |
|----------|--------|--------|----------|
| **O*NET-derived** | Work Styles | Theoretical mapping | Occupation-based personas |
| **IPIP-NEO** | Direct assessment | Item responses | Individual persona profiles |

### Facet-Level Granularity

IPIP-NEO provides **30 facets** (6 per domain), enabling fine-grained persona differentiation:

```
Example: Two "High Conscientiousness" personas can differ on facets:
  - Persona A: High Orderliness, Low Achievement-Striving (neat but unambitious)
  - Persona B: Low Orderliness, High Achievement-Striving (messy high-achiever)
```

### Public Domain Status

| Aspect | IPIP-NEO | HEXACO | NEO-PI-R (Commercial) |
|--------|----------|--------|----------------------|
| License | Public Domain | Academic only | Commercial license required |
| Redistribution | Yes | Restricted | No |
| Modification | Yes | Restricted | No |
| Citation | Recommended | Required | Required |

### Use Cases in LM-VECTOR

1. **Persona Generation**: Generate personas with specific Big Five profiles at facet level
2. **Validation**: Compare IPIP-NEO profiles against O*NET-derived Big Five scores
3. **Profile Interpolation**: Create gradient personas between extreme facet values
4. **Research Replication**: Use public domain items for reproducible research

---

## 3. Instrument Overview

### IPIP-NEO-120 Structure

```
IPIP-NEO-120 (5 domains × 6 facets × 4 items = 120 items)
│
├── N: Neuroticism
│   ├── N1: Anxiety (4 items)
│   ├── N2: Anger (4 items)
│   ├── N3: Depression (4 items)
│   ├── N4: Self-Consciousness (4 items)
│   ├── N5: Immoderation (4 items)
│   └── N6: Vulnerability (4 items)
│
├── E: Extraversion
│   ├── E1: Friendliness (4 items)
│   ├── E2: Gregariousness (4 items)
│   ├── E3: Assertiveness (4 items)
│   ├── E4: Activity Level (4 items)
│   ├── E5: Excitement-Seeking (4 items)
│   └── E6: Cheerfulness (4 items)
│
├── O: Openness to Experience
│   ├── O1: Imagination (4 items)
│   ├── O2: Artistic Interests (4 items)
│   ├── O3: Emotionality (4 items)
│   ├── O4: Adventurousness (4 items)
│   ├── O5: Intellect (4 items)
│   └── O6: Liberalism (4 items)
│
├── A: Agreeableness
│   ├── A1: Trust (4 items)
│   ├── A2: Morality (4 items)
│   ├── A3: Altruism (4 items)
│   ├── A4: Cooperation (4 items)
│   ├── A5: Modesty (4 items)
│   └── A6: Sympathy (4 items)
│
└── C: Conscientiousness
    ├── C1: Self-Efficacy (4 items)
    ├── C2: Orderliness (4 items)
    ├── C3: Dutifulness (4 items)
    ├── C4: Achievement-Striving (4 items)
    ├── C5: Self-Discipline (4 items)
    └── C6: Cautiousness (4 items)
```

### Comparison: IPIP-NEO vs NEO-PI-R Facet Names

| Domain | IPIP-NEO Facets | NEO-PI-R Equivalent |
|--------|-----------------|---------------------|
| N1 | Anxiety | Anxiety |
| N2 | Anger | Angry Hostility |
| N3 | Depression | Depression |
| N4 | Self-Consciousness | Self-Consciousness |
| N5 | Immoderation | Impulsiveness |
| N6 | Vulnerability | Vulnerability |
| E1 | Friendliness | Warmth |
| E2 | Gregariousness | Gregariousness |
| E3 | Assertiveness | Assertiveness |
| E4 | Activity Level | Activity |
| E5 | Excitement-Seeking | Excitement-Seeking |
| E6 | Cheerfulness | Positive Emotions |
| O1 | Imagination | Fantasy |
| O2 | Artistic Interests | Aesthetics |
| O3 | Emotionality | Feelings |
| O4 | Adventurousness | Actions |
| O5 | Intellect | Ideas |
| O6 | Liberalism | Values |
| A1 | Trust | Trust |
| A2 | Morality | Straightforwardness |
| A3 | Altruism | Altruism |
| A4 | Cooperation | Compliance |
| A5 | Modesty | Modesty |
| A6 | Sympathy | Tender-Mindedness |
| C1 | Self-Efficacy | Competence |
| C2 | Orderliness | Order |
| C3 | Dutifulness | Dutifulness |
| C4 | Achievement-Striving | Achievement Striving |
| C5 | Self-Discipline | Self-Discipline |
| C6 | Cautiousness | Deliberation |

### Response Scale

```
1 = Very Inaccurate
2 = Moderately Inaccurate
3 = Neither Accurate Nor Inaccurate
4 = Moderately Accurate
5 = Very Accurate
```

### Reverse Scoring

Approximately 40% of items are reverse-scored. The formula is:

```
reversed_score = 6 - original_score
```

For a 1-5 scale, this transforms: 1→5, 2→4, 3→3, 4→2, 5→1

---

## 4. Implementation Tasks

### Task 4.1: Create Data Directories

```bash
mkdir -p data/psychometrics/ipip_neo/{raw,items}
```

**Output Structure**:
```
data/psychometrics/ipip_neo/
├── items/
│   └── ipip_neo_120.json       # Item database with scoring key
└── README.md                   # Data source documentation
```

Note: Unlike HEXACO, no manual download is required - items are public domain and embedded in the loader.

---

### Task 4.2: Create Item Database JSON

The IPIP-NEO-120 items are already documented in [PSYCHOMETRICS_DATA.md](../reference/PSYCHOMETRICS_DATA.md). We will embed them directly in the codebase.

**File: `data/psychometrics/ipip_neo/items/ipip_neo_120.json`**

This file should contain:
- All 120 items with text
- Scoring key (domain, facet, reverse-scored flag)
- Response scale metadata
- Citation information

See [Section 5.2](#52-item-database-schema) for the full schema.

---

### Task 4.3: Build Scoring Key from Items

The scoring key is derived from item metadata:

```python
# Items are grouped by facet code (N1, N2, ..., C6)
# Each facet has exactly 4 items
# Reverse-scored items are flagged in the item data

IPIP_NEO_120_STRUCTURE = {
    "N": {
        "name": "Neuroticism",
        "facets": {
            "N1": {"name": "Anxiety", "items": [1, 2, 3, 4], "reverse": []}},
            "N2": {"name": "Anger", "items": [5, 6, 7, 8], "reverse": [8]},
            # ... etc
        }
    },
    # ... other domains
}
```

**Note**: Item numbers are 1-indexed and sequential within facet groups.

---

### Task 4.4: Document Data Sources

**File: `data/psychometrics/ipip_neo/README.md`**

```markdown
# IPIP-NEO-120 Data

## Source
International Personality Item Pool (IPIP)
https://ipip.ori.org/

## License
PUBLIC DOMAIN - No restrictions on use.

## Citation
Johnson, J. A. (2014). Measuring thirty facets of the Five Factor Model
with a 120-item public domain inventory: Development of the IPIP-NEO-120.
Journal of Research in Personality, 51, 78-89.

## Validation
- Sample size: 619,150 participants
- Cronbach's alpha: 0.75-0.89 across facets

## Items
All 120 items are embedded in `items/ipip_neo_120.json`.
Items are public domain and may be freely used, modified, and redistributed.
```

---

## 5. Data Artifacts

### 5.1 Required Files

| Artifact | Location | Format |
|----------|----------|--------|
| Item database | `data/psychometrics/ipip_neo/items/ipip_neo_120.json` | JSON |
| README | `data/psychometrics/ipip_neo/README.md` | Markdown |

### 5.2 Item Database Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["instrument", "version", "items", "structure", "response_scale"],
  "properties": {
    "instrument": {"type": "string"},
    "version": {"type": "string"},
    "source": {"type": "string"},
    "url": {"type": "string", "format": "uri"},
    "license": {"type": "string"},
    "citation": {"type": "string"},
    "validation_sample_size": {"type": "integer"},
    "response_scale": {
      "type": "object",
      "properties": {
        "type": {"type": "string"},
        "points": {"type": "integer"},
        "labels": {"type": "object"}
      }
    },
    "structure": {
      "type": "object",
      "description": "Domain and facet hierarchy"
    },
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["item_id", "text", "domain", "facet", "reverse_scored"],
        "properties": {
          "item_id": {"type": "integer", "minimum": 1, "maximum": 120},
          "text": {"type": "string", "minLength": 1},
          "domain": {"type": "string", "enum": ["N", "E", "O", "A", "C"]},
          "domain_name": {"type": "string"},
          "facet": {"type": "string", "pattern": "^[NEOAC][1-6]$"},
          "facet_name": {"type": "string"},
          "reverse_scored": {"type": "boolean"}
        }
      }
    }
  }
}
```

---

## 6. Code Implementation

### 6.1 File: `src/pvx/data/ipip_neo_loader.py`

```python
"""
IPIP-NEO Personality Inventory Loader

Provides access to the public domain IPIP-NEO-120 questionnaire items
and scoring logic for Big Five personality assessment.

Usage:
    from pvx.data.ipip_neo_loader import IPIPNEOLoader

    loader = IPIPNEOLoader()
    items = loader.get_items()
    scores = loader.score_responses({1: 4, 2: 3, ...})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

# Big Five domain metadata
BIG_FIVE_DOMAINS = {
    "N": {
        "name": "Neuroticism",
        "description": "Tendency to experience negative emotions",
        "facets": ["Anxiety", "Anger", "Depression", "Self-Consciousness",
                   "Immoderation", "Vulnerability"]
    },
    "E": {
        "name": "Extraversion",
        "description": "Tendency toward sociability and positive emotions",
        "facets": ["Friendliness", "Gregariousness", "Assertiveness",
                   "Activity Level", "Excitement-Seeking", "Cheerfulness"]
    },
    "O": {
        "name": "Openness to Experience",
        "description": "Tendency toward intellectual curiosity and creativity",
        "facets": ["Imagination", "Artistic Interests", "Emotionality",
                   "Adventurousness", "Intellect", "Liberalism"]
    },
    "A": {
        "name": "Agreeableness",
        "description": "Tendency toward cooperation and trust",
        "facets": ["Trust", "Morality", "Altruism", "Cooperation",
                   "Modesty", "Sympathy"]
    },
    "C": {
        "name": "Conscientiousness",
        "description": "Tendency toward self-discipline and organization",
        "facets": ["Self-Efficacy", "Orderliness", "Dutifulness",
                   "Achievement-Striving", "Self-Discipline", "Cautiousness"]
    }
}


class IPIPNEOLoader:
    """Loader for IPIP-NEO-120 personality inventory."""

    def __init__(self, data_dir: Path | str | None = None):
        """
        Initialize IPIP-NEO loader.

        Args:
            data_dir: Path to psychometrics data directory.
                      Defaults to data/psychometrics/ipip_neo/
        """
        if data_dir is None:
            # Assume running from project root
            self.data_dir = Path("data/psychometrics/ipip_neo")
        else:
            self.data_dir = Path(data_dir)

        self._items_cache: dict | None = None

    def _load_items(self) -> dict:
        """Load item database from JSON file."""
        if self._items_cache is not None:
            return self._items_cache

        filepath = self.data_dir / "items" / "ipip_neo_120.json"
        if not filepath.exists():
            raise FileNotFoundError(
                f"IPIP-NEO-120 items not found at {filepath}. "
                "Run the data preparation workflow first."
            )

        with open(filepath) as f:
            data = json.load(f)

        self._items_cache = data
        return data

    def get_items(self) -> list[dict]:
        """
        Get all questionnaire items.

        Returns:
            List of item dictionaries with keys:
            item_id, text, domain, domain_name, facet, facet_name, reverse_scored
        """
        data = self._load_items()
        return data.get("items", [])

    def get_item_text(self, item_id: int) -> str:
        """Get text for a specific item ID (1-120)."""
        items = self.get_items()
        for item in items:
            if item["item_id"] == item_id:
                return item["text"]
        raise ValueError(f"Item {item_id} not found in IPIP-NEO-120")

    def get_items_by_facet(self, facet: str) -> list[dict]:
        """
        Get all items for a specific facet.

        Args:
            facet: Facet code (e.g., "N1", "E2", "O5")

        Returns:
            List of item dictionaries for that facet
        """
        items = self.get_items()
        return [item for item in items if item["facet"] == facet]

    def get_items_by_domain(self, domain: str) -> list[dict]:
        """
        Get all items for a specific domain.

        Args:
            domain: Domain code (N, E, O, A, C)

        Returns:
            List of item dictionaries for that domain (24 items)
        """
        if domain not in BIG_FIVE_DOMAINS:
            raise ValueError(f"Unknown domain: {domain}. Valid: {list(BIG_FIVE_DOMAINS.keys())}")
        items = self.get_items()
        return [item for item in items if item["domain"] == domain]

    def get_domain_names(self) -> dict[str, str]:
        """Get mapping of domain codes to full names."""
        return {code: info["name"] for code, info in BIG_FIVE_DOMAINS.items()}

    def get_facets(self, domain: str) -> list[str]:
        """Get facet names for a domain."""
        if domain not in BIG_FIVE_DOMAINS:
            raise ValueError(f"Unknown domain: {domain}. Valid: {list(BIG_FIVE_DOMAINS.keys())}")
        return BIG_FIVE_DOMAINS[domain]["facets"]

    def get_facet_codes(self, domain: str) -> list[str]:
        """Get facet codes for a domain (e.g., ['N1', 'N2', ..., 'N6'])."""
        if domain not in BIG_FIVE_DOMAINS:
            raise ValueError(f"Unknown domain: {domain}")
        return [f"{domain}{i}" for i in range(1, 7)]

    @staticmethod
    def reverse_score(value: int, scale_max: int = 5) -> int:
        """
        Reverse score an item.

        For a 1-5 scale: 1→5, 2→4, 3→3, 4→2, 5→1
        """
        return scale_max + 1 - value

    def score_responses(
        self,
        responses: dict[int, int],
        method: Literal["sum", "mean"] = "mean"
    ) -> dict:
        """
        Score IPIP-NEO-120 responses.

        Args:
            responses: Dict mapping item_id (1-120) to response (1-5)
            method: "sum" for raw sums, "mean" for averages

        Returns:
            Dict with:
                - domains: {domain_code: score}
                - facets: {facet_code: score}
                - items_scored: number of items used
                - method: scoring method used
        """
        items = self.get_items()

        # Group scores by facet
        facet_scores: dict[str, list[int]] = {}

        for item in items:
            item_id = item["item_id"]
            if item_id not in responses:
                continue

            value = responses[item_id]
            if item["reverse_scored"]:
                value = self.reverse_score(value)

            facet = item["facet"]
            if facet not in facet_scores:
                facet_scores[facet] = []
            facet_scores[facet].append(value)

        # Calculate facet scores
        results = {
            "domains": {},
            "facets": {},
            "items_scored": sum(len(v) for v in facet_scores.values()),
            "method": method
        }

        for facet, scores in facet_scores.items():
            if method == "sum":
                results["facets"][facet] = sum(scores)
            else:
                results["facets"][facet] = round(sum(scores) / len(scores), 3)

        # Calculate domain scores
        for domain in BIG_FIVE_DOMAINS:
            domain_facets = [f"{domain}{i}" for i in range(1, 7)]
            facet_vals = [results["facets"][f] for f in domain_facets if f in results["facets"]]

            if facet_vals:
                if method == "sum":
                    results["domains"][domain] = sum(facet_vals)
                else:
                    results["domains"][domain] = round(sum(facet_vals) / len(facet_vals), 3)

        return results

    def compare_with_onet(
        self,
        ipip_scores: dict[str, float],
        onet_scores: dict[str, float]
    ) -> dict:
        """
        Compare IPIP-NEO domain scores with O*NET-derived Big Five scores.

        Args:
            ipip_scores: Domain scores from score_responses()["domains"]
            onet_scores: Big Five scores from ONETLoader.get_big_five_scores()

        Returns:
            Dict with correlation and per-domain differences
        """
        # Map O*NET keys to IPIP keys
        onet_to_ipip = {
            "openness": "O",
            "conscientiousness": "C",
            "extraversion": "E",
            "agreeableness": "A",
            "neuroticism": "N"
        }

        differences = {}
        for onet_key, ipip_key in onet_to_ipip.items():
            if onet_key in onet_scores and ipip_key in ipip_scores:
                # Normalize O*NET scores (0-100) to IPIP scale (1-5)
                onet_normalized = 1 + (onet_scores[onet_key] / 100) * 4
                diff = ipip_scores[ipip_key] - onet_normalized
                differences[ipip_key] = {
                    "ipip": ipip_scores[ipip_key],
                    "onet_normalized": round(onet_normalized, 3),
                    "difference": round(diff, 3)
                }

        return differences

    def get_response_scale(self) -> dict:
        """Get the response scale metadata."""
        data = self._load_items()
        return data.get("response_scale", {})

    def get_citation(self) -> str:
        """Get the citation for the instrument."""
        data = self._load_items()
        return data.get("citation", "")

    def get_license(self) -> str:
        """Get the license information."""
        return "PUBLIC DOMAIN - No restrictions on use"
```

### 6.2 Integration with `__init__.py`

**Update: `src/pvx/data/__init__.py`**

```python
from pvx.data.onet_loader import ONETLoader
from pvx.data.ipip_neo_loader import IPIPNEOLoader

__all__ = ["ONETLoader", "IPIPNEOLoader"]
```

---

## 7. Testing Strategy

### 7.1 Test File: `tests/unit/test_ipip_neo_loader.py`

```python
"""Unit tests for IPIPNEOLoader."""

import pytest
from pathlib import Path
import json

from pvx.data.ipip_neo_loader import IPIPNEOLoader, BIG_FIVE_DOMAINS


class TestIPIPNEOLoader:
    """Tests for IPIPNEOLoader class."""

    @pytest.fixture
    def loader(self, tmp_path: Path) -> IPIPNEOLoader:
        """Create loader with test data."""
        items_dir = tmp_path / "items"
        items_dir.mkdir()

        # Create minimal test data with all 120 items
        test_items = []
        item_id = 1
        for domain in ["N", "E", "O", "A", "C"]:
            for facet_num in range(1, 7):
                for i in range(4):
                    test_items.append({
                        "item_id": item_id,
                        "text": f"Test item {item_id}",
                        "domain": domain,
                        "domain_name": BIG_FIVE_DOMAINS[domain]["name"],
                        "facet": f"{domain}{facet_num}",
                        "facet_name": BIG_FIVE_DOMAINS[domain]["facets"][facet_num - 1],
                        "reverse_scored": (i == 3)  # Last item of each facet is reversed
                    })
                    item_id += 1

        test_data = {
            "instrument": "IPIP-NEO-120",
            "version": "120-item",
            "license": "PUBLIC DOMAIN",
            "response_scale": {
                "type": "likert",
                "points": 5,
                "labels": {"1": "Very Inaccurate", "5": "Very Accurate"}
            },
            "items": test_items
        }

        with open(items_dir / "ipip_neo_120.json", "w") as f:
            json.dump(test_data, f)

        return IPIPNEOLoader(data_dir=tmp_path)

    def test_get_items_count(self, loader: IPIPNEOLoader):
        """Test that we get all 120 items."""
        items = loader.get_items()
        assert len(items) == 120

    def test_get_items_structure(self, loader: IPIPNEOLoader):
        """Test item structure has required fields."""
        items = loader.get_items()
        required_fields = ["item_id", "text", "domain", "facet", "reverse_scored"]
        for item in items:
            for field in required_fields:
                assert field in item

    def test_get_item_text(self, loader: IPIPNEOLoader):
        """Test getting specific item text."""
        text = loader.get_item_text(1)
        assert text == "Test item 1"

    def test_get_item_text_not_found(self, loader: IPIPNEOLoader):
        """Test error for invalid item number."""
        with pytest.raises(ValueError, match="Item 999 not found"):
            loader.get_item_text(999)

    def test_get_items_by_domain(self, loader: IPIPNEOLoader):
        """Test getting items for a domain."""
        items = loader.get_items_by_domain("N")
        assert len(items) == 24  # 6 facets × 4 items
        assert all(item["domain"] == "N" for item in items)

    def test_get_items_by_domain_invalid(self, loader: IPIPNEOLoader):
        """Test error for invalid domain."""
        with pytest.raises(ValueError, match="Unknown domain"):
            loader.get_items_by_domain("X")

    def test_get_items_by_facet(self, loader: IPIPNEOLoader):
        """Test getting items for a facet."""
        items = loader.get_items_by_facet("N1")
        assert len(items) == 4
        assert all(item["facet"] == "N1" for item in items)

    def test_get_domain_names(self, loader: IPIPNEOLoader):
        """Test domain name mapping."""
        names = loader.get_domain_names()
        assert names["N"] == "Neuroticism"
        assert names["E"] == "Extraversion"
        assert names["O"] == "Openness to Experience"
        assert names["A"] == "Agreeableness"
        assert names["C"] == "Conscientiousness"
        assert len(names) == 5

    def test_get_facets(self, loader: IPIPNEOLoader):
        """Test getting facets for a domain."""
        facets = loader.get_facets("N")
        assert len(facets) == 6
        assert "Anxiety" in facets
        assert "Depression" in facets

    def test_get_facet_codes(self, loader: IPIPNEOLoader):
        """Test getting facet codes."""
        codes = loader.get_facet_codes("E")
        assert codes == ["E1", "E2", "E3", "E4", "E5", "E6"]

    def test_reverse_score(self, loader: IPIPNEOLoader):
        """Test reverse scoring logic."""
        assert loader.reverse_score(1) == 5
        assert loader.reverse_score(2) == 4
        assert loader.reverse_score(3) == 3
        assert loader.reverse_score(4) == 2
        assert loader.reverse_score(5) == 1

    def test_score_responses_all_neutral(self, loader: IPIPNEOLoader):
        """Test scoring with neutral responses."""
        responses = {i: 3 for i in range(1, 121)}
        scores = loader.score_responses(responses)

        assert "domains" in scores
        assert "facets" in scores
        assert "items_scored" in scores
        assert scores["items_scored"] == 120

        # All neutral should give ~3.0 for all domains
        for domain, score in scores["domains"].items():
            assert 2.9 <= score <= 3.1

    def test_score_responses_mean_vs_sum(self, loader: IPIPNEOLoader):
        """Test mean vs sum scoring methods."""
        responses = {i: 4 for i in range(1, 121)}

        mean_scores = loader.score_responses(responses, method="mean")
        sum_scores = loader.score_responses(responses, method="sum")

        # Mean should be ~4 (except for reversed items which become 2)
        # Sum should be much larger
        for domain in ["N", "E", "O", "A", "C"]:
            assert mean_scores["domains"][domain] < sum_scores["domains"][domain]

    def test_score_responses_with_reverse_scoring(self, loader: IPIPNEOLoader):
        """Test that reverse scoring is applied correctly."""
        # Item 4 is reverse-scored (last in first facet)
        # Give it 1, should become 5 after reverse
        responses = {4: 1}  # Only answer one reverse-scored item

        scores = loader.score_responses(responses)

        # N1 facet should have high score (5 after reversal)
        assert scores["facets"]["N1"] == 5.0

    def test_score_responses_partial(self, loader: IPIPNEOLoader):
        """Test scoring with partial responses."""
        responses = {1: 4, 2: 3, 3: 5}  # Only 3 items
        scores = loader.score_responses(responses)

        assert scores["items_scored"] == 3
        assert "N1" in scores["facets"]

    def test_get_license(self, loader: IPIPNEOLoader):
        """Test license returns public domain."""
        license_text = loader.get_license()
        assert "PUBLIC DOMAIN" in license_text


class TestIPIPNEOFacetStructure:
    """Tests verifying IPIP-NEO facet structure correctness."""

    def test_all_domains_have_six_facets(self):
        """Verify each domain has exactly 6 facets."""
        for domain, info in BIG_FIVE_DOMAINS.items():
            assert len(info["facets"]) == 6, f"Domain {domain} should have 6 facets"

    def test_domain_descriptions_exist(self):
        """Verify all domains have descriptions."""
        for domain, info in BIG_FIVE_DOMAINS.items():
            assert "description" in info
            assert len(info["description"]) > 10


class TestIPIPNEOCompareWithONET:
    """Tests for O*NET comparison functionality."""

    @pytest.fixture
    def loader(self, tmp_path: Path) -> IPIPNEOLoader:
        """Create loader with minimal test data."""
        items_dir = tmp_path / "items"
        items_dir.mkdir()

        test_items = [
            {"item_id": i, "text": f"Item {i}", "domain": "N",
             "facet": "N1", "reverse_scored": False}
            for i in range(1, 5)
        ]

        test_data = {"instrument": "IPIP-NEO-120", "items": test_items}
        with open(items_dir / "ipip_neo_120.json", "w") as f:
            json.dump(test_data, f)

        return IPIPNEOLoader(data_dir=tmp_path)

    def test_compare_with_onet(self, loader: IPIPNEOLoader):
        """Test comparison with O*NET scores."""
        ipip_scores = {"N": 3.0, "E": 4.0, "O": 2.5, "A": 3.5, "C": 4.5}
        onet_scores = {
            "neuroticism": 50,  # Should normalize to 3.0
            "extraversion": 75,  # Should normalize to 4.0
            "openness": 37.5,  # Should normalize to 2.5
            "agreeableness": 62.5,  # Should normalize to 3.5
            "conscientiousness": 87.5  # Should normalize to 4.5
        }

        comparison = loader.compare_with_onet(ipip_scores, onet_scores)

        assert len(comparison) == 5
        # When scores match, difference should be ~0
        for domain, data in comparison.items():
            assert abs(data["difference"]) < 0.01
```

### 7.2 Test Commands

```bash
# Run all IPIP-NEO tests
uv run pytest tests/unit/test_ipip_neo_loader.py -v

# Run with coverage
uv run pytest tests/unit/test_ipip_neo_loader.py --cov=pvx.data.ipip_neo_loader --cov-report=term-missing

# Run specific test
uv run pytest tests/unit/test_ipip_neo_loader.py::TestIPIPNEOLoader::test_score_responses_with_reverse_scoring -v
```

---

## 8. Acceptance Criteria

### 8.1 Data Artifacts

- [ ] `data/psychometrics/ipip_neo/items/ipip_neo_120.json` contains all 120 items
- [ ] JSON file includes item text, domain, facet, and reverse_scored flag
- [ ] JSON file passes schema validation
- [ ] README documents data source and license

### 8.2 Code Quality

- [ ] `IPIPNEOLoader` class implemented in `src/pvx/data/ipip_neo_loader.py`
- [ ] All public methods have docstrings
- [ ] Type hints on all parameters and return types
- [ ] No `ty` type checker errors

### 8.3 Functionality

- [ ] `get_items()` returns 120 items
- [ ] `score_responses()` produces valid domain scores (1.0-5.0 range for mean method)
- [ ] Reverse scoring correctly transforms items
- [ ] Both "sum" and "mean" scoring methods work
- [ ] `compare_with_onet()` normalizes and compares scores

### 8.4 Testing

- [ ] Unit tests cover all public methods
- [ ] Tests for edge cases (partial responses, invalid inputs)
- [ ] All tests pass
- [ ] Test coverage > 80%

### 8.5 Integration

- [ ] Loader exported from `pvx.data` package
- [ ] Works alongside existing `ONETLoader`

---

## 9. Verification Commands

### Final Verification Script

```bash
#!/bin/bash
echo "=== IPIP-NEO Phase 3 Verification ==="

echo -e "\n--- Data Files ---"
ls -la data/psychometrics/ipip_neo/
ls -la data/psychometrics/ipip_neo/items/

echo -e "\n--- Item Count Verification ---"
python3 -c "
import json
with open('data/psychometrics/ipip_neo/items/ipip_neo_120.json') as f:
    data = json.load(f)
    print(f'IPIP-NEO-120: {len(data[\"items\"])} items')

    # Count by domain
    domains = {}
    for item in data['items']:
        d = item['domain']
        domains[d] = domains.get(d, 0) + 1
    print(f'Items per domain: {domains}')

    # Count reverse-scored
    reverse_count = sum(1 for item in data['items'] if item['reverse_scored'])
    print(f'Reverse-scored items: {reverse_count}')
"

echo -e "\n--- Loader Test ---"
python3 -c "
from pvx.data.ipip_neo_loader import IPIPNEOLoader

loader = IPIPNEOLoader()
items = loader.get_items()
print(f'Items loaded: {len(items)}')

# Test scoring
responses = {i: 3 for i in range(1, 121)}
scores = loader.score_responses(responses)
print(f'Domain scores: {scores[\"domains\"]}')

# Show license
print(f'License: {loader.get_license()}')
"

echo -e "\n--- Unit Tests ---"
uv run pytest tests/unit/test_ipip_neo_loader.py -v --tb=short

echo -e "\n--- Type Check ---"
uv run ty check src/pvx/data/ipip_neo_loader.py

echo -e "\n=== Verification Complete ==="
```

---

## Related Documents

- [PSYCHOMETRICS_DATA.md § Phase 3](../reference/PSYCHOMETRICS_DATA.md#phase-3-ipip-neo-big-five) - Reference implementation
- [STATUS_psychometrics.md](./STATUS_psychometrics.md) - Progress tracking
- [DESIGN_psychometrics_mapping.md](./DESIGN_psychometrics_mapping.md) - Big Five mapping from O*NET
- [WORKFLOW_phase2_hexaco.md](./WORKFLOW_phase2_hexaco.md) - Related workflow (HEXACO)
- [onet_loader.py](../../src/pvx/data/onet_loader.py) - Pattern reference

---

## Appendix A: Complete IPIP-NEO-120 Items

The full 120 items with reverse-scoring flags are documented in [PSYCHOMETRICS_DATA.md](../reference/PSYCHOMETRICS_DATA.md#32-parse-ipip-neo-120-items).

Key statistics:
- **Total items**: 120
- **Items per domain**: 24
- **Items per facet**: 4
- **Reverse-scored items**: ~40% (varies by facet)

---

## Appendix B: Facet-Level Reverse Scoring Summary

| Facet | Items | Reverse-Scored Items |
|-------|-------|---------------------|
| N1: Anxiety | 4 | 0 |
| N2: Anger | 4 | 1 (item 8) |
| N3: Depression | 4 | 1 (item 12) |
| N4: Self-Consciousness | 4 | 1 (item 16) |
| N5: Immoderation | 4 | 2 (items 19, 20) |
| N6: Vulnerability | 4 | 1 (item 24) |
| E1: Friendliness | 4 | 1 (item 28) |
| E2: Gregariousness | 4 | 2 (items 31, 32) |
| E3: Assertiveness | 4 | 1 (item 36) |
| E4: Activity Level | 4 | 1 (item 40) |
| E5: Excitement-Seeking | 4 | 0 |
| E6: Cheerfulness | 4 | 0 |
| O1: Imagination | 4 | 0 |
| O2: Artistic Interests | 4 | 1 (item 52) |
| O3: Emotionality | 4 | 1 (item 56) |
| O4: Adventurousness | 4 | 1 (item 60) |
| O5: Intellect | 4 | 3 (items 62, 63, 64) |
| O6: Liberalism | 4 | 2 (items 67, 68) |
| A1: Trust | 4 | 1 (item 72) |
| A2: Morality | 4 | 2 (items 75, 76) |
| A3: Altruism | 4 | 0 |
| A4: Cooperation | 4 | 1 (item 84) |
| A5: Modesty | 4 | 1 (item 88) |
| A6: Sympathy | 4 | 1 (item 92) |
| C1: Self-Efficacy | 4 | 0 |
| C2: Orderliness | 4 | 1 (item 100) |
| C3: Dutifulness | 4 | 0 |
| C4: Achievement-Striving | 4 | 1 (item 108) |
| C5: Self-Discipline | 4 | 1 (item 112) |
| C6: Cautiousness | 4 | 1 (item 116) |

**Note**: Item numbers in this table are approximate and based on sequential ordering. Verify against the authoritative item list in the JSON database.
